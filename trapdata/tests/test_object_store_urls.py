"""
Tests for where model weights and sample trap images are downloaded from.

The model_base_url and image_base_url settings (AMI_MODEL_BASE_URL and
AMI_IMAGE_BASE_URL) decide where downloads come from. These tests pin that the settings
can be overridden from the environment or a ".env" file, that a value which would
produce broken or tamperable download URLs is rejected, that model paths are joined to
the configured base URL when they are resolved, that a file already in the model cache
is used as-is, and that no module outside the settings writes out the object store
host in full, which would bypass a deployment's override.
"""

import pathlib

import pytest
from pydantic import ValidationError

import trapdata.settings
from trapdata.ml.utils import get_or_download_file, resolve_model_url
from trapdata.settings import DEFAULT_OBJECT_STORE_URL, Settings, read_settings

PACKAGE_DIR = pathlib.Path(trapdata.settings.__file__).parent
# The only files allowed to spell out the object store host: the settings module, where
# the default lives, and this test. Anywhere else it would bypass a deployment's override.
HOST_ALLOWED_IN = {PACKAGE_DIR / "settings.py", pathlib.Path(__file__).resolve()}
SOURCE_FILES = sorted(
    path for path in PACKAGE_DIR.rglob("*.py") if path.resolve() not in HOST_ALLOWED_IN
)
OTHER_STORE = "https://models.example.org/ami-models/"


def make_settings(env_file=None) -> Settings:
    """Build settings from the environment, and from env_file if one is given."""
    return Settings(_env_file=env_file)


@pytest.fixture
def fresh_settings_cache():
    """Make read_settings pick up environment changes made by the test."""
    read_settings.cache_clear()
    yield
    read_settings.cache_clear()


def test_defaults_point_at_public_object_store(monkeypatch):
    monkeypatch.delenv("AMI_MODEL_BASE_URL", raising=False)
    monkeypatch.delenv("AMI_IMAGE_BASE_URL", raising=False)
    settings = make_settings()
    assert settings.model_base_url == f"{DEFAULT_OBJECT_STORE_URL}ami-models/"
    assert settings.image_base_url == f"{DEFAULT_OBJECT_STORE_URL}ami-trapdata/"


def test_environment_overrides_default(monkeypatch):
    monkeypatch.setenv("AMI_MODEL_BASE_URL", OTHER_STORE)
    assert make_settings().model_base_url == OTHER_STORE


def test_env_file_overrides_default(monkeypatch, tmp_path):
    """An override in ".env" applies, as it does for every other AMI_ setting."""
    monkeypatch.delenv("AMI_MODEL_BASE_URL", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text(f"AMI_MODEL_BASE_URL={OTHER_STORE}\n")
    assert make_settings(env_file).model_base_url == OTHER_STORE


def test_missing_trailing_slash_is_added(monkeypatch):
    monkeypatch.setenv("AMI_MODEL_BASE_URL", OTHER_STORE.rstrip("/"))
    assert make_settings().model_base_url == OTHER_STORE


def test_empty_value_is_rejected(monkeypatch):
    monkeypatch.setenv("AMI_MODEL_BASE_URL", "  ")
    with pytest.raises(ValidationError):
        make_settings()


@pytest.mark.parametrize("env_var", ["AMI_MODEL_BASE_URL", "AMI_IMAGE_BASE_URL"])
@pytest.mark.parametrize(
    "value",
    [
        "http://models.example.org/ami-models/",
        "ftp://models.example.org/",
        "/srv/models/",
    ],
)
def test_non_https_remote_url_is_rejected(monkeypatch, env_var, value):
    """Any remote store must be reached over HTTPS, so downloads cannot be swapped."""
    monkeypatch.setenv(env_var, value)
    with pytest.raises(ValidationError):
        make_settings()


@pytest.mark.parametrize(
    "value",
    [
        "http://localhost:9000/ami-models/",
        "http://127.0.0.1:9000/ami-models",
        "http://[::1]:9000/ami-models/",
    ],
)
def test_plain_http_is_accepted_for_a_loopback_host(monkeypatch, value):
    """A local object store during development may use plain HTTP."""
    monkeypatch.setenv("AMI_MODEL_BASE_URL", value)
    assert make_settings().model_base_url == value.rstrip("/") + "/"


@pytest.mark.parametrize(
    "value",
    [
        "https:///ami-models/",
        "https://models.example.org/ami-models/?token=abc",
        "https://models.example.org/ami-models/#models",
    ],
)
def test_url_without_host_or_with_query_is_rejected(monkeypatch, value):
    """File paths are appended to the base URL, so it must be a plain host and path."""
    monkeypatch.setenv("AMI_MODEL_BASE_URL", value)
    with pytest.raises(ValidationError):
        make_settings()


def test_relative_model_path_is_joined_to_base_url():
    path = "moths/classification/weights.pth"
    assert resolve_model_url(path, base_url=OTHER_STORE) == f"{OTHER_STORE}{path}"


@pytest.mark.parametrize(
    "path",
    [
        "https://elsewhere.example.org/models/weights.pth",
        # A pre-signed URL carries its credentials in the query string, which must survive.
        "https://elsewhere.example.org/models/weights.pth?X-Amz-Signature=abc&X-Amz-Expires=1",
        "http://localhost:9000/models/weights.pth",
        "/srv/models/weights.pth",
        None,
    ],
)
def test_full_url_absolute_path_or_none_is_unchanged(path):
    """Files hosted elsewhere or placed on disk by hand are used as given."""
    assert resolve_model_url(path, base_url=OTHER_STORE) == path


@pytest.mark.parametrize(
    "path",
    [
        "http://elsewhere.example.org/models/weights.pth",
        "ftp://elsewhere.example.org/models/weights.pth",
        # A scheme with no host is not a URL the downloader can fetch; it must not fall
        # through and be treated as a local path.
        "https:weights.pth",
    ],
)
def test_insecure_or_malformed_full_url_in_model_path_is_rejected(path):
    """The HTTPS rule for the base URL applies to a model that names a full URL too."""
    with pytest.raises(ValueError):
        resolve_model_url(path, base_url=OTHER_STORE)


@pytest.mark.usefixtures("fresh_settings_cache")
def test_resolution_uses_configured_base_url(monkeypatch):
    monkeypatch.setenv("AMI_MODEL_BASE_URL", OTHER_STORE)
    resolved = resolve_model_url("insect_orders/map.json")
    assert resolved == f"{OTHER_STORE}insect_orders/map.json"


def test_model_classes_resolve_against_the_configured_store():
    """Model classes keep relative paths, so the override reaches every model."""
    from trapdata.ml.models.classification import MothNonMothClassifier

    for path in (MothNonMothClassifier.weights_path, MothNonMothClassifier.labels_path):
        resolved = resolve_model_url(path, base_url=OTHER_STORE)
        assert resolved is not None and resolved.startswith(OTHER_STORE)


def test_cached_file_named_after_the_url_is_used_without_downloading(
    tmp_path, monkeypatch
):
    """
    A file placed in the model cache by hand is used as long as it is named exactly as
    the last part of its URL. No request is made for it.
    """

    def refuse(*args, **kwargs):
        raise AssertionError("a cached file must not be downloaded again")

    monkeypatch.setattr("trapdata.ml.utils.requests.get", refuse)
    cache_dir = tmp_path / "models"
    cache_dir.mkdir()
    (cache_dir / "weights.pth").write_bytes(b"weights")

    url = resolve_model_url("moths/classification/weights.pth", base_url=OTHER_STORE)
    local_path = get_or_download_file(url, tmp_path, prefix="models")

    assert local_path == cache_dir / "weights.pth"


class FakeResponse:
    """Stand-in for a streamed requests.Response, with the redirect chain it followed."""

    def __init__(self, url, history=()):
        self.url = url
        self.history = [FakeResponse(hop) for hop in history]
        self.closed = False

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size):
        yield b"weights"

    def close(self):
        self.closed = True


def test_model_file_is_downloaded_from_the_configured_store_into_the_cache(
    tmp_path, monkeypatch
):
    """The joined URL is what gets requested, and the file lands in the models cache."""
    requested = []

    def fake_get(url, **kwargs):
        requested.append(url)
        return FakeResponse(url)

    monkeypatch.setattr("trapdata.ml.utils.requests.get", fake_get)

    url = resolve_model_url("moths/classification/weights.pth", base_url=OTHER_STORE)
    local_path = get_or_download_file(url, tmp_path, prefix="models")

    assert requested == [f"{OTHER_STORE}moths/classification/weights.pth"]
    assert local_path == tmp_path / "models" / "weights.pth"
    assert local_path.read_bytes() == b"weights"


def test_download_that_redirects_to_plain_http_is_refused(tmp_path, monkeypatch):
    """
    An HTTPS URL that redirects to plain HTTP loses the protection HTTPS gives against
    a swapped file, so the download is refused and nothing is written to the cache.
    """
    url = f"{OTHER_STORE}moths/classification/weights.pth"
    downgraded = "http://elsewhere.example.org/moths/classification/weights.pth"
    response = FakeResponse(downgraded, history=[url])
    monkeypatch.setattr("trapdata.ml.utils.requests.get", lambda *a, **kw: response)

    with pytest.raises(ValueError, match="redirected to a plain http://"):
        get_or_download_file(url, tmp_path, prefix="models")

    assert response.closed
    assert not (tmp_path / "models" / "weights.pth").exists()


def test_download_that_redirects_within_https_is_accepted(tmp_path, monkeypatch):
    """A redirect between HTTPS URLs, as object stores and Hugging Face do, is fine."""
    url = f"{OTHER_STORE}moths/classification/weights.pth"
    response = FakeResponse("https://cdn.example.org/weights.pth", history=[url])
    monkeypatch.setattr("trapdata.ml.utils.requests.get", lambda *a, **kw: response)

    local_path = get_or_download_file(url, tmp_path, prefix="models")

    assert local_path.read_bytes() == b"weights"


@pytest.mark.parametrize(
    "source_path", SOURCE_FILES, ids=lambda p: str(p.relative_to(PACKAGE_DIR))
)
def test_no_module_hardcodes_the_object_store_host(source_path):
    """Writing out the host anywhere but the settings would bypass a deployment's override."""
    assert "object-arbutus" not in source_path.read_text()
