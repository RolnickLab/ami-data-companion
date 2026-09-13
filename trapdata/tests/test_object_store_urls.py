"""
Tests for where model weights and sample trap images are downloaded from.

The model_base_url and image_base_url settings (AMI_MODEL_BASE_URL and
AMI_IMAGE_BASE_URL) decide where downloads come from. These tests pin that the settings
can be overridden from the environment or a ".env" file, that a value which would
produce broken or tamperable download URLs is rejected, that model paths are joined to
the configured base URL when they are resolved, and that no model module writes out an
object store host in full, which would bypass a deployment's override.
"""

import pathlib

import pytest
from pydantic import ValidationError

import trapdata.settings
from trapdata.ml.utils import resolve_model_url
from trapdata.settings import DEFAULT_OBJECT_STORE_URL, Settings, read_settings

PACKAGE_DIR = pathlib.Path(trapdata.settings.__file__).parent
MODEL_MODULES = [
    PACKAGE_DIR / "ml" / "models" / name
    for name in ("classification.py", "localization.py")
]
MIRROR = "https://mirror.example.org/ami-models/"


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
    monkeypatch.setenv("AMI_MODEL_BASE_URL", MIRROR)
    assert make_settings().model_base_url == MIRROR


def test_env_file_overrides_default(monkeypatch, tmp_path):
    """An override in ".env" applies, as it does for every other AMI_ setting."""
    monkeypatch.delenv("AMI_MODEL_BASE_URL", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text(f"AMI_MODEL_BASE_URL={MIRROR}\n")
    assert make_settings(env_file).model_base_url == MIRROR


def test_missing_trailing_slash_is_added(monkeypatch):
    monkeypatch.setenv("AMI_MODEL_BASE_URL", MIRROR.rstrip("/"))
    assert make_settings().model_base_url == MIRROR


def test_empty_value_is_rejected(monkeypatch):
    monkeypatch.setenv("AMI_MODEL_BASE_URL", "  ")
    with pytest.raises(ValidationError):
        make_settings()


@pytest.mark.parametrize("env_var", ["AMI_MODEL_BASE_URL", "AMI_IMAGE_BASE_URL"])
@pytest.mark.parametrize(
    "value",
    [
        "http://mirror.example.org/ami-models/",
        "ftp://mirror.example.org/",
        "/srv/models/",
    ],
)
def test_non_https_remote_url_is_rejected(monkeypatch, env_var, value):
    """Weights are unpickled on load, so a remote mirror must be reached over HTTPS."""
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
        "https://mirror.example.org/ami-models/?token=abc",
        "https://mirror.example.org/ami-models/#models",
    ],
)
def test_url_without_host_or_with_query_is_rejected(monkeypatch, value):
    """File paths are appended to the base URL, so it must be a plain host and path."""
    monkeypatch.setenv("AMI_MODEL_BASE_URL", value)
    with pytest.raises(ValidationError):
        make_settings()


def test_relative_model_path_is_joined_to_base_url():
    path = "moths/classification/weights.pth"
    assert resolve_model_url(path, base_url=MIRROR) == f"{MIRROR}{path}"


@pytest.mark.parametrize(
    "path",
    [
        "https://elsewhere.example.org/models/weights.pth",
        "/srv/models/weights.pth",
        None,
    ],
)
def test_full_url_absolute_path_or_none_is_unchanged(path):
    """Files hosted elsewhere or placed on disk by hand are used as given."""
    assert resolve_model_url(path, base_url=MIRROR) == path


@pytest.mark.usefixtures("fresh_settings_cache")
def test_resolution_uses_configured_base_url(monkeypatch):
    monkeypatch.setenv("AMI_MODEL_BASE_URL", MIRROR)
    resolved = resolve_model_url("insect_orders/map.json")
    assert resolved == f"{MIRROR}insect_orders/map.json"


def test_model_classes_resolve_against_the_configured_store():
    """Model classes keep relative paths, so the override reaches every model."""
    from trapdata.ml.models.classification import MothNonMothClassifier

    for path in (MothNonMothClassifier.weights_path, MothNonMothClassifier.labels_path):
        resolved = resolve_model_url(path, base_url=MIRROR)
        assert resolved is not None and resolved.startswith(MIRROR)


@pytest.mark.parametrize("module_path", MODEL_MODULES, ids=lambda p: p.name)
def test_model_modules_do_not_hardcode_the_host(module_path):
    """Writing out the host in a model module would bypass a deployment's override."""
    assert "object-arbutus" not in module_path.read_text()
