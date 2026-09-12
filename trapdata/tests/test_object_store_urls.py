"""
Tests for where model weights and sample trap images are downloaded from.

Deployments can override the download location with AMI_MODEL_BASE_URL and
AMI_IMAGE_BASE_URL, set either in the environment or in a ".env" file. These tests pin
that both sources are honoured, that a missing trailing slash cannot produce a malformed
URL, that model weights are never fetched over plain HTTP from a remote host, and that no
model module writes out an object store host in full, which would bypass the override for
that model.
"""

import pathlib

import pytest
from pydantic import ValidationError

from trapdata.common import constants
from trapdata.common.constants import ObjectStoreSettings

MODEL_MODULES = [
    pathlib.Path(constants.__file__).parents[1] / "ml" / "models" / name
    for name in ("classification.py", "localization.py")
]


def test_defaults_point_at_public_object_store(monkeypatch):
    monkeypatch.delenv("AMI_MODEL_BASE_URL", raising=False)
    monkeypatch.delenv("AMI_IMAGE_BASE_URL", raising=False)
    settings = ObjectStoreSettings(_env_file=None)
    assert settings.model_base_url == f"{constants.OBJECT_STORE_BASE_URL}ami-models/"
    assert settings.image_base_url == f"{constants.OBJECT_STORE_BASE_URL}ami-trapdata/"


def test_environment_overrides_default(monkeypatch):
    monkeypatch.setenv("AMI_MODEL_BASE_URL", "https://mirror.example.org/models/")
    settings = ObjectStoreSettings(_env_file=None)
    assert settings.model_base_url == "https://mirror.example.org/models/"


def test_env_file_overrides_default(monkeypatch, tmp_path):
    """An override in ".env" applies, as it does for every other AMI_ setting."""
    monkeypatch.delenv("AMI_MODEL_BASE_URL", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("AMI_MODEL_BASE_URL=https://mirror.example.org/models/\n")
    settings = ObjectStoreSettings(_env_file=env_file)
    assert settings.model_base_url == "https://mirror.example.org/models/"


def test_missing_trailing_slash_is_added(monkeypatch):
    monkeypatch.setenv("AMI_MODEL_BASE_URL", "https://mirror.example.org/models")
    settings = ObjectStoreSettings(_env_file=None)
    assert settings.model_base_url == "https://mirror.example.org/models/"


def test_empty_value_is_rejected(monkeypatch):
    monkeypatch.setenv("AMI_MODEL_BASE_URL", "  ")
    with pytest.raises(ValidationError):
        ObjectStoreSettings(_env_file=None)


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
        ObjectStoreSettings(_env_file=None)


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
    settings = ObjectStoreSettings(_env_file=None)
    assert settings.model_base_url == value.rstrip("/") + "/"


@pytest.mark.parametrize("module_path", MODEL_MODULES, ids=lambda p: p.name)
def test_model_modules_do_not_hardcode_the_host(module_path):
    """Writing out the host in a model module would bypass a deployment's override."""
    assert "object-arbutus" not in module_path.read_text()
