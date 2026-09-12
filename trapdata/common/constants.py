from urllib.parse import urlparse

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

SUPPORTED_IMAGE_EXTENSIONS = (".jpg", ".jpeg")

POSITIVE_BINARY_LABEL = "moth"
NEGATIVE_BINARY_LABEL = "nonmoth"
NULL_DETECTION_LABELS = [NEGATIVE_BINARY_LABEL]
TRACKING_COST_THRESHOLD = 1.0

POSITIVE_COLOR = [0, 100 / 255, 1, 1]  # Blue
# POSITIVE_COLOR = [1, 0, 162 / 255, 1]  # Pink
# NEUTRAL_COLOR = [1, 1, 1, 0.5]  # White
# NEUTRAL_COLOR = [1, 0, 162 / 255, 0.2]  # Pink, semi-transparent
NEUTRAL_COLOR = [0, 100 / 255, 1, 0.4]  # Blue
NEGATIVE_COLOR = [1, 1, 1, 0]  # Transparent

SUMMARY_REFRESH_SECONDS = 5

# Public object store that holds the model weights, label maps and sample trap images.
# The Swift path form is used because the equivalent S3 path form puts a "<tenant>:"
# prefix on the bucket name, and the colon trips some URL parsers and caches.
OBJECT_STORE_BASE_URL = (
    "https://object-arbutus.alliancecan.ca/swift/v1/"
    "AUTH_3c987b8fc90743469d42899b1fdb48eb/"
)

# Hosts where a plain http:// base URL is accepted, such as a local object store used
# during development. There is no network path to tamper with on a loopback address.
_LOOPBACK_HOSTS = ("localhost", "127.0.0.1", "::1")


class ObjectStoreSettings(BaseSettings):
    """
    Where model weights and sample trap images are downloaded from.

    Both locations default to the public object store and can be overridden per
    deployment, for example to serve models from a mirror close to a compute cluster.
    Values are read from the environment and from the ".env" file with the same "AMI_"
    prefix as the other settings, so an override placed in ".env" takes effect. They are
    resolved once at import time, because model classes build their download URLs as
    class attributes.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="ami_",
        extra="ignore",
        protected_namespaces=(),
    )

    model_base_url: str = f"{OBJECT_STORE_BASE_URL}ami-models/"
    image_base_url: str = f"{OBJECT_STORE_BASE_URL}ami-trapdata/"

    @field_validator("model_base_url", "image_base_url")
    @classmethod
    def validate_base_url(cls, value: str) -> str:
        """
        Require HTTPS, and accept a base URL with or without a trailing slash.

        Model weights are unpickled by torch.load, so fetching them over plain HTTP would
        let anyone on the network path substitute a file that runs code when loaded.
        Plain HTTP is accepted only for a loopback host. The trailing slash is added when
        missing because file paths are appended to the base URL directly.
        """
        value = value.strip()
        if not value:
            raise ValueError("must not be empty")
        parsed = urlparse(value)
        is_local_http = parsed.scheme == "http" and parsed.hostname in _LOOPBACK_HOSTS
        if parsed.scheme != "https" and not is_local_http:
            raise ValueError(
                "must be an https:// URL (plain http:// is accepted only for localhost), "
                f"got {value!r}"
            )
        return value if value.endswith("/") else f"{value}/"


_object_store = ObjectStoreSettings()
MODEL_BASE_URL = _object_store.model_base_url
IMAGE_BASE_URL = _object_store.image_base_url
