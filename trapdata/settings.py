import configparser
import pathlib
import sys
from functools import lru_cache
from typing import Optional, Union
from urllib.parse import urlparse

import pydantic
import sqlalchemy
from pydantic import Field, ValidationError
from pydantic_settings import BaseSettings
from rich import print as rprint

from trapdata import ml
from trapdata.common import constants
from trapdata.common.filemanagement import default_database_dsn, get_app_dir
from trapdata.common.schemas import FilePath

# Hosts where a plain http:// download base URL is accepted, such as a local object store
# used during development. There is no network path to tamper with on a loopback address.
_LOOPBACK_HOSTS = ("localhost", "127.0.0.1", "::1")


def validate_object_store_base_url(value: str) -> str:
    """
    Check a model or image download base URL, and add a trailing slash if missing.

    The URL must use HTTPS: model weights are unpickled by torch.load, so fetching them
    over plain HTTP would let anyone on the network path substitute a file that runs
    code when loaded. Plain HTTP is accepted only for a loopback host. File paths are
    appended to the base URL directly, so it must have a host and no query string or
    fragment.
    """
    value = value.strip()
    if not value:
        raise ValueError("must not be empty")
    parsed = urlparse(value)
    if not parsed.hostname or parsed.query or parsed.fragment:
        raise ValueError(
            "must be an absolute URL with a host and no query string or fragment, "
            f"got {value!r}"
        )
    is_local_http = parsed.scheme == "http" and parsed.hostname in _LOOPBACK_HOSTS
    if parsed.scheme != "https" and not is_local_http:
        raise ValueError(
            "must be an https:// URL (plain http:// is accepted only for localhost), "
            f"got {value!r}"
        )
    return value if value.endswith("/") else f"{value}/"


class Settings(BaseSettings):
    # Can't use PyDantic DSN validator for database_url if sqlite filepath has spaces, see custom validator below
    database_url: Union[str, sqlalchemy.engine.URL] = default_database_dsn()
    user_data_path: pathlib.Path = get_app_dir()
    image_base_path: Optional[pathlib.Path] = None
    localization_model: ml.models.ObjectDetectorChoice = Field(
        default=ml.models.DEFAULT_OBJECT_DETECTOR
    )
    binary_classification_model: ml.models.BinaryClassifierChoice = Field(
        default=ml.models.DEFAULT_BINARY_CLASSIFIER
    )
    species_classification_model: ml.models.SpeciesClassifierChoice = Field(
        default=ml.models.DEFAULT_SPECIES_CLASSIFIER
    )
    feature_extractor: ml.models.FeatureExtractorChoice = Field(
        default=ml.models.DEFAULT_FEATURE_EXTRACTOR
    )
    classification_threshold: float = 0.6
    localization_batch_size: int = 8
    classification_batch_size: int = 20
    num_workers: int = 4

    # Antenna API worker settings
    antenna_api_base_url: str = "http://localhost:8000/api/v2"
    antenna_api_auth_token: str = ""
    antenna_service_name: str = "AMI Data Companion"
    antenna_api_batch_size: int = 24
    # Maximum size (in bytes) of a single result POST body to the Antenna API.
    # The results for one processed batch are split across multiple POSTs so that
    # no single request exceeds this limit. Wide-taxonomy classifiers (e.g. the
    # global moths model with ~29k classes) emit ~2 MB per detection because each
    # classification carries full-length labels/scores/logits arrays, so a dense
    # batch can otherwise produce a 100+ MB body that reverse proxies reject (413).
    # Default 25 MB leaves headroom under common proxy limits (typically 100 MB).
    antenna_result_post_max_bytes: int = 25 * 1024 * 1024

    # Where model weights and sample trap images are downloaded from. Model classes give
    # their files as paths relative to model_base_url; see resolve_model_url in
    # trapdata/ml/utils.py.
    model_base_url: str = f"{constants.OBJECT_STORE_BASE_URL}ami-models/"
    image_base_url: str = f"{constants.OBJECT_STORE_BASE_URL}ami-trapdata/"

    @pydantic.field_validator("image_base_path", "user_data_path")
    def validate_path(cls, v):
        """
        Expand relative paths into a normalized path.

        This is important because the `image_base_path` is currently
        stored in the database for objects and must be an exact match.
        """
        if v:
            return pathlib.Path(v).expanduser().resolve()
        else:
            return None

    @pydantic.field_validator("database_url")
    def validate_database_dsn(cls, v):
        return sqlalchemy.engine.url.make_url(v)

    @pydantic.field_validator("model_base_url", "image_base_url")
    def validate_base_url(cls, v):
        return validate_object_store_base_url(v)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        env_prefix = "ami_"
        extra = "ignore"
        protected_namespaces = ()

        fields = {
            "image_base_path": {
                "title": "Trap images",
                "description": "The root folder containing images from all nights that will be processed. It is recommended to start with a small sample.",
                "kivy_type": "path",
                "kivy_section": "paths",
            },
            "database_url": {
                "title": "Database connection string",
                "description": "Defaults to a local SQLite database that will automatically be created. Supports PostgreSQL.",
                "kivy_type": "string",
                "kivy_section": "paths",
            },
            "user_data_path": {
                "title": "Local directory for models, thumbnails & reports",
                "description": "Model weights are between 100-200Mb and will be downloaded the first time a model is used.",
                "kivy_type": "path",
                "kivy_section": "paths",
            },
            "localization_model": {
                "title": "Localization model",
                "description": "Model & settings to use for object detection in original images from camera trap.",
                "kivy_type": "options",
                "kivy_section": "models",
            },
            "binary_classification_model": {
                "title": "Binary classification model",
                "description": "Model & settings to use for moth / non-moth classification of cropped images after object detection.",
                "kivy_type": "options",
                "kivy_section": "models",
            },
            "species_classification_model": {
                "title": "Species classification model",
                "description": (
                    "Model & settings to use for fine-grained species or taxon-level classification of cropped images after moth/non-moth detection."
                ),
                "kivy_type": "options",
                "kivy_section": "models",
            },
            # "tracking_algorithm": {
            #     "title": "Occurrence tracking algorithm (de-duplication)",
            #     "description": "Method of identifying and tracking the same individual moth across multiple images.",
            #     "kivy_type": "options",
            #     "kivy_section": "models",
            # },
            "feature_extractor": {
                "title": "Feature extractor used for image similarity search and occurrence tracking",
                "description": "CNN model for extracting the embedded feature vector of an image used for similarity comparisons.",
                "kivy_type": "options",
                "kivy_section": "models",
            },
            "classification_threshold": {
                "title": "Classification threshold",
                "description": "Only show results with a confidence score greater or equal to this value.",
                "kivy_type": "numeric",
                "kivy_section": "models",
            },
            "localization_batch_size": {
                "title": "Localization batch size",
                "description": (
                    "Number of images to process per-batch during localization. "
                    "These are large images (e.g. 4096x2160px), smaller batch sizes are appropriate (1-10). "
                    "Reduce this if you run out of memory."
                ),
                "kivy_type": "numeric",
                "kivy_section": "performance",
            },
            "classification_batch_size": {
                "title": "Classification batch size",
                "description": (
                    "Number of images to process per-batch during classification. "
                    "These are small images (e.g. 50x100px), larger batch sizes are appropriate (10-200). "
                    "Reduce this if you run out of memory."
                ),
                "kivy_type": "numeric",
                "kivy_section": "performance",
            },
            "num_workers": {
                "title": "DataLoader workers",
                "description": (
                    "Number of parallel subprocesses for the PyTorch DataLoader (image downloading & preprocessing). "
                    "See https://pytorch.org/docs/stable/data.html"
                ),
                "kivy_type": "numeric",
                "kivy_section": "performance",
            },
            "antenna_api_base_url": {
                "title": "Antenna API Base URL",
                "description": "URL to the Antenna platform API for worker processing (should include /api/v2)",
                "kivy_type": "string",
                "kivy_section": "antenna",
            },
            "antenna_api_auth_token": {
                "title": "Antenna API Token",
                "description": "Authentication token for your Antenna project",
                "kivy_type": "string",
                "kivy_section": "antenna",
            },
            "antenna_api_batch_size": {
                "title": "Antenna API Batch Size",
                "description": "Number of tasks to fetch from Antenna per batch",
                "kivy_type": "numeric",
                "kivy_section": "antenna",
            },
            "antenna_result_post_max_bytes": {
                "title": "Antenna Result POST Max Bytes",
                "description": (
                    "Maximum size in bytes of a single result POST body; results "
                    "for a batch are split across multiple POSTs to stay under it"
                ),
                "kivy_type": "numeric",
                "kivy_section": "antenna",
            },
            "antenna_service_name": {
                "title": "Antenna Service Name",
                "description": "Name for the processing service registration (hostname will be added automatically)",
                "kivy_type": "string",
                "kivy_section": "antenna",
            },
        }

        @classmethod
        def customise_sources(  # UK spelling
            cls,
            init_settings,
            env_settings,
            file_secret_settings,
        ):
            return (
                init_settings,
                env_settings,
                kivy_settings_source,
                file_secret_settings,
            )


class PipelineSettings(Settings):
    image_base_path: FilePath  # Override default settings to enforce image_base_path


def kivy_settings_path() -> pathlib.Path:
    project_root = pathlib.Path(__file__).parent
    kivy_settings_path = project_root / "ui" / "trapdata.ini"
    return kivy_settings_path


def kivy_settings_source(settings: BaseSettings) -> dict[str, str]:
    """
    Load settings set by user in the Kivy GUI app.
    """
    path = kivy_settings_path()
    if not path.exists():
        return {}
    else:
        config = configparser.ConfigParser()
        config.read(kivy_settings_path())
        kivy_settings = [config.items(section) for section in config.sections()]
        kivy_settings_flat = dict(
            [item for section in kivy_settings for item in section]
        )
        null_values = ["None"]
        kivy_settings_flat = {
            k: v for k, v in kivy_settings_flat.items() if v not in null_values
        }
        return kivy_settings_flat


cli_help_message = f"""
    Configuration for the CLI is currently set in the following sources, in order of priority:
        - The system environment (os.environ)
        - ".env" file (see ".env.example"), prefix settings with "AMI_"
        - Kivy settings panel in the GUI app
        - Directly in the Kivy settings file: {kivy_settings_path()}
    """


@lru_cache
def read_settings(*args, **kwargs):
    try:
        return Settings(*args, **kwargs)
    except ValidationError as e:
        # @TODO the validation errors could be printed in a more helpful way:
        rprint(cli_help_message)
        rprint(e)
        sys.exit(1)


if __name__ == "__main__":
    rprint(read_settings())  # .schema_json(indent=2))
