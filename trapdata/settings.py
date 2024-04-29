import pathlib
import sys
from functools import lru_cache
from typing import Optional, Union

import sqlalchemy
from pydantic import Field, ValidationError, validator
from pydantic_settings import BaseSettings
from rich import print as rprint

from trapdata import ml
from trapdata.common.filemanagement import default_database_dsn, get_app_dir
from trapdata.common.schemas import FilePath


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
    localization_batch_size: int = 2
    classification_batch_size: int = 20
    num_workers: int = 1
    api_base_url: str | None = "http://localhost:8000/api/v2/"
    api_username: str | None = None
    api_password: str | None = None
    s3_access_key_id: str | None = None
    s3_secret_access_key: str | None = None
    s3_endpoint_url: str | None = None
    s3_destination_bucket: str | None = None

    @validator("api_base_url")
    def validate_base_url(cls, v):
        if v and not v.endswith("/"):
            return v + "/"
        else:
            return v

    @validator("image_base_path", "user_data_path")
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

    @validator("database_url")
    def validate_database_dsn(cls, v):
        return sqlalchemy.engine.url.make_url(v)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        env_prefix = "ami_"
        extra = "ignore"

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
                "title": "Number of workers",
                "description": "Number of parallel workers for the PyTorch dataloader. See https://pytorch.org/docs/stable/data.html",
                "kivy_type": "numeric",
                "kivy_section": "performance",
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
                file_secret_settings,
            )


class PipelineSettings(Settings):
    image_base_path: FilePath  # Override default settings to enforce image_base_path


cli_help_message = """
    Configuration for the CLI is currently set in the following sources, in order of priority:
        - The system environment (os.environ)
        - ".env" file (see ".env.example"), prefix settings with "AMI_"
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
