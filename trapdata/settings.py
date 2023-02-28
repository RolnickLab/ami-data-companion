# @TODO use this settings module in Kivy
import sys
from typing import Union, Optional, Any
import configparser

import pathlib

from pydantic import (
    BaseSettings,
    Field,
    FileUrl,
    PostgresDsn,
    ValidationError,
)
from rich import print as rprint

from trapdata import ml


class SqliteDsn(FileUrl):
    allowed_schemes = {
        "sqlite",
        "sqlite+pysqlite",
        "sqlite+aiosqlite",
        "sqlite+pysqlcipher",
    }


class Settings(BaseSettings):
    database_url: Union[SqliteDsn, PostgresDsn]
    user_data_path: pathlib.Path
    # local_weights_path: pathlib.Path
    localization_model: Optional[ml.models.ObjectDetectorChoice]
    binary_classification_model: Optional[ml.models.BinaryClassifierChoice]
    species_classification_model: Optional[ml.models.SpeciesClassifierChoice]
    tracking_algorithm: Optional[ml.models.TrackingAlgorithmChoice]
    localization_batch_size: int = Field(2)
    classification_batch_size: int = Field(20)
    num_workers: int = Field(1)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        env_prefix = "ami_"
        extra = "ignore"

        fields = {
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
                "description": "Model & settings to use for fine-grained species or taxon-level classification of cropped images after moth/non-moth detection.",
                "kivy_type": "options",
                "kivy_section": "models",
            },
            "tracking_algorithm": {
                "title": "Occurence tracking algorithm (de-duplication)",
                "description": "Method of identifying and tracking the same individual moth across multiple images.",
                "kivy_type": "options",
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
        def customise_sources(
            cls,
            init_settings,
            env_settings,
            file_secret_settings,
        ):
            return (
                init_settings,
                kivy_settings_source,
                env_settings,
                file_secret_settings,
            )


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


def read_settings(*args, **kwargs):
    try:
        settings = Settings(*args, **kwargs)  # type: ignore
    except ValidationError as e:
        rprint(
            f"""
            Configuration for the CLI is currently set in the following sources, in order of priority:
             - Kivy settings panel in the GUI app
             - Directly in the Kivy settings file: {kivy_settings_path()}
             - ".env" file (see ".env.example"), prefix settings with "AMI_"
             - The system environment (os.environ)
            """
        )
        # @TODO can we make this output more friendly with the rich library?
        rprint(e)
        print(e)
        sys.exit(1)
    else:
        return settings


settings = read_settings()


if __name__ == "__main__":
    rprint(settings)  # .schema_json(indent=2))
