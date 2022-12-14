from typing import Union, Optional

import pathlib

from pydantic import (
    BaseSettings,
    Field,
    FileUrl,
    PostgresDsn,
)

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


if __name__ == "__main__":
    settings = Settings()  # type: ignore
    print(settings.schema_json(indent=2))
