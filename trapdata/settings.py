from typing import Union, Literal

import pathlib

from pydantic import (
    BaseModel,
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
    local_weights_path: pathlib.Path
    localization_model: Literal[tuple(ml.models.object_detectors.keys())] = Field(None)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        env_prefix = "ami_"

        fields = {
            "user_data_path": {
                "title": "Local directory for models, thumbnails & reports",
                "description": "Model weights are between 100-200Mb and will be downloaded the first time a model is used.",
                "kivy_type": "path",
                "kivy_section": "paths",
            },
            "num_workers": {
                "title": "Number of workers",
                "description": "Number of parallel workers for the PyTorch dataloader. See https://pytorch.org/docs/stable/data.html",
                "kivy_type": "numeric",
                "kivy_section": "performance",
            },
        }


settings = Settings()
print(settings.schema_json(indent=2))
