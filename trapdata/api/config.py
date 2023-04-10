import pathlib
from typing import Any, Dict, List, Optional

from pydantic import BaseSettings, HttpUrl, PostgresDsn, validator
from pydantic.networks import AnyHttpUrl

from trapdata.cli import read_settings
from trapdata.settings import Settings as BaseSettings


class Settings(BaseSettings):
    PROJECT_NAME: str = "AMI Data Manager"

    SENTRY_DSN: Optional[HttpUrl] = None

    API_PATH: str = "/api/v1"

    ACCESS_TOKEN_EXPIRE_MINUTES: int = 7 * 24 * 60  # 7 days

    BACKEND_CORS_ORIGINS: List[AnyHttpUrl] = []

    # The following variables need to be defined in environment

    TEST_DATABASE_URL: Optional[PostgresDsn]

    SECRET_KEY: str
    #  END: required environment variables

    # STATIC_ROOT: str = "static"

    # @validator("STATIC_ROOT")
    # def validate_static_root(cls, v):
    #     path = cls.user_data_path / v
    #     path.mkdir(parents=True, exist_ok=True)
    #     return path


# settings = read_settings(SettingsClass=Settings, SECRET_KEY="secret")
settings = Settings(SECRET_KEY="secret")
