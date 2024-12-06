# Can these be imported from the OpenAPI spec yaml?
import datetime
import pathlib

import PIL.Image
import pydantic

from trapdata.common.logs import logger
from trapdata.ml.utils import get_image


class BoundingBox(pydantic.BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float

    @classmethod
    def from_coords(cls, coords: list[float]):
        return cls(x1=coords[0], y1=coords[1], x2=coords[2], y2=coords[3])

    def to_string(self):
        return f"{self.x1},{self.y1},{self.x2},{self.y2}"

    def to_path(self):
        return "-".join([str(int(x)) for x in [self.x1, self.y1, self.x2, self.y2]])

    def to_tuple(self):
        return (self.x1, self.y1, self.x2, self.y2)


class SourceImage(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="ignore", arbitrary_types_allowed=True)

    id: str
    url: str | None = None
    b64: str | None = None
    filepath: str | pathlib.Path | None = None
    _pil: PIL.Image.Image | None = None
    width: int | None = None
    height: int | None = None
    timestamp: datetime.datetime | None = None

    # Validate that there is at least one of the following fields
    @pydantic.model_validator(mode="after")
    def validate_source(self):
        if not any([self.url, self.b64, self.filepath, self._pil]):
            raise ValueError(
                "At least one of the following fields must be provided: "
                "url, b64, filepath, pil"
            )
        return self

    def open(self, raise_exception=False) -> PIL.Image.Image | None:
        if not self._pil:
            logger.warn(f"Opening image {self.id} for the first time")
            self._pil = get_image(
                url=self.url,
                b64=self.b64,
                filepath=self.filepath,
                raise_exception=raise_exception,
            )
        else:
            logger.info(f"Using already loaded image {self.id}")
        if self._pil:
            self.width, self.height = self._pil.size
        return self._pil


class Classification(pydantic.BaseModel):
    classification: str
    labels: list[str] = []
    scores: list[float] = []
    logits: list[float] = []
    inference_time: float | None = None
    algorithm: str | None = None
    terminal: bool = True
    timestamp: datetime.datetime


class Detection(pydantic.BaseModel):
    source_image_id: str
    bbox: BoundingBox
    inference_time: float | None = None
    algorithm: str | None = None
    timestamp: datetime.datetime
    crop_image_url: str | None = None
    classifications: list[Classification] = []
