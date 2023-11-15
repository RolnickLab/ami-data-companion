# Can these be imported from the OpenAPI spec yaml?
import pathlib

import PIL.Image
import pydantic

from trapdata.ml.utils import get_image


class BoundingBox(pydantic.BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    source_width: int | None = None
    source_height: int | None = None


class SourceImage(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="ignore", arbitrary_types_allowed=True)

    id: int
    url: str | None = None
    b64: str | None = None
    filepath: str | pathlib.Path | None = None
    pil: PIL.Image.Image | None = None
    detections: list[BoundingBox] | None = None
    width: int | None = None
    height: int | None = None

    # Validate that there is at least one of the following fields
    @pydantic.model_validator(mode="after")
    def validate_source(self):
        if not any([self.url, self.b64, self.filepath, self.pil]):
            raise ValueError(
                "At least one of the following fields must be provided: url, b64, filepath, pil"
            )
        return self

    def open(self, raise_exception=False) -> PIL.Image.Image | None:
        if not self.pil:
            self.pil = get_image(
                url=self.url,
                b64=self.b64,
                filepath=self.filepath,
                raise_exception=raise_exception,
            )
        if self.pil:
            self.width, self.height = self.pil.size
        return self.pil


class Detection(pydantic.BaseModel):
    source_image: SourceImage
    bbox: BoundingBox
