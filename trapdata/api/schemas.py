# Can these be imported from the OpenAPI spec yaml?
import pathlib

import PIL.Image
import pydantic

from trapdata.ml.utils import get_image


class IncomingSourceImage(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="ignore")

    id: int
    url: str | None = None
    b64: str | None = None
    filepath: str | pathlib.Path | None = None
    _pil: PIL.Image.Image | None = None

    # Validate that there is at least one of the following fields
    @pydantic.model_validator(mode="after")
    def validate_source(self):
        if not any([self.url, self.b64, self.filepath]):
            raise ValueError(
                "At least one of the following fields must be provided: url, b64, filepath"
            )
        return self

    def open(self, raise_exception=False) -> PIL.Image.Image | None:
        if not self._pil:
            self._pil = get_image(
                url=self.url,
                b64=self.b64,
                filepath=self.filepath,
                raise_exception=raise_exception,
            )
        return self._pil
