import pathlib
import time

import PIL.Image

from ..common.utils import slugify
from .schemas import BoundingBox, SourceImage


def render_crop(source_image: SourceImage, bbox: BoundingBox | None) -> PIL.Image.Image:
    """ """
    source_image.open(raise_exception=True)
    assert source_image._pil is not None
    if not bbox:
        # Use full image
        return source_image._pil
    coords = (bbox.x1, bbox.y1, bbox.x2, bbox.y2)
    return source_image._pil.crop(coords)  # type: ignore


def render_crops(
    source_image: SourceImage, bboxes: list[BoundingBox]
) -> list[PIL.Image.Image]:
    """
    Is there an efficient way to crop multiple boxes from a single image using numpy?
    """
    raise NotImplementedError


def get_crop_fname(source_image: SourceImage, bbox: BoundingBox) -> str:
    assert source_image.url, "Auto naming only works for images with a URL"
    source_name = slugify(pathlib.Path(source_image.url).stem)
    bbox_name = bbox.to_path()
    timestamp = int(time.time())  # @TODO use pipeline name/version instead
    return f"{source_name}/{bbox_name}-{timestamp}.jpg"
