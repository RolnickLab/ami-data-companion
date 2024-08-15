import io
import pathlib
import time

import PIL.Image

from ..common.logs import logger
from ..common.s3 import S3Config, file_exists, public_url, write_file
from ..common.utils import slugify
from . import settings
from .schemas import BoundingBox, SourceImage


def get_s3_config() -> S3Config:
    """ """
    return S3Config(
        endpoint_url=settings.s3_endpoint_url,
        bucket_name=settings.s3_destination_bucket,
        access_key_id=settings.s3_access_key_id,
        secret_access_key=settings.s3_secret_access_key,
        public_base_url=None,
        prefix="crops-dev/",
    )


def upload_image(image: PIL.Image.Image, name: str):
    """
    @TODO can we do this with a presigned URL from the API request?
    """
    s3_config = get_s3_config()
    with io.BytesIO() as f:
        image.save(f, format="JPEG")
        f.seek(0)
        key = write_file(s3_config, name, f)
        url = public_url(s3_config, key)
        # img = read_image(s3_config, key)
        # assert img.width == image.width
        return url


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


def upload_crop(
    source_image: SourceImage, bbox: BoundingBox, overwrite=False
) -> str | None:
    """ """
    if (
        not settings.s3_destination_bucket
        and not settings.s3_access_key_id
        and not settings.s3_secret_access_key
    ):
        logger.debug("Skipping crop upload because S3 is not configured")
        return None
    crop_fname = get_crop_fname(source_image, bbox)
    if not overwrite:
        if file_exists(get_s3_config(), crop_fname):
            return public_url(get_s3_config(), crop_fname)
    crop_image = render_crop(source_image, bbox)
    url = upload_image(crop_image, crop_fname)
    return url
