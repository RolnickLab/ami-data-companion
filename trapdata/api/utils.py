import pathlib
import time

import PIL.Image
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

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


def get_http_session(auth_token: str | None = None) -> requests.Session:
    """
    Create an HTTP session with retry logic for transient failures.

    Configures a requests.Session with HTTPAdapter and urllib3.Retry to automatically
    retry failed requests with exponential backoff. Only retries on server errors (5XX)
    and network failures, NOT on client errors (4XX). Only GET requests are retried.

    TODO: This will likely become part of an AntennaClient class that encapsulates
    base_url, auth_token, and session management. See docs/claude/planning/antenna-client.md

    Args:
        auth_token: Optional API token. If provided, adds "Token {auth_token}" header.

    Returns:
        Configured requests.Session with retry adapter mounted
    """
    session = requests.Session()

    retry_strategy = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=(500, 502, 503, 504),
        allowed_methods=["GET"],
        raise_on_status=False,
    )

    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    if auth_token:
        session.headers["Authorization"] = f"Token {auth_token}"

    return session
