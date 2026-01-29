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


def get_http_session(
    auth_token: str | None = None,
    max_retries: int | None = None,
    backoff_factor: float | None = None,
    status_forcelist: tuple[int, ...] = (500, 502, 503, 504),
) -> requests.Session:
    """
    Create an HTTP session with retry logic for transient failures.

    Configures a requests.Session with HTTPAdapter and urllib3.Retry to automatically
    retry failed requests with exponential backoff. Only retries on server errors (5XX)
    and network failures, NOT on client errors (4XX).

    Args:
        auth_token: Optional authentication token (adds "Token {token}" to Authorization header)
        max_retries: Maximum number of retry attempts (default: from settings.antenna_api_retry_max)
        backoff_factor: Exponential backoff multiplier in seconds (default: from settings.antenna_api_retry_backoff)
                       Delays will be: backoff_factor * (2 ** retry_number)
                       e.g., 0.5s, 1s, 2s for default settings
        status_forcelist: HTTP status codes that trigger a retry (default: 500, 502, 503, 504)

    Returns:
        Configured requests.Session with retry adapter mounted

    Example:
        >>> session = get_http_session(max_retries=3, backoff_factor=0.5)
        >>> response = session.get("https://api.example.com/data")
        >>> # With authentication:
        >>> session = get_http_session(auth_token="abc123")
        >>> response = session.get("https://api.example.com/data")
    """
    # Read defaults from settings if not explicitly provided
    if max_retries is None or backoff_factor is None:
        from trapdata.settings import read_settings

        settings = read_settings()
        if max_retries is None:
            max_retries = settings.antenna_api_retry_max
        if backoff_factor is None:
            backoff_factor = settings.antenna_api_retry_backoff

    session = requests.Session()

    retry_strategy = Retry(
        total=max_retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        allowed_methods=["GET", "POST"],
        raise_on_status=False,  # Don't raise exception, let caller handle status codes
    )

    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    # Add auth header if token provided
    if auth_token:
        session.headers["Authorization"] = f"Token {auth_token}"

    return session
