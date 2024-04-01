import base64
import binascii
import io
import os
import pathlib
import re
import tempfile

from ..logs import logger

if TYPE_CHECKING:
    from _typeshed import SupportsRead


def get_or_download_file(
    path, destination_dir=None, prefix=None, suffix=None
) -> pathlib.Path:
    """
    >>> filename, headers = get_weights("https://drive.google.com/file/d/1KdQc56WtnMWX9PUapy6cS0CdjC8VSdVe/view?usp=sharing")

    """
    if not path:
        raise Exception("Specify a URL or path to fetch file from.")

    # If path is a local path instead of a URL then urlretrieve will just return that path
    destination_dir = destination_dir or os.environ.get("LOCAL_WEIGHTS_PATH")
    fname = path.rsplit("/", 1)[-1]
    if destination_dir:
        destination_dir = pathlib.Path(destination_dir)
        if prefix:
            destination_dir = destination_dir / prefix
        if not destination_dir.exists():
            logger.info(f"Creating local directory {str(destination_dir)}")
            destination_dir.mkdir(parents=True, exist_ok=True)
        local_filepath = pathlib.Path(destination_dir) / fname
        if suffix:
            local_filepath = local_filepath.with_suffix(suffix)
    else:
        raise Exception(
            "No destination directory specified by LOCAL_WEIGHTS_PATH or app settings."
        )

    if local_filepath and local_filepath.exists():
        logger.info(f"Using existing {local_filepath}")
        return local_filepath

    else:
        logger.info(f"Downloading {path} to {local_filepath}")
        resulting_filepath, headers = urllib.request.urlretrieve(
            url=path, filename=local_filepath
        )
        resulting_filepath = pathlib.Path(resulting_filepath)
        logger.info(f"Downloaded to {resulting_filepath}")
        return resulting_filepath


def decode_base64_string(string) -> io.BytesIO:
    image_data = re.sub("^data:image/.+;base64,", "", string)
    decoded = base64.b64decode(image_data)
    buffer = io.BytesIO(decoded)
    buffer.seek(0)
    return buffer


def open_image(
    fp: str | bytes | pathlib.Path | SupportsRead[bytes], raise_exception: bool = True
) -> PIL.Image.Image | None:
    """
    Wrapper from PIL.Image.open that handles errors and converts to RGB.
    """
    img = None
    try:
        img = PIL.Image.open(fp)
    except PIL.UnidentifiedImageError:
        logger.warn(f"Unidentified image: {str(fp)[:100]}...")
        if raise_exception:
            raise
    except OSError:
        logger.warn(f"Could not open image: {str(fp)[:100]}...")
        if raise_exception:
            raise
    else:
        # Convert to RGB if necessary
        if img.mode != "RGB":
            img = img.convert("RGB")

    return img


def get_image(
    url: str | None = None,
    filepath: str | pathlib.Path | None = None,
    b64: str | None = None,
    raise_exception: bool = True,
) -> PIL.Image.Image | None:
    """
    Given a URL, local file path or base64 image, return a PIL image.
    """

    if url:
        logger.info(f"Fetching image from URL: {url}")
        tempdir = tempfile.TemporaryDirectory(prefix="ami_images")
        img_path = get_or_download_file(url, destination_dir=tempdir.name)
        return open_image(img_path, raise_exception=raise_exception)

    elif filepath:
        logger.info(f"Loading image from local filesystem: {filepath}")
        return open_image(filepath, raise_exception=raise_exception)

    elif b64:
        logger.info(f"Loading image from base64 string: {b64[:30]}...")
        try:
            buffer = decode_base64_string(b64)
        except binascii.Error as e:
            logger.warn(f"Could not decode base64 image: {e}")
            if raise_exception:
                raise
            else:
                return None
        else:
            return open_image(buffer, raise_exception=raise_exception)

    else:
        raise Exception("Specify a URL, path or base64 image.")
