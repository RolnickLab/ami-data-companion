from __future__ import annotations

import base64
import binascii
import datetime
import io
import json
import os
import pathlib
import re
import requests
import tempfile
import time
import urllib.error
from urllib.parse import urlparse
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import pandas as pd
import PIL.Image
import PIL.ImageFile
import torch
import torchvision

if TYPE_CHECKING:
    from _typeshed import SupportsRead

from trapdata import logger

PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True


def get_device(device_str=None) -> torch.device:
    """
    Select CUDA if available.

    @TODO add macOS Metal?
    @TODO check Kivy settings to see if user forced use of CPU
    """
    if not device_str:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    logger.info(f"Using device '{device}' for inference")
    return device


def get_or_download_file(
    path, destination_dir=None, prefix=None, suffix=None
) -> pathlib.Path:
    """
    Get or download a file from a given path or URL using the requests library.

    Args:
    path (str): URL or local path to the file
    destination_dir (str, optional): Directory to save the downloaded file
    prefix (str, optional): Prefix to add to the destination directory
    suffix (str, optional): Suffix to add to the filename

    Returns:
    pathlib.Path: Path to the local file

    >>> filename = get_or_download_file("https://example.com/file with spaces.zip")
    """
    if not path:
        raise ValueError("Specify a URL or path to fetch file from.")

    # If path is a local path instead of a URL, just return it
    if os.path.exists(path):
        return pathlib.Path(path)

    destination_dir = destination_dir or os.environ.get("LOCAL_WEIGHTS_PATH")
    
    if not destination_dir:
        raise ValueError(
            "No destination directory specified by LOCAL_WEIGHTS_PATH or app settings."
        )

    destination_dir = pathlib.Path(destination_dir)
    if prefix:
        destination_dir = destination_dir / prefix
    if not destination_dir.exists():
        logger.info(f"Creating local directory {str(destination_dir)}")
        destination_dir.mkdir(parents=True, exist_ok=True)

    # Extract filename from URL
    fname = path.split("/")[-1]
    local_filepath = destination_dir / fname

    if suffix:
        local_filepath = local_filepath.with_suffix(suffix)

    if local_filepath.exists():
        logger.info(f"Using existing {local_filepath}")
        return local_filepath

    logger.info(f"Downloading {path} to {local_filepath}")
    
    try:
        response = requests.get(path, stream=True)
        response.raise_for_status()  # Raises an HTTPError for bad responses

        with open(local_filepath, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)

        logger.info(f"Downloaded to {local_filepath}")
        return local_filepath

    except requests.exceptions.RequestException as e:
        logger.error(f"Error downloading file: {e}")
        raise


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


def synchronize_clocks():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    else:
        pass


def bbox_relative(bbox_absolute, img_width, img_height):
    """
    Convert bounding box from absolute coordinates (x1, y1, x2, y2)
    like those used by pytorch, to coordinates that are relative
    percentages of the original image size like those used by
    the COCO cameratraps format.
    https://github.com/Microsoft/CameraTraps/blob/main/data_management/README.md#coco-cameratraps-format
    """

    box_numpy = bbox_absolute.detach().cpu().numpy()
    bbox_percent = [
        round(box_numpy[0] / img_width, 4),
        round(box_numpy[1] / img_height, 4),
        round(box_numpy[2] / img_width, 4),
        round(box_numpy[3] / img_height, 4),
    ]
    return bbox_percent


def crop_bbox(image, bbox):
    """
    Create cropped image from region specified in a bounding box.

    Bounding boxes are assumed to be in the format:
    [(top-left-coordinate-pair), (bottom-right-coordinate-pair)]
    or: [x1, y1, x2, y2]

    The image is assumed to be a numpy array that can be indexed using the
    coordinate pairs.
    """

    x1, y1, x2, y2 = bbox

    cropped_image = image[
        :,
        int(y1) : int(y2),
        int(x1) : int(x2),
    ]
    transform_to_PIL = torchvision.transforms.ToPILImage()
    cropped_image = transform_to_PIL(cropped_image)
    yield cropped_image


def get_user_data_dir() -> pathlib.Path:
    """
    Return the path to the user data directory if possible.
    Otherwise return the system temp directory.
    """
    try:
        from trapdata.settings import read_settings

        settings = read_settings()
        return settings.user_data_path
    except Exception:
        import tempfile

        return pathlib.Path(tempfile.gettempdir())


@dataclass
class Taxon:
    gbif_id: int
    name: Optional[str]
    genus: Optional[str]
    family: Optional[str]
    source: Optional[str]


def fetch_gbif_species(gbif_id: int) -> Optional[Taxon]:
    """
    Look up taxon name from GBIF API. Cache results in user_data_path.
    """

    logger.info(f"Looking up species name for GBIF id {gbif_id}")
    base_url = "https://api.gbif.org/v1/species/{gbif_id}"
    url = base_url.format(gbif_id=gbif_id)

    try:
        taxon_data = get_or_download_file(
            url, destination_dir=get_user_data_dir(), prefix="taxa/gbif", suffix=".json"
        )
        data: dict = json.load(taxon_data.open())
    except urllib.error.HTTPError:
        logger.warn(f"Could not find species with gbif_id {gbif_id} in {url}")
        return None
    except json.decoder.JSONDecodeError:
        logger.warn(f"Could not parse JSON response from {url}")
        return None

    taxon = Taxon(
        gbif_id=gbif_id,
        name=data["canonicalName"],
        genus=data["genus"],
        family=data["family"],
        source="gbif",
    )
    return taxon


def lookup_gbif_species(species_list_path: str, gbif_id: int) -> Taxon:
    """
    Look up taxa names from a Darwin Core Archive file (DwC-A).

    Example:
    https://docs.google.com/spreadsheets/d/1E3-GAB0PSKrnproAC44whigMvnAkbkwUmwXUHMKMOII/edit#gid=1916842176

    @TODO Optionally look up species name from GBIF API
    Example https://api.gbif.org/v1/species/5231190
    """
    local_path = get_or_download_file(
        species_list_path, destination_dir=get_user_data_dir(), prefix="taxa"
    )
    df = pd.read_csv(local_path)
    taxon = None
    # look up single row by gbif_id
    try:
        row = df.loc[df["taxon_key_gbif_id"] == gbif_id].iloc[0]
    except IndexError:
        logger.warn(
            f"Could not find species with gbif_id {gbif_id} in {species_list_path}"
        )
    else:
        taxon = Taxon(
            gbif_id=gbif_id,
            name=row["search_species_name"],
            genus=row["genus_name"],
            family=row["family_name"],
            source=row["source"],
        )

    if not taxon:
        taxon = fetch_gbif_species(gbif_id)

    if not taxon:
        return Taxon(
            gbif_id=gbif_id, name=str(gbif_id), genus=None, family=None, source=None
        )

    return taxon


def replace_gbif_id_with_name(name) -> str:
    """
    If the name appears to be a GBIF ID, then look up the species name from GBIF.
    """
    try:
        gbif_id = int(name)
    except ValueError:
        return name
    else:
        taxon = fetch_gbif_species(gbif_id)
        if taxon and taxon.name:
            return taxon.name
        else:
            return name


class StopWatch:
    """
    Measure inference time with GPU support.

    >>> with stopwatch() as t:
    >>>     sleep(5)
    >>> int(t.duration)
    >>> 5
    """

    def __enter__(self):
        synchronize_clocks()
        # self.start = time.perf_counter()
        self.start = time.time()
        return self

    def __exit__(self, type, value, traceback):
        synchronize_clocks()
        # self.end = time.perf_counter()
        self.end = time.time()
        self.duration = self.end - self.start

    def __repr__(self):
        start = datetime.datetime.fromtimestamp(self.start).strftime("%H:%M:%S")
        end = datetime.datetime.fromtimestamp(self.end).strftime("%H:%M:%S")
        seconds = int(round(self.duration, 1))
        return f"Started: {start}, Ended: {end}, Duration: {seconds} seconds"
