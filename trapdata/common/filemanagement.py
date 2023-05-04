import collections
import datetime
import hashlib
import math
import os
import pathlib
import re
import tempfile
import time
from sys import platform as _sys_platform
from typing import Literal, Optional, Union

import dateutil.parser
import imagesize
import PIL.ExifTags
import PIL.Image
from dateutil.parser import ParserError

from trapdata.tests import TEST_IMAGES_BASE_PATH  # noqa: F401

from . import constants
from .logs import logger

APP_NAME_SLUG = "AMI"
EXIF_DATETIME_STR_FORMAT = "%Y:%m:%d %H:%M:%S"


def absolute_path(
    path: str, base_path: Union[pathlib.Path, str, None]
) -> Union[pathlib.Path, None]:
    """
    Turn a relative path into an absolute path.
    """
    if not path:
        return None
    elif not base_path:
        return pathlib.Path(path)
    elif str(base_path) in path:
        return pathlib.Path(path)
    else:
        return pathlib.Path(base_path) / path


def archive_file(filepath):
    """
    Rename an existing file to `<filepath>/<filename>.bak.<timestamp>`
    """
    filepath = pathlib.Path(filepath)
    if filepath.exists():
        suffix = f".{filepath.suffix}.backup.{str(int(time.time()))}"
        backup_filepath = filepath.with_suffix(suffix)
        logger.info(f"Moving existing file to {backup_filepath}")
        filepath.rename(backup_filepath)
        return backup_filepath


def find_timestamped_folders(path):
    """
    Find all directories in a given path that have
    dates / timestamps in the name.

    This should be the nightly folders from the trap data.

    >>> pathlib.Path("./tmp/2022_05_14").mkdir(exist_ok=True, parents=True)
    >>> pathlib.Path("./tmp/nope").mkdir(exist_ok=True, parents=True)
    >>> find_timestamped_folders("./tmp")
    OrderedDict([(datetime.datetime(2022, 5, 14, 0, 0), PosixPath('tmp/2022_05_14'))])
    """
    logger.debug("Looking for nightly timestamped folders")
    nights = collections.OrderedDict()

    def _preprocess(name):
        return name.replace("_", "-")

    dirs = sorted(list(pathlib.Path(path).iterdir()))
    for d in dirs:
        # @TODO use yield?
        try:
            date = dateutil.parser.parse(_preprocess(d.name))
        except Exception:
            # except dateutil.parser.ParserError:
            pass
        else:
            logger.debug(f"Found nightly folder for {date}: {d}")
            nights[date] = d

    return nights


def get_exif(img_path):
    """
    Read the EXIF tags in an image file
    """
    img = PIL.Image.open(img_path)
    img_exif = img.getexif()
    tags = {}

    for key, val in img_exif.items():
        if key in PIL.ExifTags.TAGS:
            name = PIL.ExifTags.TAGS[key]
            # logger.debug(f"EXIF tag found'{name}': '{val}'")
            tags[name] = val

    return tags


def construct_exif(
    timestamp: Optional[datetime.datetime] = None,
    description: Optional[str] = None,
    location: Optional[dict] = None,
    other_tags: Optional[dict] = None,
    existing_exif: Optional[PIL.Image.Exif] = None,
) -> PIL.Image.Exif:
    """
    Construct an EXIF class using human readable keys.
    Can be save to a Pillow image using:

    >>> image = PIL.Image.open("./trapdata/tests/images/denmark/20220811005907-00-78.jpg")
    >>> existing_exif = image.getexif()
    >>> exif_data = construct_exif(description="hi!", existing_exif=existing_exif)
    >>> image.save("test_with_exif.jpg", exif=exif_data)
    """

    exif = existing_exif or PIL.Image.Exif()

    name_to_code = {name: code for code, name in PIL.ExifTags.TAGS.items()}

    if timestamp:
        timestamp_str = timestamp.strftime(EXIF_DATETIME_STR_FORMAT)
        exif[name_to_code["DateTime"]] = timestamp_str
        exif[name_to_code["DateTimeOriginal"]] = timestamp_str
        exif[name_to_code["DateTimeDigitized"]] = timestamp_str

    if description:
        exif[name_to_code["ImageDescription"]] = description
        exif[name_to_code["UserComment"]] = description

    if location:
        raise NotImplementedError

    if other_tags:
        for name in other_tags:
            if name not in name_to_code:
                raise Exception(f"Unknown EXIF tag '{name}'")
            else:
                code = name_to_code[name]
                value = other_tags[name]
                logger.debug(f"Adding EXIF tag {name} ({code}) with value {value}")
                exif[code] = value

    return exif


def get_image_dimensions(img_path):
    """
    Get the dimensions of an image without loading it into memory.
    """
    return imagesize.get(img_path)


def get_image_filesize(img_path):
    """
    Return the filesize of an image in bytes.
    """
    return pathlib.Path(img_path).stat().st_size


def get_image_timestamp_from_filename(img_path) -> datetime.datetime:
    """
    Parse the date and time a photo was taken from its filename.

    The timestamp must be in the format `YYYYMMDDHHMMSS` but can be
    preceded or followed by other characters (e.g. `84-20220916202959-snapshot.jpg`).

    >>> out_fmt = "%Y-%m-%d %H:%M:%S"
    >>> # Aarhus date format
    >>> get_image_timestamp_from_filename("20220810231507-00-07.jpg").strftime(out_fmt)
    '2022-08-10 23:15:07'
    >>> # Diopsis date format
    >>> get_image_timestamp_from_filename("20230124191342.jpg").strftime(out_fmt)
    '2023-01-24 19:13:42'
    >>> # Snapshot date format in Vermont traps
    >>> get_image_timestamp_from_filename("20220622000459-108-snapshot.jpg").strftime(out_fmt)
    '2022-06-22 00:04:59'
    >>> # Snapshot date format in Cyprus traps
    >>> get_image_timestamp_from_filename("84-20220916202959-snapshot.jpg").strftime(out_fmt)
    '2022-09-16 20:29:59'

    """
    name = pathlib.Path(img_path).stem
    date = None

    # Extract date from a filename using regex in the format %Y%m%d%H%M%S
    matches = re.search(r"(\d{14})", name)
    if matches:
        date = datetime.datetime.strptime(matches.group(), "%Y%m%d%H%M%S")
    else:
        date = dateutil.parser.parse(
            name, fuzzy=False
        )  # Fuzzy will interpret "DSC_1974" as 1974-01-01

    if date:
        return date
    else:
        raise ValueError(f"Could not parse date from filename '{img_path}'")


def get_image_timestamp_from_exif(img_path):
    """
    Parse the date and time a photo was taken from its EXIF data.

    This ignores the TimeZoneOffset and creates a datetime that is
    timezone naive.
    """
    exif = get_exif(img_path)
    datestring = exif["DateTime"].replace(":", "-", 2)
    date = dateutil.parser.parse(datestring)
    return date


def get_image_timestamp(
    img_path, check_exif=True, assert_exists=True
) -> datetime.datetime:
    """
    Parse the date and time a photo was taken from its filename or EXIF data.

    Reading the exif data is slow, so only do it if necessary.
    It is set to True for backwards compatibility.

    >>> images = pathlib.Path(TEST_IMAGES_BASE_PATH)
    >>> # Use filename
    >>> get_image_timestamp(images / "cyprus/84-20220916202959-snapshot.jpg").strftime("%Y-%m-%d %H:%M:%S")
    '2022-09-16 20:29:59'
    >>> # Fallback to EXIF
    >>> get_image_timestamp(images / "DSLR/DSC_0390.JPG").strftime("%Y-%m-%d %H:%M:%S")
    '2022-07-19 14:28:16'
    """
    if assert_exists:
        assert pathlib.Path(img_path).exists(), f"Image file does not exist: {img_path}"
    try:
        date = get_image_timestamp_from_filename(img_path)
    except (ValueError, ParserError) as e:
        if check_exif:
            logger.debug(f"Could not parse date from filename: {e}. Trying EXIF.")
            try:
                date = get_image_timestamp_from_exif(img_path)
            except dateutil.parser.ParserError:
                logger.error(
                    f"Could not parse image timestamp from filename or EXIF tags: {e}."
                )
                raise
        else:
            raise
    return date


def get_image_timestamp_with_timezone(img_path, default_offset="+0"):
    """
    Parse the date and time a photo was taken from its EXIF data.

    Also sets the timezone based on the TimeZoneOffset field if available.
    Example EXIF offset: "-4". Some libaries expect the format to be: "-04:00"
    However dateutil.parse seems to handle "-4" or "+4" just fine.
    """
    exif = get_exif(img_path)
    datestring = exif["DateTime"].replace(":", "-", 2)
    offset = exif.get("TimeZoneOffset") or str(default_offset)
    if int(offset) > 0:
        offset = f"+{offset}"
    datestring = f"{datestring} {offset}"
    date = dateutil.parser.parse(datestring)
    return date


def find_images(
    base_directory,
    absolute_paths=False,
    check_exif=True,
    skip_missing_timestamps=True,
):
    logger.info(f"Scanning '{base_directory}' for images")
    base_directory = pathlib.Path(base_directory).expanduser().resolve()
    if not base_directory.exists():
        raise Exception(f"Directory does not exist: {base_directory}")
    extensions_list = "|".join(
        [f.lstrip(".") for f in constants.SUPPORTED_IMAGE_EXTENSIONS]
    )
    pattern = rf"\.({extensions_list})$"
    for walk_path, dirs, files in os.walk(base_directory):
        for name in files:
            if re.search(pattern, name, re.IGNORECASE):
                relative_path = pathlib.Path(walk_path) / name
                full_path = base_directory / relative_path
                path = full_path if absolute_paths else relative_path
                shape = get_image_dimensions(path)
                filesize = get_image_filesize(path)

                try:
                    date = get_image_timestamp(full_path, check_exif=check_exif)
                except Exception as e:
                    logger.error(
                        f"Skipping image, could not determine timestamp for: {full_path}\n {e}"
                    )
                    if skip_missing_timestamps:
                        continue
                    else:
                        date = None

                yield {
                    "path": path,
                    "timestamp": date,
                    "shape": shape,
                    "filesize": filesize,
                    # "hash": None,
                }


def group_images_by_day(images, maximum_gap_minutes=6 * 60):
    """
    Find consecutive images and group them into daily/nightly monitoring sessions.
    If the time between two photos is greater than `maximum_time_gap` (in minutes)
    then start a new session group. Each new group uses the first photo's day
    as the day of the session even if consecutive images are taken past midnight.
    # @TODO add other group by methods? like image size, camera model, random sample batches, etc. Add to UI settings

    @TODO make fake images for this test
    >>> images = find_images(TEST_IMAGES_BASE_PATH, skip_missing_timestamps=True)
    >>> sessions = group_images_by_day(images)
    >>> len(sessions)
    7
    """
    logger.info(
        f"Grouping images into date-based groups with a maximum gap of {maximum_gap_minutes} minutes"
    )
    images = sorted(images, key=lambda image: image["timestamp"])
    if not images:
        return {}

    groups = collections.OrderedDict()

    last_timestamp = None
    current_day = None

    for image in images:
        if last_timestamp:
            delta = (image["timestamp"] - last_timestamp).seconds / 60
        else:
            delta = maximum_gap_minutes

        logger.debug(f"{image['timestamp']}, {round(delta, 2)}")

        if delta >= maximum_gap_minutes:
            current_day = image["timestamp"].date()
            logger.debug(
                f"Gap of {round(delta/60, 1)} hours detected. Starting new session for date: {current_day}"
            )
            groups[current_day] = []

        groups[current_day].append(image)
        last_timestamp = image["timestamp"]

    # This is for debugging
    for day, images in groups.items():
        first_date = images[0]["timestamp"]
        last_date = images[-1]["timestamp"]
        delta = last_date - first_date
        hours = round(delta.seconds / 60 / 60, 1)
        logger.debug(
            f"Found session on {day} with {len(images)} images that ran for {hours} hours.\n"
            f"From {first_date.strftime('%c')} to {last_date.strftime('%c')}."
        )

    return groups
    # yield relative_path, get_image_timestamp(full_path)


def save_image(
    image: PIL.Image.Image,
    base_path=None,
    subdir=None,
    name=None,
    suffix=".jpg",
    exif_data: Optional[PIL.Image.Exif] = None,
):
    """
    Accepts a PIL image, returns fpath Path object
    """
    if not name:
        name = hashlib.md5(image.tobytes()).hexdigest()

    if base_path:
        base_path = pathlib.Path(base_path)
    else:
        base_path = pathlib.Path(tempfile.TemporaryDirectory().name)
        logger.info(f"Saving image to temporary directory: {base_path}")

    if subdir:
        base_path = base_path / subdir

    if not base_path.exists():
        base_path.mkdir(parents=True)

    fpath = (base_path / name).with_suffix(suffix)
    logger.debug(f"Saving image to {fpath}")
    if exif_data:
        image.save(fpath, exif=exif_data)
    else:
        image.save(fpath)
    return fpath


def dd_coordinate_to_dms(deg: float) -> tuple[int, int, float]:
    """
    Convert a single decimal degree coordinate to "degree minute seconds".

    The first digit returned may be positive or negative, which infers
    whether the direction is north (+) vs. south (-) or west (+) vs. east (-).
    """
    decimals, number = math.modf(deg)
    d = int(number)
    m = int(decimals * 60)
    s = (deg - d - m / 60) * 3600.00
    return d, m, s


def dd_location_to_dms(
    latitude: float, longitude: float
) -> dict[str, tuple[tuple[int, int, float], Literal["N", "S", "E", "W"]]]:
    """
    Format decimal degrees format to degrees, minutes, seconds (DMS)
    for the image EXIF data.
    """

    lat = dd_coordinate_to_dms(latitude)
    lat_ref = "N" if lat[0] >= 1 else "S"
    lon = dd_coordinate_to_dms(longitude)
    lon_ref = "E" if lon[0] >= 1 else "W"

    return {"latitude": (lat, lat_ref), "longitude": (lon, lon_ref)}


def get_platform() -> Literal["win", "macosx", "linux", "unknown"]:
    """
    A string identifying the current operating system. It is one
    of: `'win'`, `'linux'`,  `'macosx'`, or `'unknown'`.
    Adapted from kivy.utils.platform()
    https://github.com/kivy/kivy/blob/master/kivy/utils.py
    """
    if _sys_platform in ("win32", "cygwin"):
        return "win"
    elif _sys_platform == "darwin":
        return "macosx"
    elif _sys_platform.startswith("linux"):
        return "linux"
    elif _sys_platform.startswith("freebsd"):
        return "linux"
    return "unknown"


def get_app_dir(app_name: Optional[str] = None) -> pathlib.Path:
    """
    Returns the path to the directory in the users file system which the
    application can use to store additional data.

    Different platforms have different conventions with regards to where
    the user can store data such as preferences, saved games and settings.
    This function implements these conventions. The <app_name> directory
    is created when the property is called, unless it already exists.

    On Windows, `%APPDATA%/<app_name>` is returned.
    On OS X, `~/Library/Application Support/<app_name>` is returned.
    On Linux, `$XDG_CONFIG_HOME/<app_name>` is returned.

    Adapted from kivy.app._get_user_data_dir()
    https://github.com/kivy/kivy/blob/master/kivy/app.py
    """
    app_name = app_name or APP_NAME_SLUG
    platform = get_platform()
    data_dir = ""
    if platform == "win":
        data_dir = pathlib.Path(os.environ["APPDATA"])
    elif platform == "macosx":
        data_dir = pathlib.Path("~/Library/Application Support/")
    else:  # _platform == 'linux' or anything else...:
        data_dir = pathlib.Path(os.environ.get("XDG_CONFIG_HOME", "~/.config"))
    data_dir = data_dir.expanduser().resolve() / app_name
    if not data_dir.exists():
        data_dir.mkdir()
    return data_dir


def default_database_dsn(db_name: str = "trapdata") -> str:
    return f"sqlite+pysqlite:///{get_app_dir() / db_name}.db"


def initial_directory_choice():
    """
    Default to the user's home directory if no path to their
    trap data has been selected yet.

    This path likely will never be used, but it is helpful to
    have a placeholder `path.Path` for type annotations
    and it is a better starting directory for file dialog prompts
    than "." which is the directory of this python package.
    """
    return pathlib.Path("~/")
