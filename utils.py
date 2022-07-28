import pathlib
import logging
import pathlib
import random
import tempfile
import dateutil.parser
import logging
import requests
import base64
import io

import PIL
from plyer import filechooser


logger = logging.getLogger(__name__)


SUPPORTED_IMAGE_EXTENSIONS = (".jpg", ".jpeg")
SUPPORTED_ANNOTATION_PATTERNS = ("detections.json", "megadetections.json")
TEMPORARY_BASE_PATH = "/media/michael/LaCie/AMI/"


def cache_dir(path=None):
    # If fails, use temp dir?
    # d = tempfile.TemporaryDirectory(delete=False)
    path = path or ".cache"
    d = pathlib.Path(".cache")
    d.mkdir(exist_ok=True)
    return d


def save_setting(key, val):
    """
    >>> save_setting("last_test", "now")
    'now'
    >>> read_setting("last_test")
    'now'
    """
    f = cache_dir() / key
    logger.debug(f"Writing to cache: {f}")
    f.write_text(val)
    return val


def read_setting(key):
    f = cache_dir() / key
    logger.debug(f"Checking cache: {f}")
    if f.exists():
        return f.read_text()
    else:
        return None


def delete_setting(key):
    f = cache_dir() / key
    logger.debug(f"Deleting cache: {f}")
    if f.exists():
        return f.unlink()
    else:
        return None


def choose_root_directory(cache=True):
    """
    Prompt the user to select a directory where trap data has been saved.
    The subfolders of this directory should be timestamped directories
    with nightly trap images.

    The user's selection is saved and reused on the subsequent launch.
    """
    # @TODO Look for SDCARD / USB Devices first?

    setting_key = "last_root_dir"

    if cache:
        root_dir = read_setting(setting_key)
    else:
        root_dir = None

    if root_dir:
        root_dir = pathlib.Path(root_dir)

        if root_dir.is_dir():
            return root_dir
        else:
            delete_setting(setting_key)

    selection = filechooser.choose_dir(
        title="Choose the root directory for your nightly trap data",
        path=TEMPORARY_BASE_PATH,
    )

    if selection:
        root_dir = selection[0]
    else:
        return None

    save_setting(setting_key, root_dir)

    return root_dir


def find_timestamped_folders(path):
    """
    Find all directories in a given path that have
    dates / timestamps in the name.

    This should be the nightly folders from the trap data.

    >>> pathlib.Path("./tmp/2022_05_14").mkdir(exist_ok=True, parents=True)
    >>> pathlib.Path("./tmp/nope").mkdir(exist_ok=True, parents=True)
    >>> find_timestamped_folders("./tmp")
    [PosixPath('tmp/2022_05_14')]
    """
    nights = {}

    def _preprocess(name):
        return name.replace("_", "-")

    for d in pathlib.Path(path).iterdir():
        # @TODO use yield?
        try:
            date = dateutil.parser.parse(_preprocess(d.name))
        except Exception:
            # except dateutil.parser.ParserError:
            pass
        else:
            nights[date] = d

    # @TODO should be sorted by date
    return nights


def find_images(path):
    images = [
        f for f in path.iterdir() if f.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    ]
    return images


def find_annotations(path):
    annotations = [
        f for f in path.iterdir() if f.name.lower() in SUPPORTED_ANNOTATION_PATTERNS
    ]
    return annotations


def predict_image(path):
    img = PIL.Image.open(path)
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue())
    resp = requests.post("http://localhost:5000/predict", data={"b64": img_str})
    resp.raise_for_status()
    results = resp.json()
    return results
