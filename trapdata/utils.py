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
import json
import collections

import PIL
from plyer import filechooser


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


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


def choose_directory(
    cache=True, setting_key="last_root_directory", starting_path=TEMPORARY_BASE_PATH
):
    """
    Prompt the user to select a directory where trap data has been saved.
    The subfolders of this directory should be timestamped directories
    with nightly trap images.

    The user's selection is saved and reused on the subsequent launch.
    """
    # @TODO Look for SDCARD / USB Devices first?

    if cache:
        selected_dir = read_setting(setting_key)
    else:
        selected_dir = None

    if selected_dir:
        selected_dir = pathlib.Path(selected_dir)

        if selected_dir.is_dir():
            return selected_dir
        else:
            delete_setting(setting_key)

    selection = filechooser.choose_dir(
        title="Choose the root directory for your nightly trap data",
        path=starting_path,
    )

    if selection:
        selected_dir = selection[0]
    else:
        return None

    save_setting(setting_key, selected_dir)

    return selected_dir


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
    print("Looking for nightly timestamped folders")
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
            logger.debug(f"Found nightly folder for {date}: {d}")
            nights[date] = d

    # @TODO should be sorted by date
    return nights


def find_images(path):
    """
    @TODO speed this up!
    """
    print("Looking for images")
    images = list(path.glob("*.jpg"))
    # images = [
    #     f for f in path.iterdir() if f.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    # ]
    return images


def find_annotations(path):
    """
    @TODO sort by date modified?
    """
    # annotations = [f for f in path.glob(SUPPORTED_ANNOTATION_PATTERNS)]
    annotations = [f for f in path.glob("*.json")]
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


def choose_sample(images, direction, last_sample=None):

    if last_sample:
        last_sample = pathlib.Path(last_sample).name
        last_idx = images.index(last_sample)
    else:
        last_idx = 0

    if direction > 0:
        idx = last_idx + 1
    elif direction < 0:
        idx = last_idx - 1
    print("Sample index:", idx)
    sample = images[idx % len(images)]
    print("Last sample :", last_sample)
    print("Loading sample :", sample)
    return sample


def parse_annotations(annotations, format="aditya"):
    if format == "aditya":
        bboxes, _, binary_labels, labels, scores = annotations
        annotations = [
            {"bbox": bbox, "label": label, "score": score}
            for bbox, label, score in zip(bboxes, labels, scores)
        ]
        return annotations
    else:
        raise NotImplementedError


def get_sequential_sample(direction, source_dir, last_sample=None):
    source_dir = pathlib.Path(source_dir)
    annotation_files = find_annotations(source_dir)
    if annotation_files:
        annotations = json.load(open(annotation_files[0]))
    else:
        raise Exception(f"No annotations found in directory: {source_dir}")

    if "images" in annotations:
        # This is from MegaDetector

        samples = list(annotations["images"])
        # sample = random.choice(list(annotations["images"]))
        samples = {s["file"]: s for s in samples}
        filenames = list(samples.keys())
        img_name = choose_sample(filenames, direction, last_sample)
        img_path = source_dir / img_name
        sample = samples[img_name]
        bboxes = []
        for detection in sample["detections"]:
            img_width, img_height = PImage.open(img_path).size
            print("PIL image:", img_width, img_height)
            x, y, width, height = detection["bbox"]
            x1 = x * img_width
            y1 = y * img_height
            x2 = (width * img_width) + x1
            y2 = (height * img_height) + y1
            bbox = [x1, y1, x2, y2]
            print("MegaDetector bbox:", detection["bbox"])
            print("MegaDetector bbox converted:", bbox)
            bboxes.append(bbox)
    else:
        # Assume this is Aditya's format

        images = list(annotations.keys())
        # sample = random.choice(list(annotations.keys()))
        sample = choose_sample(images, direction, last_sample)
        img_path = source_dir / sample
        parsed_annotations = parse_annotations(annotations[sample])

    return img_path, parsed_annotations


def summarize_species(path):
    """
    Summarize the data in an annotations file
    """
    data = json.load(open(path))
    species = collections.defaultdict(list)

    for img_path, annotations in data.items():
        # Aditya's format
        ants = parse_annotations(annotations)
        for ant in ants:
            label = ant["label"]
            species[label].append((img_path, ant))

    return species
