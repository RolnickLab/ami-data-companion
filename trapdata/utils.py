import sys
import os
import re
import pathlib
import logging
import pathlib
import random
import tempfile
import dateutil.parser
from dateutil import tz
import logging
import requests
import base64
import io
import json
import csv
import collections
import time
import datetime

import PIL.ExifTags, PIL.Image
from plyer import filechooser


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())

# if hasattr(sys.modules["__main__"], "_SpoofOut"):
#     # If running tests,
#     logger.setLevel(logging.DEBUG)
#     logger.addHandler(logging.StreamHandler())


SUPPORTED_IMAGE_EXTENSIONS = (".jpg", ".jpeg")
SUPPORTED_ANNOTATION_PATTERNS = ("detections.json", "megadetections.json")
# TEST_IMAGES_BASE_PATH = "/media/michael/LaCie/AMI/"
TEST_IMAGES_BASE_PATH = "/home/michael/Projects/AMI/TRAPDATA/Moth Week/"
# TEST_IMAGES_BASE_PATH = "/home/michael/Projects/AMI/TRAPDATA/Quebec/"
NULL_DETECTION_LABELS = ["nonmoth"]


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


def choose_directory(cache=True, setting_key="last_root_directory", starting_path=None):
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


def find_images(path):
    """
    @TODO speed this up!
    """
    logger.debug(f"Looking for images in {path}")
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
    path = pathlib.Path(path)
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

        samples = sorted(list(annotations["images"]))
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

        images = sorted(list(annotations.keys()))
        # sample = random.choice(list(annotations.keys()))
        sample = choose_sample(images, direction, last_sample)
        img_path = source_dir / sample
        parsed_annotations = parse_annotations(annotations[sample])

    return img_path, parsed_annotations


def summarize_species(path, best_only=False):
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

    def prune(species):
        # Pick only the annotation with the top score
        print("Pruning species")
        filtered_species = {}
        for label, ants in species.items():
            # print(label, ants)
            top_score = -10000
            top_pixels = 0
            for img_path, ant in ants:
                x1, y1, x2, y2 = ant["bbox"]
                pixels = (x2 - x1) * (y2 - y1)
                if (
                    ant["score"] > top_score
                    and ant["label"] not in NULL_DETECTION_LABELS
                ):
                    # if pixels > top_pixels:
                    filtered_species[label] = {
                        "image": img_path,
                        "count": len(ants),
                        "bbox": ant["bbox"],
                        "score": ant["score"],
                    }
                    top_score = ant["score"]
                    top_pixels = pixels

        # if "nonmoth" not in filtered_species:
        #     raise Exception("WHERE ARE YOU")

        return filtered_species

    if best_only:
        species = prune(species)

    return species


def slugify(s):
    return s.replace(" ", "_").lower()


def parse_annotations_to_kivy_atlas(path):
    atlas = {}
    path = pathlib.Path(path)
    data = json.load(open(path))
    species = summarize_species(path, best_only=True)
    for name, ant in species.items():
        # print(name, ant)
        # Aditya's format
        # img_path = str(path / img_path)
        # img_name = pathlib.Path(img_path).name
        if ant["image"] not in atlas:
            atlas[ant["image"]] = {}

        img_width, img_height = PIL.Image.open(path.parent / ant["image"]).size

        label = slugify(name)
        x1, y1, x2, y2 = ant["bbox"]
        # Reference from bottom left instead of top left
        y1 = img_height - y1
        y2 = img_height - y2
        w = x2 - x1
        h = y2 - y1
        atlas[ant["image"]][label] = [x1, y1, w, h]

    fpath = path.parent / "trapdata.atlas"
    json.dump(atlas, open(fpath, "w"), indent=2)

    # print("Atlas data:")
    # print(json.dumps(atlas, indent=2))

    # Test

    # from kivy.atlas import Atlas
    # atlas_ready = Atlas(str(fpath))
    # print(atlas_ready.textures.keys())
    # import ipdb

    # ipdb.set_trace()
    return fpath


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


def get_image_timestamp(img_path):
    """
    Parse the date and time a photo was taken from its EXIF data.

    Also sets the timezone based on the TimeZoneOffset field if available.
    Example EXIF offset: "-4". Some libaries expect the format to be: "-04:00"
    However dateutil.parse seems to handle "-4" or "+4" just fine.
    """
    exif = get_exif(img_path)
    datestring = exif["DateTime"].replace(":", "-", 2)
    offset = exif.get("TimeZoneOffset")
    if offset:
        datestring = f"{datestring} {offset}"
    date = dateutil.parser.parse(datestring)
    # print(date.strftime("%c %z"))
    return date


def export_report(path, *args):
    path = pathlib.Path(path)
    annotation_files = find_annotations(path)
    if not annotation_files:
        print("No annotation files found at", path)
        data = {}
    else:
        data = json.load(open(annotation_files[0]))

    fname = path / "report.csv"
    records = []

    print("Generating report")
    for img_fname, annotations in data.items():
        # Aditya's format
        ants = parse_annotations(annotations)
        for ant in ants:
            ant["image"] = img_fname
            ant["time"] = get_image_timestamp(path / img_fname)
            records.append(ant)

    if records:
        header = records[0].keys()
    else:
        header = []

    print("Saving report to", fname)
    with open(fname, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for record in records:
            if record["label"] not in NULL_DETECTION_LABELS:
                writer.writerow(record.values())

    return fname


def find_images(
    path,
    absolute_paths=False,
    include_timestamps=True,
    skip_bad_exif=False,
):

    extensions_list = "|".join([f.lstrip(".") for f in SUPPORTED_IMAGE_EXTENSIONS])
    pattern = f"\.({extensions_list})$"
    for root, dirs, files in os.walk(path):
        for name in files:
            if re.search(pattern, name, re.IGNORECASE):
                full_path = os.path.join(root, name)
                relative_path = full_path.split(path)[-1]
                path = full_path if absolute_paths else relative_path

                if include_timestamps:
                    try:
                        date = get_image_timestamp(full_path)
                    except Exception as e:
                        print("Could not get EXIF date for image", full_path, e)
                        if skip_bad_exif:
                            continue
                        else:
                            date = None
                    finally:
                        yield path, date
                else:
                    yield path


def group_images_by_session(images, maximum_gap_minutes=6 * 60):
    """
    Find consecutive images and group them into daily/nightly monitoring sessions.
    If the time between two photos is greater than `maximumm_time_gap` (in minutes)
    then start a new session group. Each new group uses the first photo's day
    as the day of the session even if consecutive images are taken past midnight.

    @TODO make fake images for this test
    >>> images = find_images(TEST_IMAGES_BASE_PATH, skip_bad_exif=True)
    >>> sessions = group_images_by_session(images)
    >>> len(sessions)
    11
    """
    images = sorted(images, key=lambda image_and_date: image_and_date[1])
    if not images:
        return []

    sessions = collections.OrderedDict()

    first_image, first_timestamp = images[0]

    last_timestamp = None
    current_day = None

    for image, timestamp in images:
        if last_timestamp:
            delta = (timestamp - last_timestamp).seconds / 60
        else:
            delta = maximum_gap_minutes

        logger.debug(f"{timestamp}, {round(delta, 2)}")

        if delta >= maximum_gap_minutes:
            current_day = timestamp.date()
            logger.debug(
                f"Gap of {round(delta/60, 1)} hours detected. Starting new session for date: {current_day}"
            )
            sessions[current_day] = []

        sessions[current_day].append((image, timestamp))
        last_timestamp = timestamp

    # This is for debugging
    for day, images in sessions.items():
        _, first_date = images[0]
        _, last_date = images[-1]
        delta = last_date - first_date
        hours = round(delta.seconds / 60 / 60, 1)
        logger.debug(
            f"Found session on {day} with {len(images)} images that ran for {hours} hours.\n"
            f"From {first_date.strftime('%c')} to {last_date.strftime('%c')}."
        )

    return sessions

    # yield relative_path, get_image_timestamp(full_path)
