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
from kivy.logger import Logger as logger

from . import db

# logger = logging.getLogger().getChild(__name__)
logger.setLevel(logging.DEBUG)
# logger.addHandler(logging.StreamHandler())

# if hasattr(sys.modules["__main__"], "_SpoofOut"):
#     # If running tests,
#     logger.setLevel(logging.DEBUG)
#     logger.addHandler(logging.StreamHandler())


SUPPORTED_IMAGE_EXTENSIONS = (".jpg", ".jpeg")
SUPPORTED_ANNOTATION_PATTERNS = ("detections.json", "megadetections.json")
# TEST_IMAGES_BASE_PATH = "/media/michael/LaCie/AMI/"
TEST_IMAGES_BASE_PATH = "/home/michael/Projects/AMI/TRAPDATA/Test/"
# TEST_IMAGES_BASE_PATH = "/home/michael/Projects/AMI/TRAPDATA/Quebec/"

POSITIVE_BINARY_LABEL = "moth"
NEGATIVE_BINARY_LABEL = "nonmoth"
NULL_DETECTION_LABELS = [NEGATIVE_BINARY_LABEL]

POSITIVE_COLOR = [0, 100 / 255, 1, 0.8]  # Blue
NEGATIVE_COLOR = [1, 0, 162 / 255, 1]  # Pink


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


def predict_image_from_service(path):
    """
    Simple method for getting predictions from a webservice.

    No longer used.
    """
    img = PIL.Image.open(path)
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue())
    resp = requests.post("http://localhost:5000/predict", data={"b64": img_str})
    resp.raise_for_status()
    results = resp.json()
    return results


def choose_sample_from_filesystem(images, direction, last_sample=None):

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


def get_sequential_sample(direction, images, last_sample=None):
    if not images:
        return None

    if last_sample:
        last_idx = images.index(last_sample)
    else:
        last_idx = 0

    if direction > 0:
        idx = last_idx + 1
    elif direction < 0:
        idx = last_idx - 1

    sample = images[idx % len(images)]
    return sample


def get_sequential_sample_from_filesystem(direction, source_dir, last_sample=None):
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
        img_name = choose_sample_from_filesystem(filenames, direction, last_sample)
        img_path = source_dir / img_name
        sample = samples[img_name]
        bboxes = []
        for detection in sample["detections"]:
            img_width, img_height = PIL.Image.open(img_path).size
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
        sample = choose_sample_from_filesystem(images, direction, last_sample)
        img_path = source_dir / sample
        parsed_annotations = parse_annotations(annotations[sample])

    return img_path, parsed_annotations


def summarize_species(ms, best_only=False):
    """
    Summarize the data in an annotations file
    """

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


def summarize_species_from_file(path, best_only=False):
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
    base_directory,
    absolute_paths=False,
    include_timestamps=True,
    skip_bad_exif=True,
):
    logger.info(f"Scanning '{base_directory}' for images")
    base_directory = pathlib.Path(base_directory)
    extensions_list = "|".join([f.lstrip(".") for f in SUPPORTED_IMAGE_EXTENSIONS])
    pattern = f"\.({extensions_list})$"
    for walk_path, dirs, files in os.walk(base_directory):
        for name in files:
            if re.search(pattern, name, re.IGNORECASE):
                relative_path = pathlib.Path(walk_path) / name
                full_path = base_directory / relative_path
                path = full_path if absolute_paths else relative_path

                if include_timestamps:
                    try:
                        date = get_image_timestamp(full_path)
                    except Exception as e:
                        logger.error(
                            f"Could not get EXIF date for image: {full_path}\n {e}"
                        )
                        if skip_bad_exif:
                            continue
                        else:
                            date = None
                else:
                    date = None

                yield {"path": path, "timestamp": date}


def bbox_area(bbox):
    """
    Return the area of a bounding box.

    Bounding boxes are assumed to be in the format:
    [(top-left-coordinate-pair), (bottom-right-coordinate-pair)]
    or: [x1, y1, x2, y2]
    """
    x1, y1, x2, y2 = bbox
    area = (y2 - y1) * (x2 - x1)
    return area


# @TODO add other group by methods? like image size, camera model, random sample batches, etc. Add to UI settings
def group_images_by_day(images, maximum_gap_minutes=6 * 60):
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
    logger.info(
        f"Grouping images into date-based groups with a maximum gap of {maximum_gap_minutes} minutes"
    )
    images = sorted(images, key=lambda image: image["timestamp"])
    if not images:
        return []

    groups = collections.OrderedDict()

    first_image = images[0]

    last_timestamp = None
    current_day = None

    for image in images:
        if last_timestamp:
            delta = (image["timestamp"] - last_timestamp).seconds / 60
        else:
            delta = maximum_gap_minutes

        # logger.debug(f"{timestamp}, {round(delta, 2)}")

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


def get_monitoring_sessions_from_filesystem(base_directory):
    # @TODO can we use the sqlalchemy classes for sessions & images before
    # they are saved to the DB?
    images = find_images(base_directory)
    sessions = []
    groups = group_images_by_day(images)
    for day, images in groups.items():
        sessions.append(
            {
                "base_directory": str(base_directory),
                "day": day,
                "num_images": len(images),
                "start_time": images[0]["timestamp"],
                "end_time": images[-1]["timestamp"],
                "images": images,
            }
        )
    sessions.sort(key=lambda s: s["day"])
    return sessions


def save_monitoring_session(base_directory, session):
    with db.get_session(base_directory) as sess:
        ms_kwargs = {"base_directory": str(base_directory), "day": session["day"]}
        ms = sess.query(db.MonitoringSession).filter_by(**ms_kwargs).one_or_none()
        if ms:
            logger.debug(f"Found existing Monitoring Session in db: {ms}")
        else:
            ms = db.MonitoringSession(**ms_kwargs)
            logger.debug(f"Adding new Monitoring Session to db: {ms}")
            sess.add(ms)
            sess.flush()

        num_existing_images = (
            sess.query(db.Image).filter_by(monitoring_session_id=ms.id).count()
        )
        if session["num_images"] > num_existing_images:
            logger.info(
                f"session images: {session['num_images']}, saved count: {num_existing_images}"
            )
            # Compare the number of images known in this session
            # Only scan & add images if there is a difference.
            # This does not delete missing images.
            ms_images = []
            for image in session["images"]:
                path = pathlib.Path(image["path"]).relative_to(base_directory)
                img_kwargs = {
                    "monitoring_session_id": ms.id,
                    "path": str(image["path"]),  # @TODO these should be relative paths!
                    "timestamp": image["timestamp"],
                }
                db_img = sess.query(db.Image).filter_by(**img_kwargs).one_or_none()
                if db_img:
                    # logger.debug(f"Found existing Image in db: {img}")
                    pass
                else:
                    db_img = db.Image(**img_kwargs)
                    logger.debug(f"Adding new Image to db: {db_img}")
                ms_images.append(db_img)
            sess.bulk_save_objects(ms_images)

            # Manually update aggregate & cached values after bulk update
            ms.update_aggregates()

        logger.debug("Comitting changes to DB")
        sess.commit()
        logger.debug("Done committing")


def save_monitoring_sessions(base_directory, sessions):

    for session in sessions:
        save_monitoring_session(base_directory, session)

    return get_monitoring_sessions_from_db(base_directory)


def get_monitoring_sessions_from_db(base_directory):
    logger.info("Quering existing sessions in DB")
    with db.get_session(base_directory) as sess:
        results = (
            sess.query(db.MonitoringSession)
            .filter_by(base_directory=str(base_directory))
            .all()
        )
    return list(results)


def get_monitoring_session_images(ms):
    # @TODO this is likely to slow things down. Some monitoring sessions have thousands of images.
    with db.get_session(ms.base_directory) as sess:
        images = list(sess.query(db.Image).filter_by(monitoring_session_id=ms.id).all())
    logger.info(f"Found {len(images)} images in Monitoring Session: {ms}")
    return images


def get_monitoring_session_image_ids(ms):
    # Get a list of image IDs in order of timestamps as quickly as possible
    # This could be in the thousands
    with db.get_session(ms.base_directory) as sess:
        images = list(
            sess.query(db.Image.id)
            .filter_by(monitoring_session_id=ms.id)
            .order_by(db.Image.timestamp)
            .all()
        )
    logger.info(f"Found {len(images)} images in Monitoring Session: {ms}")
    return images


def save_detected_objects(monitoring_session, image_paths, detected_objects_data):
    logger.debug(
        f"Callback was called! {monitoring_session}, {image_paths}, {detected_objects_data}"
    )
    base_directory = monitoring_session.base_directory
    with db.get_session(base_directory) as sess:
        timestamp = datetime.datetime.now()
        for image_path, detected_objects in zip(image_paths, detected_objects_data):
            image_kwargs = {
                "path": str(image_path),
                "monitoring_session_id": monitoring_session.id,
            }
            image = sess.query(db.Image).filter_by(**image_kwargs).one()
            image.last_processed = timestamp
            sess.add(image)
            for object_data in detected_objects:
                detection = db.DetectedObject(
                    last_detected=timestamp,
                    in_queue=True,
                )

                if "bbox" in object_data:
                    area_pixels = bbox_area(object_data["bbox"])
                    object_data["area_pixels"] = area_pixels

                for k, v in object_data.items():
                    logger.debug(f"Adding {k}: {v} to detected object {detection.id}")
                    setattr(detection, k, v)

                logger.debug(f"Saving detected object {detection} for image {image}")
                sess.add(detection)
                detection.monitoring_session_id = monitoring_session.id
                detection.image_id = image.id
        sess.commit()


def save_classified_objects(monitoring_session, object_ids, classified_objects_data):
    logger.debug(
        f"Callback was called! {monitoring_session}, {object_ids}, {classified_objects_data}"
    )
    base_directory = monitoring_session.base_directory
    with db.get_session(base_directory) as sess:
        timestamp = datetime.datetime.now()
        for object_id, object_data in zip(object_ids, classified_objects_data):
            obj = sess.get(db.DetectedObject, object_id)
            obj.last_processed = timestamp
            sess.add(obj)

            for k, v in object_data.items():
                logger.debug(f"Adding {k}: {v} to detected object {obj.id}")
                setattr(obj, k, v)

            logger.debug(f"Saving classifed object {obj}")

        sess.commit()


def get_detected_objects(monitoring_session):
    base_directory = monitoring_session.base_directory
    with db.get_session(base_directory) as sess:
        query_kwargs = {
            "monitoring_session_id": monitoring_session.id,
        }
        for obj in sess.query(db.DetectedObject).filter_by(**query_kwargs).all():
            yield obj


def get_image_with_objects(monitoring_session, image_id):
    base_directory = monitoring_session.base_directory
    with db.get_session(base_directory) as sess:
        image_kwargs = {
            "id": image_id,
            # "path": str(image_path),
            # "monitoring_session_id": monitoring_session.id,
        }
        image = (
            sess.query(db.Image)
            .filter_by(**image_kwargs)
            .options(db.orm.joinedload(db.Image.detected_objects))
            .one_or_none()
        )
        logger.debug(
            f"Found image {image} with {len(image.detected_objects)} detected objects"
        )
        return image


def add_sample_to_queue(monitoring_session, sample_size=10):
    # if sample_size:
    #     order = db.func.random()
    # else:
    #     order = db.Image.timestamp

    ms = monitoring_session

    # image_ids = get_monitoring_session_image_ids(ms)
    # logger.info(f"Adding {len(image_ids)} images into queue")

    # with db.get_session(ms.base_directory) as sess:
    #     sess.execute(
    #         db.sa.update(db.Image)
    #         .where(db.Image.monitoring_session_id == ms.id)
    #         .values(in_queue=True)
    #     )

    # Add random images to queue if the queue has fewer than n samples

    with db.get_session(ms.base_directory) as sess:
        num_in_queue = (
            sess.query(db.Image)
            .filter_by(in_queue=True, monitoring_session_id=ms.id)
            .count()
        )
        if num_in_queue < sample_size:
            for image in (
                sess.query(db.Image)
                .filter_by(
                    in_queue=False,
                    monitoring_session_id=ms.id,
                )
                .order_by(db.sa.func.random())
                .limit(sample_size - num_in_queue)
                .all()
            ):
                image.in_queue = True
                sess.add(image)
            sess.commit()


def clear_queue():
    pass
