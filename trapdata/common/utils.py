import csv
import datetime
import pathlib
import random
import string
from typing import Any, Union


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


def slugify(s):
    # Quick method to make an acceptable attribute name or url part from a title
    # install python-slugify for handling unicode chars, numbers at the beginning, etc.
    separator = "_"
    acceptable_chars = list(string.ascii_letters) + list(string.digits) + [separator]
    return (
        "".join(
            [
                chr
                for chr in s.replace(" ", separator).lower()
                if chr in acceptable_chars
            ]
        )
        .strip(separator)
        .replace(separator * 2, separator)
    )


def bbox_area(bbox: tuple[float, float, float, float]) -> float:
    """
    Return the area of a bounding box.

    Bounding boxes are assumed to be in the format:
    [(top-left-coordinate-pair), (bottom-right-coordinate-pair)]
    or: [x1, y1, x2, y2]


    >>> bbox_area([0, 0, 1, 1])
    1
    """
    x1, y1, x2, y2 = bbox
    area = (y2 - y1) * (x2 - x1)
    return area


def bbox_center(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    """
    Return the center coordinates of a bounding box.
    """
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    center_x = x1 + (width / 2)
    center_y = y1 + (height / 2)
    return (center_x, center_y)


def export_report(
    records: list[dict[str, Any]],
    report_name: str,
    directory: Union[pathlib.Path, str],
) -> Union[pathlib.Path, None]:
    if not records:
        return None

    filepath = (pathlib.Path(directory) / "reports" / report_name).with_suffix(".csv")
    if not filepath.parent.exists():
        filepath.parent.mkdir(parents=True, exist_ok=True)

    header = records[0].keys()

    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
        writer.writerow(header)
        for record in records:
            writer.writerow(record.values())

    return filepath


def format_timedelta(td: datetime.timedelta) -> str:
    minutes, seconds = divmod(td.seconds + td.days * 86400, 60)
    hours, minutes = divmod(minutes, 60)
    return "{:d}:{:02d}:{:02d}".format(hours, minutes, seconds)


def format_timedelta_hours(td: datetime.timedelta) -> str:
    minutes, seconds = divmod(td.seconds + td.days * 86400, 60)
    hours, minutes = divmod(minutes, 60)
    display_parts = []
    if hours:
        display_parts.append(f"{str(hours).lstrip('0')} hours")
    if minutes:
        display_parts.append(f"{str(minutes).lstrip('0')} min")
    if not hours and not minutes:
        display_parts.append(f"{str(seconds).lstrip('0') or '0'} seconds")

    display_str = ", ".join(display_parts)
    return display_str


def random_color():
    color = [random.random() for _ in range(3)]
    color.append(0.8)  # alpha
    return color
