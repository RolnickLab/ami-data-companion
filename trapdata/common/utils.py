import csv
import pathlib
from typing import Union, Any


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
    return s.replace(" ", "_").lower()


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


def bbox_center(bbox):
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
