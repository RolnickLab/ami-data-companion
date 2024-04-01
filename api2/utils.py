from __future__ import annotations

import datetime
import pathlib
import string
import time

import pandas as pd
import PIL.Image
import PIL.ImageFile
import torch
import torchvision

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
