import pathlib
import json
import urllib.request

import torch
import torchvision

from trapdata import logger

LOCAL_WEIGHTS_PATH = pathlib.Path(torch.hub.get_dir())
logger.info(f"LOCAL WEIGHTS PATH: {LOCAL_WEIGHTS_PATH}")


def get_device(device_str=None):
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


def get_or_download_file(path, destination_dir=None):
    """
    >>> filename, headers = get_weights("https://drive.google.com/file/d/1KdQc56WtnMWX9PUapy6cS0CdjC8VSdVe/view?usp=sharing")

    """
    # If path is a local path instead of a URL then urlretrieve will just return that path
    fname = path.rsplit("/", 1)[-1]
    if destination_dir:
        destination_dir = pathlib.Path(destination_dir)
        if not destination_dir.exists():
            logger.info(f"Creating local directory {str(destination_dir)}")
            destination_dir.mkdir(parents=True, exist_ok=True)
        local_filepath = pathlib.Path(destination_dir) / fname
    else:
        # If local_filepath is None, file will be downloaded to a temporary location
        # But will likely be downloaded every time the model is invoked
        local_filepath = None

    if local_filepath and local_filepath.exists():
        logger.info(f"Using existing {local_filepath}")
        return local_filepath

    else:
        logger.info(f"Downloading {path}")
        resulting_filepath, headers = urllib.request.urlretrieve(
            url=path, filename=local_filepath
        )
        logger.info(f"Downloaded to {resulting_filepath}")
        return resulting_filepath


def get_category_map(labels_path):
    with open(LOCAL_WEIGHTS_PATH / labels_path) as f:
        labels = json.load(f)

    # @TODO would this be faster as a list? especially when getting the labels of multiple
    # indexes in one prediction
    index_to_label = {index: label for label, index in labels.items()}

    return index_to_label


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
