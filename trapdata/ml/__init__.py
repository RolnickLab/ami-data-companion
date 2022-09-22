import pathlib
import glob
import torch
import sys
import importlib

from ..utils import logger
from . import localization


LOCALIZATION_MODELS = [
    "fasterrcnn",
    "fasterrcnn_mobilenet",
]

BINARY_CLASSIFICATION_MODELS = [
    "moth_nonmoth",
    "insect_noninsect",
]

TAXON_CLASSIFICATION_MODELS = [
    "efficientnetv4",
]

# Original model weight paths:
# model_localize=MODEL_BASE_PATH / "v1_localizmodel_2021-08-17-12-06.pt",
# model_moth=MODEL_BASE_PATH / "mothsv2_20220421_110638_30.pth",
# model_moth_nonmoth=MODEL_BASE_PATH / "moth-nonmoth-effv2b3_20220506_061527_30.pth",
# category_map_moth=MODEL_BASE_PATH / "03-mothsv2_category_map.json",
# category_map_moth_nonmoth=MODEL_BASE_PATH / "05-moth-nonmoth_category_map.json",
# LOCAL_WEIGHTS_PATH = pathlib.Path(torch.hub.get_dir())


def detect_objects(model_name, **kwargs):

    logger.debug(f"Loading object detection model: {model_name}")
    model_module = importlib.import_module("." + model_name, "trapdata.ml.localization")

    logger.debug(f"Calling predict with arguments: {kwargs}")
    model_module.predict(**kwargs)


def classify_objects(model_name, **kwargs):
    pass


def watch_queue(db):
    """
    query db for new images to detect ojects in
    create a batch, remove from initial queue.

    query db for objects that need cropping & initial classification

    query db that need fine-grained taxon classification
    """
    pass


if __name__ == "__main__":
    source_dir = sys.argv[1]
    image_list = glob(source_dir)
    detect_objects(source_dir, image_list)
