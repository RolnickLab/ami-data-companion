import pathlib
import glob
import torch
import sys
import importlib

from ..utils import logger
from . import localization


LOCALIZATION_MODELS = {
    "FasterRCNN MobileNet": "trapdata.ml.localization.fasterrcnn_mobilenet",
    "Custom FasterRCNN": "trapdata.ml.localization.fasterrcnn",
    "MegaDectector v5": "trapdata.ml.localization.megadetectorv5",
}


BINARY_CLASSIFICATION_MODELS = {
    "Moth / Non-Moth": "trapdata.ml.classification.moth_nonmoth",
}

TAXON_CLASSIFICATION_MODELS = {
    "Ottawa & Vermont Macromoth Species": "trapdata.ml.classification.ottawa_macromoths",
}

# Original model weight paths:
# model_localize=MODEL_BASE_PATH / "v1_localizmodel_2021-08-17-12-06.pt",
# model_moth=MODEL_BASE_PATH / "mothsv2_20220421_110638_30.pth",
# model_moth_nonmoth=MODEL_BASE_PATH / "moth-nonmoth-effv2b3_20220506_061527_30.pth",
# category_map_moth=MODEL_BASE_PATH / "03-mothsv2_category_map.json",
# category_map_moth_nonmoth=MODEL_BASE_PATH / "05-moth-nonmoth_category_map.json",
# LOCAL_WEIGHTS_PATH = pathlib.Path(torch.hub.get_dir())


def detect_objects(model_name, **kwargs):

    module_path = LOCALIZATION_MODELS[model_name]
    logger.debug(f"Loading object detection model: {module_path}")
    model_module = importlib.import_module(module_path)

    logger.debug(f"Calling predict with arguments: {kwargs}")
    model_module.predict(**kwargs)
    logger.debug("Predict complete")


def classify_objects(model_name, **kwargs):
    module_path = BINARY_CLASSIFICATION_MODELS[model_name]
    logger.debug(f"Loading classification model: {module_path}")
    model_module = importlib.import_module(module_path)

    logger.debug(f"Calling predict with arguments: {kwargs}")
    model_module.predict(**kwargs)
    logger.debug("Predict complete")


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
