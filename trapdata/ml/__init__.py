import glob
import sys
import importlib

from trapdata import logger


# @TODO move this to the model registry
LOCALIZATION_MODELS = {
    "FasterRCNN MobileNet": "trapdata.ml.localization.fasterrcnn_mobilenet",
    "Custom FasterRCNN": "trapdata.ml.localization.fasterrcnn_full",
    "MegaDectector v5": "trapdata.ml.localization.megadetectorv5",
}


# These are separate for the settings choices
BINARY_CLASSIFICATION_MODELS = {
    "Moth / Non-Moth": "trapdata.ml.classification.moth_nonmoth",
}

TAXON_CLASSIFICATION_MODELS = {
    "Quebec & Vermont Species": "trapdata.ml.classification.quebec_vermont_species",
    "UK & Denmark Species": "trapdata.ml.classification.uk_denmark_species",
}

# These are combined for selecting a model with the same function
CLASSIFICATION_MODELS = {}
CLASSIFICATION_MODELS.update(BINARY_CLASSIFICATION_MODELS)
CLASSIFICATION_MODELS.update(TAXON_CLASSIFICATION_MODELS)


def detect_objects(model_name, **kwargs):

    module_path = LOCALIZATION_MODELS[model_name]
    logger.debug(f"Loading object detection model: {module_path}")
    model_module = importlib.import_module(module_path)

    logger.debug(f"Calling predict with arguments: {kwargs}")
    model_module.predict(**kwargs)
    logger.debug("Predict complete")


def classify_objects(model_name, **kwargs):
    module_path = CLASSIFICATION_MODELS[model_name]
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
