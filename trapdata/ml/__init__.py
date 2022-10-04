import torch
import torchvision

from trapdata import logger
from trapdata.ml.utils import get_device, StopWatch


class InferenceModel:
    key = None
    title = None
    description = None
    device = None
    weights = None
    labels = None
    model = None
    transforms = None
    batch_size = 4
    num_workers = 2
    description = str()
    db_path = None

    def __init__(self, *args, **kwargs):
        for k, v in kwargs:
            setattr(self, k, v)

        logger.info(
            f"Initializing model {self.name} on device: {self.device} with weights: {self.weights}"
        )

        self.device = self.device or get_device()
        self.model = self.get_model(self.weights)
        self.transforms = self.get_transforms()
        self.dataset = self.get_dataset()
        self.dataloader = self.get_dataloader()

    def get_model(self, weights):
        model = torch.nn.Module()
        model.load_state_dict(weights)
        model = model.to(self.device)
        model.eval()
        return model

    def get_transforms(self):
        transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
            ]
        )
        return transforms

    def get_dataset(self):
        dataset = torch.utils.data.Dataset(
            image_transforms=self.get_transforms(),
        )
        return dataset

    def get_dataloader(self):
        logger.info(
            f"Preparing dataloader with batch size of {self.batch_size} and {self.num_workers} workers."
        )
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
            shuffle=False,
            pin_memory=True,  # @TODO review this
        )
        return self.dataloader

    def predict_batch(self, batch):
        batch_input = batch.to(
            self.device,
            non_blocking=True,  # Block while in development, are we already in a background process?
        )
        batch_output = self.model(batch_input)
        return batch_output

    def post_process(self, batch_output):
        return batch_output

    def format_output(self, batch_output):
        return batch_output

    def save_results(self, batch_output):
        logger.warn(
            "No save method configured for model. Doing nothing with model output!"
        )
        return None

    def run(self):
        with torch.no_grad():
            for i, batch_input in enumerate(self.dataloader):

                logger.info(f"Running batch {i+1} out of {len(self.dataloader)}")

                with StopWatch() as batch_time:
                    batch_output = self.predict_batch(batch_input)

                seconds_per_item = batch_time.duration / len(batch_output)

                batch_output = self.post_process(batch_output)

                logger.info(
                    f"Inference time for batch: {batch_time}, "
                    f"Seconds per item: {round(seconds_per_item, 1)}"
                )

                self.save_results(batch_output)


import importlib

from trapdata import logger


# @TODO move this to the model registry
LOCALIZATION_MODELS = {
    "FasterRCNN MobileNet": "trapdata.ml.localization.fasterrcnn_mobilenet",
    "Custom FasterRCNN": "trapdata.ml.localization.fasterrcnn_full",
    "SSDlite": "trapdata.ml.localization.ssdlite",
    # "MegaDectector v5": "trapdata.ml.localization.megadetectorv5",
    "Disabled": None,
}


# These are separate for the settings choices
BINARY_CLASSIFICATION_MODELS = {
    "Moth / Non-Moth": "trapdata.ml.classification.moth_nonmoth",
    "Disabled": None,
}

TAXON_CLASSIFICATION_MODELS = {
    "Quebec & Vermont Species": "trapdata.ml.classification.quebec_vermont_species",
    "UK & Denmark Species": "trapdata.ml.classification.uk_denmark_species",
    "Disabled": None,
}

# These are combined for selecting a model with the same function
CLASSIFICATION_MODELS = {}
CLASSIFICATION_MODELS.update(BINARY_CLASSIFICATION_MODELS)
CLASSIFICATION_MODELS.update(TAXON_CLASSIFICATION_MODELS)


def detect_objects(model_name, **kwargs):

    module_path = LOCALIZATION_MODELS[model_name]
    if not module_path:
        logger.info("Skipping classification")
        return None
    logger.debug(f"Loading object detection model: {module_path}")
    model_module = importlib.import_module(module_path)

    logger.debug(f"Calling predict with arguments: {kwargs}")
    model_module.predict(**kwargs)
    logger.debug("Predict complete")


def classify_objects(model_name, **kwargs):
    module_path = CLASSIFICATION_MODELS[model_name]
    if not module_path:
        logger.info("Skipping classification")
        return None

    logger.debug(f"Loading classification model: {module_path}")
    model_module = importlib.import_module(module_path)

    logger.debug(f"Calling predict with arguments: {kwargs}")
    model_module.predict(**kwargs)
    logger.debug("Predict complete")
