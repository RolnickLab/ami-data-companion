import json

import torch
from sentry_sdk import start_transaction

from trapdata import logger
from trapdata.ml.utils import (
    get_device,
    get_or_download_file,
    StopWatch,
)


class InferenceBaseClass:
    """
    Base class for all batch-inference models.

    This outlines a common interface for all classifiers and object detectors.
    Generic methods like `get_weights_from_url` are defined here, but
    methods that return NotImplementedError must be overridden in a subclass
    that is specific to each inference model.

    See examples in `classification.py` and `localization.py`
    """

    db_path = None
    name = "Unknown Inference Model"
    description = str()
    model_type = None
    device = None
    weights_path = None
    weights = None
    labels_path = None
    category_map = {}
    model = None
    transforms = None
    batch_size = 4
    num_workers = 1
    user_data_path = None
    type = "unknown"
    stage = 0

    def __init__(self, db_path, **kwargs):
        self.db_path = db_path

        for k, v in kwargs.items():
            setattr(self, k, v)

        logger.info(f"Initializing inference class {self.name}")

        self.device = self.device or get_device()
        self.category_map = self.get_labels(self.labels_path)
        self.weights = self.get_weights(self.weights_path)
        self.transforms = self.get_transforms()
        self.dataset = self.get_dataset()
        self.dataloader = self.get_dataloader()
        logger.info(
            f"Loading {self.type} model (stage: {self.stage}) for {self.name} with {len(self.category_map or [])} categories"
        )
        self.model = self.get_model()

    def get_weights(self, weights_path):
        if weights_path:
            return get_or_download_file(
                weights_path, self.user_data_path, prefix="models"
            )
        else:
            logger.warn(f"No weights specified for model {self.name}")

    def get_labels(self, labels_path):
        if labels_path:
            local_path = get_or_download_file(
                labels_path, self.user_data_path, prefix="models"
            )

            with open(local_path) as f:
                labels = json.load(f)

            # @TODO would this be faster as a list? especially when getting the labels of multiple
            # indexes in one prediction
            index_to_label = {index: label for label, index in labels.items()}

            return index_to_label

    def get_model(self):
        """
        # This method must be implemented by a subclass.
        # Example:

        model = torch.nn.Module()
        checkpoint = torch.load(self.weights, map_location=self.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(self.device)
        model.eval()
        return model
        """
        raise NotImplementedError

    def get_transforms(self):
        """
        # This method must be implemented by a subclass.
        # Example:

        transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
            ]
        )
        return transforms
        """
        raise NotImplementedError

    def get_dataset(self):
        """
        # This method must be implemented by a subclass.
        # Example:

        dataset = torch.utils.data.Dataset()
        return dataset
        """
        raise NotImplementedError

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
            non_blocking=False,  # Block while in development, are we already in a background process?
        )
        batch_output = self.model(batch_input)
        return batch_output

    def post_process_single(self, item):
        return item

    def post_process_batch(self, batch_output):
        for item in batch_output:
            yield self.post_process_single(item)

    def save_results(self, item_ids, batch_output):
        logger.warn("No save method configured for model. Doing nothing with results")
        return None

    def run(self):
        with torch.no_grad():
            for i, (item_ids, batch_input) in enumerate(self.dataloader):

                logger.info(
                    f"Processing batch {i+1}, about {len(self.dataloader)} remaining"
                )

                # @TODO the StopWatch doesn't seem to work for the classifier batches,
                # it always returns 0 seconds
                with StopWatch() as batch_time:
                    with start_transaction(op="inference_batch", name=self.name):
                        batch_output = self.predict_batch(batch_input)

                seconds_per_item = batch_time.duration / len(batch_output)
                logger.info(
                    f"Inference time for batch: {batch_time}, "
                    f"Seconds per item: {round(seconds_per_item, 2)}"
                )

                batch_output = self.post_process_batch(batch_output)
                item_ids = item_ids.tolist()
                self.save_results(item_ids, batch_output)
