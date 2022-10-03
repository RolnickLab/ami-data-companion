import torch
import torchvision

from trapdata import logger
from trapdata.ml.utils import (
    get_device,
    get_or_download_file,
    get_category_map,
    StopWatch,
)


class InferenceModel:
    key = None
    title = None
    description = None
    model_type = None
    device = None
    weights_path = None
    weights = None
    labels_path = None
    category_map = None
    model = None
    transforms = None
    batch_size = 4
    num_workers = 2
    description = str()
    db_path = None

    def __init__(self, db_path, **kwargs):
        self.db_path = db_path

        for k, v in kwargs:
            setattr(self, k, v)

        self.device = self.device or get_device()
        self.weights = self.get_weights(self.weights_path)
        self.transforms = self.get_transforms()
        self.dataset = self.get_dataset()
        self.dataloader = self.get_dataloader()

    def get_weights(self, weights_path):
        if not weights_path:
            raise Exception(
                "Missing parameter `weights_path`. "
                "Specify a URL or local path to a model checkpoint"
            )
        return get_or_download_file(weights_path)

    def load_model(self):
        logger.info(
            f"Loading model {self.name} on device: {self.device} with weights: {self.weights}"
        )
        model = torch.nn.Module()
        checkpoint = torch.load(self.weights, map_location=self.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(self.device)
        model.eval()
        self.model = model
        return self.model

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

    def post_process_single(self, item):
        return item

    def post_process_batch(self, batch_output):
        for item in batch_output:
            yield self.post_process_single(item)

    def format_output(self, batch_output):
        return batch_output

    def save_results(self, batch_output):
        logger.warn("No save method configured for model. Doing nothing with results")
        return None

    def run(self):
        with torch.no_grad():
            for i, (item_ids, batch_input) in enumerate(self.dataloader):

                logger.info(f"Running batch {i+1} out of {len(self.dataloader)}")

                with StopWatch() as batch_time:
                    batch_output = self.predict_batch(batch_input)

                seconds_per_item = batch_time.duration / len(batch_output)
                logger.info(
                    f"Inference time for batch: {batch_time}\n"
                    f"{round(seconds_per_item, 1)} seconds per item"
                )

                batch_output = self.post_process_batch(batch_output)
                item_ids = item_ids.tolist()
                self.save_results(item_ids, batch_output)
