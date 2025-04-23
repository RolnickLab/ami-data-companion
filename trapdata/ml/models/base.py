import json
import pandas as pd
from typing import Union

import numpy as np
import sqlalchemy
import torch
import torch.utils.data
import torchvision.transforms
from sentry_sdk import start_transaction

from trapdata import logger
from trapdata.common.schemas import FilePath
from trapdata.common.utils import slugify
from trapdata.db.models.queue import QueueManager
from trapdata.ml.utils import StopWatch, get_device, get_or_download_file

from dataclasses import dataclass


class BatchEmptyException(Exception):
    pass


def zero_okay_collate(batch):
    """
    If the queue is cleared or shortened before the original batch count is complete
    then the dataloader will crash. This catches the empty batch more gracefully.

    @TODO switch to streaming IterableDataset type.
    """
    if any(not item for item in batch):
        logger.debug(f"There's a None in the batch of len {len(batch)}")
        return None
    else:
        return torch.utils.data.default_collate(batch)


imagenet_normalization = torchvision.transforms.Normalize(
    # "torch preprocessing"
    mean=[0.485, 0.456, 0.406],  # RGB
    std=[0.229, 0.224, 0.225],  # RGB
)

tensorflow_normalization = torchvision.transforms.Normalize(
    # -1 to 1
    mean=[0.5, 0.5, 0.5],  # RGB
    std=[0.5, 0.5, 0.5],  # RGB
)

generic_normalization = torchvision.transforms.Normalize(
    # 0 to 1
    mean=[0.5, 0.5, 0.5],  # RGB
    std=[0.5, 0.5, 0.5],  # RGB
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

    db_path: Union[str, sqlalchemy.engine.URL]
    image_base_path: FilePath
    name = "Unknown Inference Model"
    description = str()
    model_type = None
    device = None
    weights_path = None
    weights = None
    labels_path = None
    category_map = {}
    num_classes: Union[int, None] = None  # Will use len(category_map) if None
    lookup_gbif_names: bool = False
    model: torch.nn.Module
    normalization = tensorflow_normalization
    transforms: torchvision.transforms.Compose
    batch_size = 4
    num_workers = 1
    user_data_path = None
    type = "unknown"
    stage = 0
    single = True
    queue: QueueManager
    dataset: torch.utils.data.Dataset
    dataloader: torch.utils.data.DataLoader
    training_csv_path: str | None = None

    def __init__(
        self,
        db_path: Union[str, sqlalchemy.engine.URL],
        image_base_path: FilePath,
        **kwargs,
    ):
        self.db_path = db_path
        self.image_base_path = image_base_path

        for k, v in kwargs.items():
            setattr(self, k, v)

        logger.info(f"Initializing inference class {self.name}")

        self.device = self.device or get_device()
        self.category_map = self.get_labels(self.labels_path)
        self.class_prior = self.get_class_prior(self.training_csv_path)
        self.num_classes = self.num_classes or len(self.category_map)
        self.weights = self.get_weights(self.weights_path)
        self.transforms = self.get_transforms()
        self.queue = self.get_queue()
        self.dataset = self.get_dataset()
        self.dataloader = self.get_dataloader()
        logger.info(
            f"Loading {self.type} model (stage: {self.stage}) for {self.name} with {len(self.category_map or [])} categories"
        )
        self.model = self.get_model()

    @classmethod
    def get_key(cls):
        if hasattr(cls, "key") and cls.key:  # type: ignore
            return cls.key  # type: ignore
        else:
            return slugify(cls.name)

    def get_weights(self, weights_path):
        if weights_path:
            return get_or_download_file(
                weights_path,
                self.user_data_path or torch.hub.get_dir(),
                prefix="models",
            )
        else:
            logger.warn(f"No weights specified for model {self.name}")

    def get_labels(self, labels_path) -> dict[int, str]:
        if labels_path:
            local_path = get_or_download_file(
                labels_path,
                self.user_data_path or torch.hub.get_dir(),
                prefix="models",
            )

            with open(local_path) as f:
                labels = json.load(f)

            if self.lookup_gbif_names:
                """
                Use this if you want to store name strings instead of taxon IDs.
                Taxon IDs are helpful for looking up additional information about the species
                such as the genus and family.
                """
                import concurrent.futures

                from trapdata.ml.utils import replace_gbif_id_with_name

                def fetch_gbif_ids(labels):
                    string_labels = {}
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        futures = []
                        for label, _index in labels.items():
                            future = executor.submit(replace_gbif_id_with_name, label)
                            futures.append(future)
                        for future, (_label, index) in zip(futures, labels.items()):
                            string_label = future.result()
                            string_labels[string_label] = index

                    return string_labels

                string_labels = fetch_gbif_ids(labels)

                logger.info(f"Replacing GBIF IDs with names in {local_path}")
                # Backup the original file
                local_path.rename(local_path.with_suffix(".bak"))
                with open(local_path, "w") as f:
                    json.dump(string_labels, f)

            # @TODO would this be faster as a list? especially when getting the labels of multiple
            # indexes in one prediction
            index_to_label = {index: label for label, index in labels.items()}

            return index_to_label
        else:
            return {}

    def get_class_prior(self, training_csv_path):
        if training_csv_path:
            local_path = get_or_download_file(
                training_csv_path,
                self.user_data_path or torch.hub.get_dir(),
                prefix="models",
            )
            df_train = pd.read_csv(local_path)
            categories = sorted(list(df_train["speciesKey"].unique()))
            categories_map = {categ: id for id, categ in enumerate(categories)}
            df_train["label"] = df_train["speciesKey"].map(categories_map)
            cls_idx = df_train["label"].astype(int).values
            num_classes = df_train["label"].nunique()
            cls_num = np.bincount(cls_idx, minlength=num_classes)
            targets = cls_num / cls_num.sum()
            return targets
        else:
            return None

    def get_features(
        self, batch_input: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Default get_features method for models that don't implement  feature extraction.
        """

        return None

    def get_model(self) -> torch.nn.Module:
        """
        This method must be implemented by a subclass.

        Example:

        model = torch.nn.Module()
        checkpoint = torch.load(self.weights, map_location=self.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(self.device)
        model.eval()
        return model
        """
        raise NotImplementedError

    def get_transforms(self) -> torchvision.transforms.Compose:
        """
        This method must be implemented by a subclass.

        Example:

        transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
            ]
        )
        return transforms
        """
        raise NotImplementedError

    def get_queue(self) -> QueueManager:
        """
        This method must be implemented by a subclass.
        Example:

        from trapdata.db.models.queue import DetectedObjectQueue
        def get_queue(self):
            return DetectedObjectQueue(self.db_path, self.image_base_path)
        """
        raise NotImplementedError

    def get_dataset(self) -> torch.utils.data.Dataset:
        """
        This method must be implemented by a subclass.

        Example:

        dataset = torch.utils.data.Dataset()
        return dataset
        """
        raise NotImplementedError

    def get_dataloader(self):
        """
        Prepare dataloader for streaming/iterable datasets from database
        """
        if self.single:
            logger.info(
                f"Preparing dataloader with batch size of {self.batch_size} in single worker mode."
            )
        else:
            logger.info(
                f"Preparing dataloader with batch size of {self.batch_size} and {self.num_workers} workers."
            )
        dataloader_args = {
            "num_workers": 0 if self.single else self.num_workers,
            "persistent_workers": False if self.single else True,
            "shuffle": False,
            "pin_memory": False if self.single else True,  # @TODO review this
        }
        if isinstance(self.dataset, torch.utils.data.IterableDataset):
            # Batch size and sample should be None for streaming datasets
            dataloader_args.update(
                {
                    "batch_size": None,
                    "batch_sampler": None,
                }
            )
        else:
            dataloader_args.update(
                {
                    "batch_size": self.batch_size,
                }
            )
        self.dataloader = torch.utils.data.DataLoader(self.dataset, **dataloader_args)
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
        return [self.post_process_single(item) for item in batch_output]
        # Had problems with this generator and multiprocessing
        # for item in batch_output:
        #     yield self.post_process_single(item)

    def save_results(
        self, item_ids, batch_output, seconds_per_item: float | None = None
    ):
        logger.warn("No save method configured for model. Doing nothing with results")
        return None

    @torch.no_grad()
    def run(self):
        torch.cuda.empty_cache()

        for i, batch in enumerate(self.dataloader):
            if not batch:
                # @TODO review this once we switch to streaming IterableDataset
                logger.info(f"Batch {i+1} is empty, skipping")
                continue

            item_ids, batch_input = batch

            logger.info(
                f"Processing batch {i+1}, about {len(self.dataloader)} remaining"
            )

            # @TODO the StopWatch doesn't seem to work when there are multiple workers,
            # it always returns 0 seconds.
            with StopWatch() as batch_time:
                with start_transaction(op="inference_batch", name=self.name):
                    batch_output = self.predict_batch(batch_input)

            seconds_per_item = batch_time.duration / len(batch_output)
            logger.info(
                f"Inference time for batch: {batch_time}, "
                f"Seconds per item: {round(seconds_per_item, 2)}"
            )

            batch_output = list(self.post_process_batch(batch_output))
            if isinstance(item_ids, (np.ndarray, torch.Tensor)):
                item_ids = item_ids.tolist()
            logger.info(f"Saving results from {len(item_ids)} items")

            self.save_results(item_ids, batch_output, seconds_per_item=seconds_per_item)
            logger.info(f"{self.name} Batch -- Done")

        logger.info(f"{self.name} -- Done")


@dataclass
class ClassifierResult:
    # TODO: add types
    feature: None
    labels: None
    logit: None
    scores: None
    ood_score: float
