import functools
import logging
import time
import typing
from io import BytesIO
from typing import Callable, Tuple

import requests
import torch
import torch.utils.data
import torchvision
from PIL import Image

from trapdata.common.logs import logger

from .schemas import DetectionResponse, SourceImage


class LocalizationImageDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        source_images: typing.Iterable[SourceImage],
        image_transforms: torchvision.transforms.Compose,
        batch_size: int = 1,
    ):
        super().__init__()
        self.source_images: list[SourceImage] = list(source_images)
        self.image_transforms: torchvision.transforms.Compose = image_transforms
        self.batch_size: int = batch_size

    def __len__(self):
        return len(list(self.source_images))

    def __getitem__(self, idx):
        worker_info = torch.utils.data.get_worker_info()
        logger.info(f"Using worker: {worker_info}")

        source_image: SourceImage = self.source_images[idx]
        image_data = source_image.open()
        if not image_data:
            return None

        image_data = self.image_transforms(image_data)

        # ids_batch = torch.utils.data.default_collate([source_image.id])
        # image_batch = torch.utils.data.default_collate([image_data])

        # logger.info(f"Batch data: {ids_batch}, {image_batch}")

        # return (ids_batch, image_batch)
        return source_image.id, image_data


class ClassificationImageDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        source_images: typing.Iterable[SourceImage],
        detections: typing.Iterable[DetectionResponse],
        image_transforms: torchvision.transforms.Compose,
        batch_size: int = 1,
    ):
        super().__init__()
        self.detections = list(detections)
        self.image_transforms: torchvision.transforms.Compose = image_transforms
        self.batch_size: int = batch_size
        self.source_images: dict[str, SourceImage] = {
            img.id: img for img in source_images
        }

    def __len__(self):
        # Append all detections to a single list
        return len(self.detections)

    def __getitem__(self, idx):
        worker_info = torch.utils.data.get_worker_info()
        logger.info(f"Using worker: {worker_info}")

        detection_idx = idx
        detection: DetectionResponse = self.detections[idx]
        source_image = self.source_images[detection.source_image_id]
        image_data = source_image.open()
        if not image_data:
            return None
        bbox = detection.bbox
        coords = bbox.x1, bbox.y1, bbox.x2, bbox.y2
        assert all(coord is not None for coord in coords)
        image_data = image_data.crop(coords)  # type: ignore
        image_data = self.image_transforms(image_data)

        # ids_batch = torch.utils.data.default_collate([source_image.id])
        # image_batch = torch.utils.data.default_collate([image_data])

        # logger.info(f"Batch data: {ids_batch}, {image_batch}")

        # return (ids_batch, image_batch)
        return (source_image.id, detection_idx), image_data


class RESTDataset(torch.utils.data.IterableDataset):
    """
    An IterableDataset that fetches tasks from a REST API endpoint and loads images.

    The dataset continuously polls the API for tasks, loads the associated images,
    and yields them as PyTorch tensors along with metadata.
    """

    def __init__(
        self,
        base_url: str,
        job_id: int,
        batch_size: int = 1,
        image_transforms: typing.Optional[torchvision.transforms.Compose] = None,
    ):
        """
        Initialize the REST dataset.

        Args:
            base_url: Base URL for the API (e.g., "http://localhost:8000")
            job_id: The job ID to fetch tasks for
            batch_size: Number of tasks to request per batch
            image_transforms: Optional transforms to apply to loaded images
        """
        super().__init__()
        self.base_url = base_url.rstrip("/")
        self.job_id = job_id
        self.batch_size = batch_size
        self.image_transforms = image_transforms or torchvision.transforms.ToTensor()

    def _fetch_tasks(self) -> list[dict]:
        """
        Fetch a batch of tasks from the REST API.

        Returns:
            List of task dictionaries from the API response
        """
        url = f"{self.base_url}/api/v2/jobs/{self.job_id}/tasks"
        params = {"batch": self.batch_size}

        try:
            response = requests.get(
                url,
                params=params,
                timeout=30,
                headers={
                    "Authorization": "",
                },
            )
            response.raise_for_status()
            data = response.json()
            return data.get("tasks", [])
        except requests.RequestException as e:
            logger.error(f"Failed to fetch tasks from {url}: {e}")
            return []

    def _load_image(self, image_url: str) -> typing.Optional[torch.Tensor]:
        """
        Load an image from a URL and convert it to a PyTorch tensor.

        Args:
            image_url: URL of the image to load

        Returns:
            Image as a PyTorch tensor, or None if loading failed
        """
        try:
            response = requests.get(image_url, timeout=30)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content))

            # Convert to RGB if necessary
            if image.mode != "RGB":
                image = image.convert("RGB")

            # Apply transforms
            image_tensor = self.image_transforms(image)
            return image_tensor
        except Exception as e:
            logger.error(f"Failed to load image from {image_url}: {e}")
            return None

    def __iter__(self):
        """
        Iterate over tasks from the REST API.

        Yields:
            Dictionary containing:
                - image: PyTorch tensor of the loaded image
                - reply_subject: Reply subject for the task
                - batch_index: Index of the image in the batch
                - job_id: Job ID
                - image_id: Image ID
        """
        try:
            # Get worker info for debugging
            worker_info = torch.utils.data.get_worker_info()
            worker_id = worker_info.id if worker_info else 0
            num_workers = worker_info.num_workers if worker_info else 1

            logger.info(
                f"Worker {worker_id}/{num_workers} starting iteration for job {self.job_id}"
            )

            while True:
                tasks = self._fetch_tasks()
                # _, t = log_time()
                # _, t = t(f"Worker {worker_id}: Fetched {len(tasks)} tasks from API")

                # If no tasks returned, dataset is finished
                if not tasks:
                    logger.info(
                        f"Worker {worker_id}: No more tasks for job {self.job_id}, terminating"
                    )
                    break

                for task in tasks:
                    body = task.get("body", {})
                    image_url = body.get("image_url")

                    if not image_url:
                        logger.warning(
                            f"Task {task.get('id')} missing image_url, skipping"
                        )
                        continue

                    # Load the image
                    # _, t = log_time()
                    image_tensor = self._load_image(image_url)
                    # _, t = t(f"Loaded image from {image_url}")

                    if image_tensor is None:
                        logger.warning(
                            f"Failed to load image for task {task.get('id')}, skipping"
                        )
                        continue

                    # Yield the data row
                    # yield {
                    #     "image": image_tensor,
                    #     "reply_subject": task.get("reply_subject"),
                    #     "batch_index": body.get("batch_index"),
                    #     "job_id": body.get("job_id"),
                    #     "image_id": body.get("image_id"),
                    # }
                    yield str(body.get("image_id")), image_tensor

            logger.info(f"Worker {worker_id}: Iterator finished")
        except Exception as e:
            logger.error(f"Worker {worker_id}: Exception in iterator: {e}")
            raise


def log_time(start: float = 0, msg: str = None) -> Tuple[float, Callable]:
    """
    Small helper to measure time between calls.

    Returns: elapsed time since the last call, and a partial function to measure from the current call
    Usage:

    _, tlog = log_time()
    # do something
    _, tlog = tlog("Did something") # will log the time taken by 'something'
    # do something else
    t, tlog = tlog("Did something else") # will log the time taken by 'something else', returned as 't'
    """
    end = time.perf_counter()
    if start == 0:
        dur = 0.0
    else:
        dur = end - start
    if msg and start > 0:
        logger.info(f"{msg}: {dur:.3f}s")
    new_start = time.perf_counter()
    return dur, functools.partial(log_time, new_start)


def main():
    # Initialize console logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )

    dataset = RESTDataset(base_url="http://localhost:8000", job_id=11, batch_size=10)

    _, t = log_time()
    # for data in dataset:
    #     image_tensor = data["image"]
    #     # reply_subject = data["reply_subject"]
    #     logger.info(f"Image tensor shape: {image_tensor.shape}")
    #     # logger.info(f"Reply subject: {reply_subject}")
    # _, t = t("Processed all images via dataset")
    # time.sleep(40)

    _, t = t("Starting dataloader")

    # Use 'spawn' instead of 'fork' to avoid hanging on macOS
    # import multiprocessing

    # ctx = multiprocessing.get_context("spawn")

    dl = torch.utils.data.DataLoader(
        dataset,
        batch_size=4,
        num_workers=4,
        # multiprocessing_context=ctx,  # Use spawn method
    )

    c = 0
    for batch in dl:
        images = batch["image"]
        c += len(images)
        logger.info(f"Batch image tensor shape: {images.shape}: {c} images processed")

    _, t = t(f"Processed all images via dataloader: {c} images")


if __name__ == "__main__":
    print("Running REST dataset test... ")
    main()
    print("Done.")
