import os
import time
import typing
from io import BytesIO
from urllib.parse import urljoin

import requests
import torch
import torch.utils.data
import torchvision
from PIL import Image

from trapdata.common.logs import logger

from .schemas import (
    AntennaPipelineProcessingTask,
    AntennaTasksListResponse,
    DetectionResponse,
    SourceImage,
)

if typing.TYPE_CHECKING:
    from trapdata.settings import Settings


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

    IMPORTANT: This dataset assumes the API endpoint atomically removes tasks from
    the queue when fetched (like RabbitMQ, SQS, Redis LPOP). This means multiple
    DataLoader workers are SAFE and won't process duplicate tasks. Each worker
    independently fetches different tasks from the shared queue.

    With num_workers > 0:
        Worker 1: GET /tasks → receives [1,2,3,4], removed from queue
        Worker 2: GET /tasks → receives [5,6,7,8], removed from queue
        No duplicates, safe for parallel processing
    """

    def __init__(
        self,
        base_url: str,
        job_id: int,
        batch_size: int = 1,
        image_transforms: typing.Optional[torchvision.transforms.Compose] = None,
        auth_token: typing.Optional[str] = None,
    ):
        """
        Initialize the REST dataset.

        Args:
            base_url: Base URL for the API including /api/v2 (e.g., "http://localhost:8000/api/v2")
            job_id: The job ID to fetch tasks for
            batch_size: Number of tasks to request per batch
            image_transforms: Optional transforms to apply to loaded images
            auth_token: API authentication token
        """
        super().__init__()
        # Ensure base_url has trailing slash for proper urljoin behavior
        self.base_url = base_url if base_url.endswith("/") else base_url + "/"
        self.job_id = job_id
        self.batch_size = batch_size
        self.image_transforms = image_transforms or torchvision.transforms.ToTensor()
        self.auth_token = auth_token or os.environ.get("ANTENNA_API_TOKEN")

    def _fetch_tasks(self) -> list[AntennaPipelineProcessingTask]:
        """
        Fetch a batch of tasks from the REST API.

        Returns:
            List of tasks (possibly empty if queue is drained)

        Raises:
            requests.RequestException: If the request fails (network error, etc.)
        """
        url = urljoin(self.base_url, f"jobs/{self.job_id}/tasks")
        params = {"batch": self.batch_size}

        headers = {}
        if self.auth_token:
            headers["Authorization"] = f"Token {self.auth_token}"

        response = requests.get(
            url,
            params=params,
            timeout=30,
            headers=headers,
        )
        response.raise_for_status()

        # Parse and validate response with Pydantic
        tasks_response = AntennaTasksListResponse.model_validate(response.json())
        return tasks_response.tasks  # Empty list is valid (queue drained)

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
                try:
                    tasks = self._fetch_tasks()
                except requests.RequestException as e:
                    # Fetch failed - retry after delay
                    logger.warning(
                        f"Worker {worker_id}: Fetch failed ({e}), retrying in 5s"
                    )
                    time.sleep(5)
                    continue

                if not tasks:
                    # Queue is empty - job complete
                    logger.info(
                        f"Worker {worker_id}: No more tasks for job {self.job_id}"
                    )
                    break

                for task in tasks:
                    errors = []
                    # Load the image
                    # _, t = log_time()
                    image_tensor = (
                        self._load_image(task.image_url) if task.image_url else None
                    )
                    # _, t = t(f"Loaded image from {image_url}")

                    if image_tensor is None:
                        errors.append("failed to load image")

                    if errors:
                        logger.warning(
                            f"Worker {worker_id}: Errors in task for image '{task.image_id}': {', '.join(errors)}"
                        )

                    # Yield the data row
                    row = {
                        "image": image_tensor,
                        "reply_subject": task.reply_subject,
                        "image_id": task.image_id,
                        "image_url": task.image_url,
                    }
                    if errors:
                        row["error"] = "; ".join(errors) if errors else None
                    yield row

            logger.info(f"Worker {worker_id}: Iterator finished")
        except Exception as e:
            logger.error(f"Worker {worker_id}: Exception in iterator: {e}")
            raise


def rest_collate_fn(batch: list[dict]) -> dict:
    """
    Custom collate function that separates failed and successful items.

    Returns a dict with:
        - image: Stacked tensor of valid images (only present if there are successful items)
        - reply_subject: List of reply subjects for valid images
        - image_id: List of image IDs for valid images
        - image_url: List of image URLs for valid images
        - failed_items: List of dicts with metadata for failed items

    When all items in the batch have failed, the returned dict will only contain:
        - reply_subject: empty list
        - image_id: empty list
        - failed_items: list of failure metadata
    """
    successful = []
    failed = []

    for item in batch:
        if item["image"] is None or item.get("error"):
            # Failed item
            failed.append(
                {
                    "reply_subject": item["reply_subject"],
                    "image_id": item["image_id"],
                    "image_url": item.get("image_url"),
                    "error": item.get("error", "Unknown error"),
                }
            )
        else:
            # Successful item
            successful.append(item)

    # Collate successful items
    if successful:
        result = {
            "image": torch.stack([item["image"] for item in successful]),
            "reply_subject": [item["reply_subject"] for item in successful],
            "image_id": [item["image_id"] for item in successful],
            "image_url": [item.get("image_url") for item in successful],
        }
    else:
        # Empty batch - all failed
        result = {
            "reply_subject": [],
            "image_id": [],
        }

    result["failed_items"] = failed

    return result


def get_rest_dataloader(
    job_id: int,
    settings: "Settings",
) -> torch.utils.data.DataLoader:
    """
    Create a DataLoader that fetches tasks from Antenna API.

    Note: num_workers > 0 is SAFE here (unlike local file reading) because:
    - Antenna API provides atomic task dequeue (work queue pattern)
    - No shared file handles between workers
    - Each worker gets different tasks automatically
    - Parallel downloads improve throughput for I/O-bound work

    Args:
        job_id: Job ID to fetch tasks for
        settings: Settings object with antenna_api_* configuration
    """
    dataset = RESTDataset(
        base_url=settings.antenna_api_base_url,
        job_id=job_id,
        batch_size=settings.antenna_api_batch_size,
        auth_token=settings.antenna_api_auth_token,
    )

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=settings.antenna_api_batch_size,
        num_workers=settings.num_workers,
        collate_fn=rest_collate_fn,
    )
