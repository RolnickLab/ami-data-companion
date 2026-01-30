"""Dataset classes for streaming tasks from the Antenna API."""

import typing
from io import BytesIO

import requests
import torch
import torch.utils.data
import torchvision
from PIL import Image

from trapdata.antenna.schemas import (
    AntennaPipelineProcessingTask,
    AntennaTasksListResponse,
)
from trapdata.api.utils import get_http_session
from trapdata.common.logs import logger

if typing.TYPE_CHECKING:
    from trapdata.settings import Settings


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
        auth_token: str,
        job_id: int,
        batch_size: int = 1,
        image_transforms: torchvision.transforms.Compose | None = None,
    ):
        """
        Initialize the REST dataset.

        Args:
            base_url: Base URL for the API including /api/v2 (e.g., "http://localhost:8000/api/v2")
            auth_token: API authentication token
            job_id: The job ID to fetch tasks for
            batch_size: Number of tasks to request per batch
            image_transforms: Optional transforms to apply to loaded images
        """
        super().__init__()
        self.base_url = base_url
        self.job_id = job_id
        self.batch_size = batch_size
        self.image_transforms = image_transforms or torchvision.transforms.ToTensor()

        # Create persistent sessions for connection pooling
        self.api_session = get_http_session(auth_token)
        self.image_fetch_session = get_http_session()  # No auth for external image URLs

    def __del__(self):
        """Clean up HTTP sessions on dataset destruction."""
        if hasattr(self, "api_session"):
            self.api_session.close()
        if hasattr(self, "image_fetch_session"):
            self.image_fetch_session.close()

    def _fetch_tasks(self) -> list[AntennaPipelineProcessingTask]:
        """
        Fetch a batch of tasks from the REST API.

        Returns:
            List of tasks (possibly empty if queue is drained)

        Raises:
            requests.RequestException: If the request fails (network error, etc.)
        """
        url = f"{self.base_url.rstrip('/')}/jobs/{self.job_id}/tasks"
        params = {"batch": self.batch_size}

        response = self.api_session.get(url, params=params, timeout=30)
        response.raise_for_status()

        # Parse and validate response with Pydantic
        tasks_response = AntennaTasksListResponse.model_validate(response.json())
        return tasks_response.tasks  # Empty list is valid (queue drained)

    def _load_image(self, image_url: str) -> torch.Tensor | None:
        """
        Load an image from a URL and convert it to a PyTorch tensor.

        Args:
            image_url: URL of the image to load

        Returns:
            Image as a PyTorch tensor, or None if loading failed
        """
        try:
            # Use dedicated session without auth for external images
            response = self.image_fetch_session.get(image_url, timeout=30)
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
        worker_id = 0  # Initialize before try block to avoid UnboundLocalError
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
                    # Fetch failed after retries - log and stop
                    logger.error(
                        f"Worker {worker_id}: Fetch failed after retries ({e}), stopping"
                    )
                    break

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
        - images: Stacked tensor of valid images (only present if there are successful items)
        - reply_subjects: List of reply subjects for valid images
        - image_ids: List of image IDs for valid images
        - image_urls: List of image URLs for valid images
        - failed_items: List of dicts with metadata for failed items

    When all items in the batch have failed, the returned dict will only contain:
        - reply_subjects: empty list
        - image_ids: empty list
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
            "images": torch.stack([item["image"] for item in successful]),
            "reply_subjects": [item["reply_subject"] for item in successful],
            "image_ids": [item["image_id"] for item in successful],
            "image_urls": [item.get("image_url") for item in successful],
        }
    else:
        # Empty batch - all failed
        result = {
            "reply_subjects": [],
            "image_ids": [],
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
        auth_token=settings.antenna_api_auth_token,
        job_id=job_id,
        batch_size=settings.antenna_api_batch_size,
    )

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=settings.localization_batch_size,
        num_workers=settings.num_workers,
        collate_fn=rest_collate_fn,
    )
