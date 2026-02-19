"""Dataset and DataLoader for streaming tasks from the Antenna API.

Data loading pipeline overview
==============================

The pipeline has three layers of concurrency. Each layer is controlled by a
different setting and targets a different bottleneck.

::

    ┌──────────────────────────────────────────────────────────────────┐
    │  GPU process  (_worker_loop in worker.py)                       │
    │  One per GPU. Runs detection → classification on batches.       │
    │  Controlled by: automatic (one per torch.cuda.device_count())   │
    ├──────────────────────────────────────────────────────────────────┤
    │  DataLoader workers  (num_workers subprocesses)                  │
    │  Each subprocess runs its own RESTDataset.__iter__ loop:        │
    │    1. GET /tasks  → fetch batch of task metadata from Antenna   │
    │    2. Download images (threaded, see below)                     │
    │    3. Yield individual (image_tensor, metadata) rows            │
    │  The DataLoader collates rows into GPU-sized batches.           │
    │  Controlled by: settings.num_workers  (AMI_NUM_WORKERS)         │
    │  Default: 2.  Safe >0 because Antenna dequeues atomically.      │
    ├──────────────────────────────────────────────────────────────────┤
    │  Thread pool  (ThreadPoolExecutor inside each DataLoader worker) │
    │  Downloads images concurrently *within* one API fetch batch.    │
    │  Each thread: HTTP GET → PIL open → RGB convert → ToTensor().   │
    │  Controlled by: ThreadPoolExecutor(max_workers=8) on the class. │
    │  Note: RGB conversion and ToTensor are GIL-bound (CPU). Only    │
    │  the network wait truly runs in parallel. A future optimisation  │
    │  could move transforms out of the thread.                       │
    └──────────────────────────────────────────────────────────────────┘

Settings quick-reference (prefix with AMI_ as env vars):

    localization_batch_size  (default 8)
        How many images the GPU processes at once (detection). Larger =
        more GPU memory. These are full-resolution images (~4K).

    num_workers  (default 2)
        DataLoader subprocesses. Each independently fetches tasks and
        downloads images. More workers = more images prefetched for the
        GPU, at the cost of CPU/RAM. With 0 workers, fetching and
        inference are sequential (useful for debugging).

    antenna_api_batch_size  (default 16)
        How many task URLs to request from Antenna per API call.
        Determines how many images are downloaded concurrently per
        thread pool invocation. Should be >= localization_batch_size
        so one API call can fill at least one GPU batch without an
        extra round trip.

    prefetch_factor  (PyTorch default: 2 when num_workers > 0)
        Batches prefetched per worker. Not overridden here — the
        default was tested and no improvement was measured by
        increasing it (it just adds memory pressure).

What has NOT been benchmarked yet (as of 2026-02):
    - Optimal num_workers / thread count combination
    - Whether moving transforms out of threads helps throughput
    - Whether multiple DataLoader workers + threads overlap well
      or contend on the GIL
"""

import typing
from concurrent.futures import ThreadPoolExecutor
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

    With DataLoader num_workers > 0 (I/O subprocesses, not AMI instances):
        Subprocess 1: GET /tasks → receives [1,2,3,4], removed from queue
        Subprocess 2: GET /tasks → receives [5,6,7,8], removed from queue
        No duplicates, safe for parallel processing
    """

    def __init__(
        self,
        base_url: str,
        auth_token: str,
        job_id: int,
        batch_size: int = 1,
        image_transforms: torchvision.transforms.Compose | None = None,
        processing_service_name: str = "",
    ):
        """
        Initialize the REST dataset.

        Args:
            base_url: Base URL for the API including /api/v2 (e.g., "http://localhost:8000/api/v2")
            auth_token: API authentication token
            job_id: The job ID to fetch tasks for
            batch_size: Number of tasks to request per batch
            image_transforms: Optional transforms to apply to loaded images
            processing_service_name: Name of the processing service
        """
        super().__init__()
        self.base_url = base_url
        self.auth_token = auth_token
        self.job_id = job_id
        self.batch_size = batch_size
        self.image_transforms = image_transforms or torchvision.transforms.ToTensor()
        self.processing_service_name = processing_service_name

        # These are created lazily in _ensure_sessions() because they contain
        # unpicklable objects (ThreadPoolExecutor has a SimpleQueue) and
        # PyTorch DataLoader with num_workers>0 pickles the dataset to send
        # it to worker subprocesses.
        self._api_session: requests.Session | None = None
        self._image_fetch_session: requests.Session | None = None
        self._executor: ThreadPoolExecutor | None = None

    def _ensure_sessions(self) -> None:
        """Lazily create HTTP sessions and thread pool.

        Called once per worker process on first use. This avoids pickling
        issues with num_workers > 0 (SimpleQueue, socket objects, etc.).
        """
        if self._api_session is None:
            self._api_session = get_http_session(self.auth_token)
        if self._image_fetch_session is None:
            self._image_fetch_session = get_http_session()
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=8)

    def __del__(self):
        """Clean up HTTP sessions and thread pool on dataset destruction."""
        if self._executor is not None:
            self._executor.shutdown(wait=False)
        if self._api_session is not None:
            self._api_session.close()
        if self._image_fetch_session is not None:
            self._image_fetch_session.close()

    def _fetch_tasks(self) -> list[AntennaPipelineProcessingTask]:
        """
        Fetch a batch of tasks from the REST API.

        Returns:
            List of tasks (possibly empty if queue is drained)

        Raises:
            requests.RequestException: If the request fails (network error, etc.)
        """
        url = f"{self.base_url.rstrip('/')}/jobs/{self.job_id}/tasks"
        params = {
            "batch": self.batch_size,
            "processing_service_name": self.processing_service_name,
        }

        self._ensure_sessions()
        assert self._api_session is not None
        response = self._api_session.get(url, params=params, timeout=30)
        response.raise_for_status()

        # Parse and validate response with Pydantic
        tasks_response = AntennaTasksListResponse.model_validate(response.json())
        return tasks_response.tasks  # Empty list is valid (queue drained)

    def _load_image(self, image_url: str) -> torch.Tensor | None:
        """Load an image from a URL and convert it to a PyTorch tensor.

        Called from threads inside ``_load_images_threaded``. The HTTP
        fetch is truly concurrent (network I/O releases the GIL), but
        PIL decode, RGB conversion, and ``image_transforms`` (ToTensor)
        are CPU-bound and serialised by the GIL.

        Args:
            image_url: URL of the image to load

        Returns:
            Image as a PyTorch tensor, or None if loading failed
        """
        try:
            # Use dedicated session without auth for external images
            self._ensure_sessions()
            assert self._image_fetch_session is not None
            response = self._image_fetch_session.get(image_url, timeout=30)
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

    def _load_images_threaded(
        self,
        tasks: list[AntennaPipelineProcessingTask],
    ) -> dict[str, torch.Tensor | None]:
        """Download images for a batch of tasks using concurrent threads.

        Image downloads are I/O-bound (network latency, not CPU), so threads
        provide near-linear speedup without the overhead of extra processes.
        Note: ``requests.Session`` is not formally thread-safe, but the
        underlying urllib3 connection pool handles concurrent socket access.
        In practice shared read-only sessions work fine for GET requests;
        if issues arise, switch to per-thread sessions.

        Args:
            tasks: List of tasks whose images should be downloaded.

        Returns:
            Mapping from image_id to tensor (or None on failure), preserving
            the order needed by the caller.
        """

        def _download(
            task: AntennaPipelineProcessingTask,
        ) -> tuple[str, torch.Tensor | None]:
            tensor = self._load_image(task.image_url) if task.image_url else None
            return (task.image_id, tensor)

        self._ensure_sessions()
        assert self._executor is not None
        return dict(self._executor.map(_download, tasks))

    def __iter__(self):
        """
        Iterate over tasks from the REST API.

        Each API fetch returns a batch of tasks. Images for the entire batch
        are downloaded concurrently using threads (see _load_images_threaded),
        then yielded one at a time for the DataLoader to collate.

        Yields:
            Dictionary containing:
                - image: PyTorch tensor of the loaded image
                - reply_subject: Reply subject for the task
                - image_id: Image ID
                - image_url: Source URL
        """
        worker_id = 0  # Initialize before try block to avoid UnboundLocalError
        try:
            # Get worker info for debugging
            worker_info = torch.utils.data.get_worker_info()
            worker_id = worker_info.id if worker_info else 0
            num_workers = worker_info.num_workers if worker_info else 1

            logger.info(
                f"DataLoader subprocess {worker_id}/{num_workers} starting iteration for job {self.job_id}"
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

                # Download all images concurrently
                image_map = self._load_images_threaded(tasks)

                for task in tasks:
                    image_tensor = image_map.get(task.image_id)
                    errors = []

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
        - images: List of image tensors (only present if there are successful items)
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
            "images": [item["image"] for item in successful],
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
    processing_service_name: str,
) -> torch.utils.data.DataLoader:
    """Create a DataLoader that fetches tasks from Antenna API.

    See the module docstring for an overview of the three concurrency
    layers (GPU processes → DataLoader workers → thread pool) and which
    settings control each.

    DataLoader num_workers > 0 is safe here because Antenna dequeues
    tasks atomically — each worker subprocess gets a unique set of tasks.

    Args:
        job_id: Job ID to fetch tasks for
        settings: Settings object. Relevant fields:
            - antenna_api_base_url / antenna_api_auth_token
            - antenna_api_batch_size  (tasks per API call)
            - localization_batch_size (images per GPU batch)
            - num_workers            (DataLoader subprocesses)
            - processing_service_name  (name of this worker)
    """
    dataset = RESTDataset(
        base_url=settings.antenna_api_base_url,
        auth_token=settings.antenna_api_auth_token,
        job_id=job_id,
        batch_size=settings.antenna_api_batch_size,
        processing_service_name=processing_service_name,
    )

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=settings.localization_batch_size,
        num_workers=settings.num_workers,
        collate_fn=rest_collate_fn,
    )
