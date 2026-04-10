"""Dataset and DataLoader for streaming tasks from the Antenna API.

Data loading pipeline overview
==============================

The pipeline has three layers of concurrency. Each layer is controlled by a
different setting and targets a different bottleneck.

::

    ┌──────────────────────────────────────────────────────────────────┐
    │  GPU process  (_worker_loop in worker.py)                        │
    │  One per GPU. Runs detection → classification on batches.        │
    │  Controlled by: automatic (one per torch.cuda.device_count())    │
    ├──────────────────────────────────────────────────────────────────┤
    │  DataLoader workers  (num_workers subprocesses)                  │
    │  Each subprocess runs its own RESTDataset.__iter__ loop:         │
    │    1. POST /tasks  → fetch batch of task metadata from Antenna   │
    │    2. Download images (threaded, see below)                      │
    │    3. Yield individual (image_tensor, metadata) rows             │
    │  The DataLoader collates rows into GPU-sized batches.            │
    │  Controlled by: settings.num_workers  (AMI_NUM_WORKERS)          │
    │  Safe >0 because Antenna dequeues atomically.                    │
    ├──────────────────────────────────────────────────────────────────┤
    │  Thread pool  (ThreadPoolExecutor inside each DataLoader worker) │
    │  Downloads images concurrently *within* one API fetch batch.    │
    │  Each thread: HTTP GET → PIL open → RGB convert → ToTensor().    │
    │  Controlled by: AMI_ANTENNA_API_DATALOADER_DOWNLOAD_THREADS (8). │
    │  Note: RGB conversion and ToTensor are GIL-bound (CPU). Only     │
    │  the network wait truly runs in parallel.                        │
    └──────────────────────────────────────────────────────────────────┘

Request and memory flow (per DataLoader subprocess, one job)
============================================================

This is the flow that drives peak RAM. See issue #138 for the full analysis.

::

    ┌─ RESTDataset.__iter__ ────────────────────────────────────────────┐
    │                                                                    │
    │  loop until no more tasks:                                         │
    │                                                                    │
    │    ┌─ _fetch_tasks() ──────────────────────────────────────────┐   │
    │    │  POST /api/v2/jobs/{id}/tasks/                            │   │
    │    │  body: {"batch_size": AMI_ANTENNA_API_BATCH_SIZE}         │   │
    │    │  → returns up to N task dicts (image_url + metadata)      │   │
    │    └───────────────────────────────────────────────────────────┘   │
    │                              │                                    │
    │                              ▼                                    │
    │    ┌─ _load_images_threaded() ────────────────────────────────┐   │
    │    │  ThreadPoolExecutor(DOWNLOAD_THREADS) downloads N JPEGs  │   │
    │    │  Each thread: HTTP GET → PIL decode → ToTensor (float32) │   │
    │    │                                                           │   │
    │    │  RAM cost at this point: N × tensor_size                 │   │
    │    │    where tensor_size ≈ 4 × decoded_bytes                 │   │
    │    │    (24 MB JPEG → ~144 MB float32 CHW tensor)             │   │
    │    └───────────────────────────────────────────────────────────┘   │
    │                              │                                    │
    │                              ▼                                    │
    │    yield N rows → collated into a batch of size N                 │
    │                              │                                    │
    └──────────────────────────────┼─────────────────────────────────────┘
                                   │
                                   ▼
    ┌─ DataLoader queue (per subprocess) ────────────────────────────────┐
    │  Holds up to PREFETCH_FACTOR batches ready for the main process.   │
    │                                                                     │
    │  If AMI_ANTENNA_API_DATALOADER_PIN_MEMORY=True:                    │
    │    Each queued batch is also in pinned (unswappable) shmem for IPC │
    │    → effective cost ≈ 2× (pageable + pinned copy)                  │
    │                                                                     │
    │  Peak ingest RAM per subprocess ≈                                   │
    │    PREFETCH_FACTOR × API_BATCH_SIZE × tensor_size × (2 if pinned)  │
    │                                                                     │
    │  Total across worker = above × num_workers + 1 active batch on GPU │
    └─────────────────────────────────────────────────────────────────────┘

When peak RAM is the problem, the knob that scales the hardest is
AMI_ANTENNA_API_BATCH_SIZE. Turning off AMI_ANTENNA_API_DATALOADER_PIN_MEMORY
roughly halves the cost on HTTP-sourced workloads (where the pinned-memory
DMA speedup is negligible compared to download time). Lowering
AMI_ANTENNA_API_DATALOADER_PREFETCH_FACTOR trades network-latency hiding
for lower peak RAM.

Settings quick-reference (prefix with AMI_ as env vars):

    localization_batch_size  (default 8)
        How many images the GPU processes at once (detection). Larger =
        more VRAM. These are full-resolution images.
        This is a model-side knob — does NOT limit ingest RAM.

    classification_batch_size  (default 20)
        How many crops the classifier processes at once. Model-side.

    num_workers  (default 4)
        DataLoader subprocesses per AMI worker instance. Each independently
        fetches tasks and downloads images. More workers = more images
        prefetched, at the cost of CPU/RAM. 0 makes fetching and inference
        sequential (useful for debugging).

    antenna_api_batch_size  (default 24)
        Tasks requested per POST to /api/v2/jobs/{id}/tasks/. The biggest
        lever on peak ingest RAM. See the flow diagram above.

    antenna_api_dataloader_pin_memory  (default True)
        Whether the DataLoader puts prefetched tensors in page-locked system
        RAM. Helpful when CPU→GPU transfer is a meaningful fraction of wall
        time, harmful when data loading dominates wall time. Affects system
        RAM, NOT VRAM.

    antenna_api_dataloader_prefetch_factor  (default 4)
        Batches each DataLoader subprocess keeps queued ahead of the main
        process. Hides data-loading latency; multiplies peak RAM.

    antenna_api_dataloader_download_threads  (default 8)
        ThreadPoolExecutor size for concurrent image downloads inside one
        DataLoader subprocess.

What has NOT been benchmarked yet:
    - Optimal num_workers / thread count combination
    - Whether moving transforms out of threads helps throughput
    - Whether multiple DataLoader workers + threads overlap well or
      contend on the GIL
    - Actual memory profile under a memory profiler (the numbers in
      issue #138 are estimates from code reading, not measurements)
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
    AntennaTasksRequest,
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
        Subprocess 1: POST /tasks → receives [1,2,3,4], removed from queue
        Subprocess 2: POST /tasks → receives [5,6,7,8], removed from queue
        No duplicates, safe for parallel processing
    """

    def __init__(
        self,
        base_url: str,
        auth_token: str,
        job_id: int,
        batch_size: int = 1,
        image_transforms: torchvision.transforms.Compose | None = None,
        download_threads: int = 8,
    ):
        """
        Initialize the REST dataset.

        Args:
            base_url: Base URL for the API including /api/v2 (e.g., "http://localhost:8000/api/v2")
            auth_token: API authentication token
            job_id: The job ID to fetch tasks for
            batch_size: Number of tasks to request per batch
            image_transforms: Optional transforms to apply to loaded images
            download_threads: ThreadPoolExecutor size for concurrent image downloads
        """
        super().__init__()
        self.base_url = base_url
        self.auth_token = auth_token
        self.job_id = job_id
        self.batch_size = batch_size
        self.image_transforms = image_transforms or torchvision.transforms.ToTensor()
        self.download_threads = download_threads

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
            self._executor = ThreadPoolExecutor(max_workers=self.download_threads)

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
        url = f"{self.base_url.rstrip('/')}/jobs/{self.job_id}/tasks/"
        request_body = AntennaTasksRequest(batch_size=self.batch_size)

        self._ensure_sessions()
        assert self._api_session is not None
        response = self._api_session.post(
            url, json=request_body.model_dump(), timeout=30
        )
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
        then as a pre-collated batch.

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
                    logger.debug(
                        f"Worker {worker_id}: No more tasks for job {self.job_id}"
                    )
                    break

                # Download all images concurrently
                image_map = self._load_images_threaded(tasks)
                pre_batch = []
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
                    pre_batch.append(row)
                batch = rest_collate_fn(
                    pre_batch
                )  # Collate before yielding to GPU process
                yield batch

            logger.debug(f"Worker {worker_id}: Iterator finished")
        except Exception as e:
            logger.error(f"Worker {worker_id}: Exception in iterator: {e}")
            raise


def rest_collate_fn(batch: list[dict]) -> dict:
    """
    Custom collate function that separates failed and successful items.

    Returns a dict with:
        - images: Stacked tensor when all images share the same shape, or a list
          of tensors when sizes differ (only present if there are successful items)
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
        image_tensors = [item["image"] for item in successful]
        # Stack into a single tensor when all images are the same size (fast path).
        # Fall back to a list of tensors for mixed sizes — the detector handles both
        # but the list path is slower (individual GPU transfers instead of one bulk copy).
        # To avoid this, sort source images by resolution or request same-size batches.
        shapes = {t.shape for t in image_tensors}
        if len(shapes) > 1:
            logger.warning(
                f"Batch contains {len(shapes)} different image sizes: {shapes}. "
                "Falling back to per-image GPU transfer (slower). "
                "Consider sorting source images by resolution."
            )
        images = torch.stack(image_tensors) if len(shapes) == 1 else image_tensors
        result = {
            "images": images,
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


def _no_op_collate_fn(batch: list[dict]) -> dict:
    """
    A no-op collate function that unwraps a single-element batch.

    This can be used when the dataset already returns batches in the desired format,
    and no further collation is needed. It simply returns the input list of dicts
    without modification.
    """
    return batch[0]


def get_rest_dataloader(
    job_id: int,
    settings: "Settings",
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
            - antenna_api_batch_size                  (tasks per API call)
            - num_workers                             (DataLoader subprocesses)
            - antenna_api_dataloader_pin_memory       (see issue #138)
            - antenna_api_dataloader_prefetch_factor  (see issue #138)
            - antenna_api_dataloader_download_threads (per-subprocess HTTP pool)
    """
    dataset = RESTDataset(
        base_url=settings.antenna_api_base_url,
        auth_token=settings.antenna_api_auth_token,
        job_id=job_id,
        batch_size=settings.antenna_api_batch_size,
        download_threads=settings.antenna_api_dataloader_download_threads,
    )

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=1,  # We collate manually in rest_collate_fn, so set batch_size=1 here
        num_workers=settings.num_workers,
        collate_fn=_no_op_collate_fn,
        pin_memory=settings.antenna_api_dataloader_pin_memory,
        persistent_workers=settings.num_workers > 0,
        prefetch_factor=(
            settings.antenna_api_dataloader_prefetch_factor
            if settings.num_workers > 0
            else None
        ),
    )


class CUDAPrefetcher:
    def __init__(self, loader: torch.utils.data.DataLoader, device: torch.device):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.device = device
        self.next_batch = None
        self._preload()

    def _preload(self):
        try:
            batch = next(self.loader)
        except StopIteration:
            self.next_batch = None
            return

        with torch.cuda.stream(self.stream):
            self.next_batch = {
                k: (
                    v.to(self.device, non_blocking=True)
                    if isinstance(v, torch.Tensor)
                    else v
                )
                for k, v in batch.items()
            }

    def __iter__(self):
        return self

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.next_batch
        if batch is None:
            raise StopIteration
        self._preload()
        return batch
