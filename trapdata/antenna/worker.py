"""Worker loop for processing jobs from Antenna API."""

from __future__ import annotations

import datetime
import time
from collections.abc import Callable

import numpy as np
import torch
import torch.multiprocessing as mp

from trapdata.antenna.client import get_full_service_name, get_jobs
from trapdata.antenna.datasets import CUDAPrefetcher, get_rest_dataloader
from trapdata.antenna.result_posting import ResultPoster
from trapdata.antenna.schemas import AntennaTaskResult, AntennaTaskResultError
from trapdata.api.api import CLASSIFIER_CHOICES, should_filter_detections
from trapdata.api.models.classification import MothClassifierBinary
from trapdata.api.models.localization import APIMothDetector
from trapdata.api.schemas import (
    DetectionResponse,
    PipelineResultsResponse,
    SourceImageResponse,
)
from trapdata.common.logs import logger
from trapdata.common.utils import log_time
from trapdata.settings import Settings, read_settings

MAX_PENDING_POSTS = 5  # Maximum number of concurrent result posts before blocking
SLEEP_TIME_SECONDS = 5


def run_worker(pipelines: list[str]):
    """Run the worker to process images from the REST API queue.

    Automatically spawns one AMI worker instance process per available GPU.
    On single-GPU or CPU-only machines, runs in-process (no overhead).
    """
    settings = read_settings()

    # Validate auth token
    if not settings.antenna_api_auth_token:
        raise ValueError(
            "AMI_ANTENNA_API_AUTH_TOKEN environment variable must be set. "
            "Get your auth token from your Antenna project settings."
        )

    # Validate service name
    if not settings.antenna_service_name or not settings.antenna_service_name.strip():
        raise ValueError(
            "AMI_ANTENNA_SERVICE_NAME configuration setting must be set. "
            "Configure it via environment variable or .env file."
        )

    gpu_count = torch.cuda.device_count()
    if gpu_count > 1:
        logger.info(f"Found {gpu_count} GPUs, spawning one AMI worker instance per GPU")
        # Don't pass settings through mp.spawn — Settings contains enums that
        # can't be pickled. Each child process calls read_settings() itself.
        mp.spawn(
            _worker_loop,
            args=(pipelines,),
            nprocs=gpu_count,
            join=True,
        )
    else:
        if gpu_count == 1:
            logger.info(f"Found 1 GPU: {torch.cuda.get_device_name(0)}")
        else:
            logger.info("No GPUs found, running on CPU")
        _worker_loop(0, pipelines)


def _worker_loop(gpu_id: int, pipelines: list[str]):
    """Main polling loop for a single AMI worker instance, pinned to a specific GPU.

    Args:
        gpu_id: GPU index to pin this AMI worker instance to (0 for CPU-only).
        pipelines: List of pipeline slugs to poll for jobs.
    """
    settings = read_settings()
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        torch.cuda.set_device(gpu_id)
        logger.info(
            f"AMI worker instance {gpu_id} pinned to GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}"
        )

    # Build full service name with hostname
    full_service_name = get_full_service_name(settings.antenna_service_name)
    logger.info(f"Running worker as: {full_service_name}")

    while True:
        # TODO CGJS: Support pulling and prioritizing single image tasks, which are used in interactive testing
        # These should probably come from a dedicated endpoint and should preempt batch jobs under the assumption that they
        # would run on the same GPU.
        any_jobs = False
        logger.info(
            f"[GPU {gpu_id}] Checking for jobs for pipelines: {', '.join(pipelines)}"
        )
        jobs = get_jobs(
            base_url=settings.antenna_api_base_url,
            auth_token=settings.antenna_api_auth_token,
            pipeline_slugs=pipelines,
            processing_service_name=full_service_name,
        )
        for job_id, pipeline in jobs:
            logger.info(
                f"[GPU {gpu_id}] Processing job {job_id} with pipeline {pipeline}"
            )
            try:
                any_work_done = _process_job(
                    pipeline=pipeline,
                    job_id=job_id,
                    settings=settings,
                    processing_service_name=full_service_name,
                    device=device,
                )
                any_jobs = any_jobs or any_work_done
            except Exception as e:
                logger.error(
                    f"[GPU {gpu_id}] Failed to process job {job_id} with pipeline {pipeline}: {e}",
                    exc_info=True,
                )
                # Continue to next job rather than crashing the worker

        if not any_jobs:
            logger.info(
                f"[GPU {gpu_id}] No jobs found, sleeping for {SLEEP_TIME_SECONDS} seconds"
            )
            time.sleep(SLEEP_TIME_SECONDS)


def _apply_binary_classification(
    binary_filter: "MothClassifierBinary",
    detector_results: list[DetectionResponse],
    image_tensors: dict[str, torch.Tensor],
    image_detections: dict[str, list[DetectionResponse]],
) -> tuple[list[DetectionResponse], list[DetectionResponse]]:
    """Apply binary classification to filter moth vs non-moth detections.

    Args:
        binary_filter: The binary classifier instance
        detector_results: List of detections from the object detector
        image_tensors: Mapping of image IDs to tensor data
        image_detections: Mapping to store detections by image ID

    Returns:
        Tuple of (moth_detections, non_moth_detections)
    """
    binary_filter.reset(detector_results)

    # Process binary classification crops
    binary_crops = []
    binary_valid_indices = []
    binary_transforms = binary_filter.get_transforms()

    for idx, dresp in enumerate(detector_results):
        image_tensor = image_tensors[dresp.source_image_id]
        bbox = dresp.bbox
        y1, y2 = int(bbox.y1), int(bbox.y2)
        x1, x2 = int(bbox.x1), int(bbox.x2)
        if y1 >= y2 or x1 >= x2:
            logger.warning(
                f"Skipping binary classification {idx} with invalid bbox: "
                f"({x1},{y1})->({x2},{y2})"
            )
            continue
        crop = image_tensor[:, y1:y2, x1:x2]
        crop_transformed = binary_transforms(crop)
        binary_crops.append(crop_transformed)
        binary_valid_indices.append(idx)

    moth_detections = []
    non_moth_detections = []

    if binary_crops:
        batched_binary_crops = torch.stack(binary_crops)
        binary_out = binary_filter.predict_batch(batched_binary_crops)
        binary_out = binary_filter.post_process_batch(binary_out)

        for crop_i, idx in enumerate(binary_valid_indices):
            dresp = detector_results[idx]
            detection = binary_filter.update_detection_classification(
                seconds_per_item=0,
                image_id=dresp.source_image_id,
                detection_idx=idx,
                predictions=binary_out[crop_i],
            )

            # Separate moth from non-moth detections
            for classification in detection.classifications:
                if classification.classification == binary_filter.positive_binary_label:
                    moth_detections.append(detection)
                elif (
                    classification.classification == binary_filter.negative_binary_label
                ):
                    non_moth_detections.append(detection)
                    image_detections[detection.source_image_id].append(detection)
                break

    return moth_detections, non_moth_detections


def _process_batch(
    batch: dict,
    batch_num: int,
    detector: APIMothDetector,
    classifier,
    pipeline: str,
    binary_filter: "MothClassifierBinary | None",
    use_binary_filter: bool,
) -> tuple[int, int, list[AntennaTaskResult], float, float]:
    """Process a single batch of images through detection and classification.

    All large intermediates (image_tensors, crops, batched_crops, image_detections)
    are local to this function and freed by Python's reference counting when it
    returns, preventing memory accumulation across batches.

    Args:
        batch: Dictionary with images, image_ids, reply_subjects, image_urls, failed_items
        batch_num: 0-based batch index (for logging)
        detector: APIMothDetector instance (reset before call)
        classifier: Terminal species classifier instance
        pipeline: Pipeline slug for response payload
        binary_filter: Binary moth/non-moth classifier, or None
        use_binary_filter: Whether to run binary classification step

    Returns:
        (n_items, n_detections, batch_results, detect_time, classify_time)
    """
    images = batch.get("images", [])
    image_ids = batch.get("image_ids", [])
    reply_subjects = batch.get("reply_subjects", [None] * len(images))
    image_urls = batch.get("image_urls", [None] * len(images))

    batch_results: list[AntennaTaskResult] = []

    try:
        # Validate all arrays have same length before zipping
        if len(image_ids) != len(images):
            raise ValueError(
                f"Length mismatch: image_ids ({len(image_ids)}) != images ({len(images)})"
            )
        if len(image_ids) != len(reply_subjects) or len(image_ids) != len(image_urls):
            raise ValueError(
                f"Length mismatch: image_ids ({len(image_ids)}), "
                f"reply_subjects ({len(reply_subjects)}), image_urls ({len(image_urls)})"
            )

        batch_start_time = datetime.datetime.now()

        # output is dict of "boxes", "labels", "scores"
        batch_output = []
        if len(images) > 0:
            batch_output = detector.predict_batch(images)

        n_items = len(batch_output)
        batch_output = list(detector.post_process_batch(batch_output))

        # Convert image_ids to list if needed
        if isinstance(image_ids, (np.ndarray, torch.Tensor)):
            image_ids = image_ids.tolist()

        # TODO CGJS: Add seconds per item calculation for both detector and classifier
        detector.save_results(
            item_ids=image_ids,
            batch_output=batch_output,
            seconds_per_item=0,
        )
        detect_time = (datetime.datetime.now() - batch_start_time).total_seconds()

        # Group detections by image_id
        image_detections: dict[str, list[DetectionResponse]] = {
            img_id: [] for img_id in image_ids
        }
        image_tensors = dict(zip(image_ids, images, strict=True))

        # Apply binary classification filter if needed
        detector_results = detector.results

        if use_binary_filter:
            assert binary_filter is not None, "Binary filter not initialized"
            (
                detections_for_terminal_classifier,
                detections_to_return,
            ) = _apply_binary_classification(
                binary_filter,
                detector_results,
                image_tensors,
                image_detections,
            )
        else:
            detections_for_terminal_classifier = detector_results
            detections_to_return = []

        # Run terminal classifier on filtered detections
        classifier.reset(detections_for_terminal_classifier)
        classify_transforms = classifier.get_transforms()

        # Collect and transform all crops for batched classification
        crops = []
        valid_indices = []
        n_detections = 0
        for idx, dresp in enumerate(detections_for_terminal_classifier):
            image_tensor = image_tensors[dresp.source_image_id]
            bbox = dresp.bbox
            y1, y2 = int(bbox.y1), int(bbox.y2)
            x1, x2 = int(bbox.x1), int(bbox.x2)
            if y1 >= y2 or x1 >= x2:
                logger.warning(
                    f"Skipping detection {idx} with invalid bbox: "
                    f"({x1},{y1})->({x2},{y2})"
                )
                continue
            crop = image_tensor[:, y1:y2, x1:x2]
            crop_transformed = classify_transforms(crop)
            crops.append(crop_transformed)
            valid_indices.append(idx)

        classify_start = datetime.datetime.now()
        if crops:
            batched_crops = torch.stack(crops)
            classifier_out = classifier.predict_batch(batched_crops)
            classifier_out = classifier.post_process_batch(classifier_out)

            for crop_i, idx in enumerate(valid_indices):
                dresp = detections_for_terminal_classifier[idx]
                detection = classifier.update_detection_classification(
                    seconds_per_item=0,
                    image_id=dresp.source_image_id,
                    detection_idx=idx,
                    predictions=classifier_out[crop_i],
                )
                image_detections[dresp.source_image_id].append(detection)
                n_detections += 1

        classify_time = (datetime.datetime.now() - classify_start).total_seconds()
        # Count non-moth detections returned from binary filter
        n_detections += len(detections_to_return)

        # Calculate batch processing time
        batch_end_time = datetime.datetime.now()
        batch_elapsed = (batch_end_time - batch_start_time).total_seconds()

        # Post results back to the API with PipelineResponse for each image
        for reply_subject, image_id, image_url in zip(
            reply_subjects, image_ids, image_urls, strict=True
        ):
            source_image = SourceImageResponse(id=image_id, url=image_url)
            pipeline_response = PipelineResultsResponse(
                pipeline=pipeline,
                source_images=[source_image],
                detections=image_detections[image_id],
                total_time=batch_elapsed / len(image_ids),
            )
            batch_results.append(
                AntennaTaskResult(
                    reply_subject=reply_subject,
                    result=pipeline_response,
                )
            )
    except Exception as e:
        logger.error(
            f"Batch {batch_num + 1} failed during processing: {e}", exc_info=True
        )
        # Report errors back to Antenna so tasks aren't stuck in the queue
        batch_results = []
        for reply_subject, image_id in zip(reply_subjects, image_ids, strict=True):
            batch_results.append(
                AntennaTaskResult(
                    reply_subject=reply_subject,
                    result=AntennaTaskResultError(
                        error=f"Batch processing error: {e}",
                        image_id=str(image_id) if image_id is not None else None,
                    ),
                )
            )
        n_items = 0
        n_detections = 0
        detect_time = 0.0
        classify_time = 0.0

    failed_items = batch.get("failed_items")
    if failed_items:
        for failed_item in failed_items:
            batch_results.append(
                AntennaTaskResult(
                    reply_subject=failed_item.get("reply_subject"),
                    result=AntennaTaskResultError(
                        error=failed_item.get("error", "Unknown error"),
                        image_id=failed_item.get("image_id"),
                    ),
                )
            )

    torch.cuda.empty_cache()
    return n_items, n_detections, batch_results, detect_time, classify_time


@torch.no_grad()
def _process_job(
    pipeline: str,
    job_id: int,
    settings: Settings,
    processing_service_name: str,
    device: torch.device | None = None,
    on_batch_complete: Callable | None = None,
) -> bool:
    """Run the worker to process images from the REST API queue.

    Args:
        pipeline: Pipeline name to use for processing (e.g., moth_binary, panama_moths_2024)
        job_id: Job ID to process
        settings: Settings object with antenna_api_* configuration
        processing_service_name: Name of the processing service
        device: The device to use for processing. Auto-detected if None.
        on_batch_complete: Optional callback invoked after each batch, with kwargs
            batch_num (int) and items (int, cumulative items processed so far).
    Returns:
        True if any work was done, False otherwise
    """
    did_work = False
    loader = get_rest_dataloader(
        job_id=job_id,
        settings=settings,
        processing_service_name=processing_service_name,
    )
    classifier = None
    detector = None

    # Check if binary filtering is needed once for the entire job
    classifier_class = CLASSIFIER_CHOICES[pipeline]
    use_binary_filter = should_filter_detections(classifier_class)
    binary_filter = None

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    items = 0

    total_detection_time = 0.0
    total_classification_time = 0.0
    total_dl_time = 0.0
    total_detections = 0
    _, t = log_time()
    result_poster: ResultPoster | None = None
    # Conditionally use CUDA prefetcher; fall back to plain iterator on CPU
    if torch.cuda.is_available():
        batch_source = CUDAPrefetcher(
            loader, device
        )  # __init__ already calls preload()
    else:
        batch_source = iter(loader)

    _, t_total = log_time()
    try:
        for i, batch in enumerate(batch_source):
            cls_time = 0.0
            det_time = 0.0
            load_time, t = t()
            total_dl_time += load_time
            if not batch:
                logger.warning(f"Batch {i + 1} is empty, skipping")
                continue

            # Defer instantiation of poster, detector and classifiers until we have data
            if not classifier:
                classifier = classifier_class(source_images=[], detections=[])
                detector = APIMothDetector([])
                result_poster = ResultPoster(max_pending=MAX_PENDING_POSTS)

                if use_binary_filter:
                    binary_filter = MothClassifierBinary(
                        source_images=[],
                        detections=[],
                        terminal=False,
                    )

            assert detector is not None, "Detector not initialized"
            assert classifier is not None, "Classifier not initialized"
            assert result_poster is not None, "ResultPoster not initialized"
            assert not (
                use_binary_filter and binary_filter is None
            ), "Binary filter not initialized"
            detector.reset([])
            did_work = True

            n_items, n_detections, batch_results, det_time, cls_time = _process_batch(
                batch,
                i,
                detector,
                classifier,
                pipeline,
                binary_filter,
                use_binary_filter,
            )
            items += n_items
            total_detections += n_detections
            total_detection_time += det_time
            total_classification_time += cls_time

            # Post results asynchronously (non-blocking)
            result_poster.post_async(
                settings.antenna_api_base_url,
                settings.antenna_api_auth_token,
                job_id,
                batch_results,
                processing_service_name,
            )
            batch_total, t_total = t_total()
            logger.info(
                f"Batch {i + 1}: {batch_total/max(n_items, 1):.2f}s/image, "
                f"Classification time: {cls_time:.2f}s, Detection time: {det_time:.2f}s, "
                f"Load time: {load_time:.2f}s"
            )
            (
                _,
                t,
            ) = log_time()  # reset before next() call to measure next batch's load time

            if on_batch_complete:
                on_batch_complete(batch_num=i, items=items)

        if result_poster:
            # Wait for all async posts to complete before finishing the job
            logger.info("Waiting for all pending result posts to complete...")
            result_poster.wait_for_all_posts(min_timeout=60, per_post_timeout=30)

            # Get final metrics
            post_metrics = result_poster.get_metrics()

            logger.info(
                f"Done, detections: {total_detections}. Detecting time: {total_detection_time:.2f}s, "
                f"classification time: {total_classification_time:.2f}s, dl time: {total_dl_time:.2f}s, "
                f"result posts: {post_metrics.total_posts} "
                f"(success: {post_metrics.successful_posts}, failed: {post_metrics.failed_posts}, "
                f"success rate: {post_metrics.success_rate:.1f}%, avg post time: "
                f"{post_metrics.total_post_time / post_metrics.total_posts if post_metrics.total_posts > 0 else 0:.2f}s, "
                f"max queue size: {post_metrics.max_queue_size})"
            )
        return did_work
    finally:
        if result_poster:
            result_poster.shutdown()
