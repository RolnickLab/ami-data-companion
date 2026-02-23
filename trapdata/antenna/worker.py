"""Worker loop for processing jobs from Antenna API."""

import datetime
import time

import numpy as np
import torch
import torch.multiprocessing as mp
import torchvision

from trapdata.antenna.client import get_full_service_name, get_jobs, post_batch_results
from trapdata.antenna.datasets import get_rest_dataloader
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
        for pipeline in pipelines:
            logger.info(f"[GPU {gpu_id}] Checking for jobs for pipeline {pipeline}")
            jobs = get_jobs(
                base_url=settings.antenna_api_base_url,
                auth_token=settings.antenna_api_auth_token,
                pipeline_slug=pipeline,
                processing_service_name=full_service_name,
            )
            for job_id in jobs:
                logger.info(
                    f"[GPU {gpu_id}] Processing job {job_id} with pipeline {pipeline}"
                )
                try:
                    any_work_done = _process_job(
                        pipeline=pipeline,
                        job_id=job_id,
                        settings=settings,
                        processing_service_name=full_service_name,
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
    to_pil = torchvision.transforms.ToPILImage()
    binary_transforms = binary_filter.get_transforms()

    for idx, dresp in enumerate(detector_results):
        image_tensor = image_tensors[dresp.source_image_id]
        bbox = dresp.bbox
        y1, y2 = int(bbox.y1), int(bbox.y2)
        x1, x2 = int(bbox.x1), int(bbox.x2)
        if y1 >= y2 or x1 >= x2:
            continue
        crop = image_tensor[:, y1:y2, x1:x2]
        crop_pil = to_pil(crop)
        crop_transformed = binary_transforms(crop_pil)
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


@torch.no_grad()
def _process_job(
    pipeline: str,
    job_id: int,
    settings: Settings,
    processing_service_name: str,
) -> bool:
    """Run the worker to process images from the REST API queue.

    Args:
        pipeline: Pipeline name to use for processing (e.g., moth_binary, panama_moths_2024)
        job_id: Job ID to process
        settings: Settings object with antenna_api_* configuration
        processing_service_name: Name of the processing service
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
    if use_binary_filter:
        binary_filter = MothClassifierBinary(
            source_images=[],
            detections=[],
            terminal=False,
        )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    items = 0

    total_detection_time = 0.0
    total_classification_time = 0.0
    total_save_time = 0.0
    total_dl_time = 0.0
    all_detections = []
    _, t = log_time()

    for i, batch in enumerate(loader):
        dt, t = t("Finished loading batch")
        total_dl_time += dt
        if not batch:
            logger.warning(f"Batch {i + 1} is empty, skipping")
            continue

        # Defer instantiation of detector and classifier until we have data
        if not classifier:
            classifier = classifier_class(source_images=[], detections=[])
            detector = APIMothDetector([])
        assert detector is not None, "Detector not initialized"
        assert classifier is not None, "Classifier not initialized"
        detector.reset([])
        did_work = True

        # Extract data from dictionary batch
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
            if len(image_ids) != len(reply_subjects) or len(image_ids) != len(
                image_urls
            ):
                raise ValueError(
                    f"Length mismatch: image_ids ({len(image_ids)}), "
                    f"reply_subjects ({len(reply_subjects)}), image_urls ({len(image_urls)})"
                )

            # Track start time for this batch
            batch_start_time = datetime.datetime.now()

            logger.info(f"Processing worker batch {i + 1} ({len(images)} images)")
            # output is dict of "boxes", "labels", "scores"
            batch_output = []
            if len(images) > 0:
                batch_output = detector.predict_batch(images)

            items += len(batch_output)
            logger.info(f"Total items processed so far: {items}")
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
            dt, t = t("Finished detection")
            total_detection_time += dt

            # Group detections by image_id
            image_detections: dict[str, list[DetectionResponse]] = {
                img_id: [] for img_id in image_ids
            }
            image_tensors = dict(zip(image_ids, images, strict=True))

            # Apply binary classification filter if needed
            detector_results = detector.results

            if use_binary_filter:
                assert binary_filter is not None, "Binary filter not initialized"
                detections_for_terminal_classifier, detections_to_return = (
                    _apply_binary_classification(
                        binary_filter, detector_results, image_tensors, image_detections
                    )
                )
            else:
                # No binary filtering, send all detections to terminal classifier
                detections_for_terminal_classifier = detector_results
                detections_to_return = []

            # Run terminal classifier on filtered detections
            classifier.reset(detections_for_terminal_classifier)
            to_pil = torchvision.transforms.ToPILImage()
            classify_transforms = classifier.get_transforms()

            # Collect and transform all crops for batched classification
            crops = []
            valid_indices = []
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
                crop_pil = to_pil(crop)
                crop_transformed = classify_transforms(crop_pil)
                crops.append(crop_transformed)
                valid_indices.append(idx)

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
                    all_detections.append(detection)

            # Add non-moth detections to all_detections
            all_detections.extend(detections_to_return)

            ct, t = t("Finished classification")
            total_classification_time += ct

            # Calculate batch processing time
            batch_end_time = datetime.datetime.now()
            batch_elapsed = (batch_end_time - batch_start_time).total_seconds()

            # Post results back to the API with PipelineResponse for each image
            batch_results.clear()
            for reply_subject, image_id, image_url in zip(
                reply_subjects, image_ids, image_urls, strict=True
            ):
                # Create SourceImageResponse for this image
                source_image = SourceImageResponse(id=image_id, url=image_url)

                # Create PipelineResultsResponse
                pipeline_response = PipelineResultsResponse(
                    pipeline=pipeline,
                    source_images=[source_image],
                    detections=image_detections[image_id],
                    total_time=batch_elapsed
                    / len(image_ids),  # Approximate time per image
                )

                batch_results.append(
                    AntennaTaskResult(
                        reply_subject=reply_subject,
                        result=pipeline_response,
                    )
                )
        except Exception as e:
            logger.error(f"Batch {i + 1} failed during processing: {e}", exc_info=True)
            # Report errors back to Antenna so tasks aren't stuck in the queue
            batch_results = []
            for reply_subject, image_id in zip(reply_subjects, image_ids, strict=True):
                batch_results.append(
                    AntennaTaskResult(
                        reply_subject=reply_subject,
                        result=AntennaTaskResultError(
                            error=f"Batch processing error: {e}",
                            image_id=image_id,
                        ),
                    )
                )

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

        success = post_batch_results(
            settings.antenna_api_base_url,
            settings.antenna_api_auth_token,
            job_id,
            batch_results,
            processing_service_name,
        )
        st, t = t("Finished posting results")

        if not success:
            logger.error(
                f"Failed to post {len(batch_results)} results for job {job_id} to "
                f"{settings.antenna_api_base_url}. Batch processing data lost."
            )

        total_save_time += st

    logger.info(
        f"Done, detections: {len(all_detections)}. Detecting time: {total_detection_time}, "
        f"classification time: {total_classification_time}, dl time: {total_dl_time}, save time: {total_save_time}"
    )
    return did_work
