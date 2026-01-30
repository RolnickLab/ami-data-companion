"""Worker loop for processing jobs from Antenna API."""

import datetime
import time

import numpy as np
import torch

from trapdata.antenna.client import get_jobs, post_batch_results
from trapdata.antenna.datasets import get_rest_dataloader
from trapdata.antenna.schemas import AntennaTaskResult, AntennaTaskResultError
from trapdata.api.api import CLASSIFIER_CHOICES
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
    """Run the worker to process images from the REST API queue."""
    settings = read_settings()

    # Validate auth token
    if not settings.antenna_api_auth_token:
        raise ValueError(
            "AMI_ANTENNA_API_AUTH_TOKEN environment variable must be set. "
            "Get your auth token from your Antenna project settings."
        )

    while True:
        # TODO CGJS: Support pulling and prioritizing single image tasks, which are used in interactive testing
        # These should probably come from a dedicated endpoint and should preempt batch jobs under the assumption that they
        # would run on the same GPU.
        any_jobs = False
        for pipeline in pipelines:
            logger.info(f"Checking for jobs for pipeline {pipeline}")
            jobs = get_jobs(
                base_url=settings.antenna_api_base_url,
                auth_token=settings.antenna_api_auth_token,
                pipeline_slug=pipeline,
            )
            for job_id in jobs:
                logger.info(f"Processing job {job_id} with pipeline {pipeline}")
                try:
                    any_work_done = _process_job(
                        pipeline=pipeline,
                        job_id=job_id,
                        settings=settings,
                    )
                    any_jobs = any_jobs or any_work_done
                except Exception as e:
                    logger.error(
                        f"Failed to process job {job_id} with pipeline {pipeline}: {e}",
                        exc_info=True,
                    )
                    # Continue to next job rather than crashing the worker

        if not any_jobs:
            logger.info(f"No jobs found, sleeping for {SLEEP_TIME_SECONDS} seconds")
            time.sleep(SLEEP_TIME_SECONDS)


@torch.no_grad()
def _process_job(
    pipeline: str,
    job_id: int,
    settings: Settings,
) -> bool:
    """Run the worker to process images from the REST API queue.

    Args:
        pipeline: Pipeline name to use for processing (e.g., moth_binary, panama_moths_2024)
        job_id: Job ID to process
        settings: Settings object with antenna_api_* configuration
    Returns:
        True if any work was done, False otherwise
    """
    did_work = False
    loader = get_rest_dataloader(job_id=job_id, settings=settings)
    classifier = None
    detector = None

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
            classifier_class = CLASSIFIER_CHOICES[pipeline]
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

        # Track start time for this batch
        batch_start_time = datetime.datetime.now()

        logger.info(f"Processing batch {i + 1}")
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

        classifier.reset(detector.results)

        for idx, dresp in enumerate(detector.results):
            image_tensor = image_tensors[dresp.source_image_id]
            bbox = dresp.bbox
            # crop the image tensor using the bbox
            crop = image_tensor[
                :, int(bbox.y1) : int(bbox.y2), int(bbox.x1) : int(bbox.x2)
            ]
            crop = crop.unsqueeze(0)  # add batch dimension
            classifier_out = classifier.predict_batch(crop)
            classifier_out = classifier.post_process_batch(classifier_out)
            detection = classifier.update_detection_classification(
                seconds_per_item=0,
                image_id=dresp.source_image_id,
                detection_idx=idx,
                predictions=classifier_out[0],
            )
            image_detections[dresp.source_image_id].append(detection)
            all_detections.append(detection)

        ct, t = t("Finished classification")
        total_classification_time += ct

        # Calculate batch processing time
        batch_end_time = datetime.datetime.now()
        batch_elapsed = (batch_end_time - batch_start_time).total_seconds()

        # Post results back to the API with PipelineResponse for each image
        batch_results: list[AntennaTaskResult] = []
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
                total_time=batch_elapsed / len(image_ids),  # Approximate time per image
            )

            batch_results.append(
                AntennaTaskResult(
                    reply_subject=reply_subject,
                    result=pipeline_response,
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
        )
        st, t = t("Finished posting results")

        if not success:
            error_msg = (
                f"Failed to post {len(batch_results)} results for job {job_id} to "
                f"{settings.antenna_api_base_url}. Batch processing data lost."
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        total_save_time += st

    logger.info(
        f"Done, detections: {len(all_detections)}. Detecting time: {total_detection_time}, "
        f"classification time: {total_classification_time}, dl time: {total_dl_time}, save time: {total_save_time}"
    )
    return did_work
