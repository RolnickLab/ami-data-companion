import datetime
import os

import numpy as np
import requests
import torch

from trapdata.api.models.classification import MothClassifierBinary
from trapdata.api.models.localization import APIMothDetector, RESTAPIMothDetector
from trapdata.api.schemas import (
    DetectionResponse,
    PipelineResultsResponse,
    SourceImageResponse,
)
from trapdata.common.logs import logger
from trapdata.common.utils import log_time


def post_batch_results(
    base_url: str, job_id: int, results: list[dict], auth_token: str = None
) -> bool:
    """
    Post batch results back to the API.

    Args:
        base_url: Base URL for the API
        job_id: Job ID
        results: List of dicts containing reply_subject and image_id
        auth_token: API authentication token

    Returns:
        True if successful, False otherwise
    """
    url = f"{base_url}/api/v2/jobs/{job_id}/result/"

    headers = {}
    if auth_token:
        headers["Authorization"] = f"Token {auth_token}"

    try:
        response = requests.post(url, json=results, headers=headers, timeout=60)
        response.raise_for_status()
        logger.info(f"Successfully posted {len(results)} results to {url}")
        return True
    except requests.RequestException as e:
        logger.error(f"Failed to post results to {url}: {e}")
        return False


def _get_jobs(base_url: str, auth_token: str, pipeline_id: int = 11) -> list:
    """Fetch job ids from the API for the given pipeline.

    Calls: GET {base_url}/api/v2/jobs?pipeline=<pipeline>&ids_only=1

    Returns a list of job ids (possibly empty) on error.
    """
    try:
        url = f"{base_url.rstrip('/')}/api/v2/jobs"
        params = {"pipeline": pipeline_id, "ids_only": 1, "incomplete_only": 1}

        headers = {}
        if auth_token:
            headers["Authorization"] = f"Token {auth_token}"

        resp = requests.get(url, params=params, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        job_ids = data.get("job_ids") or []
        if not isinstance(job_ids, list):
            logger.warning(f"Unexpected job_ids format from {url}: {type(job_ids)}")
            return []
        return job_ids
    except requests.RequestException as e:
        logger.error(f"Failed to fetch jobs from {base_url}: {e}")
        return []


def run_worker(pipeline: str = "moth_binary"):
    # Determine API connection details from environment or defaults
    base_url = os.environ.get("ANTENNA_API_BASE_URL", "http://localhost:8000")
    auth_token = os.environ.get("ANTENNA_API_TOKEN", "")

    while True:
        # TODO CGJS: Use the correct pipeline id mapping
        jobs = _get_jobs(base_url=base_url, auth_token=auth_token, pipeline_id=11)
        for job_id in jobs:
            logger.info(f"Processing job {job_id} with pipeline {pipeline}")
            _process_job(
                pipeline=pipeline,
                job_id=job_id,
                base_url=base_url,
                auth_token=auth_token,
            )
        if not jobs:
            SLEEP_TIME = 5  # TODO CGJS: Make configurable and put at the top
            logger.info(f"No jobs found, sleeping for {SLEEP_TIME} seconds")
            import time

            time.sleep(SLEEP_TIME)


@torch.no_grad()
def _process_job(pipeline: str, job_id: int, base_url: str, auth_token: str):
    """Run the worker to process images from the REST API queue.

    Args:
        pipeline: Pipeline name to use for processing (e.g., moth_binary, panama_moths_2024)
    """
    assert auth_token is not None, "ANTENNA_API_TOKEN environment variable not set"

    # TODO CGJS: Remove the class and just make this a helper function
    loader = RESTAPIMothDetector(
        job_id=job_id, base_url=base_url, auth_token=auth_token
    )
    classifier = MothClassifierBinary(source_images=[], detections=[])
    detector = APIMothDetector([])

    torch.cuda.empty_cache()
    items = 0

    total_detection_time = 0.0
    total_classification_time = 0.0
    total_save_time = 0.0
    total_dl_time = 0.0
    all_detections = []
    _, t = log_time()
    for i, batch in enumerate(loader.get_dataloader()):
        detector.reset([])
        dt, t = t("Finished loading batch")
        total_dl_time += dt
        if not batch:
            logger.warning(f"Batch {i+1} is empty, skipping")
            continue

        # Extract data from dictionary batch
        batch_input = batch["image"]
        item_ids = batch["image_id"]
        reply_subjects = batch.get("reply_subject", [None] * len(batch_input))
        image_urls = batch.get("image_url", [None] * len(batch_input))

        # Track start time for this batch
        batch_start_time = datetime.datetime.now()

        logger.info(f"Processing batch {i+1}")
        # output is dict of "boxes", "labels", "scores"
        batch_output = detector.predict_batch(batch_input)

        items += len(batch_output)
        logger.info(f"Total items processed so far: {items}")
        batch_output = list(detector.post_process_batch(batch_output))

        # Convert item_ids to list if needed
        if isinstance(item_ids, (np.ndarray, torch.Tensor)):
            item_ids = item_ids.tolist()

        # TODO: Add seconds per item calculation for both detector and classifier

        detector.save_results(
            item_ids=item_ids,
            batch_output=batch_output,
            seconds_per_item=0,
        )
        dt, t = t("Finished detection")
        total_detection_time += dt

        # Group detections by image_id
        image_detections: dict[str, list[DetectionResponse]] = {
            img_id: [] for img_id in item_ids
        }
        image_tensors = {
            img_id: img_tensor for img_id, img_tensor in zip(item_ids, batch_input)
        }
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
            detections = classifier.save_results(
                metadata=([dresp.source_image_id], [idx]),
                batch_output=classifier_out,
                seconds_per_item=0,
            )
            image_detections[dresp.source_image_id].extend(detections)
            all_detections.extend(detections)

        ct, t = t("Finished classification")
        total_classification_time += ct

        # Calculate batch processing time
        batch_end_time = datetime.datetime.now()
        batch_elapsed = (batch_end_time - batch_start_time).total_seconds()

        # Post results back to the API with PipelineResponse for each image
        batch_results = []
        for reply_subject, image_id, image_url in zip(
            reply_subjects, item_ids, image_urls
        ):
            # Create SourceImageResponse for this image
            source_image = SourceImageResponse(id=image_id, url=image_url)

            # Create PipelineResultsResponse
            pipeline_response = PipelineResultsResponse(
                pipeline=pipeline,
                source_images=[source_image],
                detections=image_detections[image_id],
                total_time=batch_elapsed / len(item_ids),  # Approximate time per image
            )

            batch_results.append(
                {
                    "reply_subject": reply_subject,
                    "result": pipeline_response.model_dump(mode="json"),
                }
            )

        post_batch_results(base_url, job_id, batch_results, auth_token)
        st, t = t("Finished posting results")
        total_save_time += st

    # TODO CGJS: Remove this extra call if not needed
    post_batch_results(base_url, job_id, [], auth_token)
    logger.info(
        f"Done, detections: {len(all_detections)}. Detecting time: {total_detection_time}, "
        f"classification time: {total_classification_time}, dl time: {total_dl_time}, save time: {total_save_time}"
    )
