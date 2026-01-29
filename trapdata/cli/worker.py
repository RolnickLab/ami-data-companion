"""Worker to process images from the REST API queue."""

import datetime
import socket
import time

import numpy as np
import requests
import torch

from trapdata.api.api import CLASSIFIER_CHOICES, initialize_service_info
from trapdata.api.datasets import get_rest_dataloader
from trapdata.api.models.localization import APIMothDetector
from trapdata.api.schemas import (
    AntennaJobsListResponse,
    AntennaTaskResult,
    AntennaTaskResultError,
    AsyncPipelineRegistrationRequest,
    AsyncPipelineRegistrationResponse,
    DetectionResponse,
    PipelineResultsResponse,
    SourceImageResponse,
)
from trapdata.api.utils import get_http_session
from trapdata.common.logs import logger
from trapdata.common.utils import log_time
from trapdata.settings import Settings, read_settings

SLEEP_TIME_SECONDS = 5


def post_batch_results(
    settings: Settings,
    job_id: int,
    results: list[AntennaTaskResult],
) -> bool:
    """
    Post batch results back to the API.

    Args:
        settings: Settings object with antenna_api_* configuration
        job_id: Job ID
        results: List of AntennaTaskResult objects

    Returns:
        True if successful, False otherwise
    """
    url = f"{settings.antenna_api_base_url.rstrip('/')}/jobs/{job_id}/result/"
    payload = [r.model_dump(mode="json") for r in results]

    with get_http_session(
        auth_token=settings.antenna_api_auth_token,
        max_retries=settings.antenna_api_retry_max,
        backoff_factor=settings.antenna_api_retry_backoff,
    ) as session:
        try:
            response = session.post(url, json=payload, timeout=60)
            response.raise_for_status()
            logger.info(f"Successfully posted {len(results)} results to {url}")
            return True
        except requests.RequestException as e:
            logger.error(f"Failed to post results to {url}: {e}")
            return False


def _get_jobs(
    base_url: str,
    auth_token: str,
    pipeline_slug: str,
    retry_max: int = 3,
    retry_backoff: float = 0.5,
) -> list[int]:
    """Fetch job ids from the API for the given pipeline.

    Calls: GET {base_url}/jobs?pipeline__slug=<pipeline>&ids_only=1

    Args:
        base_url: Antenna API base URL (e.g., "http://localhost:8000/api/v2")
        auth_token: API authentication token
        pipeline_slug: Pipeline slug to filter jobs
        retry_max: Maximum retry attempts for failed requests
        retry_backoff: Exponential backoff factor in seconds

    Returns:
        List of job ids (possibly empty) on success or error.
    """
    with get_http_session(
        auth_token=auth_token,
        max_retries=retry_max,
        backoff_factor=retry_backoff,
    ) as session:
        try:
            url = f"{base_url.rstrip('/')}/jobs"
            params = {
                "pipeline__slug": pipeline_slug,
                "ids_only": 1,
                "incomplete_only": 1,
            }

            resp = session.get(url, params=params, timeout=30)
            resp.raise_for_status()

            # Parse and validate response with Pydantic
            jobs_response = AntennaJobsListResponse.model_validate(resp.json())
            return [job.id for job in jobs_response.results]
        except requests.RequestException as e:
            logger.error(f"Failed to fetch jobs from {base_url}: {e}")
            return []
        except Exception as e:
            logger.error(f"Failed to parse jobs response: {e}")
            return []


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
            jobs = _get_jobs(
                base_url=settings.antenna_api_base_url,
                auth_token=settings.antenna_api_auth_token,
                pipeline_slug=pipeline,
            )
            for job_id in jobs:
                logger.info(f"Processing job {job_id} with pipeline {pipeline}")
                any_work_done = _process_job(
                    pipeline=pipeline,
                    job_id=job_id,
                    settings=settings,
                )
                any_jobs = any_jobs or any_work_done

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
        image_tensors = dict(zip(image_ids, images))

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
            reply_subjects, image_ids, image_urls
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

        post_batch_results(settings, job_id, batch_results)
        st, t = t("Finished posting results")
        total_save_time += st

    logger.info(
        f"Done, detections: {len(all_detections)}. Detecting time: {total_detection_time}, "
        f"classification time: {total_classification_time}, dl time: {total_dl_time}, save time: {total_save_time}"
    )
    return did_work


def get_user_projects(
    base_url: str,
    auth_token: str,
) -> list[dict]:
    """
    Fetch all projects the user has access to.

    Args:
        base_url: Base URL for the API (should NOT include /api/v2)
        auth_token: API authentication token

    Returns:
        List of project dictionaries with 'id' and 'name' fields
    """
    with get_http_session(auth_token=auth_token) as session:
        try:
            url = f"{base_url.rstrip('/')}/projects/"
            response = session.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()

            projects = data.get("results", [])
            if isinstance(projects, list):
                return projects
            else:
                logger.warning(f"Unexpected projects format from {url}: {type(projects)}")
                return []
        except requests.RequestException as e:
            logger.error(f"Failed to fetch projects from {base_url}: {e}")
            return []


def register_pipelines_for_project(
    base_url: str,
    auth_token: str,
    project_id: int,
    service_name: str,
    pipeline_configs: list,
) -> tuple[bool, str]:
    """
    Register all available pipelines for a specific project.

    Args:
        base_url: Base URL for the API (should NOT include /api/v2)
        auth_token: API authentication token
        project_id: Project ID to register pipelines for
        service_name: Name of the processing service
        pipeline_configs: Pre-built pipeline configuration objects

    Returns:
        Tuple of (success: bool, message: str)
    """
    with get_http_session(auth_token=auth_token) as session:
        try:
            registration_request = AsyncPipelineRegistrationRequest(
                processing_service_name=service_name, pipelines=pipeline_configs
            )

            url = f"{base_url.rstrip('/')}/projects/{project_id}/pipelines/"
            response = session.post(
                url,
                json=registration_request.model_dump(mode="json"),
                timeout=60,
            )
            response.raise_for_status()

            result = AsyncPipelineRegistrationResponse.model_validate(response.json())
            return True, f"Created {len(result.pipelines_created)} new pipelines"

        except requests.RequestException as e:
            if hasattr(e, 'response') and e.response is not None and e.response.status_code == 400:
                try:
                    error_data = e.response.json()
                    error_detail = error_data.get("detail", str(e))
                except Exception:
                    error_detail = str(e)
                return False, f"Registration failed: {error_detail}"
            else:
                return False, f"Network error during registration: {e}"
        except Exception as e:
            return False, f"Unexpected error during registration: {e}"


def register_pipelines(
    project_ids: list[int],
    service_name: str,
    settings: Settings | None = None,
) -> None:
    """
    Register pipelines for specified projects or all accessible projects.

    Args:
        project_ids: List of specific project IDs to register for. If empty, registers for all accessible projects.
        service_name: Name of the processing service
        settings: Settings object with antenna_api_* configuration (defaults to read_settings())
    """
    # Get settings from parameter or read from environment
    if settings is None:
        settings = read_settings()

    base_url = settings.antenna_api_base_url
    auth_token = settings.antenna_api_auth_token

    if not auth_token:
        logger.error("AMI_ANTENNA_API_AUTH_TOKEN environment variable not set")
        return

    if service_name is None:
        logger.error("Service name is required for registration")
        return

    # Add hostname to service name
    hostname = socket.gethostname()
    full_service_name = f"{service_name} ({hostname})"

    # Get projects to register for
    projects_to_process = []
    if project_ids:
        # Use specified project IDs
        projects_to_process = [
            {"id": pid, "name": f"Project {pid}"} for pid in project_ids
        ]
        logger.info(f"Registering pipelines for specified projects: {project_ids}")
    else:
        # Fetch all accessible projects
        logger.info("Fetching all accessible projects...")
        all_projects = get_user_projects(base_url, auth_token)
        projects_to_process = all_projects
        logger.info(f"Found {len(projects_to_process)} accessible projects")

    if not projects_to_process:
        logger.warning("No projects found to register pipelines for")
        return

    # Initialize service info once to get pipeline configurations
    logger.info("Initializing pipeline configurations...")
    service_info = initialize_service_info()
    pipeline_configs = service_info.pipelines
    logger.info(f"Generated {len(pipeline_configs)} pipeline configurations")

    # Register pipelines for each project
    successful_registrations = []
    failed_registrations = []

    logger.info(f"Available pipelines to register: {list(CLASSIFIER_CHOICES.keys())}")

    for project in projects_to_process:
        project_id = project["id"]
        project_name = project.get("name", f"Project {project_id}")

        logger.info(
            f"Registering pipelines for project {project_id} ({project_name})..."
        )

        success, message = register_pipelines_for_project(
            base_url=base_url,
            auth_token=auth_token,
            project_id=project_id,
            service_name=full_service_name,
            pipeline_configs=pipeline_configs,
        )

        if success:
            successful_registrations.append((project_id, project_name, message))
            logger.info(f"✓ Project {project_id} ({project_name}): {message}")
        else:
            failed_registrations.append((project_id, project_name, message))
            if "Processing service already exists" in message:
                logger.warning(f"⚠ Project {project_id} ({project_name}): {message}")
            else:
                logger.error(f"✗ Project {project_id} ({project_name}): {message}")

    # Summary report
    logger.info("\n=== Registration Summary ===")
    logger.info(f"Service name: {full_service_name}")
    logger.info(f"Total projects processed: {len(projects_to_process)}")
    logger.info(f"Successful registrations: {len(successful_registrations)}")
    logger.info(f"Failed registrations: {len(failed_registrations)}")

    if successful_registrations:
        logger.info("\nSuccessful registrations:")
        for project_id, project_name, message in successful_registrations:
            logger.info(f"  - Project {project_id} ({project_name}): {message}")

    if failed_registrations:
        logger.info("\nFailed registrations:")
        for project_id, project_name, message in failed_registrations:
            logger.info(f"  - Project {project_id} ({project_name}): {message}")
