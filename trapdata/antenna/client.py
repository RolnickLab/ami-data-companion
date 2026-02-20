"""Antenna API client for fetching jobs and posting results."""

import socket

import requests

from trapdata.antenna.schemas import (
    AntennaJobsListResponse,
    AntennaTaskResult,
    JobDispatchMode,
)
from trapdata.api.utils import get_http_session
from trapdata.common.logs import logger


def get_full_service_name(service_name: str) -> str:
    """Build full service name with hostname.

    Args:
        service_name: Base service name

    Returns:
        Full service name with hostname appended
    """
    hostname = socket.gethostname()
    return f"{service_name} ({hostname})"


def get_jobs(
    base_url: str,
    auth_token: str,
    pipeline_slug: str,
    processing_service_name: str,
) -> list[int]:
    """Fetch job ids from the API for the given pipeline.

    Calls: GET {base_url}/jobs?pipeline__slug=<pipeline>&ids_only=1&processing_service_name=<name>

    Args:
        base_url: Antenna API base URL (e.g., "http://localhost:8000/api/v2")
        auth_token: API authentication token
        pipeline_slug: Pipeline slug to filter jobs
        processing_service_name: Name of the processing service

    Returns:
        List of job ids (possibly empty) on success or error.
    """
    with get_http_session(auth_token) as session:
        try:
            url = f"{base_url.rstrip('/')}/jobs"
            params = {
                "pipeline__slug": pipeline_slug,
                "ids_only": 1,
                "incomplete_only": 1,
                "processing_service_name": processing_service_name,
                "dispatch_mode": JobDispatchMode.ASYNC_API,  # Only fetch async_api jobs
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


def post_batch_results(
    base_url: str,
    auth_token: str,
    job_id: int,
    results: list[AntennaTaskResult],
    processing_service_name: str,
) -> bool:
    """
    Post batch results back to the API.

    Args:
        base_url: Antenna API base URL (e.g., "http://localhost:8000/api/v2")
        auth_token: API authentication token
        job_id: Job ID
        results: List of AntennaTaskResult objects
        processing_service_name: Name of the processing service

    Returns:
        True if successful, False otherwise
    """
    url = f"{base_url.rstrip('/')}/jobs/{job_id}/result/"
    payload = [r.model_dump(mode="json") for r in results]

    with get_http_session(auth_token) as session:
        try:
            params = {"processing_service_name": processing_service_name}
            response = session.post(url, json=payload, params=params, timeout=60)
            response.raise_for_status()
            logger.debug(f"Successfully posted {len(results)} results to {url}")
            return True
        except requests.RequestException as e:
            logger.error(f"Failed to post results to {url}: {e}")
            return False


def get_user_projects(base_url: str, auth_token: str) -> list[dict]:
    """
    Fetch all projects the user has access to.

    Args:
        base_url: Base URL for the API (should NOT include /api/v2)
        auth_token: API authentication token

    Returns:
        List of project dictionaries with 'id' and 'name' fields
    """
    with get_http_session(auth_token) as session:
        try:
            url = f"{base_url.rstrip('/')}/projects/"
            response = session.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()

            projects = data.get("results", [])
            if isinstance(projects, list):
                return projects
            else:
                logger.warning(
                    f"Unexpected projects format from {url}: {type(projects)}"
                )
                return []
        except requests.RequestException as e:
            logger.error(f"Failed to fetch projects from {base_url}: {e}")
            return []
