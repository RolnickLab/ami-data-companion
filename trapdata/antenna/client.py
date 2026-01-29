"""Antenna API client for fetching jobs and posting results."""

import requests

from trapdata.api.utils import get_http_session
from trapdata.antenna.schemas import (
    AntennaJobsListResponse,
    AntennaTaskResult,
)
from trapdata.common.logs import logger
from trapdata.settings import Settings


def get_jobs(
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
                logger.warning(
                    f"Unexpected projects format from {url}: {type(projects)}"
                )
                return []
        except requests.RequestException as e:
            logger.error(f"Failed to fetch projects from {base_url}: {e}")
            return []
