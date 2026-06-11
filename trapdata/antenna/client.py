"""Antenna API client for fetching jobs and posting results."""

import json
import socket

import requests

from trapdata.antenna.schemas import (
    AntennaJobsListResponse,
    AntennaResultPostResponse,
    AntennaTaskResult,
    AntennaTaskResults,
    JobDispatchMode,
)
from trapdata.api.utils import get_http_session
from trapdata.common.logs import logger

# Default maximum size (bytes) of a single result POST body. Used when a caller
# does not pass an explicit limit (e.g. settings.antenna_result_post_max_bytes).
# Chosen to stay well under the reverse-proxy request-body limit (commonly 100 MB)
# while still allowing several wide-taxonomy detections per request.
DEFAULT_RESULT_POST_MAX_BYTES = 25 * 1024 * 1024


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
    pipeline_slugs: list[str],
) -> list[tuple[int, str]]:
    """Fetch job ids from the API for the given pipelines in a single request.

    Calls: GET {base_url}/jobs?pipeline__slug__in=<slugs>&ids_only=1

    Args:
        base_url: Antenna API base URL (e.g., "http://localhost:8000/api/v2")
        auth_token: API authentication token
        pipeline_slugs: List of pipeline slugs to filter jobs

    Returns:
        List of (job_id, pipeline_slug) tuples (possibly empty) on success or error.
    """
    with get_http_session(auth_token) as session:
        try:
            if not pipeline_slugs:
                return []
            url = f"{base_url.rstrip('/')}/jobs"
            params = {
                "pipeline__slug__in": ",".join(pipeline_slugs),
                "ids_only": 1,
                "incomplete_only": 1,
                "dispatch_mode": JobDispatchMode.ASYNC_API,  # Only fetch async_api jobs
            }

            resp = session.get(url, params=params, timeout=30)
            resp.raise_for_status()

            # Parse and validate response with Pydantic
            jobs_response = AntennaJobsListResponse.model_validate(resp.json())
            return [(job.id, job.pipeline_slug) for job in jobs_response.results]
        except requests.RequestException as e:
            logger.error(f"Failed to fetch jobs from {base_url}: {e}")
            return []
        except Exception as e:
            logger.error(f"Failed to parse jobs response: {e}")
            return []


def _result_json_size(result_json: dict) -> int:
    """Approximate the serialized byte size of one result entry.

    Uses a compact JSON encoding (no extra whitespace) so the estimate tracks
    what ``requests`` actually sends. The few bytes of array/comma framing that
    join entries in the final payload are ignored; they are negligible next to
    the per-result content for wide-taxonomy classifiers.
    """
    return len(json.dumps(result_json, separators=(",", ":")).encode("utf-8"))


def chunk_results_by_size(
    results_json: list[dict],
    max_bytes: int,
) -> list[list[dict]]:
    """Greedily pack already-serialized result dicts into byte-bounded chunks.

    Each returned chunk, once wrapped in the ``{"results": [...]}`` envelope, is
    intended to stay at or below ``max_bytes``. A single result that exceeds
    ``max_bytes`` on its own is emitted as its own chunk (it cannot be split
    further without changing the per-image API contract), and a warning is
    logged so the oversize case is visible.

    Args:
        results_json: Result entries already converted to JSON-compatible dicts.
        max_bytes: Target maximum size in bytes for one POST body.

    Returns:
        A list of chunks, where each chunk is a list of result dicts. Returns an
        empty list when given no results.
    """
    if not results_json:
        return []
    if max_bytes <= 0:
        # Defensive: a non-positive cap would otherwise loop forever. Fall back
        # to posting everything in a single chunk.
        return [list(results_json)]

    # Account for the constant envelope overhead of ``{"results":[]}``.
    envelope_overhead = len(b'{"results":[]}')
    budget = max(1, max_bytes - envelope_overhead)

    chunks: list[list[dict]] = []
    current: list[dict] = []
    current_size = 0

    for result_json in results_json:
        size = _result_json_size(result_json)
        if size > budget:
            logger.warning(
                f"Single result entry is {size} bytes, larger than the "
                f"{max_bytes}-byte POST limit; sending it in its own request. "
                "It may still be rejected by the server or proxy."
            )
        # +1 accounts for the comma joining this entry to the previous one.
        added_size = size + (1 if current else 0)
        if current and current_size + added_size > budget:
            chunks.append(current)
            current = []
            current_size = 0
            added_size = size
        current.append(result_json)
        current_size += added_size

    if current:
        chunks.append(current)
    return chunks


def post_batch_results(
    base_url: str,
    auth_token: str,
    job_id: int,
    results: list[AntennaTaskResult],
    max_bytes: int = DEFAULT_RESULT_POST_MAX_BYTES,
) -> bool:
    """
    Post batch results back to the API, splitting large batches across requests.

    The results for one processed batch are serialized once and packed into one
    or more POST bodies that each stay at or below ``max_bytes``. This prevents a
    single dense, wide-taxonomy batch from producing a request body large enough
    to be rejected by the reverse proxy (HTTP 413).

    Args:
        base_url: Antenna API base URL (e.g., "http://localhost:8000/api/v2")
        auth_token: API authentication token
        job_id: Job ID
        results: List of AntennaTaskResult objects
        max_bytes: Maximum size in bytes of a single POST body.

    Returns:
        True only if every chunk was posted successfully, False otherwise.
    """
    if not results:
        return True

    url = f"{base_url.rstrip('/')}/jobs/{job_id}/result/"

    # Serialize each result once, then group into byte-bounded chunks so we never
    # pay the serialization cost twice.
    results_json = [
        AntennaTaskResults(results=[r]).model_dump(mode="json")["results"][0]
        for r in results
    ]
    chunks = chunk_results_by_size(results_json, max_bytes)

    all_ok = True
    with get_http_session(auth_token) as session:
        for chunk_idx, chunk in enumerate(chunks):
            payload = {"results": chunk}
            try:
                response = session.post(url, json=payload, timeout=60)
                response.raise_for_status()
                result = AntennaResultPostResponse.model_validate(response.json())
                logger.debug(
                    f"Posted chunk {chunk_idx + 1}/{len(chunks)} "
                    f"({len(chunk)} results) to job {job_id}: "
                    f"{result.results_queued} queued"
                )
            except requests.RequestException as e:
                logger.error(
                    f"Failed to post result chunk {chunk_idx + 1}/{len(chunks)} "
                    f"to {url}: {e}"
                )
                all_ok = False
    return all_ok


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
