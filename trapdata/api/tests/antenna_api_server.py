"""Mock Antenna API server for integration testing.

This module provides a FastAPI application that mocks the Antenna API endpoints
used by the worker. It allows tests to validate the API contract without
requiring an actual Antenna server.
"""

from fastapi import FastAPI, HTTPException

from trapdata.api.schemas import (
    AntennaJobListItem,
    AntennaJobsListResponse,
    AntennaPipelineProcessingTask,
    AntennaTaskResult,
    AntennaTasksListResponse,
    AsyncPipelineRegistrationRequest,
    AsyncPipelineRegistrationResponse,
)

app = FastAPI()

# State management for tests
_jobs_queue: dict[int, list[AntennaPipelineProcessingTask]] = {}
_posted_results: dict[int, list[AntennaTaskResult]] = {}
_projects: list[dict] = []
_registered_pipelines: dict[int, list[str]] = {}  # project_id -> pipeline slugs


@app.get("/api/v2/jobs")
def get_jobs(pipeline__slug: str, ids_only: int, incomplete_only: int):
    """Return available job IDs.

    Args:
        pipeline__slug: Pipeline slug filter
        ids_only: If 1, return only job IDs
        incomplete_only: If 1, return only incomplete jobs

    Returns:
        AntennaJobsListResponse with list of job IDs
    """
    # Return all jobs in queue (for testing, we return all registered jobs)
    job_ids = list(_jobs_queue.keys())
    results = [AntennaJobListItem(id=job_id) for job_id in job_ids]
    return AntennaJobsListResponse(results=results)


@app.get("/api/v2/jobs/{job_id}/tasks")
def get_tasks(job_id: int, batch: int):
    """Return batch of tasks (atomically remove from queue).

    Args:
        job_id: Job ID to fetch tasks for
        batch: Number of tasks to return

    Returns:
        AntennaTasksListResponse with batch of tasks
    """
    if job_id not in _jobs_queue:
        return AntennaTasksListResponse(tasks=[])

    # Get up to `batch` tasks and remove them from queue
    tasks = _jobs_queue[job_id][:batch]
    _jobs_queue[job_id] = _jobs_queue[job_id][batch:]

    return AntennaTasksListResponse(tasks=tasks)


@app.post("/api/v2/jobs/{job_id}/result/")
def post_results(job_id: int, payload: list[dict]):
    """Store posted results for test validation.

    Args:
        job_id: Job ID to post results for
        payload: List of AntennaTaskResult dicts

    Returns:
        Success status
    """
    if job_id not in _posted_results:
        _posted_results[job_id] = []

    # Parse each result dict into AntennaTaskResult
    for result_dict in payload:
        task_result = AntennaTaskResult(**result_dict)
        _posted_results[job_id].append(task_result)

    return {"status": "ok"}


@app.get("/api/v2/projects/")
def get_projects():
    """Return list of projects the user has access to.

    Returns:
        Paginated response with list of projects
    """
    return {"results": _projects}


@app.post("/api/v2/projects/{project_id}/pipelines/")
def register_pipelines(project_id: int, payload: dict):
    """Register pipelines for a project.

    Args:
        project_id: Project ID to register pipelines for
        payload: AsyncPipelineRegistrationRequest as dict

    Returns:
        AsyncPipelineRegistrationResponse
    """
    # Validate request
    request = AsyncPipelineRegistrationRequest(**payload)

    # Check if project exists
    project_ids = [p["id"] for p in _projects]
    if project_id not in project_ids:
        raise HTTPException(status_code=404, detail="Project not found")

    # Track registered pipelines
    if project_id not in _registered_pipelines:
        _registered_pipelines[project_id] = []

    created = []
    for pipeline in request.pipelines:
        if pipeline.slug not in _registered_pipelines[project_id]:
            _registered_pipelines[project_id].append(pipeline.slug)
            created.append(pipeline.slug)

    return AsyncPipelineRegistrationResponse(
        pipelines_created=created,
        pipelines_updated=[],
        processing_service_id=1,
    )


# Test helper methods


def setup_job(job_id: int, tasks: list[AntennaPipelineProcessingTask]):
    """Populate job queue for testing.

    Args:
        job_id: Job ID to setup
        tasks: List of tasks to add to the queue
    """
    _jobs_queue[job_id] = tasks.copy()


def get_posted_results(job_id: int) -> list[AntennaTaskResult]:
    """Retrieve results posted by worker.

    Args:
        job_id: Job ID to get results for

    Returns:
        List of posted task results
    """
    return _posted_results.get(job_id, [])


def setup_projects(projects: list[dict]):
    """Setup projects for testing.

    Args:
        projects: List of project dicts with 'id' and 'name' fields
    """
    _projects.clear()
    _projects.extend(projects)


def get_registered_pipelines(project_id: int) -> list[str]:
    """Get list of pipeline slugs registered for a project.

    Args:
        project_id: Project ID to get pipelines for

    Returns:
        List of pipeline slugs
    """
    return _registered_pipelines.get(project_id, [])


def reset():
    """Clear all state between tests."""
    _jobs_queue.clear()
    _posted_results.clear()
    _projects.clear()
    _registered_pipelines.clear()
