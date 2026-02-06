"""Pydantic schemas for Antenna API requests and responses."""

import pydantic

from trapdata.api.schemas import PipelineConfigResponse, PipelineResultsResponse

# @TODO move more schemas here that are Antenna-specific from api/schemas.py


class AntennaPipelineProcessingTask(pydantic.BaseModel):
    """
    A task representing a single image or detection to be processed in an async pipeline.
    """

    id: str
    image_id: str
    image_url: str
    reply_subject: str | None = None  # The NATS subject to send the result to
    # TODO: Do we need these?
    # detections: list[DetectionRequest] | None = None
    # config: PipelineRequestConfigParameters | dict | None = None


class AntennaJobListItem(pydantic.BaseModel):
    """A single job item from the Antenna jobs list API response."""

    id: int


class AntennaJobsListResponse(pydantic.BaseModel):
    """Response from Antenna API GET /api/v2/jobs with ids_only=1."""

    results: list[AntennaJobListItem]


class AntennaTasksListResponse(pydantic.BaseModel):
    """Response from Antenna API GET /api/v2/jobs/{job_id}/tasks."""

    tasks: list[AntennaPipelineProcessingTask]


class AntennaTaskResultError(pydantic.BaseModel):
    """Error result for a single Antenna task that failed to process."""

    error: str
    image_id: str | None = None


class AntennaTaskResult(pydantic.BaseModel):
    """Result for a single Antenna task, either success or error."""

    reply_subject: str | None = None
    result: PipelineResultsResponse | AntennaTaskResultError


class AntennaTaskResults(pydantic.BaseModel):
    """Batch of task results to post back to Antenna API."""

    results: list[AntennaTaskResult] = pydantic.Field(default_factory=list)


class AsyncPipelineRegistrationRequest(pydantic.BaseModel):
    """
    Request to register pipelines from an async processing service
    """

    processing_service_name: str
    pipelines: list[PipelineConfigResponse] = []


class AsyncPipelineRegistrationResponse(pydantic.BaseModel):
    """
    Response from registering pipelines with a project.
    """

    pipelines_created: list[str] = pydantic.Field(
        default_factory=list,
        description="List of pipeline slugs that were created",
    )
    pipelines_updated: list[str] = pydantic.Field(
        default_factory=list,
        description="List of pipeline slugs that were updated",
    )
    processing_service_id: int | None = pydantic.Field(
        default=None,
        description="ID of the processing service that was created or updated",
    )

class MLBackend(str):
    """
    Backend types for ML job execution.
    """

    SYNC_API = "sync_api"
    ASYNC_API = "async_api"
