import datetime
from typing import Any, List

import sqlalchemy as sa
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import orm
from starlette.responses import Response

from trapdata.api.config import settings
from trapdata.api.deps.db import get_session
from trapdata.db import models
from trapdata.db.models.queue import QueueListItem, list_queues

router = APIRouter(prefix="/status")


@router.get("/queues", response_model=List[QueueListItem])
async def get_queues(
    response: Response,
) -> Any:
    queues = list_queues(settings.database_url, settings.image_base_path)
    return queues


class NavSummary(BaseModel):
    num_deployments: int
    num_captures: int
    num_sessions: int
    num_detections: int
    num_occurrences: int
    num_species: int
    last_updated: datetime.datetime


@router.get("/nav_summary", response_model=NavSummary)
async def get_nav_summary(
    response: Response,
    session: orm.Session = Depends(get_session),
) -> Any:
    stmt = (
        sa.select(
            sa.func.count(models.MonitoringSession.base_directory.distinct()).label(
                "num_deployments"
            ),
            sa.func.count(models.MonitoringSession.id.distinct()).label("num_sessions"),
            sa.func.sum(models.MonitoringSession.num_images).label("num_captures"),
            sa.func.sum(models.MonitoringSession.num_detected_objects).label(
                "num_detections"
            ),
            sa.func.count(models.DetectedObject.sequence_id.distinct()).label(
                "num_occurrences"
            ),  # @TODO does not filter based on classification threshold, among other things!
            sa.func.count(models.DetectedObject.specific_label.distinct()).label(
                "num_species"
            ),
        )
        .join(
            models.DetectedObject,
            models.MonitoringSession.id == models.DetectedObject.monitoring_session_id,
        )
        .group_by(models.MonitoringSession.base_directory)
    )
    summary = session.execute(stmt).first()
    if summary:
        summary = NavSummary(**summary._mapping, last_updated=datetime.datetime.now())

    return summary
