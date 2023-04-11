import datetime
from typing import Any, List, Optional

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


class SummaryCounts(BaseModel):
    num_deployments: Optional[int] = 0
    num_captures: Optional[int] = 0
    num_sessions: Optional[int] = 0
    num_detections: Optional[int] = 0
    num_occurrences: Optional[int] = 0
    num_species: Optional[int] = 0 
    last_updated: Optional[datetime.datetime] = None


@router.get("/summary", response_model=SummaryCounts)
async def get_summary_counts(
    response: Response,
    session: orm.Session = Depends(get_session),
) -> Any:
    stmt = (
        sa.select(
            sa.func.count(models.MonitoringSession.base_directory.distinct()).label(
                "num_deployments"
            ),
            sa.func.count(models.MonitoringSession.id.distinct()).label("num_sessions"),
            sa.func.count(models.TrapImage.id.distinct()).label("num_captures"),
            sa.func.count(models.DetectedObject.id.distinct()).label(
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
            models.TrapImage,
            models.MonitoringSession.id == models.TrapImage.monitoring_session_id,
        )
        .join(
            models.DetectedObject,
            models.MonitoringSession.id == models.DetectedObject.monitoring_session_id,
        )
    )
    summary = session.execute(stmt).one()
    summary = SummaryCounts(**summary._mapping, last_updated=datetime.datetime.now())

    return summary
