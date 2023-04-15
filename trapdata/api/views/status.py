import datetime
from typing import Any, List, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import orm
from starlette.responses import Response

from trapdata.api.config import settings
from trapdata.api.deps.db import get_session
from trapdata.db.models.deployments import list_deployments
from trapdata.db.models.events import list_monitoring_sessions
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
    deployments = list_deployments(session)
    # events = []
    # for deployment in deployments:
    #     events += list_monitoring_sessions(session, deployment.image_base_path)
    events = list_monitoring_sessions(session, settings.image_base_path)

    summary = SummaryCounts(
        num_deployments=len(deployments),
        num_sessions=len(events),
        num_captures=sum(e.num_captures for e in events),
        num_detections=sum(e.num_detections for e in events),
        num_occurrences=sum(e.num_occurrences for e in events),
        num_species=sum(e.num_species for e in events),
        last_updated=datetime.datetime.now(),
    )

    return summary
