from typing import Any, List

from fastapi import APIRouter, Depends
from sqlalchemy import orm
from starlette.responses import Response

from trapdata.api.config import settings
from trapdata.api.deps.db import get_session
from trapdata.db.models.events import (
    MonitoringSessionListItem,
    list_monitoring_sessions,
)

router = APIRouter(prefix="/events")


@router.get("", response_model=List[MonitoringSessionListItem])
async def get_monitoring_sessions(
    response: Response,
    session: orm.Session = Depends(get_session),
    # request_params: RequestParams = Depends(parse_react_admin_params(Base)),
) -> Any:
    items = list_monitoring_sessions(
        session, settings.image_base_path, media_url_base="/static/"
    )
    return items


# @router.post("/process", response_model=List[DeploymentListItem])
# async def process_deployment(
#     response: Response,
#     session: orm.Session = Depends(get_session),
#     # request_params: RequestParams = Depends(parse_react_admin_params(Base)),
# ) -> Any:
#     from trapdata.ml.pipeline import start_pipeline
#
#     start_pipeline(
#         session=session, image_base_path=settings.image_base_path, settings=settings
#     )
#     deployments = list_deployments(session)
#     return deployments