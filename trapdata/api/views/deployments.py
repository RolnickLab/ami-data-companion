from typing import Any, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, orm, select
from starlette.responses import Response

from trapdata.api.config import settings
from trapdata.api.deps.db import get_session
from trapdata.api.deps.request_params import parse_react_admin_params
from trapdata.api.request_params import RequestParams
from trapdata.db import Base
from trapdata.db.models.deployments import DeploymentListItem, list_deployments
from trapdata.db.models.events import update_all_aggregates

router = APIRouter(prefix="/deployments")


@router.get("", response_model=List[DeploymentListItem])
async def get_deployments(
    response: Response,
    session: orm.Session = Depends(get_session),
    # request_params: RequestParams = Depends(parse_react_admin_params(Base)),
) -> Any:
    update_all_aggregates(session, settings.image_base_path)
    deployments = list_deployments(session)
    return deployments


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
