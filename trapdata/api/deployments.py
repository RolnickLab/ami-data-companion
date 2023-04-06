from typing import Any, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, orm, select
from starlette.responses import Response

from trapdata.api.deps.db import get_async_session
from trapdata.api.deps.request_params import parse_react_admin_params
from trapdata.api.request_params import RequestParams
from trapdata.db import Base, get_session
from trapdata.db.models.deployments import DeploymentListItem, list_deployments

router = APIRouter(prefix="/deployments")


@router.get("", response_model=List[DeploymentListItem])
async def get_deployments(
    response: Response,
    session: orm.Session = Depends(get_session),
    request_params: RequestParams = Depends(parse_react_admin_params(Base)),
) -> Any:
    total = await session.scalar(select(func.count(Deployment.id)))
    deployments = list_deployments(session)
    response.headers[
        "Content-Range"
    ] = f"{request_params.skip}-{request_params.skip + len(deployments)}/{total}"
    return deployments
