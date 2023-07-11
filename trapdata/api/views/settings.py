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
from trapdata.settings import UserSettings

router = APIRouter(prefix="/settings")


@router.get("", response_model=UserSettings)
async def get_settings(
    response: Response,
) -> Any:
    return settings
