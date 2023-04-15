from typing import Any, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, orm, select
from starlette.responses import Response

from trapdata.api.config import settings
from trapdata.api.deps.db import get_session
from trapdata.api.deps.request_params import parse_react_admin_params
from trapdata.api.request_params import RequestParams
from trapdata.db import Base
from trapdata.db.models.detections import TaxonListItem, list_species

router = APIRouter(prefix="/species")


@router.get("", response_model=List[TaxonListItem])
async def get_species(
    response: Response,
    session: orm.Session = Depends(get_session),
    # request_params: RequestParams = Depends(parse_react_admin_params(Base)),
) -> Any:
    species = list_species(
        session=session,
        classification_threshold=settings.classification_threshold,
        media_url_base="/static/",
    )
    return species
