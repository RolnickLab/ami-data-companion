from fastapi import APIRouter

from trapdata.api.views import deployments, stats

api_router = APIRouter()

api_router.include_router(stats.router, tags=["stats"])
api_router.include_router(deployments.router, tags=["deployments"])
