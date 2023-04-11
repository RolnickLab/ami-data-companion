from fastapi import APIRouter

from trapdata.api.views import deployments, occurrences, settings, stats, status

api_router = APIRouter()

api_router.include_router(stats.router, tags=["stats"])
api_router.include_router(status.router, tags=["status"])
api_router.include_router(deployments.router, tags=["deployments"])
api_router.include_router(occurrences.router, tags=["occurrences"])
api_router.include_router(settings.router, tags=["settings"])