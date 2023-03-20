"""
Trap Deployment

There is currently no database model representing trap deployments. 
The name of the directory that contains the raw images from a deployment
is used as the deployment name.
"""
import pathlib

import sqlalchemy as sa
from sqlalchemy import orm

from trapdata.db import models
from trapdata.common.types import FilePath


def deployment_name(image_base_path: FilePath) -> str:
    """
    Use the directory name of an absolute file path as the deployment name.
    """
    return pathlib.Path(image_base_path).name


def list_deployments(session: orm.Session) -> list[dict]:
    """
    List all image base directories that have been scanned.
    A proxy for "registered trap deployments".
    """
    stmt = sa.select(
        models.MonitoringSession.base_directory.label("name"),
        sa.func.count(models.MonitoringSession.id).label("num_events"),
        sa.func.sum(models.MonitoringSession.num_images).label("num_source_images"),
        sa.func.sum(models.MonitoringSession.num_detected_objects).label(
            "num_detections"
        ),
    ).group_by(models.MonitoringSession.base_directory)
    deployments = [dict(d._mapping) for d in session.execute(stmt).all()]
    for deployment in deployments:
        deployment["name"] = deployment_name(deployment["name"])
    return deployments
