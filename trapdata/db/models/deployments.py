"""
Trap Deployment

There is currently no database model representing trap deployments.
The name of the directory that contains the raw images from a deployment
is used as the deployment name.
"""

import pathlib
from typing import Optional

import sqlalchemy as sa
from pydantic import BaseModel
from sqlalchemy import orm

from trapdata.common.schemas import FilePath
from trapdata.db import models


class DeploymentListItem(BaseModel):
    id: Optional[int] = None
    name: str
    image_base_path: FilePath
    num_events: int
    num_source_images: int
    num_detections: int
    # num_occurrences: int
    # num_species: int


class DeploymentDetail(DeploymentListItem):
    pass


def deployment_name(image_base_path: FilePath) -> str:
    """
    Use the directory name of an absolute file path as the deployment name.
    """
    return pathlib.Path(image_base_path).name


def list_deployments(session: orm.Session) -> list[DeploymentListItem]:
    """
    List all image base directories that have been scanned.
    A proxy for "registered trap deployments".
    """
    stmt = sa.select(
        models.MonitoringSession.base_directory.label("image_base_path"),
        sa.func.sum(models.MonitoringSession.num_images).label("num_source_images"),
        sa.func.sum(models.MonitoringSession.num_detected_objects).label(
            "num_detections"
        ),
    ).group_by(models.MonitoringSession.base_directory)
    deployments = []
    for deployment in session.execute(stmt).all():
        num_events = (
            session.scalar(
                sa.select(sa.func.count(models.MonitoringSession.id)).where(
                    models.MonitoringSession.base_directory
                    == str(deployment.image_base_path)
                )
            )
            or 0
        )
        deployments.append(
            DeploymentListItem(
                **deployment._mapping,
                num_events=num_events,
                name=deployment_name(deployment.image_base_path),
            )
        )

    return deployments
