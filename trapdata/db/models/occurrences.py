"""
Occurrence of an Individual Organism

There is currently no database model representing an occurrence.
And occurrence is a sequence of detections that are determined to be
the same individual, tracked over multiple frames in the original images
from a monitoring session.
"""
import datetime
import pathlib
from typing import Optional

import sqlalchemy as sa
from pydantic import BaseModel

from trapdata import db
from trapdata.common.filemanagement import media_url
from trapdata.common.schemas import FilePath
from trapdata.db import models


class OccurrenceNestedEvent(BaseModel):
    id: int
    day: datetime.date
    url: Optional[str] = None


class OccurrenceNestedDetection(BaseModel):
    id: int
    cropped_image_path: str


class OccurrenceListItem(BaseModel):
    id: str
    label: str
    best_score: float
    start_time: datetime.datetime
    end_time: datetime.datetime
    duration: datetime.timedelta
    deployment: str
    event: OccurrenceNestedEvent
    num_frames: int
    # cropped_image_path: pathlib.Path
    # source_image_id: int
    examples: list[dict]
    # detections: list[object]
    # deployment: object
    # captures: list[object] =
    url: Optional[str] = None


class SpeciesSummaryListItem(BaseModel):
    name: str
    count: int
    example: Optional[pathlib.Path] = None


def list_occurrences(
    db_path: str,
    image_base_path: FilePath,
    monitoring_session: Optional[models.MonitoringSession] = None,
    classification_threshold: float = -1,
    num_examples: int = 10,
    limit: Optional[int] = None,
    offset: int = 0,
    media_url_base: Optional[str] = None,
) -> list[OccurrenceListItem]:
    occurrences = []
    for item in get_unique_species_by_track(
        db_path,
        monitoring_session=monitoring_session,
        image_base_path=image_base_path,
        classification_threshold=classification_threshold,
        num_examples=num_examples,
        limit=limit,
        offset=offset,
    ):
        prepped = {k.split("sequence_", 1)[-1]: v for k, v in item.items()}
        if prepped["id"]:
            prepped["deployment"] = models.deployments.deployment_name(
                item["monitoring_session_base_directory"]
            )
            if media_url_base:
                examples = [dict(example) for example in prepped["examples"]]
                # @TODO use OccurrenceNestedDetection
                for example in examples:
                    example["cropped_image_path"] = media_url(
                        example["cropped_image_path"], "crops", media_url_base
                    )
                    example["source_image_path"] = media_url(
                        example["source_image_path"], "captures/", media_url_base
                    )
                prepped["examples"] = examples

            prepped["event"] = OccurrenceNestedEvent(
                id=item["monitoring_session_id"], day=item["monitoring_session_day"]
            )
            occur = OccurrenceListItem(**prepped)
            occurrences.append(occur)
    return occurrences


def get_valid_sequence_ids(
    monitoring_session: Optional[models.MonitoringSession] = None,
    confidence_threshold: float = 0,
) -> sa.ScalarSelect:
    """
    Sequence IDs that have a detection with a score above the confidence threshold.

    Intended to be used as a subquery in a larger query.
    """
    stmt = sa.select(
        models.DetectedObject.sequence_id.distinct().label("id"),
    ).where(models.DetectedObject.specific_label_score >= confidence_threshold)
    if monitoring_session:
        stmt = stmt.where(
            models.DetectedObject.monitoring_session_id == monitoring_session.id
        )
    stmt = (
        stmt.group_by(models.DetectedObject.sequence_id)
        .order_by(models.DetectedObject.sequence_id)
        .scalar_subquery()
    )
    return stmt


def list_species(
    db_path: str,
    image_base_path: pathlib.Path,
    classification_threshold: float = -1,
    limit: Optional[int] = None,
    offset: int = 0,
) -> list[SpeciesSummaryListItem]:
    Session = db.get_session_class(db_path)
    session = Session()
    rows = (
        session.execute(
            sa.select(
                models.DetectedObject.specific_label.label("name"),
                sa.func.count(models.DetectedObject.sequence_id).label("count"),
            )
            .where(
                (models.TrapImage.base_path == str(image_base_path))
                & (
                    models.DetectedObject.specific_label_score
                    >= classification_threshold
                )
            )
            .join(
                models.TrapImage, models.DetectedObject.image_id == models.TrapImage.id
            )
            .group_by(models.DetectedObject.specific_label)
            .limit(limit)
            .offset(offset)
            .order_by(models.DetectedObject.specific_label)
        )
        .unique()
        .all()
    )
    species = [SpeciesSummaryListItem(**dict(row._mapping)) for row in rows]
    return species


def get_unique_species_by_track(
    db_path: str,
    image_base_path: FilePath,
    monitoring_session: Optional[models.MonitoringSession] = None,
    classification_threshold: float = -1,
    num_examples: int = 3,
    limit: Optional[int] = None,
    offset: int = 0,
) -> list[dict]:
    Session = db.get_session_class(db_path)
    session = Session()

    # Select all sequences where at least one example is above the score threshold
    stmt = (
        sa.select(
            models.DetectedObject.sequence_id,
            models.DetectedObject.monitoring_session_id,
            models.MonitoringSession.day.label("monitoring_session_day"),
            models.MonitoringSession.base_directory.label(
                "monitoring_session_base_directory"
            ),
            sa.func.count(models.DetectedObject.id).label(
                "sequence_frame_count"
            ),  # frames in track
            sa.func.max(models.DetectedObject.specific_label_score).label(
                "sequence_best_score"
            ),
            sa.func.min(models.DetectedObject.timestamp).label("sequence_start_time"),
            sa.func.max(models.DetectedObject.timestamp).label("sequence_end_time"),
        )
        .join(
            models.MonitoringSession,
            models.DetectedObject.monitoring_session_id == models.MonitoringSession.id,
        )
        .where(models.MonitoringSession.base_directory == str(image_base_path))
    )
    if monitoring_session:
        stmt = stmt.where(models.MonitoringSession.id == monitoring_session.id)

    stmt = (
        stmt.group_by(
            "sequence_id",
            "monitoring_session_id",
            "monitoring_session_day",
            "monitoring_session_base_directory",
        )
        .having(
            sa.func.max(models.DetectedObject.specific_label_score)
            >= classification_threshold,
        )
        .order_by("sequence_id")
        .limit(limit)
        .offset(offset)
    )
    sequences = session.execute(stmt).all()

    rows = []
    for sequence in sequences:
        stmt = (
            sa.select(
                models.DetectedObject.id,
                models.DetectedObject.image_id.label("source_image_id"),
                models.TrapImage.path.label("source_image_path"),
                models.TrapImage.width.label("source_image_width"),
                models.TrapImage.height.label("source_image_height"),
                models.TrapImage.filesize.label("source_image_filesize"),
                models.DetectedObject.specific_label.label("label"),
                models.DetectedObject.specific_label_score.label("score"),
                models.DetectedObject.path.label("cropped_image_path"),
                models.DetectedObject.sequence_id,
                models.DetectedObject.timestamp,
                models.DetectedObject.bbox,
            )
            .join(
                models.TrapImage, models.TrapImage.id == models.DetectedObject.image_id
            )
            .where(models.DetectedObject.sequence_id == sequence.sequence_id)
        )

        if monitoring_session:
            stmt = stmt.where(
                models.DetectedObject.monitoring_session_id == monitoring_session.id
            )
        stmt = stmt.order_by(sa.desc("score")).limit(num_examples)

        frames = session.execute(stmt).all()

        row = dict(sequence._mapping)
        if frames:
            best_example = frames[0]
            row["label"] = best_example.label
            row["num_frames"] = sequence.sequence_frame_count
            row["examples"] = [example._mapping for example in frames[:num_examples]]
            row["sequence_duration"] = (
                sequence.sequence_end_time - sequence.sequence_start_time
            )
        rows.append(row)

    rows = sorted(rows, key=lambda row: row["sequence_start_time"], reverse=True)
    return rows


def sequence_display_name(sequence_id: str) -> str:
    """
    Shorter and more helpful name for user interfaces.

    >>> sequence_display_name("20220406-SEQ-123")
    SEQ-123
    """
    if not sequence_id:
        return ""
    else:
        return sequence_id.split("-", 1)[-1]
