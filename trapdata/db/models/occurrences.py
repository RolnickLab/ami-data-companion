"""
Occurrence of an Individual Organism

There is currently no database model representing an occurrence. 
And occurrence is a sequence of detections that are determined to be
the same individual, tracked over multiple frames in the original images
from a monitoring session.
"""
import datetime
import pathlib

import sqlalchemy as sa
from sqlalchemy import orm
from trapdata.db import models
from trapdata import db

from pydantic import (
    BaseModel,
)


class Occurrence(BaseModel):
    label: str
    score: float
    sequence_id: int
    start_time: datetime.datetime
    end_time: datetime.datetime
    duration: datetime.timedelta
    cropped_image_path: pathlib.Path
    # detections: list[object]
    # deployment: object
    # captures: list[object]


def get_unique_species_by_track(
    db_path: str,
    monitoring_session=None,
    classification_threshold: float = -1,
    num_examples: int = 3,
) -> list[dict]:
    Session = db.get_session_class(db_path)
    session = Session()

    # Select all sequences where at least one example is above the score threshold
    sequences = session.execute(
        sa.select(
            models.DetectedObject.sequence_id,
            sa.func.count(models.DetectedObject.id).label(
                "sequence_frame_count"
            ),  # frames in track
            sa.func.max(models.DetectedObject.specific_label_score).label(
                "sequence_best_score"
            ),
            sa.func.min(models.DetectedObject.timestamp).label("sequence_start_time"),
            sa.func.max(models.DetectedObject.timestamp).label("sequence_end_time"),
        )
        .group_by("sequence_id")
        .where((models.DetectedObject.monitoring_session_id == monitoring_session.id))
        .having(
            sa.func.max(models.DetectedObject.specific_label_score)
            >= classification_threshold,
        )
        .order_by(models.DetectedObject.specific_label)
    ).all()

    rows = []
    for sequence in sequences:
        frames = session.execute(
            sa.select(
                models.DetectedObject.image_id.label("source_image_id"),
                models.DetectedObject.specific_label.label("label"),
                models.DetectedObject.specific_label_score.label("score"),
                models.DetectedObject.path.label("cropped_image_path"),
                models.DetectedObject.sequence_id,
                models.DetectedObject.timestamp,
            )
            .where(
                (models.DetectedObject.monitoring_session_id == monitoring_session.id)
                & (models.DetectedObject.sequence_id == sequence.sequence_id)
            )
            # .order_by(sa.func.random())
            .order_by(sa.desc("score"))
            .limit(num_examples)
        ).all()
        row = dict(sequence._mapping)
        if frames:
            best_example = frames[0]
            row["label"] = best_example.label
            row["examples"] = [example._mapping for example in frames[:num_examples]]
            row["sequence_duration"] = (
                sequence.sequence_end_time - sequence.sequence_start_time
            )
        rows.append(row)

    rows = reversed(sorted(rows, key=lambda row: row["sequence_start_time"]))
    return rows
