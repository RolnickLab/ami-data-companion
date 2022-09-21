import datetime
import time
import pathlib
import contextlib

import sqlalchemy as sa
from sqlalchemy import orm
from sqlalchemy_utils import aggregated, observes

from .. import utils
from ..utils import logger
from sqlalchemy import orm

__all__ = [
    "Base",
    "MonitoringSession",
    "Image",
    "DetectedObject",
]

# Only call this once & reuse it
Base = orm.declarative_base()

# Rename to CapturePeriod? shorter? less confusing with other types of Sessions. CaptureSession?
# Or SurveyEvent or Survey?
class MonitoringSession(Base):
    __tablename__ = "monitoring_sessions"

    id = sa.Column(sa.Integer, primary_key=True)
    day = sa.Column(sa.Date)
    base_directory = sa.Column(sa.String(255))
    start_time = sa.Column(sa.DateTime(timezone=True))
    end_time = sa.Column(sa.DateTime(timezone=True))
    notes = sa.Column(sa.JSON)

    @aggregated("images", sa.Column(sa.Integer))
    def num_images(self):
        return sa.func.count("1")

    @aggregated("detected_objects", sa.Column(sa.Integer))
    def num_detected_objects(self):
        return sa.func.count("1")

    @observes("images")
    def image_observer(self, images):
        timestamps = sorted([img.timestamp for img in images if img.timestamp])
        if timestamps:
            self.start_time = timestamps[0]
            self.end_time = timestamps[-1]

    images = orm.relationship(
        "Image",
        back_populates="monitoring_session",
        cascade="all, delete-orphan",
        order_by="Image.timestamp",
        # lazy="joined",
    )

    detected_objects = orm.relationship(
        "DetectedObject",
        back_populates="monitoring_session",
        cascade="all, delete-orphan",
        # lazy="joined",
    )

    def __repr__(self):
        return (
            f"MonitoringSession("
            f"start_time={self.start_time.strftime('%c') if self.start_time else None !r}, end_time={self.end_time.strftime('%c') if self.end_time else None!r}, "
            f"num_images={self.num_images!r}, num_detected_objects={self.num_detected_objects!r})"
        )

    def update_aggregates(self):
        # Requires and active session
        logger.info(f"Updating cached values for {self}")
        self.num_images = len(self.images)
        self.num_detected_objects = len(self.detected_objects)
        self.start_time = self.images[0].timestamp
        self.end_time = self.images[-1].timestamp

    def duration(self):
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        else:
            return None

    @property
    def duration_label(self):
        if self.duration():
            hours = int(round(self.duration().seconds / 60 / 60, 0))
            unit = "hour" if hours == 1 else "hours"
            duration = f"{hours} {unit}"
        else:
            duration = "Unknown duration"
        return duration


class Image(Base):
    __tablename__ = "images"

    id = sa.Column(sa.Integer, primary_key=True)
    monitoring_session_id = sa.Column(sa.ForeignKey("monitoring_sessions.id"))
    path = sa.Column(
        sa.String(255)
    )  # @TODO store these as relative paths to the directory
    timestamp = sa.Column(sa.DateTime(timezone=True))
    last_read = sa.Column(sa.DateTime)
    last_processed = sa.Column(sa.DateTime)
    notes = sa.Column(sa.JSON)

    def absolute_path(self, directory=None):
        if not directory:
            directory = self.monitoring_session.base_directory
        return pathlib.Path(directory) / self.path

    @aggregated("detected_objects", sa.Column(sa.Integer))
    def num_detected_objects(self):
        return sa.func.count("1")

    # @TODO let's keep the precious detected objects, even if the Monitoring Session or Image is deleted?
    detected_objects = orm.relationship(
        "DetectedObject",
        back_populates="image",
        cascade="all, delete-orphan",  # @TODO no! do not delete orphans? processing time is precious
        lazy="joined",
    )

    monitoring_session = orm.relationship(
        "MonitoringSession",
        back_populates="images",
        lazy="joined",
    )

    def __repr__(self):
        return f"Image(path={self.path!r}, timestamp={self.timestamp.strftime('%c') if self.timestamp else None !r}, num_detected_objects={self.num_detected_objects!r})"


class DetectedObject(Base):
    __tablename__ = "detections"

    id = sa.Column(sa.Integer, primary_key=True)
    image_id = sa.Column(sa.ForeignKey("images.id"))
    monitoring_session_id = sa.Column(sa.ForeignKey("monitoring_sessions.id"))
    bbox = sa.Column(sa.JSON)
    area_pixels = sa.Column(sa.Integer)
    specific_label = sa.Column(sa.String(255))
    specific_label_score = sa.Column(sa.Numeric(asdecimal=False))
    binary_label = sa.Column(sa.String(255))
    binary_label_score = sa.Column(sa.Numeric(asdecimal=False))
    last_detected = sa.Column(sa.DateTime)
    model_name = sa.Column(sa.String(255))
    notes = sa.Column(sa.JSON)

    image = orm.relationship(
        "Image",
        back_populates="detected_objects",
        lazy="joined",
    )

    monitoring_session = orm.relationship(
        "MonitoringSession",
        back_populates="detected_objects",
        lazy="joined",
    )

    def __repr__(self):
        image = self.image.path if self.image else None
        return f"DetectedObject(image={image!r}, specific_label={self.specific_label!r}, bbox={self.bbox!r})"
