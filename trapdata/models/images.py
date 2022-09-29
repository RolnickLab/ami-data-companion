import pathlib

import sqlalchemy as sa
from sqlalchemy import orm
from sqlalchemy_utils import aggregated

from trapdata.db import Base, get_session
from trapdata import logger


class Image(Base):
    __tablename__ = "images"

    id = sa.Column(sa.Integer, primary_key=True)
    monitoring_session_id = sa.Column(sa.ForeignKey("monitoring_sessions.id"))
    path = sa.Column(
        sa.String(255)
    )  # @TODO store these as relative paths to the directory
    timestamp = sa.Column(sa.DateTime(timezone=True))
    # filesize = sa.Column(sa.Integer)
    last_read = sa.Column(sa.DateTime)
    last_processed = sa.Column(sa.DateTime)
    in_queue = sa.Column(sa.Boolean, default=False)
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
        return (
            f"Image(path={self.path!r}, \n"
            f"\ttimestamp={self.timestamp.strftime('%c') if self.timestamp else None !r}, \n"
            f"\tnum_detected_objects={self.num_detected_objects!r})"
        )


def get_image_with_objects(monitoring_session, image_id):
    base_directory = monitoring_session.base_directory
    with get_session(base_directory) as sess:
        image_kwargs = {
            "id": image_id,
            # "path": str(image_path),
            # "monitoring_session_id": monitoring_session.id,
        }
        image = (
            sess.query(Image)
            .filter_by(**image_kwargs)
            .options(orm.joinedload(Image.detected_objects))
            .one_or_none()
        )
        logger.debug(
            f"Found image {image} with {len(image.detected_objects)} detected objects"
        )
        return image
