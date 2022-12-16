import pathlib
from typing import Optional

import sqlalchemy as sa
from sqlalchemy import orm
from sqlalchemy_utils import aggregated

from trapdata.db import Base, get_session
from trapdata import constants


class TrapImage(Base):
    __tablename__ = "images"

    id = sa.Column(sa.Integer, primary_key=True)
    monitoring_session_id = sa.Column(sa.ForeignKey("monitoring_sessions.id"))
    base_path = sa.Column(sa.String(255))
    path = sa.Column(sa.String(255))
    timestamp = sa.Column(sa.DateTime(timezone=True))
    filesize = sa.Column(sa.Integer)
    last_read = sa.Column(sa.DateTime)
    last_processed = sa.Column(sa.DateTime)
    in_queue = sa.Column(sa.Boolean, default=False)
    notes = sa.Column(sa.JSON)
    width = sa.Column(sa.Integer)
    height = sa.Column(sa.Integer)
    # diag
    # centroid
    # cnn features

    @property
    def absolute_path(self, directory: Optional[str] = None) -> pathlib.Path:
        # @TODO this directory argument can be removed once the image has the base
        # path stored in itself
        if not directory:
            directory = str(self.base_path)
        return pathlib.Path(directory) / str(self.path)

    @aggregated("detected_objects", sa.Column(sa.Integer))
    def num_detected_objects(self):
        return sa.func.count("1")

    def previous_image(self, session: orm.Session) -> Optional["TrapImage"]:
        img = session.execute(
            sa.select(TrapImage)
            .filter(TrapImage.timestamp < self.timestamp)
            .order_by(TrapImage.timestamp.desc())
            .limit(1)
        ).scalar()
        return img

    def next_image(self, session: orm.Session) -> Optional["TrapImage"]:
        img = session.execute(
            sa.select(TrapImage)
            .filter(TrapImage.timestamp > self.timestamp)
            .order_by(TrapImage.timestamp.asc())
            .limit(1)
        ).scalar()
        return img

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

    @property
    def classified(self):
        """Have all detected objects been classified"""
        pass

    def __repr__(self):
        return (
            f"Image(path={self.path!r}, \n"
            f"\ttimestamp={self.timestamp.strftime('%c') if self.timestamp else None !r}, \n"
            f"\tnum_detected_objects={self.num_detected_objects!r})"
        )


def get_image_with_objects(db_path, image_id):
    with get_session(db_path) as sesh:
        image_kwargs = {
            "id": image_id,
            # "path": str(image_path),
            # "monitoring_session_id": monitoring_session.id,
        }
        image = (
            sesh.query(TrapImage)
            .filter_by(**image_kwargs)
            .options(orm.joinedload(TrapImage.detected_objects))
            .one_or_none()
        )
        # logger.debug(
        #     f"Found image {image} with {len(image.detected_objects)} detected objects"
        # )
        return image


def completely_classified(db_path, image_id):
    from trapdata.db.models.detections import DetectedObject

    with get_session(db_path) as sesh:
        img = sesh.query(TrapImage).get(image_id)
        if img.in_queue or not img.last_processed:
            return False

        else:
            classified_objs = (
                sesh.query(DetectedObject)
                .filter_by(
                    image_id=image_id, binary_label=constants.POSITIVE_BINARY_LABEL
                )
                .filter(
                    DetectedObject.specific_label.is_not(None),
                )
                .count()
            )
            detections = (
                sesh.query(DetectedObject)
                .filter_by(
                    image_id=image_id, binary_label=constants.POSITIVE_BINARY_LABEL
                )
                .count()
            )
            if int(classified_objs) == int(detections):
                return True
            else:
                return False
