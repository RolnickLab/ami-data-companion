import datetime

import sqlalchemy as sa
from sqlalchemy import orm

from trapdata import db
from trapdata import constants
from trapdata.models.images import Image
from trapdata.common.logs import logger
from trapdata.common.utils import bbox_area


class DetectedObject(db.Base):
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
    in_queue = sa.Column(sa.Boolean, default=False)
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
        return (
            f"DetectedObject("
            f"\timage={image!r}, \n"
            f"\tspecific_label={self.specific_label!r}, \n"
            f"\tbbox={self.bbox!r})"
        )


def save_detected_objects(db_path, image_paths, detected_objects_data):
    # logger.debug(f"Callback was called! {image_paths}, {detected_objects_data}")

    with db.get_session(db_path) as sess:
        timestamp = datetime.datetime.now()
        for image_id, detected_objects in zip(image_paths, detected_objects_data):
            image = sess.query(Image).get(image_id)
            image.last_processed = timestamp
            sess.add(image)
            for object_data in detected_objects:
                detection = DetectedObject(
                    last_detected=timestamp,
                    in_queue=True,
                )

                if "bbox" in object_data:
                    area_pixels = bbox_area(object_data["bbox"])
                    object_data["area_pixels"] = area_pixels

                for k, v in object_data.items():
                    logger.debug(f"Adding {k}: {v} to detected object {detection.id}")
                    setattr(detection, k, v)

                logger.debug(f"Saving detected object {detection} for image {image}")
                sess.add(detection)
                detection.monitoring_session_id = image.monitoring_session_id
                detection.image_id = image.id
        sess.commit()


def save_classified_objects(db_path, object_ids, classified_objects_data):
    # logger.debug(f"Callback was called! {object_ids}, {classified_objects_data}")

    with db.get_session(db_path) as sess:
        timestamp = datetime.datetime.now()
        for object_id, object_data in zip(object_ids, classified_objects_data):
            obj = sess.get(DetectedObject, object_id)
            obj.last_processed = timestamp
            sess.add(obj)

            for k, v in object_data.items():
                logger.debug(f"Adding {k}: {v} to detected object {obj.id}")
                setattr(obj, k, v)

            logger.debug(f"Saving classified object {obj}")

        sess.commit()


def get_detected_objects(monitoring_session):
    base_directory = monitoring_session.base_directory
    query_kwargs = {
        "monitoring_session_id": monitoring_session.id,
    }
    with db.get_session(base_directory) as sess:
        for obj in sess.query(DetectedObject).filter_by(**query_kwargs).all():
            yield obj


def get_objects_for_image(db_path, image_id):
    with db.get_session(db_path) as sess:
        return (
            sess.query(DetectedObject.binary_label)
            .filter_by(image_id=image_id)
            .filter(DetectedObject.binary_label.is_not(None))
        )


def get_detections_for_image(db_path, image_id):
    with db.get_session(db_path) as sess:
        return sess.query(DetectedObject.binary_label).filter_by(
            image_id=image_id, binary_label=constants.POSITIVE_BINARY_LABEL
        )


def get_species_for_image(db_path, image_id):
    with db.get_session(db_path) as sess:
        return (
            sess.query(DetectedObject.specific_label)
            .filter_by(image_id=image_id)
            .filter(DetectedObject.specific_label.is_not(None))
            .distinct()
        )
