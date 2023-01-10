import datetime
import pathlib
from typing import Iterable, Union, Optional, Any

import sqlalchemy as sa
from sqlalchemy import orm
import PIL.Image

from trapdata import db
from trapdata import constants
from trapdata.db.models.images import TrapImage
from trapdata.common.logs import logger
from trapdata.common.utils import bbox_area, bbox_center, export_report
from trapdata.common.filemanagement import (
    save_image,
    absolute_path,
    construct_exif,
    EXIF_DATETIME_STR_FORMAT,
)


class DetectedObject(db.Base):
    __tablename__ = "detections"

    id = sa.Column(sa.Integer, primary_key=True)
    image_id = sa.Column(sa.ForeignKey("images.id"))
    monitoring_session_id = sa.Column(sa.ForeignKey("monitoring_sessions.id"))
    bbox = sa.Column(sa.JSON)
    area_pixels = sa.Column(sa.Integer)
    path = sa.Column(
        sa.String(255)
    )  # @TODO currently these are absolute paths to help the pytorch dataloader, but relative would be ideal
    specific_label = sa.Column(sa.String(255))
    specific_label_score = sa.Column(sa.Numeric(asdecimal=False))
    binary_label = sa.Column(sa.String(255))
    binary_label_score = sa.Column(sa.Numeric(asdecimal=False))
    last_detected = sa.Column(sa.DateTime)
    model_name = sa.Column(sa.String(255))
    in_queue = sa.Column(sa.Boolean, default=False)
    notes = sa.Column(sa.JSON)

    image = orm.relationship(
        "TrapImage",
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
            f"DetectedObject(\n"
            f"\timage={image!r}, \n"
            f"\tpath={self.path!r}, \n"
            f"\tspecific_label={self.specific_label!r}, \n"
            f"\tbbox={self.bbox!r})"
        )

    def cropped_image_data(
        self,
        source_image: Union[TrapImage, None] = None,
        base_path: Union[pathlib.Path, str, None] = None,
    ):
        """
        Return a PIL image of this detected object.
        """
        path = absolute_path(str(self.path), base_path)
        if path and path.exists():
            logger.debug(f"Using existing image crop: {path}")
            return PIL.Image.open(path)
        else:
            source_image = source_image or self.image
            if not source_image:
                raise Exception(f"Detected object id {self.id} has no source image")
            logger.debug(
                f"Extracting cropped image data from source image {source_image.path}"
            )
            image = PIL.Image.open(str(source_image.absolute_path))
            return image.crop(self.bbox)  # type:ignore

    def save_cropped_image_data(
        self,
        base_path: Union[pathlib.Path, str, None] = None,
        source_image: Union[TrapImage, None] = None,
    ):
        """
        @TODO need consistent way of discovering the user_data_path in the application settings
        and using that for the base_path.
        """
        source_image = source_image or self.image

        exif_data: PIL.Image.Exif = PIL.Image.open(source_image.absolute_path).getexif()
        exif_data = construct_exif(
            description=f"Source image: {source_image.path}",
            timestamp=source_image.timestamp,  # type:ignore
            existing_exif=exif_data,
        )

        fpath = save_image(
            image=self.cropped_image_data(
                base_path=base_path,
                source_image=source_image,
            ),
            base_path=base_path,
            subdir="crops",
            exif_data=exif_data,
        )
        self.path = str(fpath)
        return fpath

    def report_data(self) -> dict[str, Any]:
        if self.specific_label:
            label = self.specific_label
            score = self.specific_label_score
        else:
            label = self.binary_label
            score = self.binary_label_score

        return {
            "trap": pathlib.Path(self.monitoring_session.base_directory).name,
            "event": self.monitoring_session.day,
            "source_image": self.image.absolute_path,
            "cropped_image": self.path,
            "timestamp": self.image.timestamp,
            "bbox": self.bbox,
            "bbox_center": bbox_center(self.bbox) if self.bbox else None,
            "area_pixels": self.area_pixels,
            "model_name": self.model_name,
            "category_label": label,
            "category_score": score,
        }


def save_detected_objects(
    db_path, image_ids, detected_objects_data, user_data_path=None
):

    orm_objects = []
    with db.get_session(db_path) as sesh:
        images = sesh.query(TrapImage).filter(TrapImage.id.in_(image_ids)).all()

    timestamp = datetime.datetime.now()

    for image, detected_objects in zip(images, detected_objects_data):
        image.last_processed = timestamp
        # sesh.add(image)
        orm_objects.append(image)

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

            detection.monitoring_session_id = image.monitoring_session_id
            detection.image_id = image.id

            detection.save_cropped_image_data(
                source_image=image,
                base_path=user_data_path,
            )

            logger.debug(f"Creating detected object {detection} for image {image}")

            orm_objects.append(detection)

    with db.get_session(db_path) as sesh:
        # @TODO this could be faster! Especially for sqlite
        logger.info(f"Bulk saving {len(orm_objects)} objects")
        sesh.bulk_save_objects(orm_objects)
        sesh.commit()


def save_classified_objects(db_path, object_ids, classified_objects_data):
    # logger.debug(f"Callback was called! {object_ids}, {classified_objects_data}")

    orm_objects = []
    timestamp = datetime.datetime.now()
    with db.get_session(db_path) as sesh:
        objects = (
            sesh.query(DetectedObject).filter(DetectedObject.id.in_(object_ids)).all()
        )

    for obj, object_data in zip(objects, classified_objects_data):
        obj.last_processed = timestamp

        logger.debug(f"Updating classified object {obj}")
        for k, v in object_data.items():
            logger.debug(f"Adding {k}: {v} to detected object {obj.id}")
            setattr(obj, k, v)

        orm_objects.append(obj)

    with db.get_session(db_path) as sesh:
        logger.info(f"Bulk saving {len(orm_objects)} objects")
        sesh.bulk_save_objects(orm_objects)
        sesh.commit()


def get_detected_objects(db_path, monitoring_session=None):
    query_kwargs = {}

    if monitoring_session:
        query_kwargs["monitoring_session_id"] = monitoring_session.id

    with db.get_session(db_path) as sesh:
        return sesh.query(DetectedObject).filter_by(**query_kwargs)


def get_objects_for_image(db_path, image_id):
    with db.get_session(db_path) as sesh:
        return sesh.query(DetectedObject.binary_label).filter_by(image_id=image_id)


def delete_objects_for_image(db_path, image_id):
    with db.get_session(db_path) as sesh:
        sesh.query(DetectedObject).filter_by(image_id=image_id).delete()
        sesh.commit()


def get_detections_for_image(db_path, image_id):
    with db.get_session(db_path) as sesh:
        return sesh.query(DetectedObject.binary_label).filter_by(
            image_id=image_id, binary_label=constants.POSITIVE_BINARY_LABEL
        )


def get_classifications_for_image(db_path, image_id):
    with db.get_session(db_path) as sesh:
        return (
            sesh.query(DetectedObject.specific_label)
            .filter_by(
                image_id=image_id,
                binary_label=constants.POSITIVE_BINARY_LABEL,
            )
            .filter(
                DetectedObject.specific_label.is_not(None),
            )
        )


def get_species_for_image(db_path, image_id):
    with db.get_session(db_path) as sesh:
        return (
            sesh.query(DetectedObject.specific_label)
            .filter_by(image_id=image_id)
            .filter(DetectedObject.specific_label.is_not(None))
            .distinct()
        )


def get_unique_species(db_path, monitoring_session=None):
    query = (
        sa.select(
            sa.func.coalesce(
                DetectedObject.specific_label,
                DetectedObject.binary_label,
                DetectedObject.path,
            ).label("label"),
            sa.func.count().label("count"),
        )
        .group_by("label")
        .order_by(sa.desc("count"))
    )
    if monitoring_session:
        query = query.filter_by(monitoring_session=monitoring_session)

    with db.get_session(db_path) as sesh:
        return sesh.execute(query).all()


def get_objects_for_species(db_path, species_label, monitoring_session=None):
    query = sa.select(DetectedObject).filter_by(specific_label=species_label)
    if monitoring_session:
        query = query.filter_by(monitoring_session=monitoring_session)

    with db.get_session(db_path) as sesh:
        return sesh.execute(query).unique().all()


def get_object_counts_for_image(db_path, image_id):
    # Every object detected
    num_objects = get_objects_for_image(db_path, image_id).count()

    # Every object that is a moth
    num_detections = get_detections_for_image(db_path, image_id).count()

    # Every object that has been classified to taxa level
    num_classifications = get_classifications_for_image(db_path, image_id).count()

    # Unique taxa names
    num_species = get_species_for_image(db_path, image_id).count()

    # Has every object detected in this image been fully processed?
    completely_classified = True if num_classifications == num_detections else False

    return {
        "num_objects": num_objects,
        "num_detections": num_detections,
        "num_species": num_species,
        "num_classifications": num_classifications,
        "completely_classified": completely_classified,
    }


def export_detected_objects(
    items: Iterable[DetectedObject],
    directory: Union[pathlib.Path, str],
    report_name: str = "detections",
):
    records = [item.report_data() for item in items]
    return export_report(records, report_name, directory)
