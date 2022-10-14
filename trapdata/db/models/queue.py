import pathlib

import sqlalchemy as sa

from trapdata.db import get_session
from trapdata import logger
from trapdata import constants
from trapdata import ml
from trapdata.db.models.images import TrapImage
from trapdata.db.models.detections import (
    DetectedObject,
)


class QueueManager:
    name = "Unnamed Queue"

    def __init__(self, db_path, model):
        self.db_path = db_path
        self.model = model

    def queue_count(self):
        raise NotImplementedError

    def unprocessed_count(self):
        raise NotImplementedError

    def done_count(self):
        raise NotImplementedError

    def add_unprocessed(self):
        raise NotImplementedError

    def clear_queue(self):
        raise NotImplementedError

    def status(self):
        return NotImplementedError

    def process_queue(self):
        logger.info(f"Processing {self.name} queue")
        self.model.run()
        logger.info(f"Done processing {self.name} queue")


class ImageQueue(QueueManager):
    name = "Images"

    def queue_count(self):
        with get_session(self.db_path) as sesh:
            return sesh.query(TrapImage).filter_by(in_queue=True).count()

    def unprocessed_count(self):
        with get_session(self.db_path) as sesh:
            return sesh.query(TrapImage).filter_by(last_processed=None).count()

    def done_count(self):
        with get_session(self.db_path) as sesh:
            return (
                sesh.query(TrapImage)
                .filter(TrapImage.last_processed.is_not(None))
                .count()
            )

    def add_unprocessed(self, *args):
        orm_objects = []
        with get_session(self.db_path) as sesh:
            images = (
                sesh.query(TrapImage)
                .filter_by(
                    in_queue=False,
                    last_processed=None,
                )
                .all()
            )
        for image in images:
            image.in_queue = True
            orm_objects.append(image)

        with get_session(self.db_path) as sesh:
            logger.info(f"Bulk saving {len(images)} images to queue")
            sesh.bulk_save_objects(images)
            sesh.commit()

        with get_session(self.db_path) as sesh:
            count = sesh.query(TrapImage).filter_by(last_processed=None).count()

        return count

    def clear_queue(self, *args):
        logger.info("Clearing images in queue")

        orm_objects = []
        with get_session(self.db_path) as sesh:
            images = sesh.query(TrapImage).filter_by(in_queue=True).all()

        for image in images:
            image.in_queue = False
            orm_objects.append(image)

        with get_session(self.db_path) as sesh:
            logger.info(f"Removing {len(orm_objects)} images from queue")
            sesh.bulk_save_objects(orm_objects)
            sesh.commit()


class DetectedObjectQueue(QueueManager):
    name = "Detected objects"

    def queue_count(self):
        with get_session(self.db_path) as sesh:
            return (
                sesh.query(DetectedObject)
                .filter_by(
                    in_queue=True,
                    binary_label=None,
                )
                .count()
            )

    def unprocessed_count(self):
        with get_session(self.db_path) as sesh:
            return sesh.query(DetectedObject).filter_by(binary_label=None).count()

    def done_count(self):
        with get_session(self.db_path) as sesh:
            return (
                sesh.query(DetectedObject)
                .filter(DetectedObject.binary_label.is_not(None))
                .count()
            )

    def add_unprocessed(self, *args):
        with get_session(self.db_path) as sesh:
            objects = []
            for obj in (
                sesh.query(DetectedObject)
                .filter_by(in_queue=False, binary_label=None)
                .all()
            ):
                obj.in_queue = True
                objects.append(obj)
            logger.info(f"Adding {len(objects)} objects to queue")
            sesh.bulk_save_objects(objects)
            sesh.commit()

    def clear_queue(self, *args):
        logger.info("Clearing detected objects in queue")

        orm_objects = []
        with get_session(self.db_path) as sesh:
            objects = (
                sesh.query(DetectedObject)
                .filter_by(in_queue=True, binary_label=None)
                .all()
            )

        for obj in objects:
            obj.in_queue = False
            orm_objects.append(obj)

        with get_session(self.db_path) as sesh:
            logger.info(f"Removing {len(orm_objects)} objects from queue")
            sesh.bulk_save_objects(orm_objects)
            sesh.commit()


class UnclassifiedObjectQueue(QueueManager):
    """
    Objects that have been identified as something of interest (e.g. a moth)
    but have not yet been classified to the species level.
    """

    name = "Unclassified species"

    def queue_count(self):
        with get_session(self.db_path) as sesh:
            return (
                sesh.query(DetectedObject)
                .filter_by(
                    in_queue=True,
                    specific_label=None,
                    binary_label=constants.POSITIVE_BINARY_LABEL,
                )
                .count()
            )

    def unprocessed_count(self):
        with get_session(self.db_path) as sesh:
            return (
                sesh.query(DetectedObject)
                .filter_by(
                    specific_label=None,
                    binary_label=constants.POSITIVE_BINARY_LABEL,
                )
                .count()
            )

    def done_count(self):
        with get_session(self.db_path) as sesh:
            return (
                sesh.query(DetectedObject)
                .filter(DetectedObject.specific_label.is_not(None))
                .count()
            )

    def add_unprocessed(self, *args):
        orm_objects = []
        with get_session(self.db_path) as sesh:
            objects = (
                sesh.query(DetectedObject)
                .filter_by(
                    in_queue=False,
                    specific_label=None,
                    binary_label=constants.POSITIVE_BINARY_LABEL,
                )
                .all()
            )

        for obj in objects:
            obj.in_queue = True
            orm_objects.append(obj)

        with get_session(self.db_path) as sesh:
            logger.info(f"Adding {len(orm_objects)} objects to queue")
            sesh.bulk_save_objects(orm_objects)
            sesh.commit()

    def clear_queue(self, *args):
        logger.info("Clearing unclassified objects in queue")

        with get_session(self.db_path) as sesh:
            objects = []
            for obj in (
                sesh.query(DetectedObject)
                .filter_by(
                    in_queue=True,
                    specific_label=None,
                    binary_label=constants.POSITIVE_BINARY_LABEL,
                )
                .all()
            ):
                obj.in_queue = False
                objects.append(obj)
            logger.info(f"Clearing {len(objects)} objects in queue")
            sesh.bulk_save_objects(objects)
            sesh.commit()


def all_queues(db_path):
    return {
        q.name: q
        for q in [
            ImageQueue(db_path, model=None),
            DetectedObjectQueue(db_path, model=None),
            UnclassifiedObjectQueue(db_path, model=None),
        ]
    }


def add_image_to_queue(db_path, image_id):

    with get_session(db_path) as sesh:
        image = sesh.query(TrapImage).get(image_id)
        logger.info(f"Adding image to queue: {image}")
        if not image.in_queue:
            image.in_queue = True
            sesh.add(image)
            sesh.commit()


def add_sample_to_queue(db_path, sample_size=10):

    with get_session(db_path) as sesh:
        num_in_queue = sesh.query(TrapImage).filter_by(in_queue=True).count()
        if num_in_queue < sample_size:
            images = []
            for image in (
                sesh.query(TrapImage)
                .filter_by(
                    in_queue=False,
                )
                .order_by(sa.func.random())
                .limit(sample_size - num_in_queue)
                .all()
            ):
                image.in_queue = True
                images.append(image)
            logger.info(f"Adding {len(images)} images to queue")
            sesh.bulk_save_objects(images)
            sesh.commit()


def add_monitoring_session_to_queue(db_path, monitoring_session, limit=None):
    """
    Add images captured during a give Monitoring Session to the
    processing queue. If a limit is specified, only add that many
    additional images to the queue. Will not add duplicates to the queue.
    """
    ms = monitoring_session

    # @TODO This may be a faster way to add all images to the queue
    # image_ids = get_monitoring_session_image_ids(ms)
    # logger.info(f"Adding {len(image_ids)} images into queue")

    # with get_session(db_path) as sesh:
    #     sesh.execute(
    #         sa.update(Image)
    #         .where(Image.monitoring_session_id == ms.id)
    #         .values(in_queue=True)
    #     )

    logger.info(f"Adding all images for Monitoring Session {ms.id} to queue")
    with get_session(db_path) as sesh:
        images = []
        for image in (
            sesh.query(TrapImage)
            .filter_by(
                in_queue=False,
                last_processed=None,
                monitoring_session_id=ms.id,
            )
            .order_by(TrapImage.timestamp)
            .limit(limit)
            .all()
        ):
            image.in_queue = True
            images.append(image)
        logger.info(f"Adding {len(images)} images to queue")
        sesh.bulk_save_objects(images)
        sesh.commit()


def images_in_queue(db_path):

    with get_session(db_path) as sesh:
        return sesh.query(TrapImage).filter_by(in_queue=True).count()


def queue_counts(db_path):

    counts = {}
    with get_session(db_path) as sesh:
        counts["images"] = sesh.query(TrapImage).filter_by(in_queue=True).count()
        counts["unclassified_objects"] = (
            sesh.query(DetectedObject)
            .filter_by(in_queue=True, binary_label=None)
            .count()
        )
        counts["unclassified_species"] = (
            sesh.query(DetectedObject)
            .filter_by(
                in_queue=True,
                specific_label=None,
            )
            .filter(
                DetectedObject.binary_label.is_not(None),
            )
            .count()
        )
    return counts


def unprocessed_counts(db_path):

    counts = {}
    with get_session(db_path) as sesh:
        counts["images"] = sesh.query(TrapImage).filter_by(last_processed=None).count()
        counts["unclassified_objects"] = (
            sesh.query(DetectedObject).filter_by(binary_label=None).count()
        )
        counts["unclassified_moths"] = (
            sesh.query(DetectedObject)
            .filter_by(
                specific_label=None,
            )
            .filter(
                DetectedObject.binary_label.is_not(None),
            )
            .count()
        )
    return counts


def clear_queue(db_path):

    logger.info("Clearing images in queue")

    items = []
    with get_session(db_path) as sesh:
        for image in sesh.query(TrapImage).filter_by(in_queue=True).all():
            image.in_queue = False
            items.append(image)
        for obj in sesh.query(DetectedObject).filter_by(in_queue=True).all():
            obj.in_queue = False
            items.append(obj)

    with get_session(db_path) as sesh:
        sesh.bulk_save_objects(items)
        sesh.commit()


def start_pipeline(db_path, config):

    user_data_path = pathlib.Path(config.get("paths", "user_data_path"))
    logger.info(f"Local user data path: {user_data_path}")
    num_workers = int(config.get("performance", "num_workers"))

    model_1_name = config.get("models", "localization_model")
    Model_1 = ml.models.object_detectors[model_1_name]
    model_1 = Model_1(
        db_path=db_path,
        user_data_path=user_data_path,
        batch_size=int(config.get("performance", "localization_batch_size")),
        num_workers=num_workers,
    )
    model_1.run()
    logger.info("Localization complete")

    model_2_name = config.get("models", "binary_classification_model")
    Model_2 = ml.models.binary_classifiers[model_2_name]
    model_2 = Model_2(
        db_path=db_path,
        user_data_path=user_data_path,
        batch_size=int(config.get("performance", "classification_batch_size")),
        num_workers=num_workers,
    )
    model_2.run()
    logger.info("Binary classification complete")

    model_3_name = config.get("models", "taxon_classification_model")
    Model_3 = ml.models.species_classifiers[model_3_name]
    model_3 = Model_3(
        db_path=db_path,
        user_data_path=user_data_path,
        batch_size=int(config.get("performance", "classification_batch_size")),
        num_workers=num_workers,
    )
    model_3.run()
    logger.info("Species classification complete")
