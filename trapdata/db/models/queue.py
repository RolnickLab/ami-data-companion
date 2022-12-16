from typing import Sequence, Any

import sqlalchemy as sa

from trapdata.db import get_session
from trapdata import logger
from trapdata import constants
from trapdata.db.models.images import TrapImage
from trapdata.db.models.detections import (
    DetectedObject,
)


class QueueManager:
    name = "Unnamed Queue"

    def __init__(self, db_path):
        self.db_path = db_path

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

    def pull_n_from_queue(self, n: int) -> Sequence[Any]:
        return NotImplementedError

    def process_queue(self, model):
        logger.info(f"Processing {self.name} queue")
        model.run()
        logger.info(f"Done processing {self.name} queue")


class ImageQueue(QueueManager):
    name = "Images"
    description = "Raw images from camera needing object detection"

    def queue_count(self):
        with get_session(self.db_path) as sesh:
            count = sesh.scalar(
                sa.select(sa.func.count(TrapImage.id)).where(TrapImage.in_queue == True)
            )
            logger.debug(f"Images in queue: {count}")
            return count

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

    def pull_n_from_queue(self, n: int) -> Sequence[TrapImage]:
        logger.debug(f"Attempting to pull {n} images from queue")
        select_stmt = (
            sa.select(TrapImage.id)
            .where((TrapImage.in_queue == True))
            .limit(n)
            .with_for_update()
        )
        update_stmt = (
            sa.update(TrapImage)
            .where(TrapImage.id.in_(select_stmt.scalar_subquery()))
            # .where(TrapImage.status == Status.waiting)
            .values({"in_queue": False})
            .returning(TrapImage)
        )
        with get_session(self.db_path) as sesh:
            records = sesh.execute(update_stmt).unique().all()
            sesh.commit()
            images = [record[0] for record in records]
            logger.info(f"Pulled {len(images)} images from queue")
            return images


class DetectedObjectQueue(QueueManager):
    name = "Detected objects"
    description = "Objects that were detected in an image but have not been classified"

    def queue_count(self):
        with get_session(self.db_path) as sesh:
            return (
                sesh.query(DetectedObject)
                .filter(DetectedObject.bbox.is_not(None))
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

    def pull_n_from_queue(self, n: int) -> Sequence[DetectedObject]:
        logger.debug(f"Attempting to pull {n} detected objects from queue")
        select_stmt = (
            sa.select(DetectedObject.id)
            .where(
                (DetectedObject.in_queue == True)
                & (DetectedObject.binary_label == None)
                & (DetectedObject.bbox.is_not(None))
            )
            .limit(n)
            .with_for_update()
        )
        update_stmt = (
            sa.update(DetectedObject)
            .where(DetectedObject.id.in_(select_stmt.scalar_subquery()))
            .values({"in_queue": False})
            .returning(DetectedObject.id)
        )
        with get_session(self.db_path) as sesh:
            record_ids = sesh.execute(update_stmt).unique().scalars().all()
            sesh.commit()
            records = (
                sesh.execute(
                    sa.select(DetectedObject).where(DetectedObject.id.in_(record_ids))
                )
                .unique()
                .all()
            )
            objs = [record[0] for record in records]
            logger.info(f"Pulled {len(objs)} detected objects from queue")
            return objs


class UntrackedObjectsQueue(QueueManager):
    name = "Untracked detections"
    description = """
    Objects that have been identified as something of interest (e.g. a moth)
    but have not yet been "tracked" e.g. grouped into multiple frames of the same organism.
    """

    def queue_count(self):
        with get_session(self.db_path) as sesh:
            return (
                sesh.query(DetectedObject)
                .filter_by(
                    in_queue=True,
                    binary_label=constants.POSITIVE_BINARY_LABEL,
                    sequence_id=None,
                )
                .count()
            )

    def unprocessed_count(self):
        with get_session(self.db_path) as sesh:
            return (
                sesh.query(DetectedObject)
                .filter_by(
                    binary_label=constants.POSITIVE_BINARY_LABEL,
                    sequence_id=None,
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
                    binary_label=constants.POSITIVE_BINARY_LABEL,
                    sequence_id=None,
                )
                .all()
            )

        for obj in objects:
            obj.in_queue = True
            orm_objects.append(obj)

        with get_session(self.db_path) as sesh:
            logger.info(f"Adding {len(orm_objects)} untracked detections to queue")
            sesh.bulk_save_objects(orm_objects)
            sesh.commit()

    def clear_queue(self, *args):
        logger.info("Clearing untracked detections in queue")

        with get_session(self.db_path) as sesh:
            objects = []
            for obj in (
                sesh.query(DetectedObject)
                .filter_by(
                    in_queue=True,
                    binary_label=constants.POSITIVE_BINARY_LABEL,
                    sequence_id=None,
                )
                .all()
            ):
                obj.in_queue = False
                objects.append(obj)
            logger.info(f"Clearing {len(objects)} untracked detections in queue")
            sesh.bulk_save_objects(objects)
            sesh.commit()

    def pull_n_from_queue(
        self, n: int
    ) -> Sequence[tuple[DetectedObject, Sequence[DetectedObject]]]:
        """
        Fetch detected objects that need to be assigned to a sequence / track and
        all of the objects from the previous frame that will be compared.

        This will return more DetectedObjects than specified by `n` because
        it includes all of the related objects.
        """
        logger.debug(f"Attempting to pull {n} untracked detections from queue")
        select_stmt = (
            sa.select(DetectedObject.id)
            .where(
                (DetectedObject.in_queue == True)
                & (DetectedObject.binary_label == constants.POSITIVE_BINARY_LABEL)
                & (DetectedObject.sequence_id == None)
                & (DetectedObject.bbox.is_not(None))
            )
            .limit(n)
            .with_for_update()
        )
        update_stmt = (
            sa.update(DetectedObject)
            .where(DetectedObject.id.in_(select_stmt.scalar_subquery()))
            .values({"in_queue": False})
            .returning(DetectedObject.id)
        )
        with get_session(self.db_path) as sesh:
            record_ids = sesh.execute(update_stmt).scalars().all()
            sesh.commit()
            detections = (
                sesh.execute(
                    sa.select(DetectedObject).where(DetectedObject.id.in_(record_ids))
                )
                .unique()
                .scalars()
                .all()
            )
            logger.info(f"Pulled {len(detections)} untracked detections from queue")
            objs_with_comparisons = [
                (obj, obj.previous_frame_detections(sesh)) for obj in detections
            ]
            return objs_with_comparisons


class UnclassifiedObjectQueue(QueueManager):
    name = "Unclassified species"
    description = """
    Objects that have been identified as something of interest (e.g. a moth)
    but have not yet been classified to the species level.
    """

    def queue_count(self):
        with get_session(self.db_path) as sesh:
            return (
                sesh.query(DetectedObject)
                .filter_by(
                    in_queue=True,
                    specific_label=None,
                    binary_label=constants.POSITIVE_BINARY_LABEL,
                )
                .filter(DetectedObject.sequence_id.is_not(None))
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
                .filter(DetectedObject.sequence_id.is_not(None))
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
                .filter(DetectedObject.sequence_id.is_not(None))
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
                .filter(DetectedObject.sequence_id.is_not(None))
                .all()
            ):
                obj.in_queue = False
                objects.append(obj)
            logger.info(f"Clearing {len(objects)} objects in queue")
            sesh.bulk_save_objects(objects)
            sesh.commit()

    def pull_n_from_queue(self, n: int) -> Sequence[DetectedObject]:
        logger.debug(f"Attempting to pull {n} objects of interest from queue")
        select_stmt = (
            sa.select(DetectedObject.id)
            .where(
                (DetectedObject.in_queue == True)
                & (DetectedObject.binary_label == constants.POSITIVE_BINARY_LABEL)
                & (DetectedObject.specific_label == None)
                & (DetectedObject.bbox.is_not(None))
                & (DetectedObject.sequence_id.is_not(None))
            )
            .limit(n)
            .with_for_update()
        )
        update_stmt = (
            sa.update(DetectedObject)
            .where(DetectedObject.id.in_(select_stmt.scalar_subquery()))
            .values({"in_queue": False})
            .returning(DetectedObject.id)
        )
        with get_session(self.db_path) as sesh:
            record_ids = sesh.execute(update_stmt).scalars().all()
            sesh.commit()
            records = (
                sesh.execute(
                    sa.select(DetectedObject).where(DetectedObject.id.in_(record_ids))
                )
                .unique()
                .all()
            )
            objs = [record[0] for record in records]
            logger.info(f"Pulled {len(objs)} objects of interest from queue")
            return objs


def all_queues(db_path):
    return {
        q.name: q
        for q in [
            ImageQueue(db_path),
            DetectedObjectQueue(db_path),
            UntrackedObjectsQueue(db_path),
            UnclassifiedObjectQueue(db_path),
        ]
    }


def add_image_to_queue(db_path, image_id):

    with get_session(db_path) as sesh:
        logger.info(f"Adding image id {image_id} to queue")
        stmt = sa.update(TrapImage).filter_by(id=image_id).values({"in_queue": True})
        sesh.execute(stmt)
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

    return images


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


def clear_all_queues(db_path):

    logger.info("Clearing all queues")

    for name, queue in all_queues(db_path).items():
        logger.info(f"Clearing queue: {name}")
        queue.clear_queue()
