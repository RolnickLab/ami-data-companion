from typing import Sequence, Union, Optional

import sqlalchemy as sa

from trapdata.db import get_session
from trapdata import logger
from trapdata import constants
from trapdata.common.types import FilePath
from trapdata.db.models.images import TrapImage
from trapdata.db.models.detections import DetectedObject
from trapdata.db.models.events import MonitoringSession


class QueueManager:
    name = "Unnamed Queue"
    base_directory: FilePath

    def __init__(self, db_path: str, base_directory: FilePath):
        self.db_path = db_path
        self.base_directory = base_directory

    def ids(self) -> sa.ScalarSelect:
        """
        Return subquery of all IDs managed by the scope of this queue.
        """
        return sa.select().scalar_subquery()

    def get_model(self):
        """
        The primary SQLAlchemy model for common queries.

        If this is a custom queue, you must override the default methods
        """
        raise NotImplementedError

    def queue_count(self):
        raise NotImplementedError

    def unprocessed_count(self):
        raise NotImplementedError

    def done_count(self):
        raise NotImplementedError

    def add_unprocessed(self, *_):
        raise NotImplementedError

    def clear_queue(self, *_):
        raise NotImplementedError

    def status(self):
        return NotImplementedError

    def pull_n_from_queue(self, n: int):
        return NotImplementedError

    def process_queue(self, model):
        logger.info(f"Processing {self.name} queue")
        model.run()
        logger.info(f"Done processing {self.name} queue")


class ImageQueue(QueueManager):
    name = "Images"
    description = "Raw images from camera needing object detection"

    def ids(self) -> sa.ScalarSelect:
        """
        Return subquery of all IDs managed by the scope of this queue.
        """
        return (
            sa.select(TrapImage.id)
            .where(MonitoringSession.base_directory == str(self.base_directory))
            .join(
                MonitoringSession,
                TrapImage.monitoring_session_id == MonitoringSession.id,
            )
            .scalar_subquery()
        )

    def queue_count(self) -> Union[int, None]:
        with get_session(self.db_path) as sesh:
            stmt = sa.select(sa.func.count(TrapImage.id)).where(
                (TrapImage.id.in_(self.ids()) & (TrapImage.in_queue.is_(True)))
            )
            count = sesh.execute(stmt).scalar()
            return count

    def unprocessed_count(self) -> Union[int, None]:
        with get_session(self.db_path) as sesh:
            stmt = sa.select(sa.func.count(TrapImage.id)).where(
                (TrapImage.id.in_(self.ids()) & (TrapImage.last_processed.is_(None)))
            )
            count = sesh.execute(stmt).scalar()
            return count

    def done_count(self) -> Union[int, None]:
        with get_session(self.db_path) as sesh:
            stmt = sa.select(sa.func.count(TrapImage.id)).where(
                (TrapImage.id.in_(self.ids()) & (TrapImage.last_processed.is_not(None)))
            )
            count = sesh.execute(stmt).scalar()
            return count

    def add_unprocessed(self, *_) -> None:
        logger.info("Adding all unprocessed deployment images to queue")
        with get_session(self.db_path) as sesh:
            stmt = (
                sa.update(TrapImage)
                .where(
                    TrapImage.id.in_(self.ids()) & TrapImage.last_processed.is_(None)
                )
                .values({"in_queue": True})
            )
            sesh.execute(stmt)
            sesh.commit()

    def clear_queue(self, *_) -> None:
        logger.info("Clearing all deployment images in queue")
        with get_session(self.db_path) as sesh:
            stmt = (
                sa.update(TrapImage)
                .where(TrapImage.id.in_(self.ids()) & TrapImage.in_queue.is_(True))
                .values({"in_queue": False})
            )
            sesh.execute(stmt)
            sesh.commit()

    def pull_n_from_queue(self, n: int) -> Sequence[TrapImage]:
        logger.debug(f"Attempting to pull {n} images from queue")
        select_stmt = (
            sa.select(TrapImage.id)
            .where(TrapImage.id.in_(self.ids()) & (TrapImage.in_queue.is_(True)))
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
            images = sesh.execute(update_stmt).unique().scalars().all()
            sesh.commit()
            logger.info(f"Pulled {len(images)} images from queue")
            return images


class DetectedObjectQueue(QueueManager):
    name = "Detected objects"
    description = "Objects that were detected in an image but have not been classified"

    def ids(self) -> sa.ScalarSelect:
        """
        Return subquery of all IDs managed by the scope of this queue.
        """
        return (
            sa.select(DetectedObject.id)
            .where(
                (MonitoringSession.base_directory == str(self.base_directory))
                & DetectedObject.bbox.is_not(None)
            )
            .join(
                MonitoringSession,
                DetectedObject.monitoring_session_id == MonitoringSession.id,
            )
            .scalar_subquery()
        )

    def queue_count(self):
        with get_session(self.db_path) as sesh:
            stmt = sa.select(sa.func.count(DetectedObject.id)).where(
                (
                    (DetectedObject.id.in_(self.ids()))
                    & DetectedObject.in_queue.is_(True)
                    & (DetectedObject.binary_label.is_(None))
                )
            )
            count = sesh.execute(stmt).scalar()
            return count

    def unprocessed_count(self):
        with get_session(self.db_path) as sesh:
            stmt = sa.select(sa.func.count(DetectedObject.id)).where(
                (
                    (DetectedObject.id.in_(self.ids()))
                    & (DetectedObject.binary_label.is_(None))
                )
            )
            count = sesh.execute(stmt).scalar()
            return count

    def done_count(self):
        with get_session(self.db_path) as sesh:
            stmt = sa.select(sa.func.count(DetectedObject.id)).where(
                (
                    (DetectedObject.id.in_(self.ids()))
                    & (DetectedObject.binary_label.is_not(None))
                )
            )
            count = sesh.execute(stmt).scalar()
            return count

    def add_unprocessed(self, *_) -> None:
        logger.info(f"Adding detected objects from deployment to queue")
        with get_session(self.db_path) as sesh:
            stmt = (
                sa.update(DetectedObject)
                .where(
                    (DetectedObject.id.in_(self.ids()))
                    & (DetectedObject.in_queue.is_(False))
                    & (DetectedObject.binary_label.is_(None))
                )
                .values({"in_queue": True})
            )
            sesh.execute(stmt)
            sesh.commit()

    def clear_queue(self, *_) -> None:
        logger.info("Removing detected objects from deployment from queue")
        with get_session(self.db_path) as sesh:
            stmt = (
                sa.update(DetectedObject)
                .where(
                    (DetectedObject.id.in_(self.ids()))
                    & (DetectedObject.in_queue.is_(True))
                    & (DetectedObject.binary_label.is_(None))
                )
                .values({"in_queue": False})
            )
            sesh.execute(stmt)
            sesh.commit()

    def pull_n_from_queue(self, n: int) -> Sequence[DetectedObject]:
        logger.debug(f"Attempting to pull {n} detected objects from queue")
        select_stmt = (
            sa.select(DetectedObject.id)
            .where(
                (DetectedObject.id.in_(self.ids()))
                & (DetectedObject.in_queue.is_(True))
                & (DetectedObject.binary_label.is_(None))
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
            objs = (
                sesh.execute(
                    sa.select(DetectedObject).where(DetectedObject.id.in_(record_ids))
                )
                .unique()
                .scalars()
                .all()
            )
            logger.info(f"Pulled {len(objs)} detected objects from queue")
            return objs


class UnclassifiedObjectQueue(QueueManager):
    name = "Unclassified objects"
    description = """
    Objects that have been identified as something of interest (e.g. a moth)
    but have not yet been classified to the species level.
    """

    def ids(self) -> sa.ScalarSelect:
        """
        Return subquery of all IDs managed by the scope of this queue.
        """
        return (
            sa.select(DetectedObject.id)
            .where(
                (MonitoringSession.base_directory == str(self.base_directory))
                & (DetectedObject.binary_label == constants.POSITIVE_BINARY_LABEL)
            )
            .join(
                MonitoringSession,
                DetectedObject.monitoring_session_id == MonitoringSession.id,
            )
            .scalar_subquery()
        )

    def queue_count(self) -> Union[int, None]:
        with get_session(self.db_path) as sesh:
            stmt = sa.select(sa.func.count(DetectedObject.id)).where(
                (
                    (DetectedObject.id.in_(self.ids()))
                    & DetectedObject.in_queue.is_(True)
                    & DetectedObject.specific_label.is_(None)
                )
            )
            count = sesh.execute(stmt).scalar()
            return count

    def unprocessed_count(self) -> Union[int, None]:
        with get_session(self.db_path) as sesh:
            stmt = sa.select(sa.func.count(DetectedObject.id)).where(
                (
                    (DetectedObject.id.in_(self.ids()))
                    & DetectedObject.specific_label.is_(None)
                )
            )
            count = sesh.execute(stmt).scalar()
            return count

    def done_count(self) -> Union[int, None]:
        with get_session(self.db_path) as sesh:
            stmt = sa.select(sa.func.count(DetectedObject.id)).where(
                (
                    (DetectedObject.id.in_(self.ids()))
                    & DetectedObject.specific_label.is_not(None)
                )
            )
            count = sesh.execute(stmt).scalar()
            return count

    def add_unprocessed(self, *_) -> None:
        logger.info("Adding unclassified objects from deployment to queue")
        with get_session(self.db_path) as sesh:
            stmt = (
                sa.update(DetectedObject)
                .where(
                    (DetectedObject.id.in_(self.ids()))
                    & (DetectedObject.in_queue.is_(False))
                    & (DetectedObject.specific_label.is_(None))
                )
                .values({"in_queue": True})
            )
            sesh.execute(stmt)
            sesh.commit()

    def clear_queue(self, *_) -> None:
        logger.info("Removing unclassified objects from deployment from queue")
        with get_session(self.db_path) as sesh:
            stmt = (
                sa.update(DetectedObject)
                .where(
                    (DetectedObject.id.in_(self.ids()))
                    & (DetectedObject.in_queue.is_(True))
                    & (DetectedObject.specific_label.is_(None))
                )
                .values({"in_queue": False})
            )
            sesh.execute(stmt)
            sesh.commit()

    def pull_n_from_queue(self, n: int) -> Sequence[DetectedObject]:
        logger.debug(f"Attempting to pull {n} objects of interest from queue")
        select_stmt = (
            sa.select(DetectedObject.id)
            .where(
                (DetectedObject.id.in_(self.ids()))
                & (DetectedObject.in_queue.is_(True))
                & (DetectedObject.specific_label.is_(None))
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
            objs = (
                sesh.execute(
                    sa.select(DetectedObject).where(DetectedObject.id.in_(record_ids))
                )
                .unique()
                .scalars()
                .all()
            )
            logger.info(f"Pulled {len(objs)} objects of interest from queue")
            return objs


class ObjectsWithoutFeaturesQueue(QueueManager):
    name = "Detections without features"
    description = """
    Objects that have been identified as something of interest (e.g. a moth)
    and need CNN features stored for using to generate tracks & simularity later. 
    """

    def filter_args(self):
        # args = (
        #     DetectedObject.binary_label == constants.POSITIVE_BINARY_LABEL,
        #     DetectedObject.cnn_features.is_(None),
        # )
        args = dict(
            binary_label=constants.POSITIVE_BINARY_LABEL,
            cnn_features=None,
        )
        return args

    def queue_count(self):
        with get_session(self.db_path) as sesh:
            return (
                sesh.query(DetectedObject)
                .filter_by(in_queue=True, **self.filter_args())
                .count()
            )

    def unprocessed_count(self):
        with get_session(self.db_path) as sesh:
            return sesh.query(DetectedObject).filter_by(**self.filter_args()).count()

    def done_count(self):
        with get_session(self.db_path) as sesh:
            return (
                sesh.query(DetectedObject)
                .filter(DetectedObject.cnn_features.is_not(None))
                .count()
            )

    def add_unprocessed(self, *args):
        orm_objects = []
        with get_session(self.db_path) as sesh:
            objects = sesh.query(DetectedObject).filter_by(**self.filter_args()).all()

        for obj in objects:
            obj.in_queue = True
            orm_objects.append(obj)

        with get_session(self.db_path) as sesh:
            logger.info(
                f"Adding {len(orm_objects)} detections without features to queue"
            )
            sesh.bulk_save_objects(orm_objects)
            sesh.commit()

    def clear_queue(self, *args):
        logger.info("Clearing untracked detections in queue")

        with get_session(self.db_path) as sesh:
            objects = []
            for obj in (
                sesh.query(DetectedObject)
                .filter_by(in_queue=True, **self.filter_args())
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
            .filter_by(in_queue=True, **self.filter_args())
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
            logger.info(
                f"Pulled {len(detections)} detections without features from queue"
            )
            return detections


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


def all_queues(db_path, base_directory):
    return {
        q.name: q
        for q in [
            ImageQueue(db_path, base_directory),
            DetectedObjectQueue(db_path, base_directory),
            UnclassifiedObjectQueue(db_path, base_directory),
            ObjectsWithoutFeaturesQueue(db_path, base_directory),
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


def add_monitoring_session_to_queue(
    db_path, monitoring_session, limit: Optional[int] = None
):
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


def clear_all_queues(db_path, base_directory):
    logger.info("Clearing all queues")

    for name, queue in all_queues(db_path, base_directory).items():
        logger.info(f"Clearing queue: {name}")
        queue.clear_queue()
