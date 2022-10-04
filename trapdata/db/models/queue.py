import sqlalchemy as sa
from functools import partial

from trapdata.db import get_session
from trapdata import logger
from trapdata import constants
from trapdata.db.models.images import TrapImage
from trapdata.db.models.detections import (
    DetectedObject,
    save_detected_objects,
    save_classified_objects,
)
from trapdata import ml


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

    def process_queue(self, **kwargs):
        return NotImplementedError


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
        with get_session(self.db_path) as sesh:
            # @TODO switch to bulk update method
            for image in (
                sesh.query(TrapImage)
                .filter_by(
                    in_queue=False,
                    last_processed=None,
                )
                .all()
            ):
                image.in_queue = True
                sesh.add(image)
            sesh.commit()
            return sesh.query(TrapImage).filter_by(last_processed=None).count()

    def clear_queue(self, *args):
        logger.info("Clearing images in queue")

        with get_session(self.db_path) as sesh:
            # @TODO switch to bulk update method
            for image in sesh.query(TrapImage).filter_by(in_queue=True).all():
                image.in_queue = False
                sesh.add(image)
            sesh.commit()

    def process_queue(self, model_name, base_path, models_dir, batch_size, num_workers):
        logger.info("Processing image queue")
        localization_results_callback = partial(save_detected_objects, base_path)
        ml.detect_objects(
            model_name=model_name,
            models_dir=models_dir,
            base_directory=base_path,  # base path for relative images
            results_callback=localization_results_callback,
            batch_size=int(batch_size),
            num_workers=num_workers,
        )
        logger.info("Done processing image queue")


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
            # @TODO switch to bulk update method
            for obj in (
                sesh.query(DetectedObject)
                .filter_by(in_queue=False, binary_label=None)
                .all()
            ):
                obj.in_queue = True
                sesh.add(obj)
            sesh.commit()

    def clear_queue(self, *args):
        logger.info("Clearing detected objects in queue")

        with get_session(self.db_path) as sesh:
            # @TODO switch to bulk update method
            for image in (
                sesh.query(DetectedObject)
                .filter_by(in_queue=True, binary_label=None)
                .all()
            ):
                image.in_queue = False
                sesh.add(image)
            sesh.commit()

    def process_queue(self, model_name, base_path, models_dir, batch_size, num_workers):
        logger.info("Processing detected objects queue")
        classification_results_callback = partial(save_classified_objects, base_path)
        ml.classify_objects(
            model_name=model_name,
            models_dir=models_dir,
            base_directory=base_path,
            results_callback=classification_results_callback,
            batch_size=int(batch_size),
            num_workers=num_workers,
        )
        logger.info("Done processing detected objects queue")


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
        with get_session(self.db_path) as sesh:
            # @TODO switch to bulk update method
            for obj in (
                sesh.query(DetectedObject)
                .filter_by(
                    in_queue=False,
                    specific_label=None,
                    binary_label=constants.POSITIVE_BINARY_LABEL,
                )
                .all()
            ):
                obj.in_queue = True
                sesh.add(obj)
            sesh.commit()

    def clear_queue(self, *args):
        logger.info("Clearing unclassified objects in queue")

        with get_session(self.db_path) as sesh:
            # @TODO switch to bulk update method
            for image in (
                sesh.query(DetectedObject)
                .filter_by(in_queue=True, specific_label=None)
                .all()
            ):
                image.in_queue = False
                sesh.add(image)
            sesh.commit()

    def process_queue(self, model_name, base_path, models_dir, batch_size, num_workers):
        logger.info("Processing unclassified image queue")
        classification_results_callback = partial(save_classified_objects, base_path)
        ml.classify_objects(
            model_name=model_name,
            models_dir=models_dir,
            base_directory=base_path,
            results_callback=classification_results_callback,
            batch_size=int(batch_size),
            num_workers=num_workers,
        )
        logger.info("Done processing unclassified image queue")


def all_queues(db_path):
    return {
        q.name: q
        for q in [
            ImageQueue(db_path),
            DetectedObjectQueue(db_path),
            UnclassifiedObjectQueue(db_path),
        ]
    }


def add_image_to_queue(db_path, image_id):

    with get_session(db_path) as sesh:
        image = sesh.query(TrapImage).get(image_id)
        logger.info(f"Addding image to queue: {image}")
        if not image.in_queue:
            image.in_queue = True
            sesh.add(image)
            sesh.commit()


def add_sample_to_queue(db_path, sample_size=10):

    with get_session(db_path) as sesh:
        num_in_queue = sesh.query(TrapImage).filter_by(in_queue=True).count()
        if num_in_queue < sample_size:
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
                sesh.add(image)
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
            sesh.add(image)
        sesh.commit()


def images_in_queue(db_path):

    with get_session(db_path) as sesh:
        return sesh.query(TrapImage).filter_by(in_queue=True).count()


def queue_counts(base_path):

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


def unprocessed_counts(base_path):

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

    with get_session(db_path) as sesh:
        for image in sesh.query(TrapImage).filter_by(in_queue=True).all():
            image.in_queue = False
            sesh.add(image)
        for obj in sesh.query(DetectedObject).filter_by(in_queue=True).all():
            obj.in_queue = False
            sesh.add(obj)
        sesh.commit()


class Queue:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        logger.debug("Initializing queue status and starting DB polling")

    def check_queue(self, *args):
        self.running = self.bgtask.is_alive()

    def process_queue(self):
        db_path = self.app.db_path

        models_dir = (
            pathlib.Path(self.app.config.get("paths", "user_data_path")) / "models"
        )
        logger.info(f"Local models path: {models_dir}")
        num_workers = int(self.app.config.get("performance", "num_workers"))

    def on_running(self, *args):
        if self.running:
            if not self.clock:
                logger.debug("Scheduling queue check")
                self.clock = Clock.schedule_interval(self.check_queue, 1)
            self.status_str = "Running"
        else:
            logger.debug("NOT Unscheduling queue check!")
            # logger.debug("Unscheduling queue check")
            # Clock.unschedule(self.clock)
            self.status_str = "Stopped"

    def start(self, *args):
        # @NOTE can't change a widget property from a bg thread
        if not self.running:
            logger.info("Starting queue")
            task_name = "Mr. Queue"
            self.bgtask = threading.Thread(
                target=self.process_queue,
                daemon=True,
                name=task_name,
            )
            self.bgtask.start()
            self.running = True

    def clear(self):
        clear_queue(self.app.db_path)
