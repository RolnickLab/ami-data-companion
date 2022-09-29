from trapdata import db
from trapdata import logger
from trapdata.models.images import Image
from trapdata.models.detections import DetectedObject


def add_sample_to_queue(monitoring_session, sample_size=10):
    ms = monitoring_session

    with db.get_session(ms.base_directory) as sess:
        num_in_queue = (
            sess.query(Image)
            .filter_by(in_queue=True, monitoring_session_id=ms.id)
            .count()
        )
        if num_in_queue < sample_size:
            for image in (
                sess.query(Image)
                .filter_by(
                    in_queue=False,
                    monitoring_session_id=ms.id,
                )
                .order_by(db.sa.func.random())
                .limit(sample_size - num_in_queue)
                .all()
            ):
                image.in_queue = True
                sess.add(image)
            sess.commit()


def add_monitoring_session_to_queue(monitoring_session, limit=None):
    """
    Add images captured during a give Monitoring Session to the
    processing queue. If a limit is specified, only add that many
    additional images to the queue. Will not add duplicates to the queue.
    """
    ms = monitoring_session

    # @TODO This may be a faster way to add all images to the queue
    # image_ids = get_monitoring_session_image_ids(ms)
    # logger.info(f"Adding {len(image_ids)} images into queue")

    # with db.get_session(ms.base_directory) as sess:
    #     sess.execute(
    #         db.sa.update(Image)
    #         .where(Image.monitoring_session_id == ms.id)
    #         .values(in_queue=True)
    #     )

    logger.info(f"Adding all images for Monitoring Session {ms.id} to queue")
    with db.get_session(ms.base_directory) as sess:
        for image in (
            sess.query(Image)
            .filter_by(
                in_queue=False,
                monitoring_session_id=ms.id,
            )
            .order_by(Image.timestamp)
            .limit(limit)
            .all()
        ):
            image.in_queue = True
            sess.add(image)
        sess.commit()


def images_in_queue(base_path):

    with db.get_session(base_path) as sess:
        return sess.query(Image).filter_by(in_queue=True).count()


def queue_counts(base_path):

    counts = {}
    with db.get_session(base_path) as sess:
        counts["images"] = sess.query(Image).filter_by(in_queue=True).count()
        counts["unclassified_objects"] = (
            sess.query(DetectedObject)
            .filter_by(in_queue=True, binary_label=None)
            .count()
        )
        counts["unclassified_moths"] = (
            sess.query(DetectedObject)
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
    with db.get_session(base_path) as sess:
        counts["images"] = sess.query(Image).filter_by(last_processed=None).count()
        counts["unclassified_objects"] = (
            sess.query(DetectedObject).filter_by(binary_label=None).count()
        )
        counts["unclassified_moths"] = (
            sess.query(DetectedObject)
            .filter_by(
                specific_label=None,
            )
            .filter(
                DetectedObject.binary_label.is_not(None),
            )
            .count()
        )
    return counts


def clear_queue(base_path):

    logger.info("Clearing images in queue")

    with db.get_session(base_path) as sess:
        for image in sess.query(Image).filter_by(in_queue=True).all():
            image.in_queue = False
            sess.add(image)
        sess.commit()
