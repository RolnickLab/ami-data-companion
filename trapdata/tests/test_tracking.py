# import newrelic.agent
# newrelic.agent.initialize(environment="staging")

import os
import tempfile
import pathlib

from sqlalchemy import select, orm
from PIL import Image
import torch
from rich import print

from trapdata import logger
from trapdata import constants
from trapdata.db import get_db, check_db, get_session_class
from trapdata.db.models.events import (
    get_or_create_monitoring_sessions,
    MonitoringSession,
)
from trapdata.db.models.images import TrapImage
from trapdata.db.models.queue import add_image_to_queue, clear_all_queues
from trapdata.db.models.detections import DetectedObject, get_detections_for_image

from trapdata.ml.utils import StopWatch
from trapdata.ml.models.localization import (
    MothObjectDetector_FasterRCNN,
    GenericObjectDetector_FasterRCNN_MobileNet,
)
from trapdata.ml.models.classification import (
    MothNonMothClassifier,
    UKDenmarkMothSpeciesClassifier,
)
from trapdata.ml.models.tracking import TrackingCostOriginal, image_diagonal

TRACKING_THRESHOLD = 1.0


def start_sequence(obj_current: DetectedObject, obj_previous: DetectedObject):
    # obj_current.sequence_id = uuid.uuid4() # @TODO ensure this is unique, or
    sequence_id = f"{obj_previous.id}-sequence"
    obj_previous.sequence_id = sequence_id
    obj_previous.sequence_frame = 0

    obj_current.sequence_id = sequence_id
    obj_current.sequence_frame = 1
    logger.info(
        f"Created new sequence beginning with obj {obj_previous.id}: {sequence_id}"
    )
    return sequence_id


def assign_sequence(
    obj_current: DetectedObject,
    obj_previous: DetectedObject,
    final_cost: float,
    session,
):
    obj_current.sequence_previous_cost = final_cost
    obj_current.sequence_previous_id = obj_previous.id
    if obj_previous.sequence_id:
        obj_current.sequence_id = obj_previous.sequence_id
        obj_current.sequence_frame = obj_previous.sequence_frame + 1
    else:
        start_sequence(obj_current, obj_previous)
    session.add(obj_current)
    session.add(obj_previous)
    session.flush()
    session.commit()
    return obj_current.sequence_id, obj_current.sequence_frame


def compare_objects(
    image_current: TrapImage,
    image_previous: TrapImage,
    cnn_model,
    session,
):
    logger.info(
        f"Calculating tracking costs in image {image_current.id} vs. {image_previous.id}"
    )
    objects_current: list[DetectedObject] = (
        session.execute(
            select(DetectedObject)
            .filter(DetectedObject.image == image_current)
            .where(DetectedObject.binary_label == constants.POSITIVE_BINARY_LABEL)
        )
        .unique()
        .scalars()
        .all()
    )
    objects_previous: list[DetectedObject] = (
        session.execute(
            select(DetectedObject)
            .filter(DetectedObject.image == image_previous)
            .where(DetectedObject.binary_label == constants.POSITIVE_BINARY_LABEL)
        )
        .unique()
        .scalars()
        .all()
    )

    img_shape = Image.open(image_current.absolute_path).size

    for obj_current in objects_current:
        if obj_current.sequence_id:
            logger.info(
                f"Skipping obj {obj_current.id}, already assigned to sequence {obj_current.sequence_id} as frame {obj_current.sequence_frame}"
            )
            continue

        logger.info(f"Comparing obj {obj_current.id} to all objects in previous frame")
        costs = []
        for obj_previous in objects_previous:
            cost = TrackingCostOriginal(
                obj_current.cropped_image_data(),
                obj_previous.cropped_image_data(),
                tuple(obj_current.bbox),
                tuple(obj_previous.bbox),
                source_image_diagonal=image_diagonal(img_shape[0], img_shape[1]),
                cnn_source_model=cnn_model,
            )
            final_cost = cost.final_cost()
            logger.info(
                f"\tScore for obj {obj_current.id} vs. {obj_previous.id}: {final_cost}"
            )
            costs.append((final_cost, obj_previous))
        costs.sort(key=lambda cost: cost[0])
        highest_cost, best_match = costs[-1]
        sequence_id, frame_num = assign_sequence(
            obj_current=obj_current,
            obj_previous=best_match,
            final_cost=highest_cost,
            session=session,
        )
        logger.info(
            f"Assigned {obj_current.id} to sequence {sequence_id} as frame #{frame_num}. Match score: {highest_cost}"
        )

    session.close()

    # print(list(reversed(sorted(costs, key=lambda costs: costs[0]))))


# @newrelic.agent.background_task()
def test_tracking(db_path, image_base_directory, sample_size):

    # db_path = ":memory:"
    get_db(db_path, create=True)
    Session = get_session_class(db_path)

    get_or_create_monitoring_sessions(db_path, image_base_directory)

    clear_all_queues(db_path)

    images = []
    with Session() as session:
        ms = session.execute(
            select(MonitoringSession)
            .order_by(MonitoringSession.num_images.desc())
            .limit(1)
        ).scalar()

        print("Using Monitoring Session:", ms)

        for image in ms.images:
            add_image_to_queue(db_path, image.id)

    if torch.cuda.is_available():
        object_detector = MothObjectDetector_FasterRCNN(db_path=db_path, batch_size=2)
    else:
        object_detector = GenericObjectDetector_FasterRCNN_MobileNet(
            db_path=db_path, batch_size=2
        )
    moth_nonmoth_classifier = MothNonMothClassifier(db_path=db_path, batch_size=300)
    species_classifier = UKDenmarkMothSpeciesClassifier(db_path=db_path, batch_size=300)

    check_db(db_path, quiet=False)

    object_detector.run()
    moth_nonmoth_classifier.run()
    species_classifier.run()

    logger.info("Classification complete")

    with Session() as session:
        images = (
            session.execute(
                select(TrapImage)
                .filter(TrapImage.monitoring_session == ms)
                .order_by(TrapImage.timestamp)
            )
            .unique()
            .scalars()
            .all()
        )
        for i, image in enumerate(images):
            n_current = i
            n_previous = max(n_current - 1, 0)
            image_current = images[n_current]
            image_previous = images[n_previous]
            if image_current != image_previous:
                compare_objects(
                    image_current,
                    image_previous,
                    cnn_model=species_classifier.model,
                    session=session,
                )


if __name__ == "__main__":
    image_base_directory = pathlib.Path(__file__).parent
    logger.info(f"Using test images from: {image_base_directory}")

    local_weights_path = torch.hub.get_dir()
    logger.info(f"Looking for or downloading weights in {local_weights_path}")
    os.environ["LOCAL_WEIGHTS_PATH"] = local_weights_path

    # with tempfile.NamedTemporaryFile(suffix=".db", delete=True) as db_filepath:
    db_filepath = pathlib.Path("/home/michael/Projects/AMI/tracking-test.db")
    db_path = f"sqlite+pysqlite:///{db_filepath.name}"
    logger.info(f"Using temporary DB: {db_path}")

    with StopWatch() as t:
        test_tracking(db_path, image_base_directory, sample_size=10)
    logger.info(t)
