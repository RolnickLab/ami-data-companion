# import newrelic.agent
# newrelic.agent.initialize(environment="staging")

import os
import tempfile
import pathlib

from sqlalchemy import select
from PIL import Image
import torch
from rich import print

from trapdata import logger
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
from trapdata.ml.models.tracking import TrackingCost, image_diagonal


# @newrelic.agent.background_task()
def test_tracking(db_path, image_base_directory, sample_size):

    # db_path = ":memory:"
    get_db(db_path, create=True)

    get_or_create_monitoring_sessions(db_path, image_base_directory)

    clear_all_queues(db_path)

    Session = get_session_class(db_path)
    images = []
    with Session() as session:
        ms = session.execute(
            select(MonitoringSession)
            .order_by(MonitoringSession.num_images.desc())
            .limit(1)
        ).scalar()

        print("Using Monitoring Session:", ms)

        for image in ms.images:
            print(image)
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

    with Session() as session:
        objects = (
            session.execute(
                select(DetectedObject).filter(DetectedObject.monitoring_session == ms)
            )
            .unique()
            .scalars()
            .all()
        )
        print(objects)

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
        image1 = images[0]
        image2 = images[2]

        objects1 = (
            session.execute(
                select(DetectedObject).filter(DetectedObject.image == image1)
            )
            .unique()
            .scalars()
            .all()
        )
        objects2 = (
            session.execute(
                select(DetectedObject).filter(DetectedObject.image == image2)
            )
            .unique()
            .scalars()
            .all()
        )

        img_shape = Image.open(image1.absolute_path).size

    for obj1 in objects1:
        for obj2 in objects2:
            cost = TrackingCost(
                obj1.cropped_image_data(),
                obj2.cropped_image_data(),
                tuple(obj1.bbox),
                tuple(obj2.bbox),
                source_image_diagonal=image_diagonal(img_shape[0], img_shape[1]),
                cnn_source_model=species_classifier.model,
            )
            print(cost.final_cost(), obj1.path, obj2.path)


if __name__ == "__main__":
    image_base_directory = pathlib.Path(__file__).parent
    logger.info(f"Using test images from: {image_base_directory}")

    local_weights_path = torch.hub.get_dir()
    logger.info(f"Looking for or downloading weights in {local_weights_path}")
    os.environ["LOCAL_WEIGHTS_PATH"] = local_weights_path

    with tempfile.NamedTemporaryFile(suffix=".db", delete=True) as db_filepath:
        db_path = f"sqlite+pysqlite:///{db_filepath.name}"
        logger.info(f"Using temporary DB: {db_path}")

        with StopWatch() as t:
            test_tracking(db_path, image_base_directory, sample_size=10)
        logger.info(t)
