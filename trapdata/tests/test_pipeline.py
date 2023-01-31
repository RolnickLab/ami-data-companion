# import newrelic.agent
# newrelic.agent.initialize(environment="staging")

import os
import tempfile
import pathlib

import torch

from trapdata import logger
from trapdata.db import get_db, check_db
from trapdata.db.models.events import get_or_create_monitoring_sessions
from trapdata.db.models.queue import (
    add_sample_to_queue,
    images_in_queue,
    clear_all_queues,
)

from trapdata.ml.utils import StopWatch
from trapdata.ml.models.localization import (
    MothObjectDetector_FasterRCNN,
    GenericObjectDetector_FasterRCNN_MobileNet,
)
from trapdata.ml.models.classification import (
    MothNonMothClassifier,
    UKDenmarkMothSpeciesClassifier,
)


# @newrelic.agent.background_task()
def end_to_end(db_path, image_base_directory, sample_size):

    # db_path = ":memory:"
    get_db(db_path, create=True)

    get_or_create_monitoring_sessions(db_path, image_base_directory)

    clear_all_queues(db_path)
    add_sample_to_queue(db_path, sample_size=sample_size)
    num_images = images_in_queue(db_path)
    logger.info(f"Images in queue: {num_images}")
    assert num_images == sample_size

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


def run():
    image_base_directory = pathlib.Path(__file__).parent
    logger.info(f"Using test images from: {image_base_directory}")

    local_weights_path = torch.hub.get_dir()
    logger.info(f"Looking for or downloading weights in {local_weights_path}")
    os.environ["LOCAL_WEIGHTS_PATH"] = local_weights_path

    with tempfile.NamedTemporaryFile(suffix=".db", delete=True) as db_filepath:
        db_path = f"sqlite+pysqlite:///{db_filepath.name}"
        logger.info(f"Using temporary DB: {db_path}")

        with StopWatch() as t:
            end_to_end(db_path, image_base_directory, sample_size=1)
        logger.info(t)


if __name__ == "__main__":
    run()
