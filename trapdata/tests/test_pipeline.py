import newrelic.agent

newrelic.agent.initialize(environment="staging")

import os
import sys

from trapdata.db import get_db, check_db
from trapdata.db.models.events import get_or_create_monitoring_sessions
from trapdata.db.models.queue import add_sample_to_queue, images_in_queue, clear_queue

from trapdata.ml.utils import StopWatch
from trapdata.ml.models.localization import MothObjectDetector_FasterRCNN
from trapdata.ml.models.classification import (
    MothNonMothClassifier,
    UKDenmarkMothSpeciesClassifier,
)


@newrelic.agent.background_task()
def end_to_end(db_path, image_base_directory, sample_size):

    # db_path = ":memory:"

    db = get_db(db_path, create=True)

    _ = get_or_create_monitoring_sessions(db_path, image_base_directory)

    clear_queue(db_path)
    add_sample_to_queue(db_path, sample_size=sample_size)
    num_images = images_in_queue(db_path)
    print(f"Images in queue: {num_images}")
    assert num_images == sample_size

    object_detector = MothObjectDetector_FasterRCNN(db_path=db_path, batch_size=10)
    moth_nonmoth_classifier = MothNonMothClassifier(db_path=db_path, batch_size=300)
    species_classifier = UKDenmarkMothSpeciesClassifier(db_path=db_path, batch_size=300)

    check_db(db_path, quiet=False)

    object_detector.run()
    moth_nonmoth_classifier.run()
    species_classifier.run()


if __name__ == "__main__":
    image_base_directory = sys.argv[1]
    # db_path = os.environ["DATABASE_URL"]
    db_path = "sqlite+pysqlite:///trapdata-test-1002.db"

    with StopWatch() as t:
        end_to_end(db_path, image_base_directory, sample_size=1)
    print(t)
