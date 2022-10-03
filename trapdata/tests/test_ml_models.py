import os
import sys

from trapdata.db import get_db, check_db
from trapdata.models.events import get_or_create_monitoring_sessions
from trapdata.models.queue import add_sample_to_queue, clear_queue

from trapdata.ml.models.localization import MothFasterRCNNObjectDetector
from trapdata.ml.models.classification import MothNonMothClassifier


if __name__ == "__main__":
    image_base_directory = sys.argv[1]

    db_path = os.environ["DATABASE_URL"]
    db_path = "sqlite+pysqlite:///trapdata.db"
    # db_path = ":memory:"

    db = get_db(db_path, create=True)

    monitoring_sessions = get_or_create_monitoring_sessions(
        db_path, image_base_directory
    )

    clear_queue(db_path)
    add_sample_to_queue(db_path, monitoring_sessions[0], sample_size=10)

    object_detector = MothFasterRCNNObjectDetector(db_path=db_path)
    moth_nonmoth_classifier = MothNonMothClassifier(db_path=db_path)
    # species_classifer = UKDenmarkMothSpeciesClassifer(db_path=db_path)

    check_db(db_path, quiet=False)

    # object_detector.run()
    moth_nonmoth_classifier.run()
    # species_classifier.run()
