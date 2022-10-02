import os
import sys

from trapdata.db import get_db, check_db
from trapdata.models.events import get_or_create_monitoring_sessions
from trapdata.ml.models.localization import FasterRCNN_ResNet50_FPN


if __name__ == "__main__":
    image_base_directory = sys.argv[1]

    db_path = os.environ["DATABASE_URL"]
    db_path = "sqlite+pysqlite:///trapdata.db"
    # db_path = ":memory:"

    db = get_db(db_path, create=True)

    get_or_create_monitoring_sessions(db_path, image_base_directory)

    inference_model = FasterRCNN_ResNet50_FPN(db_path=db_path)
    inference_model.load_model()

    check_db(db_path, quiet=False)

    inference_model.run()
