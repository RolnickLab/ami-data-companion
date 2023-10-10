import logging
import unittest
from unittest import TestCase

import trapdata.ml as ml

from . import auth, queries, settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestAuth(TestCase):
    def test_login(self):
        token = auth.get_token()
        self.assertIsNotNone(token)

    def test_get_session(self):
        session = auth.get_session()
        self.assertIsNotNone(session)

    def test_current_user(self):
        user = auth.get_current_user()
        self.assertEqual(user["email"], settings.api_username)


class TestSourceImages(TestCase):
    def test_get_next_batch(self):
        batch = queries.get_next_source_images(10)
        for item in batch:
            self.assertIsInstance(item, queries.IncomingSourceImage)
        return batch

    def test_save_detections(self):
        item_ids = [
            60,
            1886356,
        ]

        detected_objects_data = [
            [
                {
                    "bbox": [3980, 285, 4240, 665],
                    "model_name": "FasterRCNN for AMI Moth Traps 2021",
                },
                {
                    "bbox": [2266, 2321, 3010, 2637],
                    "model_name": "FasterRCNN for AMI Moth Traps 2021",
                },
            ],
            [
                {
                    "bbox": [3980, 285, 4240, 665],
                    "model_name": "FasterRCNN for AMI Moth Traps 2021",
                },
                {
                    "bbox": [2266, 2321, 3010, 2637],
                    "model_name": "FasterRCNN for AMI Moth Traps 2021",
                },
            ],
        ]

        queries.save_detected_objects(
            source_image_ids=item_ids, detected_objects_data=detected_objects_data
        )


class TestLocalization(TestCase):
    def test_localization(self):
        ObjectDetector = ml.models.object_detectors[settings.localization_model.value]
        object_detector = ObjectDetector(
            db_path="",  # deprecated
            image_base_path="",  # deprecated
            user_data_path=settings.user_data_path,
            batch_size=1,
            num_workers=1,
        )
        object_detector.run()
        logger.info("Localization complete")


if __name__ == "__main__":
    unittest.main()
