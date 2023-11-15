import logging
import tempfile
import unittest
from unittest import TestCase

import PIL.Image

import trapdata.ml as ml

from . import auth, queries, settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


TEST_IMAGE_SIZE = (100, 100)
TEST_BASE64_IMAGES = {
    # 10x10 pixel images
    "RED": "iVBORw0KGgoAAAANSUhEUgAAAAoAAAAKCAYAAACNMs+9AAAAFUlEQVR42mP8z8BQz0AEYBxVSF+FABJADveWkH6oAAAAAElFTkSuQmCC",
    "GREEN": "iVBORw0KGgoAAAANSUhEUgAAAAoAAAAKCAYAAACNMs+9AAAAFUlEQVR42mNk+M9Qz0AEYBxVSF+FAAhKDveksOjmAAAAAElFTkSuQmCC",
    "BLUE": "iVBORw0KGgoAAAANSUhEUgAAAAoAAAAKCAYAAACNMs+9AAAAFUlEQVR42mNkYPhfz0AEYBxVSF+FAP5FDvcfRYWgAAAAAElFTkSuQmCC",
    "BROWSER_STRING": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAoAAAAKCAYAAACNMs+9AAAAFUlEQVR42mNkYPhfz0AEYBxVSF+FAP5FDvcfRYWgAAAAAElFTkSuQmCC=",  # noqa
}


def make_image():
    # Create a fake test image and save to temporary filepath

    img = PIL.Image.new("RGB", TEST_IMAGE_SIZE, color="red")
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        img.save(f.name)
        return f.name


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


class NoTestSourceImages(TestCase):
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
            batch_size=2,
            num_workers=2,
        )
        object_detector.run()
        logger.info("Localization complete")


class TestSourceImageSchema(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.test_image = make_image()

    def test_filepath(self):
        filepath = self.test_image
        source_image = queries.IncomingSourceImage(id=1, filepath=filepath)
        self.assertEqual(source_image.filepath, filepath)
        img = source_image.open()
        self.assertIsNotNone(img)
        assert img is not None  # For type hinting
        self.assertEqual(img.size, TEST_IMAGE_SIZE)
        img.close()

    def test_url(self):
        # Don't trust placeholder image services
        url = "https://upload.wikimedia.org/wikipedia/en/thumb/8/80/Wikipedia-logo-v2.svg/103px-Wikipedia-logo-v2.svg.png"
        source_image = queries.IncomingSourceImage(id=1, url=url)
        self.assertEqual(source_image.url, url)
        img = source_image.open()
        self.assertIsNotNone(img)
        assert img is not None
        self.assertEqual(img.size, (103, 94))
        img.close()

    def test_bad_base64(self):
        base64_string = "happy birthday"
        source_image = queries.IncomingSourceImage(id=1, b64=base64_string)
        from binascii import Error

        with self.assertRaises(Error):
            source_image.open(raise_exception=True)

    def _test_base64(self, base64_string):
        source_image = queries.IncomingSourceImage(id=1, b64=base64_string)
        img = source_image.open(raise_exception=True)
        self.assertIsNotNone(img)
        assert img is not None
        self.assertEqual(img.size, (10, 10))
        img.close()

    def test_base64_images(self):
        for image in TEST_BASE64_IMAGES.values():
            self._test_base64(image)


if __name__ == "__main__":
    unittest.main()
