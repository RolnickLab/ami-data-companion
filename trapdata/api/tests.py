import datetime
import logging
import pathlib
import tempfile
import unittest
from unittest import TestCase

import PIL.Image

from trapdata.common.filemanagement import find_images
from trapdata.tests import TEST_IMAGES_BASE_PATH

from . import auth, queries, settings
from .models.classification import MothClassifierBinary, MothClassifierQuebecVermont
from .models.localization import MothDetector
from .schemas import BoundingBox, Detection, SourceImage

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


def get_empty_test_images(read: bool = False):
    assert len(TEST_BASE64_IMAGES) > 0, "No empty test images found"
    images = [SourceImage(id=name, b64=b64) for name, b64 in TEST_BASE64_IMAGES.items()]
    if read:
        for image in images:
            image.open(raise_exception=True)
    return list(images)


def get_empty_detections():
    # Return one large detection for each image
    # @TODO Also test zero sized box = [0, 0, 0, 0]
    return [
        Detection(
            source_image_id=img.id,
            bbox=BoundingBox.from_coords([0, 0, img.width, img.height]),  # type: ignore
            algorithm="Full width and height",
            timestamp=datetime.datetime.now(),
        )
        for img in get_empty_test_images(read=True)
    ]


def make_image():
    # Create a fake test image and save to temporary filepath

    img = PIL.Image.new("RGB", TEST_IMAGE_SIZE, color="red")
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        img.save(f.name)
        return f.name


def get_test_images(subdirs: list[str] = ["vermont", "panama"]) -> list[SourceImage]:
    return [
        SourceImage(id=str(img["path"].name), filepath=img["path"])
        for subdir in subdirs
        for img in find_images(pathlib.Path(TEST_IMAGES_BASE_PATH) / subdir)
    ]


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
            self.assertIsInstance(item, SourceImage)
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
    def test_localization_zero(self):
        detector = MothDetector(source_images=get_empty_test_images())
        detector.run()
        results = detector.results
        self.assertEqual(len(results), 0)

    def test_localization(self):
        test_images = get_test_images()
        detector = MothDetector(
            source_images=test_images,
        )
        detector.run()
        results = detector.results
        source_image_ids = {det.source_image_id for det in results}
        source_image_ids_expected = {img.id for img in test_images}
        self.assertEqual(source_image_ids, source_image_ids_expected)

        self.assertEqual(len(detector.results), len(test_images))
        # @TODO ensure bounding boxes are correct

        for test_image, detections in zip(test_images, detector.results):
            for detection in detections:
                assert isinstance(detection, Detection)
                self.assertEqual(detection.source_image_id, test_image.id)


class TestClassification(TestCase):
    def get_detections(self, test_images: list[SourceImage]) -> list[Detection]:
        # @TODO Reuse the results from the localization test. Or provide serialized results.
        detector = MothDetector(
            source_images=test_images,
        )
        detector.run()
        return detector.results

    def filter_detections(
        self,
        test_images: list[SourceImage],
        detections: list[Detection],
    ) -> list[Detection]:
        # Filter detections based on results of the binary classifier
        classifier = MothClassifierBinary(
            source_images=test_images,
            detections=detections,
        )
        classifier.run()
        filtered_detections = classifier.get_filtered_detections()
        self.assertLessEqual(len(filtered_detections), len(detections))
        return filtered_detections

    def test_classification_zero(self):
        classifier = MothClassifierQuebecVermont(
            source_images=get_empty_test_images(),
            detections=get_empty_detections(),
        )
        classifier.run()
        results = classifier.results
        print(results)
        self.assertGreater(len(results), 0)
        # Assert that all results have very low scores
        for result in results:
            self.assertLessEqual(result.scores[0], 0.4)  # @TODO lower this score

    def test_binary_classification_zero(self):
        classifier = MothClassifierBinary(
            source_images=get_empty_test_images(),
            detections=get_empty_detections(),
        )
        classifier.run()
        results = classifier.results
        self.assertGreater(len(results), 0)
        # Assert that all results are predicted negative and have very high scores
        for result in results:
            self.assertEqual(
                result.classification,
                MothClassifierBinary.negative_binary_label,
            )
            self.assertLessEqual(result.scores[0], 0.9)

    def test_binary_classification(self):
        test_images = get_test_images()
        detections = self.get_detections(test_images)
        classifier = MothClassifierBinary(
            source_images=test_images,
            detections=detections,
        )
        classifier.run()
        results = classifier.results
        self.assertEqual(len(results), len(detections))
        for result in results:
            self.assertIn(
                result.classification,
                (
                    MothClassifierBinary.positive_binary_label,
                    MothClassifierBinary.negative_binary_label,
                ),
            )
        # @TODO ensure classification results are correct

    def test_classification(self):
        test_images = get_test_images()
        detections = self.get_detections(test_images)
        detections = self.filter_detections(test_images, detections)
        classifier = MothClassifierQuebecVermont(
            source_images=test_images,
            detections=detections,
        )
        classifier.run()
        results = classifier.results
        # image_lookup = {img.id: img for img in test_images}
        self.assertEqual(len(results), len(detections))
        # @TODO ensure classification results are correct


class TestSourceImageSchema(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.test_image = make_image()

    def test_filepath(self):
        filepath = self.test_image
        source_image = SourceImage(id="1", filepath=filepath)
        self.assertEqual(source_image.filepath, filepath)
        img = source_image.open()
        self.assertIsNotNone(img)
        assert img is not None  # For type hinting
        self.assertEqual(img.size, TEST_IMAGE_SIZE)
        img.close()

    def test_url(self):
        # Don't trust placeholder image services
        url = "https://upload.wikimedia.org/wikipedia/en/thumb/8/80/Wikipedia-logo-v2.svg/103px-Wikipedia-logo-v2.svg.png"
        source_image = SourceImage(id="1", url=url)
        self.assertEqual(source_image.url, url)
        img = source_image.open()
        self.assertIsNotNone(img)
        assert img is not None
        self.assertEqual(img.size, (103, 94))
        img.close()

    def test_bad_base64(self):
        base64_string = "happy birthday"
        source_image = SourceImage(id="1", b64=base64_string)
        from binascii import Error

        with self.assertRaises(Error):
            source_image.open(raise_exception=True)

    def _test_base64(self, base64_string):
        source_image = SourceImage(id="1", b64=base64_string)
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
