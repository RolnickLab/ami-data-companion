import collections
import datetime
import logging
import pathlib
import tempfile
import typing
import unittest
from unittest import TestCase

import PIL.Image
import pytest

from trapdata.api.models.classification import (
    MothClassifierBinary,
    MothClassifierQuebecVermont,
)
from trapdata.api.models.localization import APIMothDetector
from trapdata.api.schemas import BoundingBox, Detection, SourceImage
from trapdata.common.filemanagement import find_images
from trapdata.tests import TEST_IMAGES_BASE_PATH

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

    img = PIL.Image.new("RGB", TEST_IMAGE_SIZE, color="red")  # type: ignore
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        img.save(f.name)
        return f.name


def get_test_images(
    subdirs: typing.Iterable[str] = ("vermont", "panama"), limit: int = 6
) -> list[SourceImage]:
    return [
        SourceImage(id=str(img["path"].name), filepath=img["path"])
        for subdir in subdirs
        for img in find_images(pathlib.Path(TEST_IMAGES_BASE_PATH) / subdir)
    ][:limit]


def check_for_duplicate_classifications(results: list[Detection]):
    """
    Ensure that there is only one classification per classifier, per bounding box
    """
    result_counts = collections.defaultdict(int)
    for result in results:
        for classification in result.classifications:
            bbox = result.bbox.to_tuple()
            unique_result = tuple(list(bbox) + [classification.algorithm])
            result_counts[unique_result] += 1

    duplicates = {k: v for k, v in result_counts.items() if v > 1}
    assert not duplicates, f"Duplicate detections found: {duplicates}"


class TestLocalization(TestCase):
    def test_localization_zero(self):
        detector = APIMothDetector(source_images=get_empty_test_images())
        detector.run()
        results = detector.results
        self.assertEqual(len(results), 0)

    def test_localization(self):
        test_images = get_test_images()
        detector = APIMothDetector(
            source_images=test_images,
        )
        detector.run()
        results = detector.results

        # Results are not grouped by image, all detections are in one list
        results_image_ids = {det.source_image_id for det in results}
        all_test_image_ids = {img.id for img in test_images}

        # If an image has no detections, it is not included in the results
        # Check that results_image_ids is a subset of all_test_image_ids
        assert set(results_image_ids).issubset(all_test_image_ids)

        # @TODO ensure bounding boxes are correct

        for detection in detector.results:
            assert isinstance(
                detection, Detection
            ), "Detection result is not a Detection object"
            self.assertIn(detection.source_image_id, all_test_image_ids)


class TestClassification(TestCase):
    def get_detections(self, test_images: list[SourceImage]) -> list[Detection]:
        # @TODO Reuse the results from the localization test. Or provide serialized results.
        detector = APIMothDetector(
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
            filter_results=True,
        )
        classifier.run()
        filtered_detections = classifier.results
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
            for classification in result.classifications:
                self.assertLessEqual(classification.scores[0], 0.4)

    @pytest.mark.skip(
        reason="The new binary classifier is classifying empty images as moths"
    )
    def test_binary_classification_zero(self):
        # @TODO
        # This is classifying empty images as moths!

        classifier = MothClassifierBinary(
            source_images=get_empty_test_images(),
            detections=get_empty_detections(),
        )
        classifier.run()
        results = classifier.results
        self.assertGreater(len(results), 0)
        # Assert that all results are predicted negative and have very high scores
        for result in results:
            for classification in result.classifications:
                self.assertEqual(
                    classification.classification,
                    MothClassifierBinary.negative_binary_label,
                )
                self.assertLessEqual(classification.scores[0], 0.9)

    def test_binary_classification(self):
        test_images = get_test_images()
        detections = self.get_detections(test_images)
        classifier = MothClassifierBinary(
            source_images=test_images,
            detections=detections,
        )
        classifier.run()
        results = classifier.results

        check_for_duplicate_classifications(results)

        self.assertEqual(len(results), len(detections))
        for result in results:
            for classification in result.classifications:
                self.assertIn(
                    classification.classification,
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

        check_for_duplicate_classifications(results)

        # Assert that each result has at least one classification
        for result in results:
            self.assertGreater(
                len(result.classifications), 0, f"{result} has no classifications"
            )

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
