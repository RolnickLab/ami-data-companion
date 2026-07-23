import collections
import datetime
import logging
import pathlib
import tempfile
import typing
import unittest
from unittest import TestCase

import numpy as np
import PIL.Image
import pytest
import torch
import torchvision.transforms

from trapdata.api.models.classification import (
    MothClassifierBinary,
    MothClassifierQuebecVermont,
)
from trapdata.api.models.localization import APIMothDetector
from trapdata.api.schemas import (
    AlgorithmReference,
    BoundingBox,
    DetectionResponse,
    SourceImage,
)
from trapdata.common.filemanagement import find_images
from trapdata.ml.models.localization import (
    AnyBugObjectDetector_YOLO26,
    MothObjectDetector_FasterRCNN_2023,
)
from trapdata.tests import TEST_IMAGES_BASE_PATH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


TEST_IMAGE_SIZE = (100, 100)
TEST_BASE64_IMAGES = {
    # 10x10 pixel images
    "RED": "iVBORw0KGgoAAAANSUhEUgAAAAoAAAAKCAYAAACNMs+9AAAAFUlEQVR42mP8z8BQz0AEYBxVSF+FABJADveWkH6oAAAAAElFTkSuQmCC",  # noqa: E501
    "GREEN": "iVBORw0KGgoAAAANSUhEUgAAAAoAAAAKCAYAAACNMs+9AAAAFUlEQVR42mNk+M9Qz0AEYBxVSF+FAAhKDveksOjmAAAAAElFTkSuQmCC",  # noqa: E501
    "BLUE": "iVBORw0KGgoAAAANSUhEUgAAAAoAAAAKCAYAAACNMs+9AAAAFUlEQVR42mNkYPhfz0AEYBxVSF+FAP5FDvcfRYWgAAAAAElFTkSuQmCC",  # noqa: E501
    "BROWSER_STRING": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAoAAAAKCAYAAACNMs+9AAAAFUlEQVR42mNkYPhfz0AEYBxVSF+FAP5FDvcfRYWgAAAAAElFTkSuQmCC=",  # noqa: E501
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
        DetectionResponse(
            source_image_id=img.id,
            bbox=BoundingBox.from_coords([0, 0, img.width, img.height]),  # type: ignore
            algorithm=AlgorithmReference(
                name="Full width and height",
                key="full_width_height",
            ),
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
    subdirs: typing.Iterable[str] = ("vermont", "panama"),
    limit: int = 6,
    with_urls: bool = False,
) -> list[SourceImage]:
    return [
        SourceImage(
            id=str(img["path"].name),
            filepath=img["path"],
            url=img["url"] if with_urls else None,
        )
        for subdir in subdirs
        for img in find_images(pathlib.Path(TEST_IMAGES_BASE_PATH) / subdir)
    ][:limit]


def check_for_duplicate_classifications(results: list[DetectionResponse]):
    """
    Ensure that there is only one classification per classifier, per bounding box
    """
    result_counts = collections.defaultdict(int)
    for result in results:
        for classification in result.classifications:
            bbox = result.bbox.to_tuple()
            unique_result = tuple(list(bbox) + [classification.algorithm.key])
            result_counts[unique_result] += 1

    duplicates = {k: v for k, v in result_counts.items() if v > 1}
    msg = f"Duplicate detections found: {duplicates}"
    assert not duplicates, msg


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
            msg = "Detection result is not a Detection object"
            assert isinstance(detection, DetectionResponse), msg
            self.assertIn(detection.source_image_id, all_test_image_ids)


class TestClassification(TestCase):
    def get_detections(self, test_images: list[SourceImage]) -> list[DetectionResponse]:
        # @TODO Reuse the results from the localization test. Or provide serialized
        # results.
        detector = APIMothDetector(
            source_images=test_images,
        )
        detector.run()
        return detector.results

    def filter_detections(
        self,
        test_images: list[SourceImage],
        detections: list[DetectionResponse],
    ) -> list[DetectionResponse]:
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

    @pytest.mark.skip(reason="Binary classifier is classifying empty images as moths")
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
        # Assert that all results are predicted negative and have high scores
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
            msg = f"{result} has no classifications"
            self.assertGreater(len(result.classifications), 0, msg)

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
        # Don't trust placeholder image services
        url = (
            "https://upload.wikimedia.org/wikipedia/en/thumb/8/80/"
            "Wikipedia-logo-v2.svg/103px-Wikipedia-logo-v2.svg.png"
        )
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


class _StubYOLO:
    """Minimal stand-in for the Ultralytics ``YOLO`` model.

    Records the exact ``images`` argument handed to ``predict`` so a test can
    assert on the array the detector produced, with no weights loaded and no
    inference run.
    """

    def __init__(self) -> None:
        self.received_images: list[np.ndarray] | None = None

    def predict(self, images, **kwargs):
        self.received_images = images
        return []  # post-processing is not exercised in these tests


# A known 2x2 RGB pattern with distinct per-channel values, so a wrong channel
# order or a transposed axis is caught by exact comparison.
_KNOWN_RGB = np.array(
    [
        [[10, 20, 30], [40, 50, 60]],
        [[70, 80, 90], [100, 110, 120]],
    ],
    dtype=np.uint8,
)  # HWC RGB, shape (2, 2, 3)


class TestAnyBugDetectorInputFormat(TestCase):
    """The YOLO26 detector must accept both dataloaders' image formats.

    The async worker's shared dataloader feeds CHW float ``ToTensor`` tensors,
    while the FastAPI ``/process`` path feeds HWC uint8 arrays. Both must reach
    Ultralytics as HWC uint8 BGR. These tests exercise only the format
    conversion in ``predict_batch`` (up to, but not including, the real model
    call, which is stubbed) — no weights, no GPU, no inference.
    """

    def _make_detector(self, stub: _StubYOLO) -> AnyBugObjectDetector_YOLO26:
        # Bypass __init__ so no weights are downloaded and ultralytics is never
        # imported; set only the attributes predict_batch reads.
        detector = AnyBugObjectDetector_YOLO26.__new__(AnyBugObjectDetector_YOLO26)
        detector.model = stub  # type: ignore[assignment]
        detector.device = "cpu"
        detector.bbox_score_threshold = 0.25
        return detector

    def test_async_chw_float_tensor_converted_to_hwc_uint8_bgr(self):
        """A CHW float ToTensor batch (async path) reaches Ultralytics as HWC
        uint8 BGR, round-tripping the known pixel pattern."""
        pil = PIL.Image.fromarray(_KNOWN_RGB, mode="RGB")
        chw_float = torchvision.transforms.ToTensor()(pil)  # (3, 2, 2) float32 in [0,1]
        self.assertEqual(tuple(chw_float.shape), (3, 2, 2))
        self.assertTrue(torch.is_floating_point(chw_float))
        # rest_collate_fn stacks same-size images into an NCHW tensor.
        batch = torch.stack([chw_float])

        stub = _StubYOLO()
        self._make_detector(stub).predict_batch(batch)

        assert stub.received_images is not None
        self.assertEqual(len(stub.received_images), 1)
        arr = stub.received_images[0]
        self.assertEqual(arr.shape, (2, 2, 3), "must be HWC, not CHW")
        self.assertEqual(arr.dtype, np.uint8, "must be uint8, not float")
        # Channels reversed RGB -> BGR, pixel values recovered from [0,1] float.
        np.testing.assert_array_equal(arr, _KNOWN_RGB[..., ::-1])

    def test_async_chw_float_list_batch_converted(self):
        """The mixed-size async fallback (a list of CHW float tensors) is also
        converted, so predict_batch does not depend on the stacked fast path."""
        pil = PIL.Image.fromarray(_KNOWN_RGB, mode="RGB")
        chw_float = torchvision.transforms.ToTensor()(pil)
        batch = [chw_float]  # list, as rest_collate_fn yields for mixed sizes

        stub = _StubYOLO()
        self._make_detector(stub).predict_batch(batch)

        assert stub.received_images is not None
        arr = stub.received_images[0]
        self.assertEqual(arr.shape, (2, 2, 3))
        self.assertEqual(arr.dtype, np.uint8)
        np.testing.assert_array_equal(arr, _KNOWN_RGB[..., ::-1])

    def test_process_hwc_uint8_tensor_unchanged(self):
        """The /process path (HWC uint8 collated into an NHWC tensor) is passed
        through unchanged apart from the RGB->BGR flip — the regression guard
        for the working FastAPI path."""
        batch = torch.from_numpy(np.stack([_KNOWN_RGB]))  # NHWC uint8

        stub = _StubYOLO()
        self._make_detector(stub).predict_batch(batch)

        assert stub.received_images is not None
        arr = stub.received_images[0]
        self.assertEqual(arr.shape, (2, 2, 3))
        self.assertEqual(arr.dtype, np.uint8)
        np.testing.assert_array_equal(arr, _KNOWN_RGB[..., ::-1])

    def test_process_hwc_uint8_ndarray_unchanged(self):
        """The /process path when the batch arrives as a raw 4D ndarray."""
        batch = np.stack([_KNOWN_RGB])  # (1, 2, 2, 3) uint8

        stub = _StubYOLO()
        self._make_detector(stub).predict_batch(batch)

        assert stub.received_images is not None
        arr = stub.received_images[0]
        self.assertEqual(arr.shape, (2, 2, 3))
        self.assertEqual(arr.dtype, np.uint8)
        np.testing.assert_array_equal(arr, _KNOWN_RGB[..., ::-1])

    def test_helper_transposes_and_rescales_chw_float(self):
        """_as_hwc_uint8_rgb converts a CHW float array to HWC uint8 directly."""
        chw_float = torchvision.transforms.ToTensor()(
            PIL.Image.fromarray(_KNOWN_RGB, mode="RGB")
        )
        out = AnyBugObjectDetector_YOLO26._as_hwc_uint8_rgb(chw_float)
        self.assertEqual(out.shape, (2, 2, 3))
        self.assertEqual(out.dtype, np.uint8)
        # Still RGB here — the flip to BGR happens in predict_batch, not the helper.
        np.testing.assert_array_equal(out, _KNOWN_RGB)

    # --- Asserted-contract guards: the conversion must fail loudly rather than
    # silently corrupt when an upstream transform changes the input format. ---

    def test_imagenet_normalized_float_raises(self):
        """A mean/std standardized float tensor (values outside [0, 1]) is
        rejected, not silently rescaled — converting it as if it were ToTensor
        output would corrupt every pixel handed to the model."""
        chw = torchvision.transforms.ToTensor()(
            PIL.Image.fromarray(_KNOWN_RGB, mode="RGB")
        )
        standardized = (chw - 0.5) / 0.25  # now well outside [0, 1]
        self.assertTrue(torch.is_floating_point(standardized))
        with self.assertRaises(ValueError):
            AnyBugObjectDetector_YOLO26._as_hwc_uint8_rgb(standardized)

    def test_already_hwc_float_raises(self):
        """An already channels-last float array (HWC, in range) is rejected
        rather than transposed into garbage, since its channel axis is not where
        the ToTensor contract puts it."""
        hwc_float = _KNOWN_RGB.astype(np.float32) / 255.0  # (2, 2, 3), in [0, 1]
        with self.assertRaises(ValueError):
            AnyBugObjectDetector_YOLO26._as_hwc_uint8_rgb(hwc_float)

    def test_integer_chw_raises(self):
        """An integer channels-first array is rejected rather than passed through
        in the wrong layout."""
        chw_uint8 = np.transpose(_KNOWN_RGB, (2, 0, 1))  # (3, 2, 2) integer CHW
        with self.assertRaises(ValueError):
            AnyBugObjectDetector_YOLO26._as_hwc_uint8_rgb(chw_uint8)

    def test_unexpected_rank_raises(self):
        """A 2D array (neither a single image nor a batch) is rejected."""
        with self.assertRaises(ValueError):
            AnyBugObjectDetector_YOLO26._as_hwc_uint8_rgb(
                np.zeros((5, 5), dtype=np.uint8)
            )

    def test_4d_nchw_float_batch_handled(self):
        """A 4D NCHW float batch is converted per-image to NHWC uint8, preserving
        the batch axis."""
        chw = torchvision.transforms.ToTensor()(
            PIL.Image.fromarray(_KNOWN_RGB, mode="RGB")
        )
        nchw = torch.stack([chw, chw])  # (2, 3, 2, 2)
        out = AnyBugObjectDetector_YOLO26._as_hwc_uint8_rgb(nchw)
        self.assertEqual(out.shape, (2, 2, 2, 3))  # NHWC
        self.assertEqual(out.dtype, np.uint8)
        np.testing.assert_array_equal(out[0], _KNOWN_RGB)
        np.testing.assert_array_equal(out[1], _KNOWN_RGB)

    def test_float_rescale_rounds_not_truncates(self):
        """The [0, 1]->0-255 rescale rounds (np.rint) rather than truncating.
        The values are deliberately NOT multiples of 1/255, and their *255 lands
        clearly above the .5 boundary, so truncation would give a different (one
        lower) uint8 on every channel."""
        # *255 -> 200.7, 10.6, 128.8 ; round -> 201, 11, 129 ; truncate -> 200, 10, 128
        chw = np.array(
            [[[200.7 / 255]], [[10.6 / 255]], [[128.8 / 255]]], dtype=np.float32
        )  # (3, 1, 1) CHW
        out = AnyBugObjectDetector_YOLO26._as_hwc_uint8_rgb(chw)
        self.assertEqual(out.shape, (1, 1, 3))
        np.testing.assert_array_equal(out, np.array([[[201, 11, 129]]], dtype=np.uint8))


class TestDetectorTransformContracts(TestCase):
    """Pin each detector's per-image transform contract so the fix cannot drift
    the FasterRCNN path or the YOLO /process path."""

    def test_fasterrcnn_transform_is_chw_float_tensor(self):
        """FasterRCNN still receives CHW float ToTensor input, unchanged."""
        detector = MothObjectDetector_FasterRCNN_2023.__new__(
            MothObjectDetector_FasterRCNN_2023
        )
        transformed = detector.get_transforms()(
            PIL.Image.fromarray(_KNOWN_RGB, mode="RGB")
        )
        self.assertIsInstance(transformed, torch.Tensor)
        self.assertTrue(torch.is_floating_point(transformed))
        self.assertEqual(tuple(transformed.shape), (3, 2, 2))  # CHW
        self.assertLessEqual(float(transformed.max()), 1.0)

    def test_yolo_transform_is_hwc_uint8_array(self):
        """The YOLO26 /process transform still yields an HWC uint8 RGB array."""
        detector = AnyBugObjectDetector_YOLO26.__new__(AnyBugObjectDetector_YOLO26)
        transformed = detector.get_transforms()(
            PIL.Image.fromarray(_KNOWN_RGB, mode="RGB")
        )
        self.assertIsInstance(transformed, np.ndarray)
        self.assertEqual(transformed.dtype, np.uint8)
        self.assertEqual(transformed.shape, (2, 2, 3))  # HWC
        np.testing.assert_array_equal(transformed, _KNOWN_RGB)


if __name__ == "__main__":
    unittest.main()
