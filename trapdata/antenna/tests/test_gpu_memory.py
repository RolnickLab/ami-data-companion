"""Unit tests for the worker's GPU memory management.

Covers the chunked-inference helpers (``_predict_in_chunks`` and
``_classify_crops_in_chunks``) and the per-job model factory that applies
the GPU batch-size settings. All tests use fake models and run on CPU; the
out-of-memory scenarios are simulated by raising the same exception types
the CUDA allocator raises.
"""

from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import MagicMock, patch

import torch

from trapdata.antenna.worker import (
    _classify_crops_in_chunks,
    _init_job_models,
    _is_cuda_memory_error,
    _predict_in_chunks,
)


class FakeModel:
    """Inference stub that records the chunk sizes ``predict_batch`` receives.

    ``post_process_batch`` negates each item so tests can verify that outputs
    pass through post-processing and stay in input order. ``fail_first_n``
    makes the first N ``predict_batch`` calls raise a simulated CUDA
    out-of-memory error before any work is recorded.
    """

    def __init__(self, batch_size: int, fail_first_n: int = 0):
        self.batch_size = batch_size
        self.seen_chunk_sizes: list[int] = []
        self._failures_remaining = fail_first_n

    def predict_batch(self, chunk):
        if self._failures_remaining > 0:
            self._failures_remaining -= 1
            raise torch.OutOfMemoryError("CUDA out of memory. (simulated)")
        self.seen_chunk_sizes.append(len(chunk))
        return list(chunk)

    def post_process_batch(self, output):
        return [-item for item in output]


class TestPredictInChunks(TestCase):
    """The detector's forward-pass peak must be bounded by its batch_size."""

    def _items(self, n: int) -> list[torch.Tensor]:
        return [torch.tensor(float(i)) for i in range(n)]

    def test_list_input_is_chunked_and_order_preserved(self):
        model = FakeModel(batch_size=4)
        results = _predict_in_chunks(model, self._items(10))

        assert model.seen_chunk_sizes == [4, 4, 2]
        assert [float(r) for r in results] == [-float(i) for i in range(10)]

    def test_tensor_input_is_sliced(self):
        model = FakeModel(batch_size=4)
        results = _predict_in_chunks(model, torch.arange(10.0))

        assert model.seen_chunk_sizes == [4, 4, 2]
        assert [float(r) for r in results] == [-float(i) for i in range(10)]

    def test_single_chunk_when_batch_size_exceeds_items(self):
        model = FakeModel(batch_size=8)
        results = _predict_in_chunks(model, self._items(3))

        assert model.seen_chunk_sizes == [3]
        assert len(results) == 3

    def test_oom_halves_chunk_size_and_retries(self):
        """A simulated allocation failure must shrink the chunk, not fail the batch."""
        model = FakeModel(batch_size=8, fail_first_n=1)
        results = _predict_in_chunks(model, self._items(8))

        # The failed 8-item attempt is not recorded; the retry runs at 4.
        assert model.seen_chunk_sizes == [4, 4]
        assert [float(r) for r in results] == [-float(i) for i in range(8)]

    def test_oom_at_chunk_size_one_reraises(self):
        model = FakeModel(batch_size=1, fail_first_n=100)

        with self.assertRaises(torch.OutOfMemoryError):
            _predict_in_chunks(model, self._items(2))

    def test_unrelated_runtime_error_propagates_without_retry(self):
        model = FakeModel(batch_size=4)
        model.predict_batch = MagicMock(side_effect=RuntimeError("size mismatch"))

        with self.assertRaises(RuntimeError):
            _predict_in_chunks(model, self._items(4))
        # No retry: a non-memory error must fail on the first call.
        assert model.predict_batch.call_count == 1


def _make_detection(image_id: str, x1: int, y1: int, x2: int, y2: int):
    """A minimal stand-in for DetectionResponse: just bbox and source image id."""
    return SimpleNamespace(
        source_image_id=image_id,
        bbox=SimpleNamespace(x1=x1, y1=y1, x2=x2, y2=y2),
    )


class FakeCropClassifier(FakeModel):
    """FakeModel plus the transform hook ``_classify_crops_in_chunks`` needs.

    The transform trims each crop to its top-left pixel, whose value encodes
    the crop's x position in the test image — so predictions can be traced
    back to detections, and crops stack regardless of bbox size.
    """

    def get_transforms(self):
        return lambda crop: crop[:, :1, :1]

    def post_process_batch(self, output):
        # Identify each crop by the value at its top-left corner.
        return [float(item[0, 0, 0]) for item in output]


class TestClassifyCropsInChunks(TestCase):
    """Crop construction itself must be chunked, not just the forward pass.

    If all crops were built before inference, a dense batch (hundreds of
    detections per image) would allocate every crop tensor up front and the
    chunked forward pass would cap nothing.
    """

    def _image_tensors(self) -> dict[str, torch.Tensor]:
        # One 3x100x100 "image" whose pixel values encode the x coordinate,
        # so each crop is identifiable by its top-left corner value.
        gradient = torch.arange(100.0).repeat(3, 100, 1)
        return {"img_a": gradient}

    def test_chunked_and_mapped_back_to_detections(self):
        detections = [
            _make_detection("img_a", x1=x, y1=0, x2=x + 10, y2=10) for x in range(5)
        ]
        model = FakeCropClassifier(batch_size=2)

        predictions, valid_indices = _classify_crops_in_chunks(
            model, detections, self._image_tensors()
        )

        assert model.seen_chunk_sizes == [2, 2, 1]
        assert valid_indices == [0, 1, 2, 3, 4]
        # Each prediction carries its crop's top-left value = detection's x1.
        assert predictions == [0.0, 1.0, 2.0, 3.0, 4.0]

    def test_invalid_bboxes_are_skipped(self):
        detections = [
            _make_detection("img_a", x1=0, y1=0, x2=10, y2=10),
            _make_detection("img_a", x1=5, y1=5, x2=5, y2=10),  # zero width
            _make_detection("img_a", x1=20, y1=0, x2=30, y2=10),
        ]
        model = FakeCropClassifier(batch_size=4)

        predictions, valid_indices = _classify_crops_in_chunks(
            model, detections, self._image_tensors()
        )

        assert valid_indices == [0, 2]
        assert predictions == [0.0, 20.0]

    def test_empty_detections(self):
        model = FakeCropClassifier(batch_size=4)
        predictions, valid_indices = _classify_crops_in_chunks(
            model, [], self._image_tensors()
        )

        assert predictions == []
        assert valid_indices == []

    def test_oom_halves_chunk_size_and_retries(self):
        detections = [
            _make_detection("img_a", x1=x, y1=0, x2=x + 10, y2=10) for x in range(4)
        ]
        model = FakeCropClassifier(batch_size=4, fail_first_n=1)

        predictions, valid_indices = _classify_crops_in_chunks(
            model, detections, self._image_tensors()
        )

        assert model.seen_chunk_sizes == [2, 2]
        assert predictions == [0.0, 1.0, 2.0, 3.0]


class TestIsCudaMemoryError(TestCase):
    def test_out_of_memory_error_is_detected(self):
        assert _is_cuda_memory_error(torch.OutOfMemoryError("CUDA out of memory."))

    def test_cublas_alloc_failure_is_detected(self):
        exc = RuntimeError(
            "CUDA error: CUBLAS_STATUS_ALLOC_FAILED when calling cublasCreate(handle)"
        )
        assert _is_cuda_memory_error(exc)

    def test_other_runtime_errors_are_not_detected(self):
        assert not _is_cuda_memory_error(RuntimeError("size mismatch"))


class TestInitJobModels(TestCase):
    """The worker must apply the GPU batch-size settings to its models.

    Guards against the regression where the worker constructed models without
    a batch size, so the whole API fetch batch went through one forward pass
    and the AMI_LOCALIZATION_BATCH_SIZE / AMI_CLASSIFICATION_BATCH_SIZE
    settings were silently ignored.
    """

    def _settings(self) -> MagicMock:
        settings = MagicMock()
        settings.localization_batch_size = 8
        settings.classification_batch_size = 20
        return settings

    @patch("trapdata.antenna.worker.MothClassifierBinary")
    @patch("trapdata.antenna.worker.APIMothDetector")
    def test_batch_sizes_from_settings(self, mock_detector, mock_binary):
        classifier_class = MagicMock()

        classifier, detector, binary_filter = _init_job_models(
            classifier_class, use_binary_filter=True, settings=self._settings()
        )

        classifier_class.assert_called_once_with(
            source_images=[], detections=[], batch_size=20
        )
        mock_detector.assert_called_once_with([], batch_size=8)
        mock_binary.assert_called_once_with(
            source_images=[], detections=[], terminal=False, batch_size=20
        )
        assert classifier is classifier_class.return_value
        assert detector is mock_detector.return_value
        assert binary_filter is mock_binary.return_value

    @patch("trapdata.antenna.worker.MothClassifierBinary")
    @patch("trapdata.antenna.worker.APIMothDetector")
    def test_no_binary_filter_when_not_used(self, mock_detector, mock_binary):
        _, _, binary_filter = _init_job_models(
            MagicMock(), use_binary_filter=False, settings=self._settings()
        )

        assert binary_filter is None
        mock_binary.assert_not_called()
