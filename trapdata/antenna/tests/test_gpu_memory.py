"""Unit tests for the worker's GPU memory management.

Covers the chunked-inference helper (``_predict_in_chunks``), the per-job
model factory that applies the GPU batch-size settings, and the default CUDA
allocator configuration. All tests use fake models and run on CPU; the
out-of-memory scenarios are simulated by raising the same exception types the
CUDA allocator raises.
"""

import os
from unittest import TestCase
from unittest.mock import MagicMock, patch

import torch

from trapdata.antenna.worker import (
    _DEFAULT_ALLOC_CONF,
    _init_job_models,
    _is_cuda_memory_error,
    _predict_in_chunks,
    _set_default_allocator_config,
)

_ALLOC_ENV_VARS = ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF")


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
    """The forward-pass peak must be bounded by the model's batch_size."""

    def _items(self, n: int) -> list[torch.Tensor]:
        return [torch.tensor(float(i)) for i in range(n)]

    def test_list_input_is_chunked_and_order_preserved(self):
        model = FakeModel(batch_size=4)
        results = _predict_in_chunks(model, self._items(10), stack_chunks=True)

        assert model.seen_chunk_sizes == [4, 4, 2]
        assert [float(r) for r in results] == [-float(i) for i in range(10)]

    def test_tensor_input_is_sliced(self):
        model = FakeModel(batch_size=4)
        results = _predict_in_chunks(model, torch.arange(10.0))

        assert model.seen_chunk_sizes == [4, 4, 2]
        assert [float(r) for r in results] == [-float(i) for i in range(10)]

    def test_single_chunk_when_batch_size_exceeds_items(self):
        model = FakeModel(batch_size=8)
        results = _predict_in_chunks(model, self._items(3), stack_chunks=True)

        assert model.seen_chunk_sizes == [3]
        assert len(results) == 3

    def test_oom_halves_chunk_size_and_retries(self):
        """A simulated allocation failure must shrink the chunk, not fail the batch."""
        model = FakeModel(batch_size=8, fail_first_n=1)
        results = _predict_in_chunks(model, self._items(8), stack_chunks=True)

        # The failed 8-item attempt is not recorded; the retry runs at 4.
        assert model.seen_chunk_sizes == [4, 4]
        assert [float(r) for r in results] == [-float(i) for i in range(8)]

    def test_oom_at_chunk_size_one_reraises(self):
        model = FakeModel(batch_size=1, fail_first_n=100)

        with self.assertRaises(torch.OutOfMemoryError):
            _predict_in_chunks(model, self._items(2), stack_chunks=True)

    def test_unrelated_runtime_error_propagates_without_retry(self):
        model = FakeModel(batch_size=4)
        model.predict_batch = MagicMock(side_effect=RuntimeError("size mismatch"))

        with self.assertRaises(RuntimeError):
            _predict_in_chunks(model, self._items(4), stack_chunks=True)
        # No retry: a non-memory error must fail on the first call.
        assert model.predict_batch.call_count == 1


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


class TestDefaultAllocatorConfig(TestCase):
    def test_sets_both_vars_when_unset(self):
        with patch.dict(os.environ):
            for var in _ALLOC_ENV_VARS:
                os.environ.pop(var, None)

            _set_default_allocator_config()

            for var in _ALLOC_ENV_VARS:
                assert os.environ[var] == _DEFAULT_ALLOC_CONF

    def test_operator_setting_is_never_overridden(self):
        for preset_var in _ALLOC_ENV_VARS:
            other_var = next(v for v in _ALLOC_ENV_VARS if v != preset_var)
            with patch.dict(os.environ):
                for var in _ALLOC_ENV_VARS:
                    os.environ.pop(var, None)
                os.environ[preset_var] = "max_split_size_mb:512"

                _set_default_allocator_config()

                # The preset value stays, and no default is layered on top of
                # it via the other variable name either.
                assert os.environ[preset_var] == "max_split_size_mb:512"
                assert other_var not in os.environ
