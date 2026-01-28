"""Tests for the REST worker and related utilities.

All ML models and network calls are mocked so tests run without GPU or network access.
"""

import datetime
from unittest.mock import MagicMock, patch

import requests
import torch

from trapdata.api.datasets import RESTDataset, rest_collate_fn
from trapdata.api.schemas import (
    AntennaTaskResult,
    AntennaTaskResultError,
    PipelineResultsResponse,
)
from trapdata.cli.worker import _get_jobs, _process_job

# ---------------------------------------------------------------------------
# TestRestCollateFn
# ---------------------------------------------------------------------------


class TestRestCollateFn:
    """Tests for rest_collate_fn which separates successful/failed items."""

    def test_all_successful(self):
        batch = [
            {
                "image": torch.rand(3, 64, 64),
                "reply_subject": "subj1",
                "image_id": "img1",
                "image_url": "http://example.com/1.jpg",
            },
            {
                "image": torch.rand(3, 64, 64),
                "reply_subject": "subj2",
                "image_id": "img2",
                "image_url": "http://example.com/2.jpg",
            },
        ]
        result = rest_collate_fn(batch)

        assert "images" in result
        assert result["images"].shape == (2, 3, 64, 64)
        assert result["image_ids"] == ["img1", "img2"]
        assert result["reply_subjects"] == ["subj1", "subj2"]
        assert result["failed_items"] == []

    def test_all_failed(self):
        batch = [
            {
                "image": None,
                "reply_subject": "subj1",
                "image_id": "img1",
                "image_url": "http://example.com/1.jpg",
                "error": "download failed",
            },
            {
                "image": None,
                "reply_subject": "subj2",
                "image_id": "img2",
                "image_url": "http://example.com/2.jpg",
                "error": "timeout",
            },
        ]
        result = rest_collate_fn(batch)

        assert "images" not in result
        assert result["image_ids"] == []
        assert result["reply_subjects"] == []
        assert len(result["failed_items"]) == 2
        assert result["failed_items"][0]["image_id"] == "img1"
        assert result["failed_items"][1]["error"] == "timeout"

    def test_mixed(self):
        batch = [
            {
                "image": torch.rand(3, 64, 64),
                "reply_subject": "subj1",
                "image_id": "img1",
                "image_url": "http://example.com/1.jpg",
            },
            {
                "image": None,
                "reply_subject": "subj2",
                "image_id": "img2",
                "image_url": "http://example.com/2.jpg",
                "error": "404",
            },
        ]
        result = rest_collate_fn(batch)

        assert result["images"].shape == (1, 3, 64, 64)
        assert result["image_ids"] == ["img1"]
        assert len(result["failed_items"]) == 1
        assert result["failed_items"][0]["image_id"] == "img2"

    def test_single_item(self):
        batch = [
            {
                "image": torch.rand(3, 32, 32),
                "reply_subject": "subj1",
                "image_id": "img1",
                "image_url": "http://example.com/1.jpg",
            },
        ]
        result = rest_collate_fn(batch)

        assert result["images"].shape == (1, 3, 32, 32)
        assert result["image_ids"] == ["img1"]
        assert result["failed_items"] == []


# ---------------------------------------------------------------------------
# TestRESTDatasetIteration
# ---------------------------------------------------------------------------


class TestRESTDatasetIteration:
    """Tests for RESTDataset.__iter__() with mocked network calls."""

    def _make_dataset(self, **kwargs):
        defaults = {
            "base_url": "http://api.test/api/v2",
            "job_id": 42,
            "batch_size": 2,
            "auth_token": "test-token",
        }
        defaults.update(kwargs)
        return RESTDataset(**defaults)

    @patch("trapdata.api.datasets.get_http_session")
    def test_normal_iteration(self, mock_get_session):
        """Fetch tasks, load images, yield rows, then empty stops iteration."""
        # First call: return tasks; second call: image download; etc.
        tasks_response = MagicMock()
        tasks_response.status_code = 200
        tasks_response.json.return_value = {
            "tasks": [
                {
                    "id": "t1",
                    "image_id": "img1",
                    "image_url": "http://images.test/1.jpg",
                    "reply_subject": "reply1",
                },
            ]
        }
        tasks_response.raise_for_status = MagicMock()

        # Create a small valid image for download
        import io

        from PIL import Image

        buf = io.BytesIO()
        Image.new("RGB", (64, 64), color="red").save(buf, format="PNG")
        image_bytes = buf.getvalue()

        image_response = MagicMock()
        image_response.status_code = 200
        image_response.content = image_bytes
        image_response.raise_for_status = MagicMock()

        empty_response = MagicMock()
        empty_response.status_code = 200
        empty_response.json.return_value = {"tasks": []}
        empty_response.raise_for_status = MagicMock()

        # Create a mock session
        mock_session = MagicMock()
        mock_session.get.side_effect = [tasks_response, image_response, empty_response]
        mock_get_session.return_value = mock_session

        ds = self._make_dataset()
        rows = list(ds)

        assert len(rows) == 1
        assert rows[0]["image_id"] == "img1"
        assert rows[0]["image"] is not None
        assert isinstance(rows[0]["image"], torch.Tensor)

    @patch("trapdata.api.datasets.get_http_session")
    def test_image_failure(self, mock_get_session):
        """Image download returns 404 → row has error, image is None."""
        tasks_response = MagicMock()
        tasks_response.json.return_value = {
            "tasks": [
                {
                    "id": "t1",
                    "image_id": "img1",
                    "image_url": "http://images.test/bad.jpg",
                    "reply_subject": "reply1",
                },
            ]
        }
        tasks_response.raise_for_status = MagicMock()

        image_response = MagicMock()
        image_response.raise_for_status.side_effect = requests.HTTPError("404")

        empty_response = MagicMock()
        empty_response.json.return_value = {"tasks": []}
        empty_response.raise_for_status = MagicMock()

        # Create a mock session
        mock_session = MagicMock()
        mock_session.get.side_effect = [tasks_response, image_response, empty_response]
        mock_get_session.return_value = mock_session

        ds = self._make_dataset()
        rows = list(ds)

        assert len(rows) == 1
        assert rows[0]["image"] is None
        assert "error" in rows[0]

    @patch("trapdata.api.datasets.get_http_session")
    def test_empty_queue(self, mock_get_session):
        """First fetch returns empty → iterator stops immediately."""
        empty_response = MagicMock()
        empty_response.json.return_value = {"tasks": []}
        empty_response.raise_for_status = MagicMock()

        # Create a mock session
        mock_session = MagicMock()
        mock_session.get.return_value = empty_response
        mock_get_session.return_value = mock_session

        ds = self._make_dataset()
        rows = list(ds)

        assert rows == []

    @patch("trapdata.api.datasets.get_http_session")
    def test_fetch_failure_stops_iteration(self, mock_get_session):
        """After max retries exhausted, iterator stops (no infinite loop)."""
        # Create a mock session that always fails
        mock_session = MagicMock()
        mock_session.get.side_effect = requests.RequestException("connection failed")
        mock_get_session.return_value = mock_session

        ds = self._make_dataset()
        rows = list(ds)

        # Iterator should stop after failure (not infinite loop)
        assert rows == []
        # Verify fetch was attempted
        assert mock_session.get.called


# ---------------------------------------------------------------------------
# TestGetJobs
# ---------------------------------------------------------------------------


class TestGetJobs:
    """Tests for _get_jobs() which fetches job IDs from the API."""

    def _make_settings(self):
        settings = MagicMock()
        settings.antenna_api_base_url = "http://api.test/api/v2"
        settings.antenna_api_auth_token = "mytoken"
        settings.antenna_api_retry_max = 3
        settings.antenna_api_retry_backoff = 0.5
        return settings

    @patch("trapdata.cli.worker.get_http_session")
    def test_returns_job_ids(self, mock_get_session):
        response = MagicMock()
        response.json.return_value = {"results": [{"id": 10}, {"id": 20}, {"id": 30}]}
        response.raise_for_status = MagicMock()

        mock_session = MagicMock()
        mock_session.get.return_value = response
        mock_get_session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_get_session.return_value.__exit__ = MagicMock(return_value=False)

        settings = self._make_settings()
        result = _get_jobs(settings, "moths_2024")
        assert result == [10, 20, 30]

    @patch("trapdata.cli.worker.get_http_session")
    def test_auth_header(self, mock_get_session):
        response = MagicMock()
        response.json.return_value = {"results": []}
        response.raise_for_status = MagicMock()

        mock_session = MagicMock()
        mock_session.get.return_value = response
        mock_get_session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_get_session.return_value.__exit__ = MagicMock(return_value=False)

        settings = self._make_settings()
        settings.antenna_api_auth_token = "secret-token"
        _get_jobs(settings, "pipeline1")

        # Verify auth_token was passed to get_http_session
        mock_get_session.assert_called_once()
        call_kwargs = mock_get_session.call_args[1]
        assert call_kwargs["auth_token"] == "secret-token"

    @patch("trapdata.cli.worker.get_http_session")
    def test_query_params(self, mock_get_session):
        response = MagicMock()
        response.json.return_value = {"results": []}
        response.raise_for_status = MagicMock()

        mock_session = MagicMock()
        mock_session.get.return_value = response
        mock_get_session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_get_session.return_value.__exit__ = MagicMock(return_value=False)

        settings = self._make_settings()
        _get_jobs(settings, "my_pipeline")

        call_kwargs = mock_session.get.call_args[1]
        params = call_kwargs["params"]
        assert params["pipeline__slug"] == "my_pipeline"
        assert params["ids_only"] == 1
        assert params["incomplete_only"] == 1

    @patch("trapdata.cli.worker.get_http_session")
    def test_network_error(self, mock_get_session):
        mock_session = MagicMock()
        mock_session.get.side_effect = requests.RequestException("timeout")
        mock_get_session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_get_session.return_value.__exit__ = MagicMock(return_value=False)

        settings = self._make_settings()
        result = _get_jobs(settings, "pipeline1")
        assert result == []

    @patch("trapdata.cli.worker.get_http_session")
    def test_invalid_response(self, mock_get_session):
        response = MagicMock()
        response.raise_for_status = MagicMock()
        response.json.return_value = {"unexpected": "format"}

        mock_session = MagicMock()
        mock_session.get.return_value = response
        mock_get_session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_get_session.return_value.__exit__ = MagicMock(return_value=False)

        settings = self._make_settings()
        result = _get_jobs(settings, "pipeline1")
        assert result == []


# ---------------------------------------------------------------------------
# TestProcessJob
# ---------------------------------------------------------------------------


class TestProcessJob:
    """Tests for _process_job() with mocked dataloader and models."""

    def _make_settings(self):
        settings = MagicMock()
        settings.antenna_api_base_url = "http://api.test/api/v2"
        settings.antenna_api_auth_token = "test-token"
        settings.antenna_api_batch_size = 4
        settings.antenna_api_retry_max = 3
        settings.antenna_api_retry_backoff = 0.5
        settings.num_workers = 0
        return settings

    def _make_batch(
        self,
        num_images=2,
        image_size=(3, 128, 128),
        failed_items=None,
    ):
        """Create a fake batch dict as produced by rest_collate_fn."""
        batch = {
            "images": torch.rand(num_images, *image_size),
            "image_ids": [f"img{i}" for i in range(num_images)],
            "reply_subjects": [f"reply{i}" for i in range(num_images)],
            "image_urls": [f"http://img.test/{i}.jpg" for i in range(num_images)],
            "failed_items": failed_items or [],
        }
        return batch

    @patch("trapdata.cli.worker.post_batch_results")
    @patch("trapdata.cli.worker.APIMothDetector")
    @patch("trapdata.cli.worker.CLASSIFIER_CHOICES", {"moths_2024": MagicMock})
    @patch("trapdata.cli.worker.get_rest_dataloader")
    def test_processes_batch_and_posts_results(
        self, mock_loader_fn, mock_detector_cls, mock_post
    ):
        """Batch with images → detection + classification → post_batch_results called."""
        batch = self._make_batch(num_images=2)
        mock_loader_fn.return_value = [batch]

        # Mock detector
        mock_detector = MagicMock()
        mock_detector_cls.return_value = mock_detector
        mock_detector.predict_batch.return_value = [
            {"boxes": torch.tensor([[10, 10, 50, 50]]), "scores": torch.tensor([0.9])},
            {"boxes": torch.tensor([[20, 20, 60, 60]]), "scores": torch.tensor([0.8])},
        ]
        mock_detector.post_process_batch.return_value = iter(
            mock_detector.predict_batch.return_value
        )
        # After save_results, detector.results has DetectionResponse objects
        from trapdata.api.schemas import (
            AlgorithmReference,
            BoundingBox,
            DetectionResponse,
        )

        det1 = DetectionResponse(
            source_image_id="img0",
            bbox=BoundingBox(x1=10, y1=10, x2=50, y2=50),
            algorithm=AlgorithmReference(name="detector", key="det_v1"),
            timestamp=datetime.datetime.now(),
        )
        det2 = DetectionResponse(
            source_image_id="img1",
            bbox=BoundingBox(x1=20, y1=20, x2=60, y2=60),
            algorithm=AlgorithmReference(name="detector", key="det_v1"),
            timestamp=datetime.datetime.now(),
        )
        mock_detector.results = [det1, det2]

        # Mock classifier
        mock_classifier_cls = MagicMock()
        with patch.dict(
            "trapdata.cli.worker.CLASSIFIER_CHOICES",
            {"moths_2024": mock_classifier_cls},
        ):
            mock_classifier = MagicMock()
            mock_classifier_cls.return_value = mock_classifier
            mock_classifier.predict_batch.return_value = [{"scores": [0.95]}]
            mock_classifier.post_process_batch.return_value = [{"scores": [0.95]}]
            mock_classifier.update_detection_classification.return_value = det1

            mock_post.return_value = True

            result = _process_job("moths_2024", 1, self._make_settings())

        assert result is True
        mock_post.assert_called_once()
        # Verify AntennaTaskResult objects were passed
        call_args = mock_post.call_args
        batch_results = call_args[0][2]  # third positional arg
        assert len(batch_results) == 2
        assert all(isinstance(r, AntennaTaskResult) for r in batch_results)
        assert all(isinstance(r.result, PipelineResultsResponse) for r in batch_results)

    @patch("trapdata.cli.worker.post_batch_results")
    @patch("trapdata.cli.worker.APIMothDetector")
    @patch("trapdata.cli.worker.CLASSIFIER_CHOICES", {"moths_2024": MagicMock})
    @patch("trapdata.cli.worker.get_rest_dataloader")
    def test_handles_failed_items(self, mock_loader_fn, mock_detector_cls, mock_post):
        """Batch with failed_items → error results in posted payload."""
        failed_items = [
            {
                "reply_subject": "reply_fail",
                "image_id": "imgX",
                "error": "404 not found",
            },
        ]
        # Batch with 1 successful image + 1 failed
        batch = self._make_batch(num_images=1, failed_items=failed_items)
        mock_loader_fn.return_value = [batch]

        mock_detector = MagicMock()
        mock_detector_cls.return_value = mock_detector
        mock_detector.predict_batch.return_value = [
            {"boxes": torch.tensor([[5, 5, 30, 30]]), "scores": torch.tensor([0.7])},
        ]
        mock_detector.post_process_batch.return_value = iter(
            mock_detector.predict_batch.return_value
        )

        from trapdata.api.schemas import (
            AlgorithmReference,
            BoundingBox,
            DetectionResponse,
        )

        det = DetectionResponse(
            source_image_id="img0",
            bbox=BoundingBox(x1=5, y1=5, x2=30, y2=30),
            algorithm=AlgorithmReference(name="det", key="det_v1"),
            timestamp=datetime.datetime.now(),
        )
        mock_detector.results = [det]

        mock_classifier_cls = MagicMock()
        with patch.dict(
            "trapdata.cli.worker.CLASSIFIER_CHOICES",
            {"moths_2024": mock_classifier_cls},
        ):
            mock_classifier = MagicMock()
            mock_classifier_cls.return_value = mock_classifier
            mock_classifier.predict_batch.return_value = [{"scores": [0.9]}]
            mock_classifier.post_process_batch.return_value = [{"scores": [0.9]}]
            mock_classifier.update_detection_classification.return_value = det

            mock_post.return_value = True
            _process_job("moths_2024", 1, self._make_settings())

        batch_results = mock_post.call_args[0][2]
        # 1 success + 1 failure
        assert len(batch_results) == 2
        error_items = [
            r for r in batch_results if isinstance(r.result, AntennaTaskResultError)
        ]
        assert len(error_items) == 1
        assert error_items[0].result.error == "404 not found"
        assert error_items[0].reply_subject == "reply_fail"

    @patch("trapdata.cli.worker.get_rest_dataloader")
    def test_empty_loader(self, mock_loader_fn):
        """No batches → returns False."""
        mock_loader_fn.return_value = []

        result = _process_job("moths_2024", 1, self._make_settings())
        assert result is False

    @patch("trapdata.cli.worker.post_batch_results")
    @patch("trapdata.cli.worker.APIMothDetector")
    @patch("trapdata.cli.worker.CLASSIFIER_CHOICES", {"moths_2024": MagicMock})
    @patch("trapdata.cli.worker.get_rest_dataloader")
    def test_multiple_batches(self, mock_loader_fn, mock_detector_cls, mock_post):
        """Results posted per-batch (post called once per batch)."""
        batch1 = self._make_batch(num_images=1)
        batch2 = self._make_batch(num_images=1)
        mock_loader_fn.return_value = [batch1, batch2]

        mock_detector = MagicMock()
        mock_detector_cls.return_value = mock_detector
        mock_detector.predict_batch.return_value = [
            {"boxes": torch.tensor([[0, 0, 10, 10]]), "scores": torch.tensor([0.5])},
        ]
        mock_detector.post_process_batch.return_value = iter(
            mock_detector.predict_batch.return_value
        )

        from trapdata.api.schemas import (
            AlgorithmReference,
            BoundingBox,
            DetectionResponse,
        )

        det = DetectionResponse(
            source_image_id="img0",
            bbox=BoundingBox(x1=0, y1=0, x2=10, y2=10),
            algorithm=AlgorithmReference(name="det", key="det_v1"),
            timestamp=datetime.datetime.now(),
        )
        mock_detector.results = [det]

        mock_classifier_cls = MagicMock()
        with patch.dict(
            "trapdata.cli.worker.CLASSIFIER_CHOICES",
            {"moths_2024": mock_classifier_cls},
        ):
            mock_classifier = MagicMock()
            mock_classifier_cls.return_value = mock_classifier
            mock_classifier.predict_batch.return_value = [{"scores": [0.8]}]
            mock_classifier.post_process_batch.return_value = [{"scores": [0.8]}]
            mock_classifier.update_detection_classification.return_value = det

            mock_post.return_value = True
            _process_job("moths_2024", 1, self._make_settings())

        assert mock_post.call_count == 2
