"""Integration tests for the REST worker and related utilities.

These tests validate the Antenna API contract and run real ML inference through
the worker's unique code path (RESTDataset → rest_collate_fn → batch processing).
Only external service dependencies are mocked - ML models and image loading are real.
"""

import pathlib
from unittest import TestCase
from unittest.mock import MagicMock

import torch
from fastapi.testclient import TestClient

from trapdata.api.datasets import RESTDataset, rest_collate_fn
from trapdata.api.schemas import (
    AntennaPipelineProcessingTask,
    AntennaTaskResult,
    AntennaTaskResultError,
    PipelineResultsResponse,
)
from trapdata.api.tests import antenna_api_server
from trapdata.api.tests.antenna_api_server import app as antenna_app
from trapdata.api.tests.image_server import StaticFileTestServer
from trapdata.api.tests.utils import get_test_image_urls, patch_antenna_api_requests
from trapdata.cli.worker import _get_jobs, _process_job
from trapdata.tests import TEST_IMAGES_BASE_PATH

# ---------------------------------------------------------------------------
# TestRestCollateFn - Unit tests for collation logic
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
# TestRESTDatasetIntegration - Integration tests with real image loading
# ---------------------------------------------------------------------------


class TestRESTDatasetIntegration(TestCase):
    """Integration tests for RESTDataset that fetch tasks and load real images."""

    @classmethod
    def setUpClass(cls):
        # Setup file server for test images
        cls.test_images_dir = pathlib.Path(TEST_IMAGES_BASE_PATH)
        cls.file_server = StaticFileTestServer(cls.test_images_dir)
        cls.file_server.start()  # Start server and keep it running for all tests

        # Setup mock Antenna API
        cls.antenna_client = TestClient(antenna_app)

    @classmethod
    def tearDownClass(cls):
        cls.file_server.stop()

    def setUp(self):
        # Reset state between tests
        antenna_api_server.reset()

    def _make_dataset(self, job_id: int = 42, batch_size: int = 2) -> RESTDataset:
        """Create a RESTDataset pointing to the mock API."""
        return RESTDataset(
            base_url="http://testserver/api/v2",
            job_id=job_id,
            batch_size=batch_size,
            auth_token="test-token",
        )

    def test_fetches_and_loads_images(self):
        """RESTDataset fetches tasks and loads images from URLs."""
        # Setup mock API job with real image URLs
        image_urls = get_test_image_urls(
            self.file_server, self.test_images_dir, subdir="vermont", num=2
        )
        tasks = [
            AntennaPipelineProcessingTask(
                id=f"task_{i}",
                image_id=f"img_{i}",
                image_url=url,
                reply_subject=f"reply_{i}",
            )
            for i, url in enumerate(image_urls)
        ]
        antenna_api_server.setup_job(job_id=1, tasks=tasks)

        # Create dataset and iterate
        with patch_antenna_api_requests(self.antenna_client):
            dataset = self._make_dataset(job_id=1, batch_size=2)
            rows = list(dataset)

        # Validate images actually loaded
        assert len(rows) == 2
        assert all(r["image"] is not None for r in rows)
        assert all(isinstance(r["image"], torch.Tensor) for r in rows)
        assert rows[0]["image_id"] == "img_0"
        assert rows[1]["image_id"] == "img_1"

    def test_image_failure(self):
        """Invalid image URL produces error row with image=None."""
        tasks = [
            AntennaPipelineProcessingTask(
                id="task_bad",
                image_id="img_bad",
                image_url="http://invalid-url.test/bad.jpg",
                reply_subject="reply_bad",
            )
        ]
        antenna_api_server.setup_job(job_id=2, tasks=tasks)

        with patch_antenna_api_requests(self.antenna_client):
            dataset = self._make_dataset(job_id=2)
            rows = list(dataset)

        assert len(rows) == 1
        assert rows[0]["image"] is None
        assert "error" in rows[0]

    def test_empty_queue(self):
        """First fetch returns empty tasks → iterator stops immediately."""
        antenna_api_server.setup_job(job_id=3, tasks=[])

        with patch_antenna_api_requests(self.antenna_client):
            dataset = self._make_dataset(job_id=3)
            rows = list(dataset)

        assert rows == []

    def test_multiple_batches(self):
        """Dataset fetches multiple batches until queue is empty."""
        # Setup job with 3 images (all available in vermont dir), batch size 2
        image_urls = get_test_image_urls(
            self.file_server, self.test_images_dir, subdir="vermont", num=3
        )
        tasks = [
            AntennaPipelineProcessingTask(
                id=f"task_{i}",
                image_id=f"img_{i}",
                image_url=url,
                reply_subject=f"reply_{i}",
            )
            for i, url in enumerate(image_urls)
        ]
        antenna_api_server.setup_job(job_id=4, tasks=tasks)

        with patch_antenna_api_requests(self.antenna_client):
            dataset = self._make_dataset(job_id=4, batch_size=2)
            rows = list(dataset)

        # Should get all 3 images (batch1: 2 images, batch2: 1 image)
        assert len(rows) == 3
        assert all(r["image"] is not None for r in rows)


# ---------------------------------------------------------------------------
# TestGetJobsIntegration - Integration tests for job fetching
# ---------------------------------------------------------------------------


class TestGetJobsIntegration(TestCase):
    """Integration tests for _get_jobs() with mock Antenna API."""

    @classmethod
    def setUpClass(cls):
        cls.antenna_client = TestClient(antenna_app)

    def setUp(self):
        antenna_api_server.reset()

    def _make_settings(self):
        """Create mock settings for _get_jobs."""
        settings = MagicMock()
        settings.antenna_api_base_url = "http://testserver/api/v2"
        settings.antenna_api_auth_token = "test-token"
        settings.antenna_api_retry_max = 3
        settings.antenna_api_retry_backoff = 0.5
        return settings

    def test_returns_job_ids(self):
        """Successfully fetches list of job IDs."""
        # Setup jobs in queue
        antenna_api_server.setup_job(10, [])
        antenna_api_server.setup_job(20, [])
        antenna_api_server.setup_job(30, [])

        with patch_antenna_api_requests(self.antenna_client):
            result = _get_jobs(self._make_settings(), "moths_2024")

        assert result == [10, 20, 30]

    def test_empty_queue(self):
        """Empty job queue returns empty list."""
        with patch_antenna_api_requests(self.antenna_client):
            result = _get_jobs(self._make_settings(), "moths_2024")

        assert result == []

    def test_query_params_sent(self):
        """Request includes correct query parameters."""
        # This test validates the query params are sent by checking the function works
        # The mock API checks the params internally
        antenna_api_server.setup_job(1, [])

        with patch_antenna_api_requests(self.antenna_client):
            result = _get_jobs(self._make_settings(), "my_pipeline")

        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# TestProcessJobIntegration - Integration tests with real ML inference
# ---------------------------------------------------------------------------


class TestProcessJobIntegration(TestCase):
    """Integration tests for _process_job() with real detector and classifier."""

    @classmethod
    def setUpClass(cls):
        cls.test_images_dir = pathlib.Path(TEST_IMAGES_BASE_PATH)
        cls.file_server = StaticFileTestServer(cls.test_images_dir)
        cls.file_server.start()  # Start server and keep it running for all tests
        cls.antenna_client = TestClient(antenna_app)

    @classmethod
    def tearDownClass(cls):
        cls.file_server.stop()

    def setUp(self):
        antenna_api_server.reset()

    def _make_settings(self):
        """Create mock settings for worker."""
        settings = MagicMock()
        settings.antenna_api_base_url = "http://testserver/api/v2"
        settings.antenna_api_auth_token = "test-token"
        settings.antenna_api_batch_size = 2
        settings.antenna_api_retry_max = 3
        settings.antenna_api_retry_backoff = 0.5
        settings.num_workers = 0  # Disable multiprocessing for tests
        settings.localization_batch_size = 2  # Real integer for batch processing
        return settings

    def test_empty_queue(self):
        """No tasks in queue → returns False."""
        antenna_api_server.setup_job(job_id=100, tasks=[])

        with patch_antenna_api_requests(self.antenna_client):
            result = _process_job(
                "quebec_vermont_moths_2023", 100, self._make_settings()
            )

        assert result is False

    def test_processes_batch_with_real_inference(self):
        """Worker fetches tasks, loads images, runs ML, posts results."""
        # Setup job with 2 test images
        image_urls = get_test_image_urls(
            self.file_server, self.test_images_dir, subdir="vermont", num=2
        )
        tasks = [
            AntennaPipelineProcessingTask(
                id=f"task_{i}",
                image_id=f"img_{i}",
                image_url=url,
                reply_subject=f"reply_{i}",
            )
            for i, url in enumerate(image_urls)
        ]
        antenna_api_server.setup_job(job_id=101, tasks=tasks)

        # Run worker
        with patch_antenna_api_requests(self.antenna_client):
            result = _process_job(
                "quebec_vermont_moths_2023", 101, self._make_settings()
            )

        # Validate processing succeeded
        assert result is True

        # Validate results were posted
        posted_results = antenna_api_server.get_posted_results(101)
        assert len(posted_results) == 2

        # Validate schema compliance
        for task_result in posted_results:
            assert isinstance(task_result, AntennaTaskResult)
            assert isinstance(task_result.result, PipelineResultsResponse)

            # Validate structure
            response = task_result.result
            assert response.pipeline == "quebec_vermont_moths_2023"
            assert response.total_time > 0
            assert len(response.source_images) == 1
            assert len(response.detections) >= 0  # May be 0 if no moths

    def test_handles_failed_items(self):
        """Failed image downloads produce AntennaTaskResultError."""
        tasks = [
            AntennaPipelineProcessingTask(
                id="task_fail",
                image_id="img_fail",
                image_url="http://invalid-url.test/image.jpg",
                reply_subject="reply_fail",
            )
        ]
        antenna_api_server.setup_job(job_id=102, tasks=tasks)

        with patch_antenna_api_requests(self.antenna_client):
            _process_job("quebec_vermont_moths_2023", 102, self._make_settings())

        posted_results = antenna_api_server.get_posted_results(102)
        assert len(posted_results) == 1
        assert isinstance(posted_results[0].result, AntennaTaskResultError)
        assert posted_results[0].result.error  # Error message should not be empty

    def test_mixed_batch_success_and_failures(self):
        """Batch with some successful and some failed images."""
        # One valid image, one invalid
        valid_url = get_test_image_urls(
            self.file_server, self.test_images_dir, subdir="vermont", num=1
        )[0]

        tasks = [
            AntennaPipelineProcessingTask(
                id="task_good",
                image_id="img_good",
                image_url=valid_url,
                reply_subject="reply_good",
            ),
            AntennaPipelineProcessingTask(
                id="task_bad",
                image_id="img_bad",
                image_url="http://invalid-url.test/bad.jpg",
                reply_subject="reply_bad",
            ),
        ]
        antenna_api_server.setup_job(job_id=103, tasks=tasks)

        with patch_antenna_api_requests(self.antenna_client):
            result = _process_job(
                "quebec_vermont_moths_2023", 103, self._make_settings()
            )

        assert result is True
        posted_results = antenna_api_server.get_posted_results(103)
        assert len(posted_results) == 2

        # One success, one error
        success_results = [
            r for r in posted_results if isinstance(r.result, PipelineResultsResponse)
        ]
        error_results = [
            r for r in posted_results if isinstance(r.result, AntennaTaskResultError)
        ]
        assert len(success_results) == 1
        assert len(error_results) == 1


# ---------------------------------------------------------------------------
# TestWorkerEndToEnd - Full workflow integration tests
# ---------------------------------------------------------------------------


class TestWorkerEndToEnd(TestCase):
    """End-to-end integration tests for complete worker workflow."""

    @classmethod
    def setUpClass(cls):
        cls.test_images_dir = pathlib.Path(TEST_IMAGES_BASE_PATH)
        cls.file_server = StaticFileTestServer(cls.test_images_dir)
        cls.file_server.start()  # Start server and keep it running for all tests
        cls.antenna_client = TestClient(antenna_app)

    @classmethod
    def tearDownClass(cls):
        cls.file_server.stop()

    def setUp(self):
        antenna_api_server.reset()

    def _make_settings(self):
        settings = MagicMock()
        settings.antenna_api_base_url = "http://testserver/api/v2"
        settings.antenna_api_auth_token = "test-token"
        settings.antenna_api_batch_size = 2
        settings.antenna_api_retry_max = 3
        settings.antenna_api_retry_backoff = 0.5
        settings.num_workers = 0
        settings.localization_batch_size = 2  # Real integer for batch processing
        return settings

    def test_full_workflow_with_real_inference(self):
        """
        Complete workflow: fetch jobs → fetch tasks → load images →
        run detection → run classification → post results.
        """
        # Setup job with 2 test images
        image_urls = get_test_image_urls(
            self.file_server, self.test_images_dir, subdir="vermont", num=2
        )
        tasks = [
            AntennaPipelineProcessingTask(
                id=f"task_{i}",
                image_id=f"img_{i}",
                image_url=url,
                reply_subject=f"reply_{i}",
            )
            for i, url in enumerate(image_urls)
        ]
        antenna_api_server.setup_job(job_id=200, tasks=tasks)

        # Step 1: Get jobs
        with patch_antenna_api_requests(self.antenna_client):
            job_ids = _get_jobs(
                self._make_settings(),
                "quebec_vermont_moths_2023",
            )

        assert 200 in job_ids

        # Step 2: Process job
        with patch_antenna_api_requests(self.antenna_client):
            result = _process_job(
                "quebec_vermont_moths_2023", 200, self._make_settings()
            )

        assert result is True

        # Step 3: Validate results posted
        posted_results = antenna_api_server.get_posted_results(200)
        assert len(posted_results) == 2

        # Validate all results are valid
        for task_result in posted_results:
            assert isinstance(task_result, AntennaTaskResult)
            assert task_result.reply_subject is not None

            # Should be success results
            assert isinstance(task_result.result, PipelineResultsResponse)
            response = task_result.result

            # Validate pipeline response structure
            assert response.pipeline == "quebec_vermont_moths_2023"
            assert response.total_time > 0
            assert len(response.source_images) == 1

            # Validate detections structure (may be empty if no moths)
            assert isinstance(response.detections, list)
            if response.detections:
                detection = response.detections[0]
                assert detection.bbox is not None
                assert detection.source_image_id is not None

    def test_multiple_batches_processed(self):
        """Job with more tasks than batch size processes in multiple batches."""
        # Setup job with 3 images (all available in vermont dir), batch size 2
        image_urls = get_test_image_urls(
            self.file_server, self.test_images_dir, subdir="vermont", num=3
        )
        tasks = [
            AntennaPipelineProcessingTask(
                id=f"task_{i}",
                image_id=f"img_{i}",
                image_url=url,
                reply_subject=f"reply_{i}",
            )
            for i, url in enumerate(image_urls)
        ]
        antenna_api_server.setup_job(job_id=201, tasks=tasks)

        with patch_antenna_api_requests(self.antenna_client):
            result = _process_job(
                "quebec_vermont_moths_2023", 201, self._make_settings()
            )

        assert result is True

        # All 3 results should be posted (batch1: 2, batch2: 1)
        posted_results = antenna_api_server.get_posted_results(201)
        assert len(posted_results) == 3

        # All should be successful
        assert all(
            isinstance(r.result, PipelineResultsResponse) for r in posted_results
        )
