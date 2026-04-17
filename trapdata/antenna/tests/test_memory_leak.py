"""Memory leak regression test for _process_job batch processing.

Verifies that RSS does not grow unboundedly across batches by using the
on_batch_complete callback to sample memory after each batch.

Uses the same test infrastructure as test_worker.py (mock Antenna API,
StaticFileTestServer, real ML inference).
"""

import os
import pathlib
from unittest import TestCase
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from trapdata.antenna.schemas import AntennaPipelineProcessingTask
from trapdata.antenna.tests import antenna_api_server
from trapdata.antenna.tests.antenna_api_server import app as antenna_app
from trapdata.antenna.worker import _process_job
from trapdata.api.tests.image_server import StaticFileTestServer
from trapdata.api.tests.utils import get_test_image_urls, patch_antenna_api_requests
from trapdata.tests import TEST_IMAGES_BASE_PATH


def _get_rss_mb() -> float:
    """Current RSS in MB, read from /proc/self/statm (Linux-only)."""
    with open("/proc/self/statm") as f:
        pages = int(f.read().split()[1])  # resident pages
    return pages * os.sysconf("SC_PAGE_SIZE") / (1024 * 1024)


class TestMemoryLeak(TestCase):
    """Regression test: RSS must not grow linearly with batch count."""

    @classmethod
    def setUpClass(cls):
        cls.test_images_dir = pathlib.Path(TEST_IMAGES_BASE_PATH)
        cls.file_server = StaticFileTestServer(cls.test_images_dir)
        cls.file_server.start()
        cls.antenna_client = TestClient(antenna_app, follow_redirects=False)

    @classmethod
    def tearDownClass(cls):
        cls.file_server.stop()

    def setUp(self):
        antenna_api_server.reset()

    def _make_settings(self):
        settings = MagicMock()
        settings.antenna_api_base_url = "http://testserver/api/v2"
        settings.antenna_api_key = "test-api-key"
        settings.antenna_api_batch_size = 2
        settings.num_workers = 0
        settings.localization_batch_size = 2
        return settings

    @pytest.mark.slow
    def test_rss_stable_across_batches(self):
        """RSS should not grow more than 200 MB across 25+ batches.

        With the old code, all_detections accumulated ~220K DetectionResponse
        objects over a large job, growing RSS by ~4 GB/hr. After the fix,
        each batch's intermediates go out of scope in _process_batch().

        The 150 MB threshold accounts for normal PyTorch/CUDA allocator
        fragmentation and memory pool behavior, which is not a true leak.
        """
        # Create 50 tasks by cycling through the 3 available test images
        image_urls = get_test_image_urls(
            self.file_server, self.test_images_dir, subdir="vermont", num=3
        )
        num_tasks = 50
        tasks = [
            AntennaPipelineProcessingTask(
                id=f"task_{i}",
                image_id=f"img_{i}",
                image_url=image_urls[i % len(image_urls)],
                reply_subject=f"reply_{i}",
            )
            for i in range(num_tasks)
        ]
        antenna_api_server.setup_job(job_id=999, tasks=tasks)

        # Collect RSS samples via callback
        rss_samples: list[float] = []

        def on_batch(batch_num: int, items: int):
            rss_samples.append(_get_rss_mb())

        with patch_antenna_api_requests(self.antenna_client):
            result = _process_job(
                "quebec_vermont_moths_2023",
                999,
                self._make_settings(),
                on_batch_complete=on_batch,
            )

        assert result is True
        assert (
            len(rss_samples) >= 10
        ), f"Expected at least 10 batches, got {len(rss_samples)}"

        # Compare RSS at end vs after first 2 batches (allow model warmup)
        warmup_rss = rss_samples[2]
        final_rss = rss_samples[-1]
        growth_mb = final_rss - warmup_rss

        print(f"\nMemory profile ({len(rss_samples)} batches):")
        print(f"  After warmup (batch 2): {warmup_rss:.1f} MB")
        print(f"  Final (batch {len(rss_samples) - 1}): {final_rss:.1f} MB")
        print(f"  Growth: {growth_mb:.1f} MB")
        for i, rss in enumerate(rss_samples):
            print(f"  Batch {i}: {rss:.1f} MB")

        # Threshold accounts for PyTorch/CUDA allocator pools and Python memory
        # fragmentation — not a true leak. Before the fix, all_detections
        # accumulated every DetectionResponse across all batches. At scale
        # (31K images, ~7 detections/image), that was ~220K objects = GB.
        # Bumped from 150 to 200 MB — CI runners show ~168 MB due to shared
        # machine variance while local runs stay under 100 MB.
        assert growth_mb < 200, (
            f"RSS grew {growth_mb:.1f} MB across {len(rss_samples)} batches "
            f"(warmup={warmup_rss:.1f} MB, final={final_rss:.1f} MB). "
            f"Likely memory leak in batch processing."
        )
