"""Regression test: the worker's open pipe count must stay flat from job to job.

Each job builds a new DataLoader with worker subprocesses. If any part of that
lifecycle keeps a pipe open after the job ends, the worker process runs out of
file descriptors after some days and every API call fails (#145).

The fake Antenna API runs as a real HTTP server in a thread, because DataLoader
worker subprocesses cannot reach the in-process ``TestClient``.

The leak in #145 needs pinned memory, which PyTorch only uses when CUDA (or
another accelerator) is available. Without one, this test still checks the
subprocess lifecycle, and ``test_dataloader_hygiene.py`` checks the setting
that caused the leak.
"""

import pathlib
import socket
import sys
import threading
import time
from types import SimpleNamespace
from unittest import TestCase

import pytest
import uvicorn

from trapdata.antenna.datasets import get_rest_dataloader
from trapdata.antenna.schemas import AntennaPipelineProcessingTask
from trapdata.antenna.tests import antenna_api_server
from trapdata.antenna.tests.antenna_api_server import app as antenna_app
from trapdata.antenna.worker import _process_job
from trapdata.api.tests.image_server import StaticFileTestServer
from trapdata.api.tests.utils import get_test_image_urls
from trapdata.tests import TEST_IMAGES_BASE_PATH

NUM_WORKERS = 2
NUM_JOBS = 5


def _count_pipes() -> int:
    """Number of pipe FDs open in this process (Linux /proc only)."""
    fd_dir = pathlib.Path("/proc/self/fd")
    count = 0
    for fd in fd_dir.iterdir():
        try:
            if fd.readlink().name.startswith("pipe:"):
                count += 1
        except OSError:
            continue  # The FD was closed while we listed the directory.
    return count


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="counts FDs in /proc/self/fd"
)
class TestDataLoaderPipeLeak(TestCase):
    @classmethod
    def setUpClass(cls):
        port = _free_port()
        cls.api_server = uvicorn.Server(
            uvicorn.Config(antenna_app, host="127.0.0.1", port=port, log_level="error")
        )
        threading.Thread(target=cls.api_server.run, daemon=True).start()
        deadline = time.monotonic() + 10
        while not cls.api_server.started:
            if time.monotonic() > deadline:
                raise RuntimeError("fake Antenna API server did not start")
            time.sleep(0.05)
        cls.base_url = f"http://127.0.0.1:{port}/api/v2"

        cls.test_images_dir = pathlib.Path(TEST_IMAGES_BASE_PATH)
        cls.file_server = StaticFileTestServer(cls.test_images_dir)
        cls.file_server.start()

    @classmethod
    def tearDownClass(cls):
        cls.file_server.stop()
        cls.api_server.should_exit = True

    def setUp(self):
        antenna_api_server.reset()

    def _settings(self) -> SimpleNamespace:
        return SimpleNamespace(
            antenna_api_base_url=self.base_url,
            antenna_api_auth_token="test-token",
            antenna_api_batch_size=2,
            num_workers=NUM_WORKERS,
        )

    def _assert_pipes_flat(self, run_job):
        # The first job may open pipes that stay open for the life of the
        # process (for example the CUDA runtime's), so measure after it.
        run_job(0)
        before = _count_pipes()
        for job_id in range(1, NUM_JOBS + 1):
            run_job(job_id)
        after = _count_pipes()
        # The leak in #145 added 2 pipes per worker per job: 20 here.
        self.assertLessEqual(
            after - before,
            1,
            f"pipe FDs grew from {before} to {after} over {NUM_JOBS} jobs",
        )

    def test_jobs_with_images_do_not_leak_pipes(self):
        image_urls = get_test_image_urls(
            self.file_server, self.test_images_dir, subdir="vermont", num=2
        )
        settings = self._settings()

        def run_job(job_id: int):
            tasks = [
                AntennaPipelineProcessingTask(
                    id=f"task_{job_id}_{i}",
                    image_id=f"img_{job_id}_{i}",
                    image_url=url,
                    reply_subject=f"reply_{job_id}_{i}",
                )
                for i, url in enumerate(image_urls)
            ]
            antenna_api_server.setup_job(job_id, tasks)
            batches = list(get_rest_dataloader(job_id=job_id, settings=settings))
            self.assertTrue(batches, "the DataLoader returned no batches")

        self._assert_pipes_flat(run_job)

    def test_polling_drained_jobs_does_not_leak_pipes(self):
        """A job with no tasks left still starts DataLoader workers. The worker
        does this on every poll while a job is listed, so it is the common case
        on an idle worker."""
        settings = self._settings()

        def run_job(job_id: int):
            antenna_api_server.setup_job(job_id, [])
            did_work = _process_job("moth_binary", job_id, settings)
            self.assertFalse(did_work)

        self._assert_pipes_flat(run_job)
