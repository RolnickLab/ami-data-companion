"""Unit tests for the worker's drain-and-exit machinery.

The worker exits at a batch boundary — never mid-batch — when an operator
sends SIGUSR1, after ``AMI_WORKER_MAX_JOBS`` jobs, or when its resident
memory, sampled between jobs, exceeds ``AMI_WORKER_MAX_RSS_MB``. These tests
cover the drain state and signal handler, the between-jobs recycle checks
and their per-job memory log line, that ``_process_job`` stops cleanly at a
batch boundary while flushing pending result posts, and that the polling
loop exits after a drain request. No models are loaded; inference is mocked
out.
"""

import os
import signal
import unittest
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import MagicMock, patch

from trapdata.antenna.worker import (
    _after_job_check,
    _current_rss_bytes,
    _DrainRequest,
    _install_drain_handler,
    _process_job,
    _worker_loop,
)

_HAS_SIGUSR1 = hasattr(signal, "SIGUSR1")


class TestDrainRequest(TestCase):
    def test_starts_unrequested(self):
        drain = _DrainRequest()
        assert not drain.requested
        assert drain.reason is None

    def test_signal_handler_sets_requested(self):
        drain = _DrainRequest()
        # The handler ignores its (signum, frame) arguments.
        drain.handle_signal(None, None)
        assert drain.requested
        assert drain.reason == "SIGUSR1"

    def test_first_reason_is_kept(self):
        drain = _DrainRequest()
        drain.request("first")
        drain.request("second")
        assert drain.reason == "first"

    @unittest.skipUnless(_HAS_SIGUSR1, "platform has no SIGUSR1")
    def test_installed_handler_receives_a_real_signal(self):
        drain = _DrainRequest()
        previous = signal.getsignal(signal.SIGUSR1)
        try:
            _install_drain_handler(drain)
            os.kill(os.getpid(), signal.SIGUSR1)
            assert drain.requested
        finally:
            signal.signal(signal.SIGUSR1, previous)


def _recycle_settings(max_rss_mb: int = 0, max_jobs: int = 0) -> SimpleNamespace:
    return SimpleNamespace(worker_max_rss_mb=max_rss_mb, worker_max_jobs=max_jobs)


class TestAfterJobCheck(TestCase):
    """Each recycle cap must trigger only past its threshold and stay off at 0.

    Both directions matter: a cap that never fires leaves memory growth
    unbounded, and a cap that fires with the triggers disabled would recycle
    every deployment that did not opt in.
    """

    @patch("trapdata.antenna.worker._current_rss_bytes", return_value=2 * 1024**3)
    def test_default_caps_never_request_drain(self, _rss):
        drain = _DrainRequest()
        _after_job_check(drain, _recycle_settings(), jobs_processed=100)
        assert not drain.requested

    @patch("trapdata.antenna.worker._current_rss_bytes", return_value=2 * 1024**3)
    def test_rss_per_job_line_is_always_logged(self, _rss):
        # The log line is the instrument for answering whether memory climbs
        # job after job in a deployment, so it must not depend on any cap
        # being enabled.
        with patch("trapdata.antenna.worker.logger") as mock_logger:
            _after_job_check(_DrainRequest(), _recycle_settings(), jobs_processed=3)
        logged = " ".join(str(c.args[0]) for c in mock_logger.info.call_args_list)
        assert "Resident memory after job 3: 2048 MiB" in logged

    @patch("trapdata.antenna.worker._current_rss_bytes", return_value=9 * 1024**3)
    def test_rss_over_cap_requests_drain(self, _rss):
        drain = _DrainRequest()
        _after_job_check(drain, _recycle_settings(max_rss_mb=8192), jobs_processed=1)
        assert drain.requested
        # 9 GiB = 9216 MiB, over an 8192 MiB cap
        assert "9216" in drain.reason

    @patch("trapdata.antenna.worker._current_rss_bytes", return_value=2 * 1024**3)
    def test_rss_under_cap_does_nothing(self, _rss):
        drain = _DrainRequest()
        _after_job_check(drain, _recycle_settings(max_rss_mb=8192), jobs_processed=1)
        assert not drain.requested

    @patch("trapdata.antenna.worker._current_rss_bytes", return_value=None)
    def test_unreadable_rss_skips_the_memory_cap(self, _rss):
        drain = _DrainRequest()
        _after_job_check(drain, _recycle_settings(max_rss_mb=8192), jobs_processed=1)
        assert not drain.requested

    @patch("trapdata.antenna.worker._current_rss_bytes", return_value=None)
    def test_job_cap_works_without_rss(self, _rss):
        drain = _DrainRequest()
        _after_job_check(drain, _recycle_settings(max_jobs=5), jobs_processed=5)
        assert drain.requested
        assert "cap 5" in drain.reason

    @patch("trapdata.antenna.worker._current_rss_bytes", return_value=2 * 1024**3)
    def test_job_cap_not_reached_does_nothing(self, _rss):
        drain = _DrainRequest()
        _after_job_check(drain, _recycle_settings(max_jobs=5), jobs_processed=4)
        assert not drain.requested

    @patch("trapdata.antenna.worker._current_rss_bytes", return_value=2 * 1024**3)
    def test_existing_drain_reason_is_kept(self, _rss):
        drain = _DrainRequest()
        drain.request("SIGUSR1")
        _after_job_check(drain, _recycle_settings(max_jobs=1), jobs_processed=1)
        assert drain.reason == "SIGUSR1"

    def test_current_rss_reads_a_positive_value_where_proc_exists(self):
        rss = _current_rss_bytes()
        if rss is None:
            self.skipTest("/proc/self/status not available on this platform")
        assert rss > 0


def _fake_batches(n: int) -> list[dict]:
    """Minimal truthy batches; contents are irrelevant with inference mocked."""
    return [{"images": [object()], "image_ids": [f"img_{i}"]} for i in range(n)]


@patch("trapdata.antenna.worker.ResultPoster")
@patch("trapdata.antenna.worker.APIMothDetector")
@patch("trapdata.antenna.worker._process_batch", return_value=(1, 0, [], 0.0, 0.0))
@patch("trapdata.antenna.worker.should_filter_detections", return_value=False)
@patch.dict(
    "trapdata.antenna.worker.CLASSIFIER_CHOICES",
    {"fake_pipeline": MagicMock()},
)
@patch("trapdata.antenna.worker.get_rest_dataloader")
class TestProcessJobStopsAtBatchBoundary(TestCase):
    """``should_stop`` must stop a job between batches, never lose posted work.

    The guard has to hold in both directions: with a stop requested after the
    first batch, later batches must not run; without one, every batch must
    run — otherwise a should_stop that always trips would pass the first
    assertion while silently truncating every job.
    """

    def _settings(self) -> MagicMock:
        settings = MagicMock()
        settings.antenna_api_base_url = "http://testserver/api/v2"
        settings.antenna_api_auth_token = "test-token"
        return settings

    def _give_poster_real_metrics(self, mock_poster_cls: MagicMock) -> None:
        # The end-of-job summary formats these with numeric format specs,
        # which a bare MagicMock attribute cannot satisfy.
        mock_poster_cls.return_value.get_metrics.return_value = SimpleNamespace(
            total_posts=1,
            successful_posts=1,
            failed_posts=0,
            success_rate=100.0,
            total_post_time=0.1,
            max_queue_size=1,
        )

    def test_stops_after_current_batch(
        self,
        mock_loader,
        mock_should_filter,
        mock_process_batch,
        mock_detector,
        mock_poster_cls,
    ):
        mock_loader.return_value = _fake_batches(3)
        self._give_poster_real_metrics(mock_poster_cls)

        stop_flag = {"stop": False}

        def on_batch(batch_num: int, items: int):
            # Simulates a drain request arriving while batch 1 is in flight.
            stop_flag["stop"] = True

        result = _process_job(
            "fake_pipeline",
            job_id=123,
            settings=self._settings(),
            on_batch_complete=on_batch,
            should_stop=lambda: stop_flag["stop"],
        )

        assert result is True
        # Batch 1 ran; batches 2 and 3 were left for the next worker.
        assert mock_process_batch.call_count == 1
        # Pending posts were flushed and the poster shut down cleanly.
        poster = mock_poster_cls.return_value
        poster.wait_for_all_posts.assert_called_once()
        poster.shutdown.assert_called_once()

    def test_without_stop_request_all_batches_run(
        self,
        mock_loader,
        mock_should_filter,
        mock_process_batch,
        mock_detector,
        mock_poster_cls,
    ):
        mock_loader.return_value = _fake_batches(3)
        self._give_poster_real_metrics(mock_poster_cls)

        result = _process_job(
            "fake_pipeline",
            job_id=123,
            settings=self._settings(),
            should_stop=lambda: False,
        )

        assert result is True
        assert mock_process_batch.call_count == 3


class TestWorkerLoopExitsOnDrain(TestCase):
    @unittest.skipUnless(_HAS_SIGUSR1, "platform has no SIGUSR1")
    @patch("trapdata.antenna.worker.get_jobs")
    @patch("trapdata.antenna.worker.read_settings")
    def test_loop_returns_after_signal(self, mock_read_settings, mock_get_jobs):
        settings = MagicMock()
        settings.antenna_service_name = "test-worker"
        settings.worker_max_rss_mb = 0
        settings.worker_max_jobs = 0
        mock_read_settings.return_value = settings

        polls: list[int] = []

        def fake_get_jobs(**kwargs):
            polls.append(1)
            if len(polls) > 1:
                raise AssertionError("worker loop kept polling after the drain request")
            # The signal is delivered to this process before get_jobs returns,
            # like an operator signalling mid-poll.
            os.kill(os.getpid(), signal.SIGUSR1)
            return []

        mock_get_jobs.side_effect = fake_get_jobs

        previous = signal.getsignal(signal.SIGUSR1)
        try:
            _worker_loop(0, ["fake_pipeline"])
        finally:
            signal.signal(signal.SIGUSR1, previous)

        assert polls == [1]
