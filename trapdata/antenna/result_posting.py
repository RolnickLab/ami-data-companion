"""Asynchronous result posting utilities for Antenna API.

This module provides utilities for posting batch results to the Antenna API with
backpressure control and comprehensive metrics tracking. The main class, ResultPoster,
manages asynchronous posting to improve worker throughput by overlapping network I/O
with compute operations.

Key features:
- Asynchronous posting using ThreadPoolExecutor
- Configurable backpressure control to prevent unbounded memory usage
- Comprehensive metrics tracking (success/failure rates, timing, queue size)
- Graceful shutdown with timeout handling
- Thread-safe operations

Usage:
    poster = ResultPoster(max_pending=5)
    poster.post_async(base_url, auth_token, job_id, results, service_name)
    metrics = poster.get_metrics()
    poster.shutdown()
"""

import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from typing import Optional

from trapdata.antenna.client import post_batch_results
from trapdata.common.logs import logger


@dataclass
class ResultPostMetrics:
    """Metrics for tracking result posting performance."""

    total_posts: int = 0
    successful_posts: int = 0
    failed_posts: int = 0
    total_post_time: float = 0.0
    max_queue_size: int = 0

    @property
    def success_rate(self) -> float:
        """Calculate success rate as a percentage."""
        return (
            (self.successful_posts / self.total_posts * 100)
            if self.total_posts > 0
            else 0.0
        )


class ResultPoster:
    """Manages asynchronous posting of batch results with backpressure control.

    This class provides asynchronous result posting to improve throughput by allowing
    the worker to continue processing while previous results are posted in background
    threads. It includes backpressure control to prevent unbounded memory usage.

    Args:
        max_pending: Maximum number of concurrent posts before blocking (default: 5)

    Example:
        poster = ResultPoster(max_pending=10)
        poster.post_async(base_url, auth_token, job_id, results, service_name)
        metrics = poster.get_metrics()
        poster.shutdown()
    """

    def __init__(
        self,
        max_pending: int = 5,
        future_timeout: float = 30.0,
    ):
        self.max_pending = max_pending
        self.future_timeout = future_timeout  # Timeout for individual future waits
        self.executor = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="result_poster"
        )
        self.pending_futures: list[Future] = []
        self.metrics = ResultPostMetrics()
        self._metrics_lock = threading.Lock()

    def post_async(
        self,
        base_url: str,
        auth_token: str,
        job_id: int,
        results: list,
        processing_service_name: str,
    ) -> None:
        """Post results asynchronously with backpressure control.

        This method will block if there are too many pending posts to prevent
        unbounded memory usage and provide backpressure.

        Args:
            base_url: Antenna API base URL
            auth_token: API authentication token
            job_id: Job ID for the results
            results: List of result objects to post
            processing_service_name: Name of the processing service
        """
        # Clean up completed futures and update metrics
        self._cleanup_completed_futures()

        # Apply backpressure: wait for pending posts to complete if we're at the limit
        while len(self.pending_futures) >= self.max_pending:
            # Wait for at least one future to complete
            done_futures, _ = wait(
                self.pending_futures,
                return_when=FIRST_COMPLETED,
                timeout=self.future_timeout,
            )
            for completed_future in done_futures:
                self.pending_futures.remove(completed_future)
            if not done_futures:
                # Force cleanup by cancelling all pending futures and clearing the list
                for future in self.pending_futures:
                    future.cancel()
                self.pending_futures.clear()
                logger.warning(
                    "Timeout waiting for pending result posts. "
                    "Cleared pending futures to prevent blocking indefinitely."
                )

        # Update queue size metric
        current_queue_size = len(self.pending_futures)
        self.metrics.max_queue_size = max(
            self.metrics.max_queue_size, current_queue_size
        )

        # Submit new post
        start_time = time.time()
        future = self.executor.submit(
            self._post_with_timing,
            base_url,
            auth_token,
            job_id,
            results,
            processing_service_name,
            start_time,
        )
        self.pending_futures.append(future)
        self.metrics.total_posts += 1

        logger.debug(
            f"Submitted result post for job {job_id}, {current_queue_size + 1} pending"
        )

    def _post_with_timing(
        self,
        base_url: str,
        auth_token: str,
        job_id: int,
        results: list,
        processing_service_name: str,
        start_time: float,
    ) -> bool:
        """Internal method that times the post operation and updates metrics.

        Args:
            base_url: Antenna API base URL
            auth_token: API authentication token
            job_id: Job ID for the results
            results: List of result objects to post
            processing_service_name: Name of the processing service
            start_time: Timestamp when the post was initiated

        Returns:
            True if successful, False otherwise
        """
        try:
            success = post_batch_results(
                base_url, auth_token, job_id, results, processing_service_name
            )
            elapsed_time = time.time() - start_time

            with self._metrics_lock:
                self.metrics.total_post_time += elapsed_time
                if success:
                    self.metrics.successful_posts += 1
                else:
                    self.metrics.failed_posts += 1
                    logger.warning(
                        f"Result post failed for job {job_id} after {elapsed_time:.2f}s"
                    )

            return success
        except Exception as e:
            elapsed_time = time.time() - start_time
            with self._metrics_lock:
                self.metrics.total_post_time += elapsed_time
                self.metrics.failed_posts += 1
            logger.error(f"Exception during result post for job {job_id}: {e}")
            return False

    def _cleanup_completed_futures(self) -> None:
        """Remove completed futures from the pending list."""
        self.pending_futures = [f for f in self.pending_futures if not f.done()]

    def wait_for_all_posts(
        self,
        min_timeout: float = 60,
        per_post_timeout: float = 30,
    ) -> None:
        """Wait for all pending posts to complete before shutting down.

        Args:
            min_timeout: Minimum timeout regardless of pending count (default: 60)
            per_post_timeout: Additional timeout per pending post (default: 30)
        """
        if not self.pending_futures:
            return

        pending_count = len(self.pending_futures)
        timeout = max(min_timeout, pending_count * per_post_timeout)
        logger.info(
            f"Waiting for {pending_count} pending result posts with dynamic timeout {timeout}s"
        )
        start_time = time.time()

        for future in self.pending_futures:
            remaining_timeout = None
            elapsed = time.time() - start_time
            remaining_timeout = max(0, timeout - elapsed)
            if remaining_timeout == 0:
                logger.warning("Timeout waiting for pending posts, some may be lost")
                break

            try:
                future.result(timeout=remaining_timeout)
            except Exception as e:
                logger.warning(f"Pending result post failed during shutdown: {e}")

        self._cleanup_completed_futures()

        # Check if any posts were abandoned and emit error
        posts_after_wait = len(self.pending_futures)
        if posts_after_wait > 0:
            metrics = self.metrics
            logger.error(
                f"Failed to complete all result posts before timeout. "
                f"{posts_after_wait} posts were abandoned. "
                f"Post metrics - Total: {metrics.total_posts}, "
                f"Successful: {metrics.successful_posts}, "
                f"Failed: {metrics.failed_posts}, "
                f"Success rate: {metrics.success_rate:.1f}%, "
                f"Max queue size: {metrics.max_queue_size}"
            )

    def get_metrics(self) -> ResultPostMetrics:
        """Get current metrics.

        Returns:
            Current ResultPostMetrics object with performance data
        """
        self._cleanup_completed_futures()
        return self.metrics

    def shutdown(self) -> None:
        """Shutdown the executor"""
        if self.pending_futures:
            raise RuntimeError(
                f"Cannot shutdown ResultPoster with {len(self.pending_futures)} pending posts. "
                "Call wait_for_all_posts() before shutdown to ensure all posts are completed."
            )
        self.executor.shutdown(wait=False)
