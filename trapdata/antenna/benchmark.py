"""Benchmarking utilities for Antenna API data loading and result posting.

This module provides a command-line benchmark tool for testing the performance
of the Antenna API data loading pipeline with asynchronous result posting.
The benchmark fetches batches from the API, processes acknowledgments, and
provides detailed performance metrics.

Usage:
    python -m trapdata.antenna.benchmark --job-id 123 --base-url http://localhost:8000/api/v2

Key metrics tracked:
- Images per second (total and successful)
- Batch processing rate
- Acknowledgment posting rate
- Result posting success/failure rates
- Queue utilization metrics
"""

import argparse
import os
import time

from trapdata.antenna.datasets import get_rest_dataloader
from trapdata.antenna.result_posting import ResultPoster
from trapdata.antenna.schemas import AntennaTaskResult, AntennaTaskResultError
from trapdata.common.logs import logger
from trapdata.common.utils import log_time
from trapdata.settings import Settings


def create_empty_result(reply_subject: str, image_id: str) -> AntennaTaskResult:
    """Create an empty/acknowledgment result for a task.

    Args:
        reply_subject: Subject for the reply
        image_id: ID of the image being acknowledged

    Returns:
        AntennaTaskResult with error acknowledgment
    """
    result = AntennaTaskResultError(
        error=f"Acknowledgment for image {image_id}",
        image_id=image_id,
    )
    return AntennaTaskResult(reply_subject=reply_subject, result=result)


def run_benchmark(
    job_id: int,
    base_url: str,
    auth_token: str,
    num_workers: int,
    batch_size: int,
    gpu_batch_size: int,
    service_name: str,
) -> None:
    """Run the benchmark with the specified parameters.

    Args:
        job_id: Job ID to process
        base_url: Antenna API base URL
        auth_token: API authentication token
        num_workers: Number of DataLoader workers
        batch_size: Batch size for API requests
        gpu_batch_size: GPU batch size for DataLoader
        service_name: Processing service name
    """
    # Create settings object
    settings = Settings()
    settings.antenna_api_base_url = base_url
    settings.antenna_api_auth_token = auth_token
    settings.antenna_api_batch_size = batch_size
    settings.localization_batch_size = gpu_batch_size
    settings.num_workers = num_workers

    print(f"Starting performance test for job {job_id}")
    print(f"Configuration:")
    print(f"  Base URL: {base_url}")
    print(f"  API batch size: {batch_size}")
    print(f"  GPU batch size: {gpu_batch_size}")
    print(f"  Num workers: {num_workers}")
    print(f"  Service name: {service_name}")
    print()

    # Create dataloader
    dataloader = get_rest_dataloader(
        job_id=job_id,
        settings=settings,
        processing_service_name=service_name,
    )

    # Initialize ResultPoster for sending acknowledgments
    result_poster = ResultPoster(max_pending=10)

    # Performance metrics
    total_batches = 0
    total_images = 0
    total_successful_images = 0
    total_failed_images = 0
    total_acks_sent = 0
    start_time = time.time()
    last_report_time = start_time
    report_interval = 10  # Report every 10 seconds

    print("Starting data consumption with acknowledgments...")
    try:
        _, t = log_time()
        for batch_idx, batch in enumerate(dataloader):
            _, t = t(
                f"Fetched batch {batch_idx} with {len(batch['reply_subjects'])} items"
            )
            current_time = time.time()
            total_batches += 1

            # Count images in this batch
            batch_size = len(batch["reply_subjects"])
            batch_failed = len(batch["failed_items"])
            batch_successful = batch_size - batch_failed

            total_images += batch_size
            total_successful_images += batch_successful
            total_failed_images += batch_failed

            # Send acknowledgments for successful items
            if batch_successful > 0:
                ack_results = []
                for i, (reply_subject, image_id) in enumerate(
                    zip(batch["reply_subjects"], batch["image_ids"])
                ):
                    if i < batch_successful:  # Only for successful items
                        ack_result = create_empty_result(reply_subject, image_id)
                        ack_results.append(ack_result)

                logger.info(f"Sending {len(ack_results)} acknowledgment(s)")
                if ack_results:
                    # Send acknowledgments asynchronously
                    result_poster.post_async(
                        base_url=base_url,
                        auth_token=auth_token,
                        job_id=job_id,
                        results=ack_results,
                        processing_service_name=service_name,
                    )
                    total_acks_sent += len(ack_results)

            # Send error results for failed items
            if batch_failed > 0:
                error_results = []
                for failed_item in batch["failed_items"]:
                    error_result = AntennaTaskResult(
                        reply_subject=failed_item["reply_subject"],
                        result=AntennaTaskResultError(
                            error=failed_item.get("error", "Image loading failed"),
                            image_id=failed_item["image_id"],
                        ),
                    )
                    error_results.append(error_result)

                if error_results:
                    result_poster.post_async(
                        base_url=base_url,
                        auth_token=auth_token,
                        job_id=job_id,
                        results=error_results,
                        processing_service_name=service_name,
                    )
                    total_acks_sent += len(error_results)

            # Report progress periodically
            if current_time - last_report_time >= report_interval:
                elapsed = current_time - start_time
                images_per_sec = total_images / elapsed if elapsed > 0 else 0
                successful_per_sec = (
                    total_successful_images / elapsed if elapsed > 0 else 0
                )
                acks_per_sec = total_acks_sent / elapsed if elapsed > 0 else 0

                # Get ResultPoster metrics
                post_metrics = result_poster.get_metrics()

                print(
                    f"Progress: {total_batches} batches, {total_images} images "
                    f"({total_successful_images} success, {total_failed_images} failed) "
                    f"- {images_per_sec:.1f} img/s, {successful_per_sec:.1f} success/s, "
                    f"{acks_per_sec:.1f} acks/s"
                )
                print(
                    f"  Posts: {post_metrics.successful_posts} success, "
                    f"{post_metrics.failed_posts} failed, "
                    f"{post_metrics.success_rate:.1f}% success rate"
                )
                last_report_time = current_time
            _, t = log_time()

    except KeyboardInterrupt:
        print("\nStopped by user")
    except Exception as e:
        print(f"\nError occurred: {e}")
        logger.error(f"DataLoader benchmark error: {e}")
    finally:
        # Wait for all pending result posts to complete
        print("Waiting for pending result posts to complete...")
        result_poster.wait_for_all_posts(timeout=30)
        result_poster.shutdown()

    # Final statistics
    end_time = time.time()
    total_elapsed = end_time - start_time
    final_post_metrics = result_poster.get_metrics()

    print("\n" + "=" * 70)
    print("PERFORMANCE SUMMARY")
    print("=" * 70)
    print(f"Total time: {total_elapsed:.2f} seconds")
    print(f"Total batches: {total_batches}")
    print(f"Total images: {total_images}")
    print(f"Successful images: {total_successful_images}")
    print(f"Failed images: {total_failed_images}")
    print(f"Acknowledgments sent: {total_acks_sent}")

    if total_elapsed > 0:
        images_per_sec = total_images / total_elapsed
        successful_per_sec = total_successful_images / total_elapsed
        batches_per_sec = total_batches / total_elapsed
        acks_per_sec = total_acks_sent / total_elapsed

        print(f"\nThroughput:")
        print(f"  {images_per_sec:.2f} images/second (total)")
        print(f"  {successful_per_sec:.2f} images/second (successful)")
        print(f"  {batches_per_sec:.2f} batches/second")
        print(f"  {acks_per_sec:.2f} acknowledgments/second")

        if total_images > 0:
            success_rate = (total_successful_images / total_images) * 100
            print(f"\nSuccess rate: {success_rate:.1f}%")

    print(f"\nResult Posting Metrics:")
    print(f"  Total posts: {final_post_metrics.total_posts}")
    print(f"  Successful posts: {final_post_metrics.successful_posts}")
    print(f"  Failed posts: {final_post_metrics.failed_posts}")
    print(f"  Post success rate: {final_post_metrics.success_rate:.1f}%")
    if final_post_metrics.total_posts > 0:
        avg_post_time = (
            final_post_metrics.total_post_time / final_post_metrics.total_posts
        )
        print(f"  Average post time: {avg_post_time:.3f} seconds")
    print(f"  Max queue size: {final_post_metrics.max_queue_size}")

    print("=" * 70)
    print("Performance benchmark completed")
    print("=" * 70)


def main():
    """Main entry point for the benchmark CLI."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Benchmark dataloader performance with acknowledgements"
    )
    parser.add_argument("--job-id", type=int, required=True, help="Job ID to process")
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:8000/api/v2",
        help="Antenna API base URL",
    )
    parser.add_argument(
        "--num-workers", type=int, default=2, help="Number of DataLoader workers"
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size for API requests"
    )
    parser.add_argument(
        "--gpu-batch-size", type=int, default=16, help="GPU batch size for DataLoader"
    )
    parser.add_argument(
        "--service-name",
        type=str,
        default="Performance Test",
        help="Processing service name",
    )

    args = parser.parse_args()

    # Get auth token from environment
    auth_token = os.getenv("AMI_ANTENNA_API_AUTH_TOKEN", "")
    if not auth_token:
        print("Warning: AMI_ANTENNA_API_AUTH_TOKEN environment variable not set")

    # Run the benchmark
    run_benchmark(
        job_id=args.job_id,
        base_url=args.base_url,
        auth_token=auth_token,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        gpu_batch_size=args.gpu_batch_size,
        service_name=args.service_name,
    )


if __name__ == "__main__":
    main()
