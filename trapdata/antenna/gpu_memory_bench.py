"""Multi-process GPU memory pressure benchmark for the Antenna worker.

Reproduces the failure mode where several worker processes share one GPU and
collectively exhaust it: each process runs the real worker code path
(``_process_job`` with the real detector, binary filter, and species
classifier) against an in-process mock Antenna API serving the bundled
full-resolution (4096x2160) test images. This is the same infrastructure the
integration tests use, so no network or external services are involved.

The point is to demonstrate *concurrent* pressure — a single process
allocating until it dies only proves the GPU is finite. Run it with the
process count and batch sizes that match a deployment to reproduce the
failure signature (a few-hundred-MiB allocation failing with little free
memory on the card), then run it again with the fix or different parameters
to demonstrate the difference.

Usage (single 24 GiB GPU, shapes similar to a shared production card)::

    python -m trapdata.antenna.gpu_memory_bench \
        --processes 4 --jobs 2 --tasks-per-job 48 \
        --api-batch-size 24 --device 0

To compare against an older revision of the worker, copy this file into a
git worktree of that revision and run it from there with the same arguments.

What it reports, per process:

- ``torch.cuda.memory_allocated`` / ``memory_reserved`` /
  ``max_memory_allocated`` and device-wide free memory at: start, after a
  reference model load, after each batch, and between jobs.
- The count of task results whose error carries a CUDA allocation-failure
  signature ("out of memory" / ``ALLOC_FAILED``), plus the first such
  message (which includes free-at-failure details reported by the
  allocator).

How to read the between-jobs numbers: if memory held after a job completes
keeps growing job over job, something is not being released (a leak). If it
returns to a stable baseline and failures only appear with several
concurrent processes, the card is oversubscribed and the levers are process
count and batch sizes.
"""

import argparse
import multiprocessing
import os
import queue
import sys
import time
from pathlib import Path

GIB = 1024**3

# The allocator env vars torch recognizes (newer and older spelling).
_ALLOC_ENV_VARS = ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-process GPU memory pressure benchmark for the worker"
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=4,
        help="Worker processes sharing the GPU (default: 4)",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=2,
        help="Sequential jobs per process (default: 2)",
    )
    parser.add_argument(
        "--tasks-per-job",
        type=int,
        default=48,
        help="Tasks (images) per job (default: 48)",
    )
    parser.add_argument(
        "--api-batch-size",
        type=int,
        default=24,
        help="antenna_api_batch_size: images fetched per API call (default: 24)",
    )
    parser.add_argument(
        "--localization-batch-size",
        type=int,
        default=8,
        help="Detector GPU batch size setting (default: 8)",
    )
    parser.add_argument(
        "--classification-batch-size",
        type=int,
        default=20,
        help="Classifier GPU batch size setting (default: 20)",
    )
    parser.add_argument(
        "--pipeline",
        default="quebec_vermont_moths_2023",
        help="Pipeline slug to run (default: quebec_vermont_moths_2023)",
    )
    parser.add_argument(
        "--alloc-conf",
        default=None,
        help=(
            "Value for the CUDA allocator env vars, e.g. "
            "'expandable_segments:True'. Default: leave the environment "
            "as-is. Applied in each child before any CUDA allocation. "
            "Note: expandable_segments raises 'CUDA driver error: operation "
            "not supported' on hardware without virtual-memory-management "
            "driver APIs (e.g. NVIDIA vGPU)."
        ),
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="CUDA device index to load (default: 0)",
    )
    return parser.parse_args(argv)


def _sample(rank: int, device: int, phase: str) -> dict:
    """Record and print the CUDA memory counters for this process."""
    import torch

    free_b, total_b = torch.cuda.mem_get_info(device)
    stats = {
        "phase": phase,
        "alloc_gib": torch.cuda.memory_allocated(device) / GIB,
        "reserved_gib": torch.cuda.memory_reserved(device) / GIB,
        "max_alloc_gib": torch.cuda.max_memory_allocated(device) / GIB,
        "free_gib": free_b / GIB,
        "total_gib": total_b / GIB,
    }
    print(
        f"[P{rank}] {phase}: "
        f"alloc={stats['alloc_gib']:.2f} GiB, "
        f"reserved={stats['reserved_gib']:.2f} GiB, "
        f"max_alloc={stats['max_alloc_gib']:.2f} GiB, "
        f"device free={stats['free_gib']:.2f}/{stats['total_gib']:.2f} GiB",
        flush=True,
    )
    return stats


def _is_memory_error_message(message: str) -> bool:
    return "out of memory" in message or "ALLOC_FAILED" in message


def _load_model_reference(pipeline: str, batch_sizes: dict, on_loaded) -> None:
    """Load the pipeline's model stack once, sample via ``on_loaded``, release.

    Gives an "after model load" baseline to compare the between-jobs numbers
    against, and measures the model footprint itself. The sample callback runs
    while the models are alive; they are released when this function returns.
    """
    from trapdata.api.api import CLASSIFIER_CHOICES, should_filter_detections
    from trapdata.api.models.classification import MothClassifierBinary
    from trapdata.api.models.localization import APIMothDetector

    classifier_class = CLASSIFIER_CHOICES[pipeline]
    models = [
        classifier_class(
            source_images=[],
            detections=[],
            batch_size=batch_sizes["classification"],
        ),
        APIMothDetector([], batch_size=batch_sizes["localization"]),
    ]
    if should_filter_detections(classifier_class):
        models.append(
            MothClassifierBinary(
                source_images=[],
                detections=[],
                terminal=False,
                batch_size=batch_sizes["classification"],
            )
        )
    on_loaded()
    del models


def _child_main(
    rank: int,
    args: argparse.Namespace,
    barrier,
    results_queue,
) -> None:
    """One simulated worker process: run jobs and report memory samples."""
    # Allocator config must be in the environment before the first CUDA
    # allocation of this process to take effect.
    if args.alloc_conf is not None:
        for var in _ALLOC_ENV_VARS:
            os.environ[var] = args.alloc_conf

    import torch
    from fastapi.testclient import TestClient

    from trapdata.antenna.schemas import AntennaPipelineProcessingTask
    from trapdata.antenna.tests import antenna_api_server
    from trapdata.antenna.tests.antenna_api_server import app as antenna_app
    from trapdata.antenna.worker import _process_job
    from trapdata.api.tests.image_server import StaticFileTestServer
    from trapdata.api.tests.utils import patch_antenna_api_requests
    from trapdata.settings import Settings
    from trapdata.tests import TEST_IMAGES_BASE_PATH

    device = args.device
    torch.cuda.set_device(device)
    effective_conf = {var: os.environ.get(var, "(unset)") for var in _ALLOC_ENV_VARS}
    print(f"[P{rank}] allocator env: {effective_conf}", flush=True)

    samples: list[dict] = []
    oom_messages: list[str] = []
    job_exceptions: list[str] = []

    samples.append(_sample(rank, device, "start"))

    batch_sizes = {
        "localization": args.localization_batch_size,
        "classification": args.classification_batch_size,
    }
    _load_model_reference(
        args.pipeline,
        batch_sizes,
        on_loaded=lambda: samples.append(_sample(rank, device, "model-load reference")),
    )
    torch.cuda.empty_cache()
    samples.append(_sample(rank, device, "post-release baseline"))

    # Each process runs its own in-process mock Antenna API and image server.
    images_dir = Path(TEST_IMAGES_BASE_PATH)
    file_server = StaticFileTestServer(images_dir)
    file_server.start()
    client = TestClient(antenna_app, follow_redirects=False)

    image_paths = sorted((images_dir / "vermont").glob("*.jpg"))
    image_urls = [file_server.get_url(p.relative_to(images_dir)) for p in image_paths]

    settings = Settings()
    settings.antenna_api_base_url = "http://testserver/api/v2"
    settings.antenna_api_auth_token = "benchmark-token"
    settings.antenna_api_batch_size = args.api_batch_size
    settings.num_workers = 0
    settings.localization_batch_size = args.localization_batch_size
    settings.classification_batch_size = args.classification_batch_size

    # Line up all processes so the jobs actually overlap.
    barrier.wait()
    start_time = time.monotonic()

    between_jobs_alloc: list[float] = []
    try:
        for j in range(args.jobs):
            job_id = 9000 + rank * 100 + j
            antenna_api_server.reset()
            tasks = [
                AntennaPipelineProcessingTask(
                    id=f"task_{rank}_{j}_{i}",
                    image_id=f"img_{rank}_{j}_{i}",
                    image_url=image_urls[i % len(image_urls)],
                    reply_subject=f"reply_{rank}_{j}_{i}",
                )
                for i in range(args.tasks_per_job)
            ]
            antenna_api_server.setup_job(job_id=job_id, tasks=tasks)

            torch.cuda.reset_peak_memory_stats(device)

            def on_batch(batch_num: int, items: int, job_index: int = j):
                label = f"job {job_index} batch {batch_num + 1}"
                if batch_num == 0:
                    label += " (models loaded)"
                samples.append(_sample(rank, device, label))

            try:
                with patch_antenna_api_requests(client):
                    _process_job(
                        args.pipeline,
                        job_id,
                        settings,
                        device=torch.device("cuda", device),
                        on_batch_complete=on_batch,
                    )
            except Exception as e:
                # A job-level failure (e.g. OOM during prefetch) is itself a
                # data point; record it and keep the process running.
                job_exceptions.append(f"job {j}: {type(e).__name__}: {e}")
                print(f"[P{rank}] job {j} raised: {e}", flush=True)

            stats = _sample(rank, device, f"between jobs (after job {j})")
            between_jobs_alloc.append(stats["alloc_gib"])

            # Collect allocation-failure signatures from the posted results.
            for result in antenna_api_server.get_posted_results(job_id):
                error = getattr(result.result, "error", None)
                if error and _is_memory_error_message(error):
                    oom_messages.append(error)
    finally:
        file_server.stop()

    elapsed = time.monotonic() - start_time
    if oom_messages:
        print(
            f"[P{rank}] first allocation-failure message: {oom_messages[0][:400]}",
            flush=True,
        )

    results_queue.put(
        {
            "rank": rank,
            "elapsed_s": elapsed,
            "samples": samples,
            "between_jobs_alloc_gib": between_jobs_alloc,
            "oom_error_count": len(oom_messages),
            "first_oom_message": oom_messages[0] if oom_messages else None,
            "job_exceptions": job_exceptions,
            "peak_alloc_gib": max(s["max_alloc_gib"] for s in samples),
        }
    )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    import torch

    if not torch.cuda.is_available():
        print("CUDA is not available; this benchmark needs a GPU.", file=sys.stderr)
        return 2

    # The parent deliberately avoids creating a CUDA context (it would hold
    # a few hundred MiB on the device and distort the measurements); the
    # children report device totals themselves.
    print(
        f"Device index {args.device}: "
        f"{args.processes} processes x {args.jobs} jobs x "
        f"{args.tasks_per_job} tasks, api_batch_size={args.api_batch_size}, "
        f"localization_batch_size={args.localization_batch_size}, "
        f"classification_batch_size={args.classification_batch_size}, "
        f"alloc_conf={args.alloc_conf!r}",
        flush=True,
    )

    ctx = multiprocessing.get_context("spawn")
    barrier = ctx.Barrier(args.processes)
    results_queue = ctx.Queue()
    procs = [
        ctx.Process(
            target=_child_main,
            args=(rank, args, barrier, results_queue),
            name=f"gpu-bench-{rank}",
        )
        for rank in range(args.processes)
    ]
    for p in procs:
        p.start()

    results = []
    for _ in procs:
        try:
            # Generous ceiling; a normal run finishes in minutes. A missing
            # result means a child died hard (e.g. CUDA abort) — report it
            # rather than hanging forever.
            results.append(results_queue.get(timeout=1800))
        except queue.Empty:
            print("Timed out waiting for a worker process result.", flush=True)
            break
    for p in procs:
        p.join(timeout=30)
        if p.is_alive():
            print(f"Terminating unresponsive process {p.name}", flush=True)
            p.terminate()

    results.sort(key=lambda r: r["rank"])
    total_oom = sum(r["oom_error_count"] for r in results)

    print("\n=== Summary ===", flush=True)
    for r in results:
        between = ", ".join(f"{a:.2f}" for a in r["between_jobs_alloc_gib"])
        print(
            f"[P{r['rank']}] peak_alloc={r['peak_alloc_gib']:.2f} GiB, "
            f"between-jobs alloc per job=[{between}] GiB, "
            f"oom_errors={r['oom_error_count']}, "
            f"job_exceptions={len(r['job_exceptions'])}, "
            f"elapsed={r['elapsed_s']:.0f}s",
            flush=True,
        )
    print(
        f"\nAllocation-failure signature reproduced: "
        f"{'YES' if total_oom else 'NO'} "
        f"({total_oom} allocation-failure task errors across "
        f"{args.processes} processes)",
        flush=True,
    )
    print(
        "Interpretation: growing between-jobs alloc across jobs suggests a "
        "leak; a stable between-jobs baseline with failures only under "
        "concurrency indicates an oversubscribed card (tune process count / "
        "batch sizes).",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
