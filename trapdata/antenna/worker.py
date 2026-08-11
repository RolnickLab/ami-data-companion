"""Worker loop for processing jobs from Antenna API."""

from __future__ import annotations

import datetime
import time
from collections.abc import Callable

import numpy as np
import torch
import torch.multiprocessing as mp

from trapdata.antenna.client import get_full_service_name, get_jobs
from trapdata.antenna.datasets import CUDAPrefetcher, get_rest_dataloader
from trapdata.antenna.result_posting import ResultPoster
from trapdata.antenna.schemas import AntennaTaskResult, AntennaTaskResultError
from trapdata.api.api import (
    PIPELINE_CHOICES,
    PipelineDefinition,
    run_classification_stages,
)
from trapdata.api.models.classification import APIMothClassifier
from trapdata.api.models.localization import APIAnyBugDetector, APIMothDetector
from trapdata.api.schemas import (
    AlgorithmConfigResponse,
    DetectionResponse,
    PipelineResultsResponse,
    SourceImageResponse,
)
from trapdata.common.logs import logger
from trapdata.common.utils import log_time
from trapdata.settings import Settings, read_settings

MAX_PENDING_POSTS = 5  # Maximum number of concurrent result posts before blocking
SLEEP_TIME_SECONDS = 5


def run_worker(pipelines: list[str]):
    """Run the worker to process images from the REST API queue.

    Automatically spawns one AMI worker instance process per available GPU.
    On single-GPU or CPU-only machines, runs in-process (no overhead).
    """
    settings = read_settings()

    # Validate auth token
    if not settings.antenna_api_auth_token:
        raise ValueError(
            "AMI_ANTENNA_API_AUTH_TOKEN environment variable must be set. "
            "Get your auth token from your Antenna project settings."
        )

    # Validate service name
    if not settings.antenna_service_name or not settings.antenna_service_name.strip():
        raise ValueError(
            "AMI_ANTENNA_SERVICE_NAME configuration setting must be set. "
            "Configure it via environment variable or .env file."
        )

    gpu_count = torch.cuda.device_count()
    if gpu_count > 1:
        logger.info(f"Found {gpu_count} GPUs, spawning one AMI worker instance per GPU")
        # Don't pass settings through mp.spawn — Settings contains enums that
        # can't be pickled. Each child process calls read_settings() itself.
        mp.spawn(
            _worker_loop,
            args=(pipelines,),
            nprocs=gpu_count,
            join=True,
        )
    else:
        if gpu_count == 1:
            logger.info(f"Found 1 GPU: {torch.cuda.get_device_name(0)}")
        else:
            logger.info("No GPUs found, running on CPU")
        _worker_loop(0, pipelines)


def _worker_loop(gpu_id: int, pipelines: list[str]):
    """Main polling loop for a single AMI worker instance, pinned to a specific GPU.

    Args:
        gpu_id: GPU index to pin this AMI worker instance to (0 for CPU-only).
        pipelines: List of pipeline slugs to poll for jobs.
    """
    settings = read_settings()
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        torch.cuda.set_device(gpu_id)
        logger.info(
            f"AMI worker instance {gpu_id} pinned to GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}"
        )

    # Build full service name with hostname
    full_service_name = get_full_service_name(settings.antenna_service_name)
    logger.info(f"Running worker as: {full_service_name}")

    while True:
        # TODO CGJS: Support pulling and prioritizing single image tasks, which are used in interactive testing
        # These should probably come from a dedicated endpoint and should preempt batch jobs under the assumption that they
        # would run on the same GPU.
        any_jobs = False
        logger.info(
            f"[GPU {gpu_id}] Checking for jobs for pipelines: {', '.join(pipelines)}"
        )
        jobs = get_jobs(
            base_url=settings.antenna_api_base_url,
            auth_token=settings.antenna_api_auth_token,
            pipeline_slugs=pipelines,
        )
        for job_id, pipeline in jobs:
            logger.info(
                f"[GPU {gpu_id}] Processing job {job_id} with pipeline {pipeline}"
            )
            try:
                any_work_done = _process_job(
                    pipeline=pipeline,
                    job_id=job_id,
                    settings=settings,
                    device=device,
                )
                any_jobs = any_jobs or any_work_done
            except Exception as e:
                logger.error(
                    f"[GPU {gpu_id}] Failed to process job {job_id} with pipeline {pipeline}: {e}",
                    exc_info=True,
                )
                # Continue to next job rather than crashing the worker

        if not any_jobs:
            logger.info(
                f"[GPU {gpu_id}] No jobs found, sleeping for {SLEEP_TIME_SECONDS} seconds"
            )
            time.sleep(SLEEP_TIME_SECONDS)


def _build_stage_models(
    pipeline_def: PipelineDefinition,
) -> dict[type[APIMothClassifier], APIMothClassifier]:
    """Instantiate every classification stage of ``pipeline_def`` once so the
    per-batch loop reuses the loaded weights instead of reloading a model for
    each batch. Intermediate gates are built non-terminal and the terminal
    classifier terminal, matching the labels the /process path assigns.

    Keyed by classifier class. The registry never uses one classifier as both a
    gate and the terminal, so there is no key collision.
    """
    stage_models: dict[type[APIMothClassifier], APIMothClassifier] = {}
    for stage in pipeline_def.intermediates:
        stage_models[stage.classifier] = stage.classifier(
            source_images=[], detections=[], terminal=False
        )
    stage_models[pipeline_def.terminal] = pipeline_def.terminal(
        source_images=[], detections=[], terminal=True
    )
    return stage_models


def _classify_detections_from_tensors(
    stage: APIMothClassifier,
    detections: list[DetectionResponse],
    image_tensors: dict[str, torch.Tensor],
) -> APIMothClassifier:
    """Run one classifier stage over crops taken from the already-loaded image
    tensors, annotating each detection in place, and return the stage with its
    ``results`` set to every detection it was handed.

    This is the worker-side counterpart to ``APIMothClassifier.run()``: rather
    than re-reading images by URL it slices each crop from the in-memory tensors
    the dataloader already produced. It honors the same contract the /process
    path relies on — ``results`` is the full input list (annotated wherever the
    crop was readable), keyed on detection identity — so a detection whose crop
    cannot be read keeps its own label instead of shifting its neighbours'. One
    function now runs every gate and the terminal, replacing the old binary-only
    crop loop, so the sync and async paths share one stage engine.

    A detection whose bbox is degenerate after clamping to the image (zero-area
    or entirely off-frame) is returned with no classification from this stage,
    and logged. It must not be given a placeholder classification: the platform
    rejects any classification whose algorithm key is absent from the pipeline's
    ``/info`` config, and it raises after the batch's detections are already
    written, so one degenerate box would cost every other detection its labels.
    """
    stage.reset(detections)
    transforms = stage.get_transforms()

    crops = []
    valid_indices = []
    for idx, detection in enumerate(detections):
        image_tensor = image_tensors[detection.source_image_id]
        height, width = int(image_tensor.shape[-2]), int(image_tensor.shape[-1])
        # Clamp to image bounds so this tensor-slicing runner and the /process
        # PIL-crop runner see the SAME in-bounds region (see
        # BoundingBox.clamp_to_bounds). Without it, tensor slicing wraps a
        # negative coordinate while PIL zero-pads, diverging the two paths.
        x1, y1, x2, y2 = detection.bbox.clamp_to_bounds(width, height)
        if y1 >= y2 or x1 >= x2:
            # Degenerate after clamping (zero-area or entirely off-image), so there
            # is no crop to classify. Skip it and leave the detection unannotated by
            # this stage; see the note in this function's docstring on why a
            # placeholder classification is not safe here.
            logger.warning(
                f"Skipping {stage.name} for detection {idx}; bbox is degenerate "
                f"after clamping to the image: ({x1},{y1})->({x2},{y2})"
            )
            continue
        crop = image_tensor[:, y1:y2, x1:x2]
        crops.append(transforms(crop))
        valid_indices.append(idx)

    if crops:
        batched_crops = torch.stack(crops)
        stage_out = stage.predict_batch(batched_crops)
        stage_out = stage.post_process_batch(stage_out)
        for crop_i, idx in enumerate(valid_indices):
            detection = detections[idx]
            stage.update_detection_classification(
                seconds_per_item=0,
                image_id=detection.source_image_id,
                detection_idx=idx,
                predictions=stage_out[crop_i],
            )

    # Identity-keyed contract shared with the /process path: return every
    # detection handed in, annotated in place where the crop was readable, in
    # the same order.
    stage.results = detections
    return stage


def _process_batch(
    batch: dict,
    batch_num: int,
    detector: APIMothDetector | APIAnyBugDetector,
    stage_models: dict[type[APIMothClassifier], APIMothClassifier],
    pipeline_def: PipelineDefinition,
    pipeline: str,
) -> tuple[int, int, list[AntennaTaskResult], float, float]:
    """Process a single batch of images through the pipeline's stages.

    The inference core is stage-driven: the detector comes from
    ``pipeline_def.detector`` (YOLO26 or FasterRCNN, whichever the pipeline
    declares) and the intermediate gate(s) and terminal classifier run through
    the SAME ``run_classification_stages`` engine as the FastAPI /process path,
    so the two code paths cannot drift. Only the inference mechanism is worker-
    specific: crops are sliced from the already-loaded image tensors instead of
    re-read from URLs.

    All large intermediates (image_tensors, crops, batched_crops, image_detections)
    are local to this function and freed by Python's reference counting when it
    returns, preventing memory accumulation across batches.

    Args:
        batch: Dictionary with images, image_ids, reply_subjects, image_urls, failed_items
        batch_num: 0-based batch index (for logging)
        detector: The pipeline's detector instance (reset before call)
        stage_models: Preloaded classifier stage instances keyed by class, one per
            intermediate gate plus the terminal classifier (see _build_stage_models)
        pipeline_def: The pipeline's stage definition driving detection/classification
        pipeline: Pipeline slug for response payload

    Returns:
        (n_items, n_detections, batch_results, detect_time, classify_time)
    """
    images = batch.get("images", [])
    image_ids = batch.get("image_ids", [])
    reply_subjects = batch.get("reply_subjects", [None] * len(images))
    image_urls = batch.get("image_urls", [None] * len(images))

    batch_results: list[AntennaTaskResult] = []

    try:
        # Validate all arrays have same length before zipping
        if len(image_ids) != len(images):
            raise ValueError(
                f"Length mismatch: image_ids ({len(image_ids)}) != images ({len(images)})"
            )
        if len(image_ids) != len(reply_subjects) or len(image_ids) != len(image_urls):
            raise ValueError(
                f"Length mismatch: image_ids ({len(image_ids)}), "
                f"reply_subjects ({len(reply_subjects)}), image_urls ({len(image_urls)})"
            )

        batch_start_time = datetime.datetime.now()

        # output is dict of "boxes", "labels", "scores"
        batch_output = []
        if len(images) > 0:
            batch_output = detector.predict_batch(images)

        n_items = len(batch_output)
        batch_output = list(detector.post_process_batch(batch_output))

        # Convert image_ids to list if needed
        if isinstance(image_ids, (np.ndarray, torch.Tensor)):
            image_ids = image_ids.tolist()

        # TODO CGJS: Add seconds per item calculation for both detector and classifier
        detector.save_results(
            item_ids=image_ids,
            batch_output=batch_output,
            seconds_per_item=0,
        )
        detect_time = (datetime.datetime.now() - batch_start_time).total_seconds()

        image_tensors = dict(zip(image_ids, images, strict=True))

        # --- Classification stages ---
        # Run the pipeline's intermediate gate(s) and terminal classifier through
        # the SAME engine as the /process path, so the two cannot drift. Each
        # stage classifies crops sliced from the already-loaded image tensors
        # (run_stage below) rather than re-reading images by URL.
        algorithms_used: dict[str, AlgorithmConfigResponse] = {}

        def run_stage(
            classifier_class: type[APIMothClassifier],
            detections: list[DetectionResponse],
            terminal: bool,
        ) -> APIMothClassifier:
            stage = stage_models[classifier_class]
            stage.terminal = terminal
            return _classify_detections_from_tensors(stage, detections, image_tensors)

        classify_start = datetime.datetime.now()
        detections_to_return = run_classification_stages(
            pipeline=pipeline_def,
            source_images=[],
            detector_results=detector.results,
            example_config_param=None,
            algorithms_used=algorithms_used,
            run_stage=run_stage,
        )
        classify_time = (datetime.datetime.now() - classify_start).total_seconds()

        # Group every returned detection (gate-dropped and tagged, plus terminal-
        # classified) by image for posting. Algorithm provenance travels on each
        # detection's own classifications, so no separate tracking is needed.
        image_detections: dict[str, list[DetectionResponse]] = {
            img_id: [] for img_id in image_ids
        }
        for detection in detections_to_return:
            image_detections[detection.source_image_id].append(detection)
        n_detections = len(detections_to_return)
        logger.debug(
            f"Batch {batch_num + 1} stages used: {list(algorithms_used.keys())}"
        )

        # Calculate batch processing time
        batch_end_time = datetime.datetime.now()
        batch_elapsed = (batch_end_time - batch_start_time).total_seconds()

        # Post results back to the API with PipelineResponse for each image
        for reply_subject, image_id, image_url in zip(
            reply_subjects, image_ids, image_urls, strict=True
        ):
            source_image = SourceImageResponse(id=image_id, url=image_url)
            pipeline_response = PipelineResultsResponse(
                pipeline=pipeline,
                source_images=[source_image],
                detections=image_detections[image_id],
                total_time=batch_elapsed / len(image_ids),
            )
            batch_results.append(
                AntennaTaskResult(
                    reply_subject=reply_subject,
                    result=pipeline_response,
                )
            )
    except Exception as e:
        logger.error(
            f"Batch {batch_num + 1} failed during processing: {e}", exc_info=True
        )
        # Report errors back to Antenna so tasks aren't stuck in the queue
        batch_results = []
        for reply_subject, image_id in zip(reply_subjects, image_ids, strict=True):
            batch_results.append(
                AntennaTaskResult(
                    reply_subject=reply_subject,
                    result=AntennaTaskResultError(
                        error=f"Batch processing error: {e}",
                        image_id=str(image_id) if image_id is not None else None,
                    ),
                )
            )
        n_items = 0
        n_detections = 0
        detect_time = 0.0
        classify_time = 0.0

    failed_items = batch.get("failed_items")
    if failed_items:
        for failed_item in failed_items:
            batch_results.append(
                AntennaTaskResult(
                    reply_subject=failed_item.get("reply_subject"),
                    result=AntennaTaskResultError(
                        error=failed_item.get("error", "Unknown error"),
                        image_id=failed_item.get("image_id"),
                    ),
                )
            )

    torch.cuda.empty_cache()
    return n_items, n_detections, batch_results, detect_time, classify_time


@torch.no_grad()
def _process_job(
    pipeline: str,
    job_id: int,
    settings: Settings,
    device: torch.device | None = None,
    on_batch_complete: Callable | None = None,
) -> bool:
    """Run the worker to process images from the REST API queue.

    Args:
        pipeline: Pipeline name to use for processing (e.g., moth_binary, panama_moths_2024)
        job_id: Job ID to process
        settings: Settings object with antenna_api_* configuration
        device: The device to use for processing. Auto-detected if None.
        on_batch_complete: Optional callback invoked after each batch, with kwargs
            batch_num (int) and items (int, cumulative items processed so far).
    Returns:
        True if any work was done, False otherwise
    """
    did_work = False
    loader = get_rest_dataloader(
        job_id=job_id,
        settings=settings,
    )
    detector: APIMothDetector | APIAnyBugDetector | None = None
    stage_models: dict[type[APIMothClassifier], APIMothClassifier] = {}

    # Look the pipeline up in the full registry (the same one /info advertises
    # and the worker subscribes to) so every advertised slug is dispatchable.
    # Every stage — detector, gate(s), terminal — is taken from this definition
    # and executed by the shared stage engine, so the worker runs whatever the
    # pipeline declares (YOLO26 + Lepidoptera order gate for anybug, FasterRCNN
    # + binary moth gate for the legacy pipelines) with no per-pipeline branching
    # here. No capability guard is needed: any pipeline the registry advertises
    # is composed of stages this path can run.
    pipeline_def = PIPELINE_CHOICES[pipeline]

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    items = 0

    total_detection_time = 0.0
    total_classification_time = 0.0
    total_dl_time = 0.0
    total_detections = 0
    _, t = log_time()
    result_poster: ResultPoster | None = None
    # Conditionally use CUDA prefetcher; fall back to plain iterator on CPU
    if torch.cuda.is_available():
        batch_source = CUDAPrefetcher(
            loader, device
        )  # __init__ already calls preload()
    else:
        batch_source = iter(loader)

    _, t_total = log_time()
    try:
        for i, batch in enumerate(batch_source):
            cls_time = 0.0
            det_time = 0.0
            load_time, t = t()
            total_dl_time += load_time
            if not batch:
                logger.warning(f"Batch {i + 1} is empty, skipping")
                continue

            # Defer instantiation of poster, detector and classifier stages until
            # we have data, so an empty job loads no model weights. Every stage
            # (detector, gate(s), terminal) comes from the pipeline definition and
            # is loaded once here, then reused across batches.
            if detector is None:
                detector = pipeline_def.detector(source_images=[])
                stage_models = _build_stage_models(pipeline_def)
                result_poster = ResultPoster(max_pending=MAX_PENDING_POSTS)

            assert detector is not None, "Detector not initialized"
            assert result_poster is not None, "ResultPoster not initialized"
            detector.reset([])
            did_work = True

            n_items, n_detections, batch_results, det_time, cls_time = _process_batch(
                batch,
                i,
                detector,
                stage_models,
                pipeline_def,
                pipeline,
            )
            items += n_items
            total_detections += n_detections
            total_detection_time += det_time
            total_classification_time += cls_time

            # Post results asynchronously (non-blocking)
            result_poster.post_async(
                settings.antenna_api_base_url,
                settings.antenna_api_auth_token,
                job_id,
                batch_results,
            )
            batch_total, t_total = t_total()
            logger.info(
                f"Batch {i + 1}: {batch_total/max(n_items, 1):.2f}s/image, "
                f"Classification time: {cls_time:.2f}s, Detection time: {det_time:.2f}s, "
                f"Load time: {load_time:.2f}s"
            )
            (
                _,
                t,
            ) = log_time()  # reset before next() call to measure next batch's load time

            if on_batch_complete:
                on_batch_complete(batch_num=i, items=items)

        if result_poster:
            # Wait for all async posts to complete before finishing the job
            logger.info("Waiting for all pending result posts to complete...")
            result_poster.wait_for_all_posts(min_timeout=60, per_post_timeout=30)

            # Get final metrics
            post_metrics = result_poster.get_metrics()

            logger.info(
                f"Done, detections: {total_detections}. Detecting time: {total_detection_time:.2f}s, "
                f"classification time: {total_classification_time:.2f}s, dl time: {total_dl_time:.2f}s, "
                f"result posts: {post_metrics.total_posts} "
                f"(success: {post_metrics.successful_posts}, failed: {post_metrics.failed_posts}, "
                f"success rate: {post_metrics.success_rate:.1f}%, avg post time: "
                f"{post_metrics.total_post_time / post_metrics.total_posts if post_metrics.total_posts > 0 else 0:.2f}s, "
                f"max queue size: {post_metrics.max_queue_size})"
            )
        return did_work
    finally:
        if result_poster:
            result_poster.shutdown()
