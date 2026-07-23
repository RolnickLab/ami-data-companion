"""
Fast API interface for processing images through the localization and classification
pipelines.
"""

import dataclasses
import enum
import time
from contextlib import asynccontextmanager

import fastapi
import pydantic
from fastapi.middleware.gzip import GZipMiddleware

from ..common.logs import logger  # noqa: F401
from . import settings
from .models.classification import (
    APIMothClassifier,
    InsectOrderClassifier,
    MothClassifierBinary,
    MothClassifierGlobal,
    MothClassifierPanama,
    MothClassifierPanama2024,
    MothClassifierQuebecVermont,
    MothClassifierTuringAnguilla,
    MothClassifierTuringCostaRica,
    MothClassifierTuringKenyaUganda,
    MothClassifierUKDenmark,
)
from .models.localization import APIAnyBugDetector, APIMothDetector
from .schemas import (
    AlgorithmCategoryMapResponse,
    AlgorithmConfigResponse,
    DetectionResponse,
    PipelineConfigResponse,
)
from .schemas import PipelineRequest as PipelineRequest_
from .schemas import PipelineResultsResponse as PipelineResponse_
from .schemas import ProcessingServiceInfoResponse, SourceImage, SourceImageResponse


@asynccontextmanager
async def lifespan(app: fastapi.FastAPI):
    # cache the service info to be built only once at startup
    app.state.service_info = initialize_service_info()
    logger.info("Initialized service info")
    yield
    # Shutdown event: Clean up resources (if necessary)
    logger.info("Shutting down API")


app = fastapi.FastAPI(lifespan=lifespan)
app.add_middleware(GZipMiddleware)


# Confirmed from the insect_orders_2025 category map
# (ami-models/insect_orders/insect_order_category_map.json): the Lepidoptera
# order is labelled exactly "Lepidoptera" (index 0). The order gate below
# matches this string against each detection's top predicted order.
LEPIDOPTERA_ORDER_LABEL = "Lepidoptera"


@dataclasses.dataclass(frozen=True)
class IntermediateStage:
    """An intermediate classifier that gates detections between the detector and
    the terminal classifier.

    Only detections whose top predicted label (from this classifier) is in
    ``pass_labels`` are forwarded to the next stage. Non-passing detections are
    returned to the caller tagged with this classifier's prediction. When
    ``pass_labels`` is ``None`` every detection passes and the stage only tags.
    """

    classifier: type[APIMothClassifier]
    pass_labels: tuple[str, ...] | None = None


@dataclasses.dataclass(frozen=True)
class PipelineDefinition:
    """Ordered stage definition for a pipeline: one detector, zero or more
    intermediate gate classifiers, then one terminal classifier. This is the
    single source of truth for how a pipeline slug is processed and advertised.
    """

    detector: type[APIMothDetector | APIAnyBugDetector]
    terminal: type[APIMothClassifier]
    intermediates: tuple[IntermediateStage, ...] = ()


# The binary moth/non-moth filter used by every species pipeline: only "moth"
# detections proceed; everything else is returned tagged non-moth.
BINARY_MOTH_FILTER = IntermediateStage(
    classifier=MothClassifierBinary,
    pass_labels=(MothClassifierBinary.positive_binary_label,),
)

# The insect-order gate used by the anybug pipeline: only detections whose top
# predicted order is Lepidoptera proceed to the species classifier; detections
# of other orders are returned tagged with their order.
LEPIDOPTERA_ORDER_GATE = IntermediateStage(
    classifier=InsectOrderClassifier,
    pass_labels=(LEPIDOPTERA_ORDER_LABEL,),
)


PIPELINE_CHOICES: dict[str, PipelineDefinition] = {
    "panama_moths_2023": PipelineDefinition(
        APIMothDetector, MothClassifierPanama, (BINARY_MOTH_FILTER,)
    ),
    "panama_moths_2024": PipelineDefinition(
        APIMothDetector, MothClassifierPanama2024, (BINARY_MOTH_FILTER,)
    ),
    "quebec_vermont_moths_2023": PipelineDefinition(
        APIMothDetector, MothClassifierQuebecVermont, (BINARY_MOTH_FILTER,)
    ),
    "uk_denmark_moths_2023": PipelineDefinition(
        APIMothDetector, MothClassifierUKDenmark, (BINARY_MOTH_FILTER,)
    ),
    "costa_rica_moths_turing_2024": PipelineDefinition(
        APIMothDetector, MothClassifierTuringCostaRica, (BINARY_MOTH_FILTER,)
    ),
    "anguilla_moths_turing_2024": PipelineDefinition(
        APIMothDetector, MothClassifierTuringAnguilla, (BINARY_MOTH_FILTER,)
    ),
    "kenya-uganda_moths_turing_2024": PipelineDefinition(
        APIMothDetector, MothClassifierTuringKenyaUganda, (BINARY_MOTH_FILTER,)
    ),
    "global_moths_2024": PipelineDefinition(
        APIMothDetector, MothClassifierGlobal, (BINARY_MOTH_FILTER,)
    ),
    # The binary and order classifiers are themselves terminal, with no upstream
    # filter (they replace the old should_filter_detections() special case).
    "moth_binary": PipelineDefinition(APIMothDetector, MothClassifierBinary),
    "insect_orders_2025": PipelineDefinition(APIMothDetector, InsectOrderClassifier),
    # NEW: YOLO26 "any-bug" detector -> Lepidoptera order gate -> global species.
    # TODO(anybug): dormant until the yolo26n.pt weight is uploaded and
    # `ultralytics` is installed. initialize_service_info() tolerates the build
    # failure so this pipeline is registered without breaking /info or startup
    # for the other pipelines. Remove that tolerance once the weight is live.
    "anybug_global_moths_2024": PipelineDefinition(
        detector=APIAnyBugDetector,
        terminal=MothClassifierGlobal,
        intermediates=(LEPIDOPTERA_ORDER_GATE,),
    ),
}
_pipeline_choices = dict(zip(PIPELINE_CHOICES.keys(), list(PIPELINE_CHOICES.keys())))


PipelineChoice = enum.Enum("PipelineChoice", _pipeline_choices)


# Backward-compatibility shim. Several call sites outside the FastAPI /process
# path (the antenna GPU worker in antenna/worker.py, the CLI, registration, and
# the test suite) still import CLASSIFIER_CHOICES as a {slug: terminal_classifier}
# mapping and should_filter_detections(). Both are derived from the new
# PIPELINE_CHOICES so those callers keep working unchanged.
#
# The anybug pipeline is intentionally excluded: it uses a different detector
# (APIAnyBugDetector) and an order gate, so the moth-detector-based GPU worker
# must not silently run it with the wrong stages.
#
# TODO(anybug): migrate the antenna/worker.py GPU path to consume PIPELINE_CHOICES
# stage definitions directly, then delete this shim.
CLASSIFIER_CHOICES: dict[str, type[APIMothClassifier]] = {
    slug: pipeline.terminal
    for slug, pipeline in PIPELINE_CHOICES.items()
    if pipeline.detector is APIMothDetector
}


def should_filter_detections(Classifier: type[APIMothClassifier]) -> bool:
    """Deprecated: whether the FasterRCNN/GPU worker path should run the binary
    moth filter ahead of ``Classifier``. The FastAPI path uses the explicit
    stage config in PIPELINE_CHOICES instead. Kept for the antenna GPU worker.
    """
    return Classifier not in (MothClassifierBinary, InsectOrderClassifier)


def top_label_for_algorithm(
    detection: DetectionResponse, algorithm_key: str
) -> str | None:
    """Return the top predicted label assigned to ``detection`` by the algorithm
    with ``algorithm_key``, or ``None`` if that algorithm did not classify it.
    Used by the intermediate gates to decide which detections proceed.
    """
    for classification in detection.classifications:
        if classification.algorithm.key == algorithm_key:
            return classification.classification
    return None


def make_category_map_response(
    model: APIMothDetector | APIMothClassifier,
) -> AlgorithmCategoryMapResponse:
    categories_sorted_by_index = sorted(model.category_map.items(), key=lambda x: x[0])
    # as list of dicts:
    categories_sorted_by_index = [
        {
            "index": index,
            "label": label,
            "taxon_rank": model.default_taxon_rank,
        }
        for index, label in categories_sorted_by_index
    ]
    label_strings_sorted_by_index = [cat["label"] for cat in categories_sorted_by_index]
    return AlgorithmCategoryMapResponse(
        data=categories_sorted_by_index,
        labels=label_strings_sorted_by_index,
        uri=model.labels_path,
    )


def make_algorithm_response(
    model: APIMothDetector | APIMothClassifier,
) -> AlgorithmConfigResponse:
    category_map = make_category_map_response(model) if model.category_map else None
    return AlgorithmConfigResponse(
        name=model.name,
        key=model.get_key(),
        task_type=model.task_type,
        description=model.description,
        category_map=category_map,
        uri=model.weights_path,
    )


def make_algorithm_config_response(
    model: APIMothDetector | APIMothClassifier,
) -> AlgorithmConfigResponse:
    category_map = make_category_map_response(model)
    return AlgorithmConfigResponse(
        name=model.name,
        key=model.get_key(),
        task_type=model.task_type,
        description=model.description,
        category_map=category_map,
        uri=model.weights_path,
    )


def make_pipeline_config_response(
    pipeline: PipelineDefinition,
    slug: str,
) -> PipelineConfigResponse:
    """
    Create a configuration for an entire pipeline by iterating its configured
    stages: detector, any intermediate gate classifiers, then the terminal
    classifier.
    """
    algorithms = []

    detector = pipeline.detector(source_images=[])
    algorithms.append(make_algorithm_config_response(detector))

    for stage in pipeline.intermediates:
        gate = stage.classifier(source_images=[], detections=[], terminal=False)
        algorithms.append(make_algorithm_config_response(gate))

    classifier = pipeline.terminal(
        source_images=[],
        detections=[],
        batch_size=settings.classification_batch_size,
        num_workers=settings.num_workers,
        terminal=True,
    )
    algorithms.append(make_algorithm_config_response(classifier))

    return PipelineConfigResponse(
        name=classifier.name,
        slug=slug,
        description=classifier.description,
        version=1,
        algorithms=algorithms,
    )


class PipelineRequest(PipelineRequest_):
    pipeline: PipelineChoice = pydantic.Field(
        description=PipelineRequest_.model_fields["pipeline"].description,
        examples=list(_pipeline_choices.keys()),
    )


class PipelineResponse(PipelineResponse_):
    pipeline: PipelineChoice = pydantic.Field(
        PipelineChoice,
        description=PipelineResponse_.model_fields["pipeline"].description,
        examples=list(_pipeline_choices.keys()),
    )


@app.get("/")
async def root():
    return fastapi.responses.RedirectResponse("/docs")


@app.post(
    "/pipeline/process/", deprecated=True, tags=["services"]
)  # old endpoint, deprecated, remove after jan 2025
@app.post("/process", tags=["services"])  # new endpoint
@app.post("/process/", tags=["services"])  # new endpoint
async def process(data: PipelineRequest) -> PipelineResponse:
    algorithms_used: dict[str, AlgorithmConfigResponse] = {}

    # Ensure that the source images are unique, filter out duplicates
    source_images_index = {
        source_image.id: source_image for source_image in data.source_images
    }
    incoming_source_images = list(source_images_index.values())
    if len(incoming_source_images) != len(data.source_images):
        logger.warning(
            f"Removed {len(data.source_images) - len(incoming_source_images)} "
            "duplicate source images"
        )

    source_image_results = [
        SourceImageResponse(**image.model_dump()) for image in incoming_source_images
    ]
    source_images = [
        SourceImage(**image.model_dump()) for image in incoming_source_images
    ]

    start_time = time.time()

    pipeline = PIPELINE_CHOICES[str(data.pipeline)]

    detector = pipeline.detector(
        source_images=source_images,
        batch_size=settings.localization_batch_size,
        num_workers=settings.num_workers,
        # single=True if len(source_images) == 1 else False,
        single=True,  # @TODO solve issues with reading images in multiprocessing
    )
    detector_results = detector.run()
    num_pre_filter = len(detector_results)
    algorithms_used[detector.get_key()] = make_algorithm_response(detector)

    detections_to_return: list[DetectionResponse] = []
    # Detections that survive every gate so far and continue to the next stage.
    detections_for_next_stage: list[DetectionResponse] = detector_results

    for stage in pipeline.intermediates:
        gate = stage.classifier(
            source_images=source_images,
            detections=detections_for_next_stage,
            batch_size=settings.classification_batch_size,
            num_workers=settings.num_workers,
            # single=True if len(detections_for_next_stage) == 1 else False,
            single=True,  # @TODO solve issues with reading images in multiprocessing
            terminal=False,
        )
        gate.run()
        algorithms_used[gate.get_key()] = make_algorithm_response(gate)

        if stage.pass_labels is None:
            # Tag-only stage: annotate the detections but let all of them pass.
            detections_for_next_stage = gate.results
            continue

        # Keep detections whose top predicted label (from this gate) is one of
        # the pass labels; return the rest tagged with this gate's prediction.
        passing: list[DetectionResponse] = []
        non_passing: list[DetectionResponse] = []
        for detection in gate.results:
            top_label = top_label_for_algorithm(detection, gate.get_key())
            if top_label in stage.pass_labels:
                passing.append(detection)
            else:
                non_passing.append(detection)
        logger.info(
            f"{gate.name}: {len(passing)} of {len(gate.results)} detections "
            f"passed the {stage.pass_labels} gate"
        )
        detections_to_return += non_passing
        detections_for_next_stage = passing

    logger.info(
        f"Sending {len(detections_for_next_stage)} of {num_pre_filter} "
        "detections to the terminal classifier"
    )

    classifier: APIMothClassifier = pipeline.terminal(
        source_images=source_images,
        detections=detections_for_next_stage,
        batch_size=settings.classification_batch_size,
        num_workers=settings.num_workers,
        # single=True if len(detections_for_next_stage) == 1 else False,
        single=True,  # @TODO solve issues with reading images in multiprocessing
        example_config_param=data.config.example_config_param,
        terminal=True,
    )
    classifier.run()
    end_time = time.time()
    seconds_elapsed = float(end_time - start_time)
    algorithms_used[classifier.get_key()] = make_algorithm_response(classifier)

    # Return all detections, including those a gate filtered out upstream.
    detections_to_return += classifier.results

    logger.info(
        f"Processed {len(source_images)} images in {seconds_elapsed:.2f} seconds"
    )
    logger.info(f"Algorithms used: {list(algorithms_used.keys())}")
    logger.info(f"Returning {len(detections_to_return)} detections")
    # print(all_detections)

    # If the number of detections is greater than 200, its suspicious. Log it.
    if len(detections_to_return) > 200:
        logger.warning(
            f"Detected {len(detections_to_return)} detections. "
            "This is suspicious and may contain duplicates."
        )

    response = PipelineResponse(
        pipeline=data.pipeline,
        source_images=source_image_results,
        detections=detections_to_return,
        total_time=seconds_elapsed,
    )
    return response


@app.get("/info", tags=["services"])
async def info() -> ProcessingServiceInfoResponse:
    return app.state.service_info


# Check if the server is online
@app.get("/livez", tags=["health checks"])
async def livez():
    return fastapi.responses.JSONResponse(status_code=200, content={"status": True})


# Check if the pipelines are ready to process data
@app.get("/readyz", tags=["health checks"])
async def readyz():
    """
    Check if the server is ready to process data.

    Returns a list of pipeline slugs that are online and ready to process data.
    @TODO may need to simplify this to just return True/False. Pipeline algorithms will
    likely be loaded into memory on-demand when the pipeline is selected.
    """
    if _pipeline_choices:
        return fastapi.responses.JSONResponse(
            status_code=200, content={"status": list(_pipeline_choices.keys())}
        )
    else:
        return fastapi.responses.JSONResponse(status_code=503, content={"status": []})


# Future methods

# batch processing
# async def process_batch(data: PipelineRequest) -> PipelineResponse:
#     pass

# render image crops and bboxes on top of the original image
# async def render(data: PipelineRequest) -> PipelineResponse:
#     pass


def initialize_service_info() -> ProcessingServiceInfoResponse:
    # @TODO This requires loading all models into memory! Can we avoid this?
    pipeline_configs = []
    for slug, pipeline in PIPELINE_CHOICES.items():
        try:
            pipeline_configs.append(make_pipeline_config_response(pipeline, slug=slug))
        except Exception as e:
            # Keep the rest of the service online if one pipeline cannot be
            # built, e.g. a stage whose weight is not uploaded yet (the anybug
            # YOLO26 detector). TODO(anybug): remove this tolerance for that
            # pipeline once `ultralytics` and the real weight are available.
            logger.warning(f"Skipping pipeline '{slug}' in /info: {e}")

    _info = ProcessingServiceInfoResponse(
        name="Antenna Inference API",
        description=(
            "The primary endpoint for processing images for the Antenna platform. "
            "This API provides access to multiple detection and classification "
            "algorithms by multiple labs for processing images of moths."
        ),
        pipelines=pipeline_configs,
        # algorithms=list(algorithm_choices.values()),
    )
    return _info


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=2000)
