"""
Fast API interface for processing images through the localization and classification
pipelines.
"""

import dataclasses
import enum
import time
from collections.abc import Callable
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


def _is_model_class(obj: object, task_type: str) -> bool:
    """True when ``obj`` is a model class whose ``task_type`` matches, e.g.
    "localization" for a detector or "classification" for a classifier. Used to
    validate that each pipeline stage is filled with a model of the right role.
    """
    return isinstance(obj, type) and getattr(obj, "task_type", None) == task_type


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

    def __post_init__(self) -> None:
        # Validate the slot's role on construction rather than trusting callers:
        # a gate must be a classification model, and pass_labels must be a tuple
        # (or None to mean "tag every detection and let all of them pass").
        if not _is_model_class(self.classifier, "classification"):
            raise TypeError(
                "IntermediateStage.classifier must be a classification model "
                f"class, got {self.classifier!r}"
            )
        if self.pass_labels is not None and not isinstance(self.pass_labels, tuple):
            raise TypeError(
                "IntermediateStage.pass_labels must be a tuple of labels or None, "
                f"got {self.pass_labels!r}"
            )


@dataclasses.dataclass(frozen=True)
class PipelineDefinition:
    """Ordered stage definition for a pipeline: one detector, zero or more
    intermediate gate classifiers, then one terminal classifier. This is the
    single source of truth for how a pipeline slug is processed and advertised.

    The fields are named by the role each stage plays (detector / intermediate
    gate / terminal), and the shape is validated on construction: exactly one
    localization model in ``detector``, exactly one classification model in
    ``terminal``, and every ``intermediates`` entry an ``IntermediateStage``.
    Because the detector and terminal are distinct single fields, "one detector
    first, one terminal last" is enforced structurally, never by list position.

    ``name`` and ``description`` are what the platform shows an operator choosing
    a pipeline. Set them whenever two pipelines share a terminal classifier: the
    advertised name falls back to the terminal classifier's, so without them two
    such pipelines are indistinguishable in the picker.
    """

    detector: type[APIMothDetector | APIAnyBugDetector]
    terminal: type[APIMothClassifier]
    intermediates: tuple[IntermediateStage, ...] = ()
    name: str | None = None
    description: str | None = None

    def __post_init__(self) -> None:
        if not _is_model_class(self.detector, "localization"):
            raise TypeError(
                "PipelineDefinition.detector must be a localization model class, "
                f"got {self.detector!r}"
            )
        if not _is_model_class(self.terminal, "classification"):
            raise TypeError(
                "PipelineDefinition.terminal must be a classification model class, "
                f"got {self.terminal!r}"
            )
        if not isinstance(self.intermediates, tuple) or not all(
            isinstance(stage, IntermediateStage) for stage in self.intermediates
        ):
            raise TypeError(
                "PipelineDefinition.intermediates must be a tuple of "
                f"IntermediateStage, got {self.intermediates!r}"
            )


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
    # gate: the stage list is the single source of truth for whether a binary
    # filter runs.
    "moth_binary": PipelineDefinition(APIMothDetector, MothClassifierBinary),
    "insect_orders_2025": PipelineDefinition(APIMothDetector, InsectOrderClassifier),
    # YOLO26 "any-bug" detector -> Lepidoptera order gate -> global species.
    # This pipeline shares its terminal classifier with "global_moths_2024", so it
    # must set its own name; otherwise both advertise the terminal classifier's
    # name and an operator sees two identical entries.
    "anybug_global_moths_2024": PipelineDefinition(
        detector=APIAnyBugDetector,
        terminal=MothClassifierGlobal,
        intermediates=(LEPIDOPTERA_ORDER_GATE,),
        name="Global moths with Anybug detector",
        description=(
            "Detects any insect with the YOLO26 'any-bug' detector, keeps the "
            "detections whose predicted order is Lepidoptera, and classifies "
            "those to species with the global moth model. Detections of other "
            "orders are returned tagged with the order that was predicted."
        ),
    ),
}
_pipeline_choices = dict(zip(PIPELINE_CHOICES.keys(), list(PIPELINE_CHOICES.keys())))


PipelineChoice = enum.Enum("PipelineChoice", _pipeline_choices)


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

    The advertised name and description come from the pipeline when it sets them
    and from the terminal classifier otherwise, so that pipelines sharing a
    terminal classifier can still be told apart in the platform's picker.
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
        name=pipeline.name or classifier.name,
        slug=slug,
        description=pipeline.description or classifier.description,
        version=1,
        algorithms=algorithms,
    )


# How one classifier stage (an intermediate gate or the terminal classifier) is
# executed on the detections it is handed. Given the stage's class, its
# detections, and whether it is the terminal stage, the runner returns the stage
# instance with its predictions attached to those detections and its ``results``
# set to every detection it was handed.
#
# Parameterizing this is what lets the FastAPI /process path and the in-process
# worker (antenna/worker.py) share the stage-orchestration engine below while
# each supplies its own inference mechanism, so the two paths cannot drift.
# /process constructs a fresh model and reads crops from image URLs; the worker
# reuses a preloaded model and slices crops from image tensors already in memory.
StageRunner = Callable[
    [type[APIMothClassifier], list[DetectionResponse], bool], APIMothClassifier
]


def _make_default_stage_runner(
    source_images: list[SourceImage],
    example_config_param: int | None,
) -> StageRunner:
    """The /process stage runner: construct each stage fresh and run it through
    its own URL-reading dataset. Only the terminal classifier receives
    ``example_config_param``, matching the request-config contract.
    """

    def run_stage(
        classifier_class: type[APIMothClassifier],
        detections: list[DetectionResponse],
        terminal: bool,
    ) -> APIMothClassifier:
        kwargs: dict = {
            "source_images": source_images,
            "detections": detections,
            "batch_size": settings.classification_batch_size,
            "num_workers": settings.num_workers,
            # "single": True if len(detections) == 1 else False,
            # @TODO solve issues with reading images in multiprocessing
            "single": True,
            "terminal": terminal,
        }
        if terminal:
            kwargs["example_config_param"] = example_config_param
        stage = classifier_class(**kwargs)
        stage.run()
        return stage

    return run_stage


def run_classification_stages(
    pipeline: PipelineDefinition,
    source_images: list[SourceImage],
    detector_results: list[DetectionResponse],
    example_config_param: int | None,
    algorithms_used: dict[str, AlgorithmConfigResponse],
    run_stage: StageRunner | None = None,
) -> list[DetectionResponse]:
    """Run each intermediate gate, then the terminal classifier, and return every
    detection exactly once.

    This is the shared classification engine for both the FastAPI /process path
    and the in-process worker. Only *how* one stage runs its inference differs
    between them, so that step is delegated to ``run_stage``; the stage order,
    the pass/fail gate split, algorithm tracking, and the final assembly all live
    here, once, so the two paths cannot drift. When ``run_stage`` is omitted the
    /process behavior (construct fresh, read crops from URLs) is used.

    A detection that fails a gate is returned tagged with that gate's prediction
    (marked non-terminal); a detection that passes every gate is returned tagged
    additionally with the terminal classifier's prediction (marked terminal), so
    the platform never shows a gate label as a passing detection's best result.

    The assembly is keyed on detection IDENTITY, never on list position: every
    stage annotates the ``DetectionResponse`` objects it is handed in place, and
    the passing/non-passing split routes each object by its own top gate label.
    A detection a gate drops, or one whose crop the terminal classifier cannot
    read, therefore keeps its own label instead of shifting the labels of the
    detections that follow it. ``algorithms_used`` is populated in place.
    """
    if run_stage is None:
        run_stage = _make_default_stage_runner(source_images, example_config_param)

    num_pre_filter = len(detector_results)
    detections_to_return: list[DetectionResponse] = []
    # Detections that survive every gate so far and continue to the next stage.
    detections_for_next_stage: list[DetectionResponse] = detector_results

    for stage in pipeline.intermediates:
        gate = run_stage(stage.classifier, detections_for_next_stage, False)
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

    classifier = run_stage(pipeline.terminal, detections_for_next_stage, True)
    algorithms_used[classifier.get_key()] = make_algorithm_response(classifier)

    # Return all detections, including those a gate filtered out upstream.
    detections_to_return += classifier.results
    return detections_to_return


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
    algorithms_used[detector.get_key()] = make_algorithm_response(detector)

    detections_to_return = run_classification_stages(
        pipeline=pipeline,
        source_images=source_images,
        detector_results=detector_results,
        example_config_param=data.config.example_config_param,
        algorithms_used=algorithms_used,
    )
    end_time = time.time()
    seconds_elapsed = float(end_time - start_time)

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
            # One pipeline that cannot be built must not take /info down with it,
            # because that would make every other pipeline unavailable too. The
            # pipeline is left out of the response and the reason is logged, so a
            # stage whose weight cannot be fetched costs only its own pipeline.
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
