"""
Fast API interface for processing images through the localization and classification
pipelines.
"""

import enum
import time

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
from .models.localization import APIMothDetector
from .schemas import (
    AlgorithmCategoryMapResponse,
    AlgorithmConfigResponse,
    DetectionResponse,
    PipelineConfigResponse,
)
from .schemas import PipelineRequest as PipelineRequest_
from .schemas import PipelineResultsResponse as PipelineResponse_
from .schemas import ProcessingServiceInfoResponse, SourceImage, SourceImageResponse

app = fastapi.FastAPI()
app.add_middleware(GZipMiddleware)


CLASSIFIER_CHOICES = {
    "panama_moths_2023": MothClassifierPanama,
    "panama_moths_2024": MothClassifierPanama2024,
    "quebec_vermont_moths_2023": MothClassifierQuebecVermont,
    "uk_denmark_moths_2023": MothClassifierUKDenmark,
    "costa_rica_moths_turing_2024": MothClassifierTuringCostaRica,
    "anguilla_moths_turing_2024": MothClassifierTuringAnguilla,
    "kenya-uganda_moths_turing_2024": MothClassifierTuringKenyaUganda,
    "global_moths_2024": MothClassifierGlobal,
    "moth_binary": MothClassifierBinary,
    "insect_orders_2025": InsectOrderClassifier,
}
_classifier_choices = dict(
    zip(CLASSIFIER_CHOICES.keys(), list(CLASSIFIER_CHOICES.keys()))
)


PipelineChoice = enum.Enum("PipelineChoice", _classifier_choices)


def should_filter_detections(Classifier: type[APIMothClassifier]) -> bool:
    if Classifier in [MothClassifierBinary, InsectOrderClassifier]:
        return False
    else:
        return True


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
    Classifier: type[APIMothClassifier],
    slug: str,
) -> PipelineConfigResponse:
    """
    Create a configuration for an entire pipeline, given a species classifier class.
    """
    algorithms = []

    detector = APIMothDetector(
        source_images=[],
    )
    algorithms.append(make_algorithm_config_response(detector))

    if should_filter_detections(Classifier):
        binary_classifier = MothClassifierBinary(
            source_images=[],
            detections=[],
            terminal=False,
        )
        algorithms.append(make_algorithm_config_response(binary_classifier))

    classifier = Classifier(
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


# @TODO This requires loading all models into memory! Can we avoid this?
PIPELINE_CONFIGS = [
    make_pipeline_config_response(classifier_class, slug=key)
    for key, classifier_class in CLASSIFIER_CHOICES.items()
]


class PipelineRequest(PipelineRequest_):
    pipeline: PipelineChoice = pydantic.Field(
        description=PipelineRequest_.model_fields["pipeline"].description,
        examples=list(_classifier_choices.keys()),
    )


class PipelineResponse(PipelineResponse_):
    pipeline: PipelineChoice = pydantic.Field(
        PipelineChoice,
        description=PipelineResponse_.model_fields["pipeline"].description,
        examples=list(_classifier_choices.keys()),
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

    Classifier = CLASSIFIER_CHOICES[str(data.pipeline)]

    detector = APIMothDetector(
        source_images=source_images,
        batch_size=settings.localization_batch_size,
        num_workers=settings.num_workers,
        # single=True if len(source_images) == 1 else False,
        single=True,  # @TODO solve issues with reading images in multiprocessing
    )
    detector_results = detector.run()
    num_pre_filter = len(detector_results)
    algorithms_used[detector.get_key()] = make_algorithm_response(detector)

    detections_for_terminal_classifier: list[DetectionResponse] = []
    detections_to_return: list[DetectionResponse] = []

    if should_filter_detections(Classifier):
        filter = MothClassifierBinary(
            source_images=source_images,
            detections=detector_results,
            batch_size=settings.classification_batch_size,
            num_workers=settings.num_workers,
            # single=True if len(detector_results) == 1 else False,
            single=True,  # @TODO solve issues with reading images in multiprocessing
            terminal=False,
        )
        filter.run()
        algorithms_used[filter.get_key()] = make_algorithm_response(filter)

        # Compare num detections with num moth detections
        num_post_filter = len(filter.results)
        logger.info(
            f"Binary classifier returned {num_post_filter} of {num_pre_filter} detections"
        )

        # Filter results based on positive_binary_label
        moth_detections = []
        non_moth_detections = []
        for detection in filter.results:
            for classification in detection.classifications:
                if classification.classification == filter.positive_binary_label:
                    moth_detections.append(detection)
                elif classification.classification == filter.negative_binary_label:
                    non_moth_detections.append(detection)
                break
        detections_for_terminal_classifier += moth_detections
        detections_to_return += non_moth_detections

    else:
        logger.info("Skipping binary classification filter")
        detections_for_terminal_classifier += detector_results

    logger.info(
        f"Sending {len(detections_for_terminal_classifier)} of {num_pre_filter} "
        "detections to the classifier"
    )

    classifier: APIMothClassifier = Classifier(
        source_images=source_images,
        detections=detections_for_terminal_classifier,
        batch_size=settings.classification_batch_size,
        num_workers=settings.num_workers,
        # single=True if len(filtered_detections) == 1 else False,
        single=True,  # @TODO solve issues with reading images in multiprocessing
        example_config_param=data.config.example_config_param,
        terminal=True,
        # critera=data.config.criteria, # @TODO another approach to intermediate filter models
    )
    classifier.run()
    end_time = time.time()
    seconds_elapsed = float(end_time - start_time)
    algorithms_used[classifier.get_key()] = make_algorithm_response(classifier)

    # Return all detections, including those that were not classified as moths
    detections_to_return += classifier.results

    logger.info(
        f"Processed {len(source_images)} images in {seconds_elapsed:.2f} seconds"
    )
    logger.info(f"Returning {len(detections_to_return)} detections")
    # print(all_detections)

    # If the number of detections is greater than 100, its suspicious. Log it.
    if len(detections_to_return) > 100:
        logger.warning(
            f"Detected {len(detections_to_return)} detections. "
            "This is suspicious and may contain duplicates."
        )

    response = PipelineResponse(
        pipeline=data.pipeline,
        algorithms=algorithms_used,
        source_images=source_image_results,
        detections=detections_to_return,
        total_time=seconds_elapsed,
    )
    return response


@app.get("/info", tags=["services"])
async def info() -> ProcessingServiceInfoResponse:
    info = ProcessingServiceInfoResponse(
        name="Antenna Inference API",
        description=(
            "The primary endpoint for processing images for the Antenna platform. "
            "This API provides access to multiple detection and classification "
            "algorithms by multiple labs for processing images of moths."
        ),
        pipelines=PIPELINE_CONFIGS,
        # algorithms=list(algorithm_choices.values()),
    )
    return info


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
    if _classifier_choices:
        return fastapi.responses.JSONResponse(
            status_code=200, content={"status": list(_classifier_choices.keys())}
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=2000)
