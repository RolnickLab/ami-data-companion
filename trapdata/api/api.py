"""
Fast API interface for processing images through the localization and classification
pipelines.
"""

import datetime
import enum
import time
from contextlib import asynccontextmanager

import fastapi
import pydantic
from fastapi.middleware.gzip import GZipMiddleware

from ..common.logs import logger  # noqa: F401
from ..ml.utils import resolve_model_url
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
from .models.feature_extraction import APIBioCLIP25FeatureExtractor, APIFeatureExtractor
from .models.localization import APIMothDetector
from .schemas import (
    AlgorithmCategoryMapResponse,
    AlgorithmConfigResponse,
    DetectionRequest,
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

# Feature extractors that can embed every detection in any pipeline, keyed by algorithm
# key. See the embedding_extractor request config and AMI_EMBEDDING_EXTRACTOR setting.
EMBEDDING_EXTRACTOR_CHOICES: dict[str, type[APIFeatureExtractor]] = {
    APIBioCLIP25FeatureExtractor.get_key(): APIBioCLIP25FeatureExtractor,
}

# Pipelines that only embed. They add no classification, and when a request carries
# existing detections they embed those boxes instead of running the detector.
FEATURE_PIPELINE_CHOICES: dict[str, type[APIFeatureExtractor]] = {
    "bioclip_2_5_features": APIBioCLIP25FeatureExtractor,
}

PIPELINE_CHOICES: dict[str, type[APIMothClassifier] | type[APIFeatureExtractor]] = {
    **CLASSIFIER_CHOICES,
    **FEATURE_PIPELINE_CHOICES,
}


def parse_pipeline_setting(value: str) -> list[str]:
    """Split the AMI_PIPELINES setting, a comma-separated list of slugs, into slugs."""
    return [slug.strip() for slug in value.split(",") if slug.strip()]


def select_pipelines(
    slugs: list[str] | None = None,
) -> dict[str, type[APIMothClassifier] | type[APIFeatureExtractor]]:
    """
    Return the pipelines this service offers, keyed by slug.

    Offering a pipeline means downloading and loading its models to describe it, which
    takes time, so a deployment can offer a subset. The slugs come from the argument when one is given,
    such as the worker's --pipeline option, and otherwise from the AMI_PIPELINES
    setting, a comma-separated list. When neither names a pipeline, every pipeline in
    PIPELINE_CHOICES is offered. An unknown slug raises ValueError, so a typo stops
    the service at startup instead of quietly leaving a pipeline out.
    """
    from_setting = slugs is None
    if slugs is None:
        slugs = parse_pipeline_setting(settings.pipelines)
    if not slugs:
        return dict(PIPELINE_CHOICES)
    unknown = [slug for slug in slugs if slug not in PIPELINE_CHOICES]
    if unknown:
        where = " in the AMI_PIPELINES setting" if from_setting else ""
        raise ValueError(
            f"Unknown pipeline(s){where}: {', '.join(unknown)}. "
            f"Must be one of: {', '.join(PIPELINE_CHOICES)}"
        )
    return {slug: PIPELINE_CHOICES[slug] for slug in slugs}


def classifier_pipelines(slugs: list[str]) -> list[str]:
    """
    Keep the pipelines the Antenna worker can run, which is every pipeline except the
    feature-only ones, since those are served by the API alone.
    """
    skipped = [slug for slug in slugs if slug in FEATURE_PIPELINE_CHOICES]
    if skipped:
        logger.info(f"The worker does not run feature-only pipelines: {skipped}")
    return [slug for slug in slugs if slug not in FEATURE_PIPELINE_CHOICES]


def resolve_embedding_extractor(
    requested: str | None = None,
) -> type[APIFeatureExtractor] | None:
    """
    Return the feature extractor that should embed every detection, or None.

    A key in the request wins; otherwise the AMI_EMBEDDING_EXTRACTOR setting applies.
    An empty key means none. An unknown key raises ValueError.
    """
    key = settings.embedding_extractor if requested is None else requested
    key = (key or "").strip()
    if not key:
        return None
    if key not in EMBEDDING_EXTRACTOR_CHOICES:
        raise ValueError(
            f"Unknown embedding extractor: {key}. "
            f"Must be one of: {', '.join(EMBEDDING_EXTRACTOR_CHOICES)}"
        )
    return EMBEDDING_EXTRACTOR_CHOICES[key]


def _offered_pipeline_slugs() -> list[str]:
    """
    Name the pipelines for the request and response schema, so that the API docs list
    only the pipelines this server offers.

    The schema is built once, when this module is imported. Every ami command imports
    this module, so an invalid AMI_PIPELINES setting must not break the import: it
    falls back to every pipeline here, and the API server still refuses to start,
    because initialize_service_info calls select_pipelines again at startup.
    """
    try:
        return list(select_pipelines())
    except ValueError:
        return list(PIPELINE_CHOICES)


_offered_pipeline_choices = {slug: slug for slug in _offered_pipeline_slugs()}
PipelineChoice = enum.Enum("PipelineChoice", _offered_pipeline_choices)


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
        uri=resolve_model_url(model.labels_path),
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
        uri=resolve_model_url(model.weights_path),
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
        uri=resolve_model_url(model.weights_path),
    )


def make_extractor_config_response(
    Extractor: type[APIFeatureExtractor],
) -> AlgorithmConfigResponse:
    """Describe a feature extractor from its class, without loading its backbone."""
    return AlgorithmConfigResponse(
        name=Extractor.name,
        key=Extractor.get_key(),
        task_type=Extractor.task_type,
        description=Extractor.description,
        category_map=None,
        uri=Extractor.model_uri,
    )


def make_pipeline_config_response(
    Classifier: type[APIMothClassifier] | type[APIFeatureExtractor],
    slug: str,
    include_embedding_extractor: bool = True,
) -> PipelineConfigResponse:
    """
    Create a configuration for an entire pipeline, given its species classifier class,
    or its feature extractor class for a feature-only pipeline.

    The AMI_EMBEDDING_EXTRACTOR setting adds its extractor to classifier pipelines
    only when include_embedding_extractor is on, because only the API runs it.
    """
    algorithms = []

    detector = APIMothDetector(
        source_images=[],
    )
    algorithms.append(make_algorithm_config_response(detector))

    if slug in FEATURE_PIPELINE_CHOICES:
        algorithms.append(make_extractor_config_response(Classifier))
        return PipelineConfigResponse(
            name=f"{Classifier.name} only",
            slug=slug,
            description=(
                f"{Classifier.description} Embeds the detections sent with the "
                "request without re-detecting; detects first when none are sent."
            ),
            version=1,
            algorithms=algorithms,
        )

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

    Extractor = resolve_embedding_extractor() if include_embedding_extractor else None
    if Extractor:
        algorithms.append(make_extractor_config_response(Extractor))

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
        examples=list(_offered_pipeline_choices.keys()),
    )


class PipelineResponse(PipelineResponse_):
    pipeline: PipelineChoice = pydantic.Field(
        PipelineChoice,
        description=PipelineResponse_.model_fields["pipeline"].description,
        examples=list(_offered_pipeline_choices.keys()),
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

    try:
        Extractor = resolve_embedding_extractor(data.config.embedding_extractor)
    except ValueError as e:
        raise fastapi.HTTPException(status_code=422, detail=str(e)) from e

    if str(data.pipeline) in FEATURE_PIPELINE_CHOICES:
        return run_feature_pipeline(
            data,
            FEATURE_PIPELINE_CHOICES[str(data.pipeline)],
            source_images,
            source_image_results,
            start_time,
        )

    Advertised = resolve_embedding_extractor()
    if Extractor and Extractor is not Advertised:
        # Antenna rejects a result that names an algorithm the pipeline does not
        # list in /info, so only the extractor from the setting may be requested.
        raise fastapi.HTTPException(
            status_code=422,
            detail=(
                f"Embedding extractor {Extractor.get_key()} is not offered by the "
                f"{data.pipeline} pipeline on this service. Offered: "
                f"{Advertised.get_key() if Advertised else 'none'}."
            ),
        )

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
    non_moth_detections: list[DetectionResponse] = []
    features_for_all_detections = (
        data.config.features_for_all_detections
        if data.config.features_for_all_detections is not None
        else settings.features_for_all_detections
    )

    if should_filter_detections(Classifier):
        filter = MothClassifierBinary(
            source_images=source_images,
            detections=detector_results,
            batch_size=settings.classification_batch_size,
            num_workers=settings.num_workers,
            # single=True if len(detector_results) == 1 else False,
            single=True,  # @TODO solve issues with reading images in multiprocessing
            terminal=False,
            # The binary gate has no backbone hook, so only logits are worth
            # passing on; asking it for features would return nothing.
            include_logits=data.config.include_logits,
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
        include_features=data.config.include_features,
        include_logits=data.config.include_logits,
        include_embeddings=features_for_all_detections,
        terminal=True,
        # critera=data.config.criteria, # @TODO another approach to intermediate filter models
    )
    classifier.run()
    # Return all detections, including those that were not classified as moths
    detections_to_return += classifier.results

    if classifier.produces_embeddings and non_moth_detections:
        # Same model as the moth detections' vectors, so every vector in the
        # response is comparable. Adds no classification to these detections.
        classifier.embed(non_moth_detections)

    if Extractor:
        embed_detections(Extractor, source_images, detections_to_return)
        algorithms_used[Extractor.get_key()] = make_extractor_config_response(Extractor)
    end_time = time.time()
    seconds_elapsed = float(end_time - start_time)
    algorithms_used[classifier.get_key()] = make_algorithm_response(classifier)

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


def embed_detections(
    Extractor: type[APIFeatureExtractor],
    source_images: list[SourceImage],
    detections: list[DetectionResponse],
) -> list[DetectionResponse]:
    """Attach the extractor's embedding to each detection, in place."""
    extractor = Extractor(
        source_images=source_images,
        batch_size=settings.classification_batch_size,
        num_workers=settings.num_workers,
        single=True,  # @TODO solve issues with reading images in multiprocessing
    )
    return extractor.embed(detections)


def detections_from_request(
    requested: list[DetectionRequest],
    source_images: list[SourceImage],
    source_image_results: list[SourceImageResponse],
) -> list[DetectionResponse]:
    """
    Turn the request's existing detections into response detections, boxes unchanged.

    A detection without a box, or with a box of no area, cannot be cropped and is
    left out. A detection whose image is missing from the request's source images
    adds that image.
    """
    known_ids = {image.id for image in source_images}
    detections = []
    for detection in requested:
        bbox = detection.bbox
        if bbox is None or bbox.x1 >= bbox.x2 or bbox.y1 >= bbox.y2:
            continue
        image = detection.source_image
        if image.id not in known_ids:
            known_ids.add(image.id)
            source_images.append(SourceImage(**image.model_dump()))
            source_image_results.append(SourceImageResponse(**image.model_dump()))
        detections.append(
            DetectionResponse(
                source_image_id=image.id,
                bbox=detection.bbox,
                algorithm=detection.algorithm,
                crop_image_url=detection.crop_image_url,
                timestamp=datetime.datetime.now(),
            )
        )
    skipped = len(requested) - len(detections)
    if skipped:
        logger.info(
            f"Skipped {skipped} requested detections without a box that has an area"
        )
    return detections


def run_feature_pipeline(
    data: PipelineRequest,
    Extractor: type[APIFeatureExtractor],
    source_images: list[SourceImage],
    source_image_results: list[SourceImageResponse],
    start_time: float,
) -> PipelineResponse:
    """
    Embed every detection and classify none of them.

    Detections sent with the request are embedded as they are, and the detector does
    not run; without them, the detector runs first.
    """
    if data.detections:
        detections = detections_from_request(
            data.detections, source_images, source_image_results
        )
    else:
        detector = APIMothDetector(
            source_images=source_images,
            batch_size=settings.localization_batch_size,
            num_workers=settings.num_workers,
            single=True,  # @TODO solve issues with reading images in multiprocessing
        )
        detections = detector.run()

    embed_detections(Extractor, source_images, detections)
    seconds_elapsed = float(time.time() - start_time)
    logger.info(
        f"Embedded {len(detections)} detections from {len(source_images)} images "
        f"in {seconds_elapsed:.2f} seconds"
    )
    return PipelineResponse(
        pipeline=data.pipeline,
        source_images=source_image_results,
        detections=detections,
        total_time=seconds_elapsed,
    )


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
    enabled_pipelines = list(select_pipelines())
    if enabled_pipelines:
        return fastapi.responses.JSONResponse(
            status_code=200, content={"status": enabled_pipelines}
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


def initialize_service_info(
    pipelines: list[str] | None = None,
    include_embedding_extractor: bool = True,
) -> ProcessingServiceInfoResponse:
    """
    Describe the pipelines this service offers, for the /info endpoint and for
    registering the pipelines with Antenna.

    Describing a pipeline loads its models into memory, so only the pipelines chosen
    by select_pipelines are included. The worker registers its pipelines with
    include_embedding_extractor off, because it does not run the extractor, and
    Antenna must not record an algorithm that never produces output.
    """
    # Check the setting here, so a typo stops the service at startup.
    resolve_embedding_extractor()
    pipeline_configs = [
        make_pipeline_config_response(
            classifier_class,
            slug=key,
            include_embedding_extractor=include_embedding_extractor,
        )
        for key, classifier_class in select_pipelines(pipelines).items()
    ]

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
