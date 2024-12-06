"""
Fast API interface for processing images through the localization and classification
pipelines.
"""

import enum
import time

import fastapi
import pydantic
from rich import print

from ..common.logs import logger  # noqa: F401
from . import settings
from .models.classification import (
    APIMothClassifier,
    MothClassifierBinary,
    MothClassifierGlobal,
    MothClassifierPanama,
    MothClassifierPanama2024,
    MothClassifierQuebecVermont,
    MothClassifierTuringAnguilla,
    MothClassifierTuringCostaRica,
    MothClassifierUKDenmark,
)
from .models.localization import APIMothDetector
from .schemas import AlgorithmCategoryMap, AlgorithmResponse
from .schemas import PipelineRequest as PipelineRequest_
from .schemas import PipelineResponse as PipelineResponse_
from .schemas import SourceImage, SourceImageResponse

app = fastapi.FastAPI()


PIPELINE_CHOICES = {
    "panama_moths_2023": MothClassifierPanama,
    "panama_moths_2024": MothClassifierPanama2024,
    "quebec_vermont_moths_2023": MothClassifierQuebecVermont,
    "uk_denmark_moths_2023": MothClassifierUKDenmark,
    "costa_rica_moths_turing_2024": MothClassifierTuringCostaRica,
    "anguilla_moths_turing_2024": MothClassifierTuringAnguilla,
    "global_moths_2024": MothClassifierGlobal,
    "moth_binary": MothClassifierBinary,
}
_pipeline_choices = dict(zip(PIPELINE_CHOICES.keys(), list(PIPELINE_CHOICES.keys())))


PipelineChoice = enum.Enum("PipelineChoice", _pipeline_choices)


def make_category_map_response(
    model_category_map: dict[int, str]
) -> AlgorithmCategoryMap:
    categories_sorted_by_index = sorted(model_category_map.items(), key=lambda x: x[0])
    # as list of dicts:
    categories_sorted_by_index = [
        {"index": index, "label": label} for index, label in categories_sorted_by_index
    ]
    label_strings_sorted_by_index = [cat["label"] for cat in categories_sorted_by_index]
    return AlgorithmCategoryMap(
        data=categories_sorted_by_index,
        labels=label_strings_sorted_by_index,
    )


def make_algorithm_response(
    model: APIMothDetector | APIMothClassifier,
) -> AlgorithmResponse:

    category_map = (
        make_category_map_response(model.category_map) if model.category_map else None
    )
    return AlgorithmResponse(
        name=model.name,
        key=model.get_key(),
        task_type=model.task_type,
        description=model.description,
        category_map=category_map,
    )


class PipelineRequest(PipelineRequest_):
    pipeline: PipelineChoice = pydantic.Field(
        PipelineChoice,
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


@app.post("/pipeline/process")
@app.post("/pipeline/process/")
async def process(data: PipelineRequest) -> PipelineResponse:
    algorithms_used: dict[str, AlgorithmResponse] = {}

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

    filter = MothClassifierBinary(
        source_images=source_images,
        detections=detector_results,
        batch_size=settings.classification_batch_size,
        num_workers=settings.num_workers,
        # single=True if len(detector_results) == 1 else False,
        single=True,  # @TODO solve issues with reading images in multiprocessing
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

    logger.info(
        f"Sending {len(moth_detections)} of {num_pre_filter} "
        "detections to the classifier"
    )

    Classifier = PIPELINE_CHOICES[str(data.pipeline)]
    classifier: APIMothClassifier = Classifier(
        source_images=source_images,
        detections=moth_detections,
        batch_size=settings.classification_batch_size,
        num_workers=settings.num_workers,
        # single=True if len(filtered_detections) == 1 else False,
        single=True,  # @TODO solve issues with reading images in multiprocessing
        example_config_param=data.config.example_config_param,
    )
    classifier.run()
    end_time = time.time()
    seconds_elapsed = float(end_time - start_time)
    algorithms_used[classifier.get_key()] = make_algorithm_response(classifier)

    # Return all detections, including those that were not classified as moths
    all_detections = classifier.results + non_moth_detections

    logger.info(
        f"Processed {len(source_images)} images in {seconds_elapsed:.2f} seconds"
    )
    logger.info(f"Returning {len(all_detections)} detections")
    print(all_detections)

    # If the number of detections is greater than 100, its suspicious. Log it.
    if len(all_detections) > 100:
        logger.warning(
            f"Detected {len(all_detections)} detections. "
            "This is suspicious and may contain duplicates."
        )

    response = PipelineResponse(
        pipeline=data.pipeline,
        algorithms=algorithms_used,
        source_images=source_image_results,
        detections=all_detections,
        total_time=seconds_elapsed,
    )
    return response


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
