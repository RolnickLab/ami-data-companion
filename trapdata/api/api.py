"""
Fast API interface for processing images through the localization and classification pipelines.
"""

import enum
import time

import fastapi
import pydantic
from rich import print

from ..common.logs import logger  # noqa: F401
from . import settings
from .models.classification import (
    MothClassifier,
    MothClassifierBinary,
    MothClassifierGlobal,
    MothClassifierPanama,
    MothClassifierPanama2024,
    MothClassifierQuebecVermont,
    MothClassifierTuringAnguilla,
    MothClassifierTuringCostaRica,
    MothClassifierUKDenmark,
)
from .models.localization import MothDetector
from .schemas import Detection, SourceImage

app = fastapi.FastAPI()


class SourceImageRequest(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="ignore")

    # @TODO bring over new SourceImage & b64 validation from the lepsAI repo
    id: str
    url: str
    # b64: str | None = None


class SourceImageResponse(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="ignore")

    id: str
    url: str


PIPELINE_CHOICES = {
    "panama_moths_2023": MothClassifierPanama,
    "panama_moths_2024": MothClassifierPanama2024,
    "quebec_vermont_moths_2023": MothClassifierQuebecVermont,
    "uk_denmark_moths_2023": MothClassifierUKDenmark,
    "costa_rica_moths_turing_2024": MothClassifierTuringCostaRica,
    "anguilla_moths_turing_2024": MothClassifierTuringAnguilla,
    "global_moths_2024": MothClassifierGlobal,
}
_pipeline_choices = dict(zip(PIPELINE_CHOICES.keys(), list(PIPELINE_CHOICES.keys())))


PipelineChoice = enum.Enum("PipelineChoice", _pipeline_choices)


class PipelineRequest(pydantic.BaseModel):
    pipeline: PipelineChoice
    source_images: list[SourceImageRequest]


class PipelineResponse(pydantic.BaseModel):
    pipeline: PipelineChoice
    total_time: float
    source_images: list[SourceImageResponse]
    detections: list[Detection]


@app.get("/")
async def root():
    return fastapi.responses.RedirectResponse("/docs")


def _get_source_image(source_images, source_image_id):
    for source_image in source_images:
        if source_image.id == source_image_id:
            return source_image
    raise ValueError(f"Source image {source_image_id} not found")


@app.post("/pipeline/process")
@app.post("/pipeline/process/")
async def process(data: PipelineRequest) -> PipelineResponse:
    # Ensure that the source images are unique, filter out duplicates
    source_images_index = {
        source_image.id: source_image for source_image in data.source_images
    }
    incoming_source_images = list(source_images_index.values())
    if len(incoming_source_images) != len(data.source_images):
        logger.warning(
            f"Removed {len(data.source_images) - len(incoming_source_images)} duplicate source images"
        )

    source_image_results = [
        SourceImageResponse(**image.model_dump()) for image in incoming_source_images
    ]
    source_images = [
        SourceImage(**image.model_dump()) for image in incoming_source_images
    ]

    start_time = time.time()
    detector = MothDetector(
        source_images=source_images,
        batch_size=settings.localization_batch_size,
        num_workers=settings.num_workers,
        # single=True if len(source_images) == 1 else False,
        single=True,  # @TODO solve issues with reading images in multiprocessing
    )
    detector_results = detector.run()
    num_pre_filter = len(detector_results)

    filter = MothClassifierBinary(
        source_images=source_images,
        detections=detector_results,
        batch_size=settings.classification_batch_size,
        num_workers=settings.num_workers,
        # single=True if len(detector_results) == 1 else False,
        single=True,  # @TODO solve issues with reading images in multiprocessing
        filter_results=False,  # Only save results with the positive_binary_label, @TODO make this configurable from request
    )
    filter.run()
    # all_binary_classifications = filter.results

    # Compare num detections with num moth detections
    num_post_filter = len(filter.results)
    logger.info(
        f"Binary classifier returned {num_post_filter} out of {num_pre_filter} detections"
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
        f"Sending {len(moth_detections)} out of {num_pre_filter} detections to the classifier"
    )

    Classifier = PIPELINE_CHOICES[data.pipeline.value]
    classifier: MothClassifier = Classifier(
        source_images=source_images,
        detections=moth_detections,
        batch_size=settings.classification_batch_size,
        num_workers=settings.num_workers,
        # single=True if len(filtered_detections) == 1 else False,
        single=True,  # @TODO solve issues with reading images in multiprocessing
    )
    classifier.run()
    end_time = time.time()
    seconds_elapsed = float(end_time - start_time)

    # Return all detections, including those that were not classified as moths

    # Delete the classifications from the detections for non-moths
    # since right now the interface will show all classifications
    for detection in non_moth_detections:
        detection.classifications = []

    all_detections = classifier.results + non_moth_detections

    # For each detection, only keep the classifications with terminal=True
    # for detection in detections_with_classifications:
    #     detection.classifications = [
    #         classification
    #         for classification in detection.classifications
    #         if classification.terminal
    #     ]
    # Remove any detections without classifications
    # detections_with_classifications = [
    #     detection
    #     for detection in detections_with_classifications
    #     if detection.classifications
    # ]
    logger.info(
        f"Processed {len(source_images)} images in {seconds_elapsed:.2f} seconds"
    )
    logger.info(f"Returning {len(all_detections)} detections")
    print(all_detections)

    # If the number of detections is greater than 100, its suspicious. Log it.
    if len(all_detections) > 100:
        logger.warning(
            f"Detected {len(all_detections)} detections. This is suspicious."
        )
        # Log the detections, order by classification label
        all_detections.sort(key=lambda x: x.classifications[0].classification)
        for detection in all_detections:
            logger.warning(
                f"{detection.source_image_id}: {detection.classifications[0].classification} at {detection.bbox}"
            )

    response = PipelineResponse(
        pipeline=data.pipeline,
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
