"""
Fast API interface for processing images through the localization and classification pipelines.
"""

import enum
import time

import fastapi
import pydantic

from ..common.logs import logger  # noqa: F401
from . import settings
from .models.classification import (
    MothClassifier,
    MothClassifierBinary,
    MothClassifierPanama,
    MothClassifierQuebecVermont,
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
    "quebec_vermont_moths_2023": MothClassifierQuebecVermont,
    "uk_denmark_moths_2023": MothClassifierUKDenmark,
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
async def process(data: PipelineRequest) -> PipelineResponse:
    source_image_results = [
        SourceImageResponse(**image.model_dump()) for image in data.source_images
    ]
    source_images = [SourceImage(**image.model_dump()) for image in data.source_images]

    start_time = time.time()
    detector = MothDetector(
        source_images=source_images,
        batch_size=settings.localization_batch_size,
        num_workers=settings.num_workers,
        # single=True if len(source_images) == 1 else False,
        single=True,  # @TODO solve issues with reading images in multiprocessing
    )
    detector_results = detector.run()

    filter = MothClassifierBinary(
        source_images=source_images,
        detections=detector_results,
        batch_size=settings.classification_batch_size,
        num_workers=settings.num_workers,
        # single=True if len(detector_results) == 1 else False,
        single=True,  # @TODO solve issues with reading images in multiprocessing
        filter_results=True,
    )
    filter.run()
    # all_binary_classifications = filter.results

    Classifier = PIPELINE_CHOICES[data.pipeline.value]
    classifier: MothClassifier = Classifier(
        source_images=source_images,
        detections=filter.results,
        batch_size=settings.classification_batch_size,
        num_workers=settings.num_workers,
        # single=True if len(filtered_detections) == 1 else False,
        single=True,  # @TODO solve issues with reading images in multiprocessing
    )
    classifier.run()
    end_time = time.time()
    seconds_elapsed = float(end_time - start_time)

    # all_classifications = all_binary_classifications + classifier.results
    detections_with_classifications = classifier.results

    response = PipelineResponse(
        pipeline=data.pipeline,
        source_images=source_image_results,
        detections=detections_with_classifications,
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
