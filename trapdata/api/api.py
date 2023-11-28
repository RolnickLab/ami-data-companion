"""
Fast API interface for processing images through the localization and classification pipelines.
"""

import enum
import time

import fastapi
import pydantic

from ..common.logs import logger
from .models.classification import (
    MothClassifierBinary,
    MothClassifierPanama,
    MothClassifierQuebecVermont,
    MothClassifierUKDenmark,
)
from .models.localization import MothDetector
from .schemas import Classification, Detection, SourceImage
from .utils import get_crop_fname, render_crop, upload_image

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
    classifications: list[Classification]


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
    detector = MothDetector(source_images=source_images)
    detector.run()

    filter = MothClassifierBinary(
        source_images=source_images, detections=detector.results
    )
    filter.run()
    # all_binary_classifications = filter.results
    filtered_detections = filter.get_filtered_detections()

    Classifier = PIPELINE_CHOICES[data.pipeline.value]
    classifier = Classifier(source_images=source_images, detections=filtered_detections)
    classifier.run()
    end_time = time.time()
    seconds_elapsed = float(end_time - start_time)

    # all_classifications = all_binary_classifications + classifier.results
    all_detections = detector.results
    all_classifications = classifier.results

    for detection in all_detections:
        source_image = _get_source_image(source_images, detection.source_image_id)
        crop = render_crop(source_image, detection.bbox)
        public_url = upload_image(
            crop, name=get_crop_fname(source_image, detection.bbox)
        )
        logger.info(f"Uploaded crop to {public_url}")
        print(public_url)
        detection.crop_image_url = public_url

    response = PipelineResponse(
        pipeline=data.pipeline,
        source_images=source_image_results,
        detections=all_detections,
        classifications=all_classifications,
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
