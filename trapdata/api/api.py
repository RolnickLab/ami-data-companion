"""
Fast API interface for processing images through the localization and classification pipelines.
"""

import logging
import time
import typing

import fastapi
import pydantic

from .models.classification import MothClassifier
from .models.localization import MothDetector
from .schemas import Classification, Detection, SourceImage

logger = logging.getLogger(__name__)

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


PipelineChoice = typing.Literal[
    "panama-moths-2023",
    "quebec-vermont-moths-2023",
    "uk-denmark-moths-2023",
]


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


@app.post("/pipeline/process")
async def process(data: PipelineRequest) -> PipelineResponse:
    source_image_results = [
        SourceImageResponse(**image.model_dump()) for image in data.source_images
    ]
    source_images = [SourceImage(**image.model_dump()) for image in data.source_images]

    start_time = time.time()
    detector = MothDetector(source_images=source_images)
    detector.run()

    classifier = MothClassifier(
        source_images=source_images, detections=detector.results
    )
    classifier.run()
    end_time = time.time()
    seconds_elapsed = float(end_time - start_time)

    response = PipelineResponse(
        pipeline=data.pipeline,
        source_images=source_image_results,
        detections=detector.results,
        classifications=classifier.results,
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
