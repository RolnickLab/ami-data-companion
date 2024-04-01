from __future__ import annotations

import datetime
import pathlib

import bentoml

from .schemas import BoundingBox, Classification, Detection, SourceImage


@bentoml.service
class DetectionService:
    def __init__(self) -> None:
        self.model = None  # initialize model here

    @bentoml.api(
        batchable=True,
    )
    def test_predict(self, images: list[SourceImage]) -> list[SourceImage]:
        for image in images:
            image.detections = [
                Detection(
                    source_image_id=image.id,
                    bbox=BoundingBox(x1=0, y1=0, x2=1, y2=1),
                    inference_time=0.0,
                    timestamp=datetime.datetime.now(),
                    algorithm="test",
                    classifications=[],
                ),
                Detection(
                    source_image_id=image.id,
                    bbox=BoundingBox(x1=1, y1=1, x2=2, y2=2),
                    inference_time=0.0,
                    timestamp=datetime.datetime.now(),
                    algorithm="test",
                    classifications=[],
                ),
            ]
        return images

    @bentoml.api
    def predict_single(self, image: pathlib.Path) -> SourceImage:
        input_image = SourceImage(id="test", url=image.as_uri())
        results = self.test_predict([input_image])
        return results[0]


@bentoml.service
class AMIMLServices:
    batch = bentoml.depends(DetectionService)

    @bentoml.api
    def detect_single(self, image: pathlib.Path) -> SourceImage:
        input_image = SourceImage(id="test", url=image.as_uri())
        results = self.batch.test_predict([input_image])
        return results[0]
