from __future__ import annotations

import datetime
import pathlib

import bentoml

from .schemas import BoundingBox, Classification, Detection, SourceImage


@bentoml.service
class DetectionService:
    @bentoml.api(batchable=True)
    def predict(self, images: list[SourceImage]) -> list[list[Detection]]:
        test_detections = [
            [
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
            for image in images
        ]
        return test_detections


@bentoml.service
class API:
    batch = bentoml.depends(DetectionService)

    @bentoml.api
    def predict(self, image: pathlib.Path) -> list[Detection]:
        input_image = SourceImage(id="test", url=image.as_uri())
        results = self.batch.predict([input_image])
        return results[0]
