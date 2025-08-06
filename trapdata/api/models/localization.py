import concurrent.futures
import datetime
import typing

from trapdata.ml.models.localization import MothObjectDetector_FasterRCNN_2023

from ..datasets import LocalizationImageDataset
from ..schemas import AlgorithmReference, BoundingBox, DetectionResponse, SourceImage
from .base import APIInferenceBaseClass


class APIMothDetector(APIInferenceBaseClass, MothObjectDetector_FasterRCNN_2023):
    task_type = "localization"

    def __init__(self, source_images: typing.Iterable[SourceImage], *args, **kwargs):
        self.source_images = source_images
        self.results: list[DetectionResponse] = []
        super().__init__(*args, **kwargs)

    def get_dataset(self):
        return LocalizationImageDataset(
            self.source_images, self.get_transforms(), batch_size=self.batch_size
        )

    def get_source_image(self, source_image_id: int) -> SourceImage:
        for source_image in self.source_images:
            if source_image.id == source_image_id:
                return source_image
        raise ValueError(f"Source image with id {source_image_id} not found")

    def save_results(self, item_ids, batch_output, seconds_per_item, *args, **kwargs):
        detections: list[DetectionResponse] = []

        def save_detection(image_id, coords):
            bbox = BoundingBox(x1=coords[0], y1=coords[1], x2=coords[2], y2=coords[3])
            detection = DetectionResponse(
                source_image_id=image_id,
                bbox=bbox,
                inference_time=seconds_per_item,
                algorithm=AlgorithmReference(name=self.name, key=self.get_key()),
                timestamp=datetime.datetime.now(),
                crop_image_url=None,
            )
            return detection

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for image_id, image_output in zip(item_ids, batch_output):
                for coords in image_output:
                    future = executor.submit(save_detection, image_id, coords)
                    futures.append(future)

            for future in concurrent.futures.as_completed(futures):
                detection = future.result()
                detections.append(detection)

        self.results += detections

    def run(self) -> list[DetectionResponse]:
        super().run()
        return self.results
