import datetime
import typing

from rich import print

from trapdata.common.logs import logger
from trapdata.ml.models.localization import (
    MothObjectDetector_FasterRCNN_MobileNet_2023,
    ObjectDetector,
)

from ..datasets import LocalizationAPIDataset, LocalizationImageDataset
from ..queries import save_detected_objects
from ..schemas import BoundingBox, Detection, SourceImage
from ..utils import upload_crop
from .base import APIInferenceBaseClass


class APIObjectDetector(APIInferenceBaseClass, ObjectDetector):
    def __init__(self, source_image_ids: list[int], *args, **kwargs):
        self.source_image_ids = source_image_ids
        super().__init__(*args, **kwargs)

    def save_results(self, item_ids, batch_output, *args, **kwargs):
        # Format data to be posted to the API
        # Here we are just saving the bboxes of detected objects
        detected_objects_data = []
        for image_output in batch_output:
            detected_objects = [
                {
                    "bbox": bbox,
                    "model_name": self.name,
                }
                for bbox in image_output
            ]
            detected_objects_data.append(detected_objects)

        # @TODO crop and post image crops to object store
        save_detected_objects(item_ids, detected_objects_data)

    def filter_objects_of_interest(self, item_ids, batch_output):
        # @TODO run the binary classifier on the crops right here dingus
        pass

    def classifiy_results(self, item_ids, batch_output):
        # @TODO run the species classifier on the crops right here boyyyyy
        pass

    def get_dataset(self):
        return LocalizationAPIDataset(
            self.source_image_ids, self.get_transforms(), batch_size=self.batch_size
        )


class APIMothObjectDetector_FasterRCNN_MobileNet_2023(
    APIObjectDetector,
    MothObjectDetector_FasterRCNN_MobileNet_2023,
):
    pass


class MothDetector(APIInferenceBaseClass, MothObjectDetector_FasterRCNN_MobileNet_2023):
    def __init__(self, source_images: typing.Iterable[SourceImage], *args, **kwargs):
        self.source_images = source_images
        self.results: list[Detection] = []
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
        detections: list[Detection] = []
        for image_id, image_output in zip(item_ids, batch_output):
            for coords in image_output:
                bbox = BoundingBox(
                    x1=coords[0], y1=coords[1], x2=coords[2], y2=coords[3]
                )
                try:
                    source_image = self.get_source_image(image_id)
                    crop_url = upload_crop(source_image, bbox)
                except Exception as e:
                    logger.error(f"Failed to upload crop: {e}")
                    crop_url = None
                detection = Detection(
                    source_image_id=image_id,
                    bbox=bbox,
                    inference_time=seconds_per_item,
                    algorithm=self.name,
                    timestamp=datetime.datetime.now(),
                    crop_image_url=crop_url,
                )
                print(detection)
                detections.append(detection)
            logger.info(f"Found {len(detections)} detected objects for item {image_id}")
        self.results += detections

    def run(self) -> list[Detection]:
        super().run()
        return self.results
