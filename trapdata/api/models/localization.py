from trapdata.ml.models.localization import (
    MothObjectDetector_FasterRCNN_MobileNet_2023,
    ObjectDetector,
)

from ..datasets import LocalizationAPIDataset, LocalizationImageDataset
from ..queries import save_detected_objects
from ..schemas import IncomingSourceImage
from .base import APIInferenceBaseClass


class APIObjectDetector(APIInferenceBaseClass, ObjectDetector):
    def __init__(self, source_image_ids: list[int], *args, **kwargs):
        self.source_image_ids = source_image_ids
        super().__init__(*args, **kwargs)

    def save_results(self, item_ids, batch_output):
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
    def __init__(self, source_images: list[IncomingSourceImage], *args, **kwargs):
        self.source_images = source_images
        self.results = []
        super().__init__(*args, **kwargs)

    def get_dataset(self):
        return LocalizationImageDataset(
            self.source_images, self.get_transforms(), batch_size=self.batch_size
        )

    def save_results(self, item_ids, batch_output):
        detected_objects_data = []
        for image_id, image_output in zip(item_ids, batch_output):
            detected_objects = [
                {
                    "source_image_id": image_id,
                    "bbox": bbox,
                }
                for bbox in image_output
            ]
            detected_objects_data.append(detected_objects)
        self.results = detected_objects_data
        return detected_objects_data

    def run(self) -> list[dict]:
        super().run()
        return self.results
