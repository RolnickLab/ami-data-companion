from trapdata.ml.models.localization import (
    MothObjectDetector_FasterRCNN_MobileNet_2023,
    ObjectDetector,
)

from .datasets import LocalizationAPIDataset
from .queries import save_detected_objects


class APIInferenceBaseClass:
    """
    Override methods and properties in the InferenceBaseClass that
    are needed or not needed for the API version.
    """

    def __init__(self, *args, **kwargs):
        # Don't need to set these for API version
        kwargs["db_path"] = None
        kwargs["image_base_path"] = None
        super().__init__(*args, **kwargs)


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

        save_detected_objects(item_ids, detected_objects_data)

    def get_dataset(self):
        return LocalizationAPIDataset(
            self.source_image_ids, self.get_transforms(), batch_size=self.batch_size
        )


class APIMothObjectDetector_FasterRCNN_MobileNet_2023(
    APIObjectDetector,
    MothObjectDetector_FasterRCNN_MobileNet_2023,
):
    pass
