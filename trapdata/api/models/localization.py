import concurrent.futures
import datetime
import typing

import numpy as np
import torch

from trapdata.api.models.classification import MothClassifierBinary
from trapdata.ml.models.localization import MothObjectDetector_FasterRCNN_2023

from ..datasets import LocalizationImageDataset, RESTDataset, log_time
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
        _, t = log_time()
        super().run()
        t("Finished detection")
        return self.results


class RESTAPIMothDetector(APIMothDetector):
    def get_dataset(self):
        return RESTDataset(base_url="http://localhost:8000", job_id=11, batch_size=4)

    def get_dataloader(self):
        assert (
            self.dataset is not None
        ), "Dataset must be initialized before getting dataloader"
        return torch.utils.data.DataLoader(
            self.dataset,
            batch_size=4,
            num_workers=2,
        )


from trapdata.common.logs import logger


@torch.no_grad()
def main():
    detector = RESTAPIMothDetector(source_images=[])
    # results = detector.run()
    # print(f"Detected {len(results)} objects")

    classifier = MothClassifierBinary(source_images=[], detections=[])
    # classified_results = classifier.run()
    # print(f"Classified {len(classified_results)} objects")

    torch.cuda.empty_cache()
    items = 0

    total_detection_time = 0.0
    total_classification_time = 0.0
    total_dl_time = 0.0
    detections = []
    _, t = log_time()
    for i, batch in enumerate(detector.get_dataloader()):
        dt, t = t("Finished loading batch")
        total_dl_time += dt
        if not batch:
            logger.warning(f"Batch {i+1} is empty, skipping")
            continue

        item_ids, batch_input = batch

        logger.info(f"Processing batch {i+1}")
        # output is dict of "boxes", "labels", "scores"
        batch_output = detector.predict_batch(batch_input)

        items += len(batch_output)
        logger.info(f"Total items processed so far: {items}")
        batch_output = list(detector.post_process_batch(batch_output))
        if isinstance(item_ids, (np.ndarray, torch.Tensor)):
            item_ids = item_ids.tolist()
        # logger.info(f"Saving results from {len(item_ids)} items")
        # classifier.predict_batch(batch_input)
        dt, t = t("Finished detection")
        total_detection_time += dt

        for image_id, boxes, image_tensor in zip(item_ids, batch_output, batch_input):
            for box in boxes:
                bbox = BoundingBox(x1=box[0], y1=box[1], x2=box[2], y2=box[3])
                # crop the image tensor using the bbox
                crop = image_tensor[
                    :, int(bbox.y1) : int(bbox.y2), int(bbox.x1) : int(bbox.x2)
                ]
                crop = crop.unsqueeze(0)  # add batch dimension
                classifier_out = classifier.predict_batch(crop)
                classifier_out = classifier.post_process_batch(classifier_out)
                detection = DetectionResponse(
                    source_image_id=image_id,
                    bbox=bbox,
                    inference_time=0,  # seconds_per_item,
                    algorithm=AlgorithmReference(
                        name=detector.name, key=detector.get_key()
                    ),
                    timestamp=datetime.datetime.now(),
                    crop_image_url=None,
                    classification=classifier_out[0] if classifier_out else None,
                )
                detections.append(detection)
        ct, t = t("Finished classification")
        total_classification_time += ct
    classifier.detections = detections
    classifier.results = detections

    logger.info(
        f"Done, detections: {len(classifier.detections)}. Detecting time: {total_detection_time}, "
        f"classification time: {total_classification_time}, dl time: {total_dl_time}"
    )


if __name__ == "__main__":
    _, t = log_time()
    main()
    t("Total time")
