import datetime

import numpy as np
import torch

from trapdata.api.models.classification import MothClassifierBinary
from trapdata.api.models.localization import RESTAPIMothDetector
from trapdata.api.schemas import AlgorithmReference, BoundingBox, DetectionResponse
from trapdata.common.logs import logger
from trapdata.common.utils import log_time


@torch.no_grad()
def run_worker():
    """Run the worker to process images from the REST API queue."""
    # TODO: Poll for new jobs from the API
    detector = RESTAPIMothDetector(job_id=11)
    classifier = MothClassifierBinary(source_images=[], detections=[])

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
                    inference_time=0,
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
