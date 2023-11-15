import numpy as np
import torch
from rich import print

from trapdata.common.logs import logger
from trapdata.ml.models.classification import (
    QuebecVermontMothSpeciesClassifierMixedResolution,
)

from ..datasets import ClassificationImageDataset
from ..schemas import BoundingBox, Classification, Detection
from .base import APIInferenceBaseClass


class MothClassifier(
    APIInferenceBaseClass,
    QuebecVermontMothSpeciesClassifierMixedResolution,
):
    def __init__(self, detections: list[Detection], *args, **kwargs):
        self.detections = detections
        self.results: list[Classification] = []
        super().__init__(*args, **kwargs)

    def get_dataset(self):
        return ClassificationImageDataset(
            self.detections, self.get_transforms(), batch_size=self.batch_size
        )

    def post_process_batch(self, output, top_n=3):
        predictions = torch.nn.functional.softmax(output, dim=1)
        predictions = predictions.cpu().numpy()

        top_n_indices = np.argpartition(predictions, -top_n, axis=1)[:, -top_n:]
        top_n_scores = predictions[
            np.arange(predictions.shape[0])[:, None], top_n_indices
        ]
        top_n_labels = np.array(
            [[self.category_map[i] for i in row] for row in top_n_indices]
        )

        result = [
            list(zip(labels, scores))
            for labels, scores in zip(top_n_labels, top_n_scores)
        ]
        result = [sorted(items, key=lambda x: x[1], reverse=True) for items in result]
        logger.debug(f"Post-processing result batch: {result}")
        return result

    def save_results(self, metadata, batch_output):
        image_ids = metadata[0]
        bboxes = [bboxes.tolist() for bboxes in metadata[1]]
        classification_objects = []
        for image_id, coords, predictions in zip(image_ids, bboxes, batch_output):
            classification = Classification(
                source_image_id=image_id,
                bbox=BoundingBox.from_coords(coords=coords),
                classification=predictions[0][0],
                labels=[label for (label, score) in list(predictions)],
                scores=[score for (label, score) in list(predictions)],
            )
            print(classification)
            classification_objects.append(classification)
        self.results.extend(classification_objects)
        logger.info(f"Saving {len(self.results)} classification results")
        return self.results

    def run(self) -> list[Classification]:
        super().run()
        return self.results
