import typing

import numpy as np
import torch
from rich import print

from trapdata.common.logs import logger
from trapdata.ml.models.classification import (
    InferenceBaseClass,
    MothNonMothClassifier,
    PanamaMothSpeciesClassifierMixedResolution2023,
    QuebecVermontMothSpeciesClassifierMixedResolution,
    UKDenmarkMothSpeciesClassifierMixedResolution,
)

from ..datasets import ClassificationImageDataset
from ..schemas import BoundingBox, Classification, Detection, SourceImage
from .base import APIInferenceBaseClass


class MothClassifier(
    APIInferenceBaseClass,
    InferenceBaseClass,
):
    def __init__(
        self,
        source_images: typing.Iterable[SourceImage],
        detections: typing.Iterable[Detection],
        *args,
        **kwargs,
    ):
        self.source_images = source_images
        self.detections = detections
        self.results: list[Classification] = []
        super().__init__(*args, **kwargs)

    def get_dataset(self):
        return ClassificationImageDataset(
            source_images=self.source_images,
            detections=self.detections,
            image_transforms=self.get_transforms(),
            batch_size=self.batch_size,
        )

    def post_process_batch(self, output, top_n=3):
        predictions = torch.nn.functional.softmax(output, dim=1)
        predictions = predictions.cpu().numpy()

        # Ensure that top_n is not greater than the number of categories
        # (e.g. binary classification will have only 2 categories)
        top_n = min(top_n, predictions.shape[1])

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

    def save_results(
        self, metadata, batch_output, seconds_per_item, *args, **kwargs
    ) -> list[Classification]:
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
                inference_time=seconds_per_item,
                algorithm=self.name,
            )
            print(classification)
            classification_objects.append(classification)
        self.results.extend(classification_objects)
        logger.info(f"Saving {len(self.results)} classification results")
        return self.results

    def run(self) -> list[Classification]:
        super().run()
        return self.results


class MothClassifierBinary(MothClassifier, MothNonMothClassifier):
    def save_results(self, *args, **kwargs) -> list[Classification]:
        """
        Override the base class method to save only the results that have the
        label we are interested in.
        """
        super().save_results(*args, **kwargs)
        for classification in self.results:
            # Assume this classifier is not the last one in the pipeline
            classification.terminal = False
        return self.results

    def get_filtered_detections(
        self, results: list[Classification] | None = None
    ) -> list[Detection]:
        """
        Return only the results that have the label we are interested in.

        This is a convenience method that returns detections instead of
        classifications, ready to be sent to the next classifier.
        """
        results = results or self.results
        detections = [
            Detection(
                source_image_id=result.source_image_id,
                bbox=result.bbox
                or BoundingBox(
                    x1=0, y1=0, x2=0, y2=0
                ),  # @TODO if there is really no bbox, use the whole image
            )
            for result in results
            if result.classification == self.positive_binary_label
        ]
        return detections


class MothClassifierPanama(
    MothClassifier, PanamaMothSpeciesClassifierMixedResolution2023
):
    pass


class MothClassifierUKDenmark(
    MothClassifier, UKDenmarkMothSpeciesClassifierMixedResolution
):
    pass


class MothClassifierQuebecVermont(
    MothClassifier, QuebecVermontMothSpeciesClassifierMixedResolution
):
    pass
