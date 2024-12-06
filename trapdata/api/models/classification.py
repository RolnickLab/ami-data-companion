import datetime
import typing

import numpy as np
import torch

from trapdata.common.logs import logger
from trapdata.ml.models.classification import (
    GlobalMothSpeciesClassifier,
    InferenceBaseClass,
    MothNonMothClassifier,
    PanamaMothSpeciesClassifier2024,
    PanamaMothSpeciesClassifierMixedResolution2023,
    QuebecVermontMothSpeciesClassifier2024,
    TuringAnguillaSpeciesClassifier,
    TuringCostaRicaSpeciesClassifier,
    UKDenmarkMothSpeciesClassifier2024,
)

from ..datasets import ClassificationImageDataset
from ..schemas import ClassificationResponse, DetectionResponse, SourceImage
from .base import APIInferenceBaseClass


class APIMothClassifier(
    APIInferenceBaseClass,
    InferenceBaseClass,
):
    def __init__(
        self,
        source_images: typing.Iterable[SourceImage],
        detections: typing.Iterable[DetectionResponse],
        *args,
        **kwargs,
    ):
        self.source_images = source_images
        self.detections = list(detections)
        self.results: list[DetectionResponse] = []
        super().__init__(*args, **kwargs)
        logger.info(
            f"Initialized {self.__class__.__name__} with {len(self.detections)} detections"
        )

    def get_dataset(self):
        return ClassificationImageDataset(
            source_images=self.source_images,
            detections=self.detections,
            image_transforms=self.get_transforms(),
            batch_size=self.batch_size,
        )

    def post_process_batch(self, logits: torch.Tensor):
        """
        Return the labels, softmax/calibrated scores, and the original logits for each image in the batch.
        """
        predictions = torch.nn.functional.softmax(logits, dim=1)
        predictions = predictions.cpu().numpy()

        indices = np.arange(predictions.shape[1])

        # @TODO Calibrate the scores here,
        scores = predictions

        labels = np.array([[self.category_map[i] for i in row] for row in indices])

        return zip(labels, scores, logits)

    def get_best_label(self, predictions):
        """
        Convenience method to get the best label from the predictions, which are a list of tuples
        in the order of the model's class index, NOT the values.

        This must not modify the predictions list!

        predictions look like:
        [
            ('label1', score1, logit1),
            ('label2', score2, logit2),
            ...
        ]
        """

        best_pred = max(predictions, key=lambda x: x[1])
        best_label = best_pred[0]
        return best_label

    def save_results(
        self, metadata, batch_output, seconds_per_item, *args, **kwargs
    ) -> list[DetectionResponse]:
        image_ids = metadata[0]
        detection_idxes = metadata[1]
        for image_id, detection_idx, predictions in zip(
            image_ids, detection_idxes, batch_output
        ):
            detection = self.detections[detection_idx]
            assert detection.source_image_id == image_id
            labels, scores, logits = zip(*predictions)
            classification = ClassificationResponse(
                classification=self.get_best_label(predictions),
                labels=labels,  # @TODO move this to the Algorithm class instead of repeating it every prediction
                scores=scores,
                logits=logits,
                inference_time=seconds_per_item,
                algorithm=self.name,
                timestamp=datetime.datetime.now(),
            )
            self.update_classification(detection, classification)

        self.results = self.detections
        logger.info(f"Saving {len(self.results)} detections with classifications")
        return self.results

    def update_classification(
        self, detection: DetectionResponse, new_classification: ClassificationResponse
    ) -> None:
        # Remove all existing classifications from this algorithm
        detection.classifications = [
            c for c in detection.classifications if c.algorithm != self.name
        ]
        # Add the new classification for this algorithm
        detection.classifications.append(new_classification)
        logger.debug(
            f"Updated classification for detection {detection.bbox}. Total classifications: {len(detection.classifications)}"
        )

    def run(self) -> list[DetectionResponse]:
        logger.info(
            f"Starting {self.__class__.__name__} run with {len(self.results)} detections"
        )
        super().run()
        logger.info(
            f"Finished {self.__class__.__name__} run. Processed {len(self.results)} detections"
        )
        return self.results


class MothClassifierBinary(APIMothClassifier, MothNonMothClassifier):
    def __init__(self, *args, **kwargs):
        self.filter_results = kwargs.get("filter_results", True)
        super().__init__(*args, **kwargs)

    def save_results(
        self, metadata, batch_output, seconds_per_item, *args, **kwargs
    ) -> list[DetectionResponse]:
        """
        Override the base class method to save only the results that have the
        label we are interested in.
        """
        logger.info(f"Saving {len(batch_output)} detections with classifications")
        image_ids = metadata[0]
        detection_idxes = metadata[1]
        for image_id, detection_idx, predictions in zip(
            image_ids, detection_idxes, batch_output
        ):
            detection = self.detections[detection_idx]
            assert detection.source_image_id == image_id
            classification = ClassificationResponse(
                classification=predictions[0][0],
                labels=[label for (label, _) in list(predictions)],
                scores=[score for (_, score) in list(predictions)],
                inference_time=seconds_per_item,
                algorithm=self.name,
                timestamp=datetime.datetime.now(),
                # Specific to binary classification / the filter model
                terminal=False,
            )
            self.update_classification(detection, classification)

        self.results = self.detections
        logger.info(f"Saving {len(self.results)} detections with classifications")
        return self.results


class MothClassifierPanama(
    APIMothClassifier, PanamaMothSpeciesClassifierMixedResolution2023
):
    pass


class MothClassifierPanama2024(APIMothClassifier, PanamaMothSpeciesClassifier2024):
    pass


class MothClassifierUKDenmark(APIMothClassifier, UKDenmarkMothSpeciesClassifier2024):
    pass


class MothClassifierQuebecVermont(
    APIMothClassifier, QuebecVermontMothSpeciesClassifier2024
):
    pass


class MothClassifierTuringCostaRica(
    APIMothClassifier, TuringCostaRicaSpeciesClassifier
):
    pass


class MothClassifierTuringAnguilla(APIMothClassifier, TuringAnguillaSpeciesClassifier):
    pass


class MothClassifierGlobal(APIMothClassifier, GlobalMothSpeciesClassifier):
    pass
