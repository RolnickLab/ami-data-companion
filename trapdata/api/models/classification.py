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
from ..schemas import (
    AlgorithmReference,
    ClassificationResponse,
    DetectionResponse,
    SourceImage,
)
from .base import APIInferenceBaseClass


class APIMothClassifier(
    APIInferenceBaseClass,
    InferenceBaseClass,
):
    type = "classification"

    def __init__(
        self,
        source_images: typing.Iterable[SourceImage],
        detections: typing.Iterable[DetectionResponse],
        terminal: bool = True,
        *args,
        **kwargs,
    ):
        self.source_images = source_images
        self.detections = list(detections)
        self.terminal = terminal
        self.results: list[DetectionResponse] = []
        super().__init__(*args, **kwargs)
        logger.info(
            f"Initialized {self.__class__.__name__} with {len(self.detections)} "
            "detections"
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
        Return the labels, softmax/calibrated scores, and the original logits for
        each image in the batch.

        Almost like the base class method, but we need to return the logits as well.
        """
        predictions = torch.nn.functional.softmax(logits, dim=1)
        predictions = predictions.cpu().numpy()

        batch_results = []
        for pred in predictions:
            # Get all class indices and their corresponding scores
            class_indices = np.arange(len(pred))
            scores = pred
            labels = [self.category_map[i] for i in class_indices]
            batch_results.append(list(zip(labels, scores, pred)))

        logger.debug(f"Post-processing result batch: {batch_results}")

        return batch_results

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
            _labels, scores, logits = zip(*predictions)
            classification = ClassificationResponse(
                classification=self.get_best_label(predictions),
                scores=scores,
                logits=logits,
                inference_time=seconds_per_item,
                algorithm=AlgorithmReference(name=self.name, key=self.get_key()),
                timestamp=datetime.datetime.now(),
                terminal=self.terminal,
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
            c for c in detection.classifications if c.algorithm.name != self.name
        ]
        # Add the new classification for this algorithm
        detection.classifications.append(new_classification)
        logger.debug(
            f"Updated classification for detection {detection.bbox}. "
            f"Total classifications: {len(detection.classifications)}"
        )

    def run(self) -> list[DetectionResponse]:
        logger.info(
            f"Starting {self.__class__.__name__} run with {len(self.results)} "
            "detections"
        )
        super().run()
        logger.info(
            f"Finished {self.__class__.__name__} run. "
            f"Processed {len(self.results)} detections"
        )
        return self.results


class MothClassifierBinary(APIMothClassifier, MothNonMothClassifier):
    pass


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
