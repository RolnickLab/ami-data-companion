import datetime
import typing

import numpy as np
import torch
import torch.utils.data
from sentry_sdk import start_transaction

from trapdata import logger
from trapdata.common.logs import logger
from trapdata.ml.models.classification import (
    GlobalMothSpeciesClassifier,
    InferenceBaseClass,
    InsectOrderClassifier2025,
    MothNonMothClassifier,
    PanamaMothSpeciesClassifier2024,
    PanamaMothSpeciesClassifierMixedResolution2023,
    QuebecVermontMothSpeciesClassifier2024,
    TuringAnguillaSpeciesClassifier,
    TuringCostaRicaSpeciesClassifier,
    TuringKenyaUgandaSpeciesClassifier,
    UKDenmarkMothSpeciesClassifier2024,
)
from trapdata.ml.utils import StopWatch

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
    task_type = "classification"

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

    def post_process_batch(
        self, logits: torch.Tensor, features: torch.Tensor | None = None
    ):
        """
        Return the labels, softmax/calibrated scores, and the original logits for
        each image in the batch, along with optional feature vectors.
        """
        predictions = torch.nn.functional.softmax(logits, dim=1)
        predictions = predictions.cpu().numpy()

        features = features.cpu() if features is not None else None
        batch_results = []
        for i, pred in enumerate(predictions):
            class_indices = np.arange(len(pred))
            scores = pred
            labels = [self.category_map[i] for i in class_indices]
            preds = list(zip(labels, scores, pred))

            if features is not None:
                batch_results.append((preds, features[i].tolist()))
            else:
                batch_results.append((preds, None))

        logger.debug(f"Post-processing result batch with {len(batch_results)} entries.")
        return batch_results

    def predict_batch(self, batch, return_features: bool = False):
        batch_input = batch.to(self.device, non_blocking=True)

        if return_features:
            features = self.get_features(batch_input)
            logits = self.model(batch_input)
            return logits, features

        logits = self.model(batch_input)
        return logits, None

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
        for image_id, detection_idx, (predictions, features_vec) in zip(
            image_ids, detection_idxes, batch_output
        ):
            detection = self.detections[detection_idx]
            assert detection.source_image_id == image_id
            _labels, scores, logits = zip(*predictions)

            classification = ClassificationResponse(
                classification=self.get_best_label(predictions),
                scores=scores,
                logits=logits,
                features=features_vec,
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

    @torch.no_grad()
    def run(self):
        logger.info(
            f"Starting {self.__class__.__name__} run with {len(self.results)} "
            "detections"
        )
        torch.cuda.empty_cache()

        for i, batch in enumerate(self.dataloader):
            if not batch:
                logger.info(f"Batch {i+1} is empty, skipping")
                continue

            item_ids, batch_input = batch

            logger.info(
                f"Processing batch {i+1}, about {len(self.dataloader)} remaining"
            )

            with StopWatch() as batch_time:
                with start_transaction(op="inference_batch", name=self.name):
                    logits, features = self.predict_batch(
                        batch_input, return_features=True
                    )

            seconds_per_item = batch_time.duration / len(logits)

            batch_output = list(self.post_process_batch(logits, features=features))
            if isinstance(item_ids, (np.ndarray, torch.Tensor)):
                item_ids = item_ids.tolist()

            logger.info(f"Saving results from {len(item_ids)} items")
            self.save_results(
                item_ids,
                batch_output,
                seconds_per_item=seconds_per_item,
            )
            logger.info(f"{self.name} Batch -- Done")

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


class MothClassifierTuringKenyaUganda(
    APIMothClassifier, TuringKenyaUgandaSpeciesClassifier
):
    pass


class MothClassifierGlobal(APIMothClassifier, GlobalMothSpeciesClassifier):
    pass


class InsectOrderClassifier(APIMothClassifier, InsectOrderClassifier2025):
    pass
