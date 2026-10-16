import contextlib
import datetime
import typing

import numpy as np
import torch

from trapdata.common.logs import logger
from trapdata.ml.models.base import ClassifierResult
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

from ..datasets import ClassificationImageDataset
from ..schemas import (
    AlgorithmReference,
    ClassificationResponse,
    DetectionResponse,
    SourceImage,
)
from .base import APIInferenceBaseClass
from .feature_extraction import attach_embedding


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
        include_features: bool = False,
        include_logits: bool = True,
        include_embeddings: bool = False,
        **kwargs,
    ):
        self.source_images = source_images
        self.detections = list(detections)
        self.terminal = terminal
        self.include_features = include_features
        self.include_logits = include_logits
        self.include_embeddings = include_embeddings
        self._embedding_only = False
        self._last_features: torch.Tensor | None = None
        self.results: list[DetectionResponse] = []
        super().__init__(*args, **kwargs)
        if (include_features or include_embeddings) and not self.supports_features():
            logger.warning(
                f"{self.__class__.__name__} has no feature extractor, so "
                "detections will be returned without features or embeddings. "
                "Only classifiers built on Resnet50TimmClassifier support them."
            )
        logger.info(
            f"Initialized {self.__class__.__name__} with {len(self.detections)} "
            "detections"
        )

    @classmethod
    def supports_features(cls) -> bool:
        """Whether this classifier can return backbone features at all.

        Lets callers tell "features were switched off" apart from "this model
        cannot produce them", which otherwise both look like ``features=None``.
        """
        return cls.forward_with_features is not InferenceBaseClass.forward_with_features

    @property
    def produces_embeddings(self) -> bool:
        """Whether this instance attaches an embedding to each detection it sees."""
        return bool(self.include_embeddings) and self.supports_features()

    def reset(self, detections: typing.Iterable[DetectionResponse]):
        self.detections = list(detections)
        self.results = []

    def get_dataset(self):
        return ClassificationImageDataset(
            source_images=self.source_images,
            detections=self.detections,
            image_transforms=self.get_transforms(),
            batch_size=self.batch_size,
        )

    @torch.no_grad()
    def predict_batch(self, batch: torch.Tensor) -> torch.Tensor:
        """Return the batch logits, stashing features for ``post_process_batch``.

        The base class ``run()`` calls these two in sequence on one batch, which
        is what lets the features be handed over on the instance. ``no_grad`` is
        declared here rather than left to the caller, since the worker drives
        these two methods directly.
        """
        batch_input = batch.to(self.device, non_blocking=True)
        if self.include_features or self.include_embeddings:
            logits, self._last_features = self.forward_with_features(batch_input)
        else:
            logits, self._last_features = self.model(batch_input), None
        return logits

    def post_process_batch(self, batch_output: torch.Tensor) -> list[ClassifierResult]:
        """
        Return ClassifierResult objects with labels, scores, and
        optional logits and feature vectors for each image in the batch.
        """
        logits = batch_output
        features = self._last_features
        self._last_features = None  # Release GPU tensor reference
        if self._embedding_only:
            # Only the vectors are used. Converting every class score to a list
            # costs far more than the backbone pass on large-vocabulary models.
            if features is None:
                raise ValueError(f"{self.name} returned no features to embed")
            return [
                ClassifierResult(labels=None, logit=None, scores=[], features=vec)
                for vec in features.cpu().tolist()
            ]
        predictions = torch.nn.functional.softmax(logits, dim=1)
        predictions = predictions.cpu().numpy()
        logits_cpu = logits.cpu() if self.include_logits else None
        if features is not None:
            features = features.cpu()

        batch_results = []

        for i, pred in enumerate(predictions):
            class_indices = np.arange(len(pred))
            labels = [self.category_map[idx] for idx in class_indices]
            logit = logits_cpu[i].tolist() if logits_cpu is not None else None
            feature_vec = features[i].tolist() if features is not None else None

            result = ClassifierResult(
                labels=labels,
                logit=logit,
                scores=pred.tolist(),
                features=feature_vec,
            )

            batch_results.append(result)

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
        best_label = predictions.labels[np.argmax(predictions.scores)]
        return best_label

    def save_results(
        self, metadata, batch_output, seconds_per_item, *args, **kwargs
    ) -> list[DetectionResponse]:
        image_ids = metadata[0]
        detection_idxes = metadata[1]
        for image_id, detection_idx, predictions in zip(
            image_ids, detection_idxes, batch_output
        ):
            if self._embedding_only:
                self.update_detection_embedding(image_id, detection_idx, predictions)
            else:
                self.update_detection_classification(
                    seconds_per_item,
                    image_id,
                    detection_idx,
                    predictions,
                )

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

    def update_detection_classification(
        self,
        seconds_per_item: float,
        image_id: str,
        detection_idx: int,
        predictions: ClassifierResult,
    ) -> DetectionResponse:
        detection = self._get_detection(image_id, detection_idx)

        classification = ClassificationResponse(
            classification=self.get_best_label(predictions),
            scores=predictions.scores,
            logits=predictions.logit,
            features=predictions.features if self.include_features else None,
            inference_time=seconds_per_item,
            algorithm=AlgorithmReference(name=self.name, key=self.get_key()),
            timestamp=datetime.datetime.now(),
            terminal=self.terminal,
        )
        self.update_classification(detection, classification)
        if self.include_embeddings and predictions.features is not None:
            self._attach_embedding(detection, predictions.features)
        return detection

    def update_detection_embedding(
        self,
        image_id: str,
        detection_idx: int,
        predictions: ClassifierResult,
    ) -> DetectionResponse:
        """Attach this model's embedding to a detection without classifying it.

        For detections the moth/non-moth filter rejected: tracking needs their
        vector, but a species label would compete with the filter's label for the
        determination downstream.
        """
        if predictions.features is None:
            raise ValueError(
                f"{self.name} returned no features for detection {detection_idx}; "
                "embeddings need a classifier with a backbone hook and "
                "include_embeddings=True."
            )
        detection = self._get_detection(image_id, detection_idx)
        self._attach_embedding(detection, predictions.features)
        return detection

    def _get_detection(self, image_id: str, detection_idx: int) -> DetectionResponse:
        detection = self.detections[detection_idx]
        if detection.source_image_id != image_id:
            raise ValueError(
                f"Detection index {detection_idx} has mismatched image_id: "
                f"expected '{image_id}', got '{detection.source_image_id}'"
            )
        return detection

    def _attach_embedding(
        self, detection: DetectionResponse, features: list[float]
    ) -> None:
        attach_embedding(
            detection, features, AlgorithmReference(name=self.name, key=self.get_key())
        )

    def embed(
        self, detections: typing.Iterable[DetectionResponse]
    ) -> list[DetectionResponse]:
        """Attach an embedding, and no classification, to each of these detections.

        Reuses the loaded model and batches the crops the same way ``run()`` does.
        Like ``reset()``, it replaces ``detections`` and ``results``, so read the
        results of an earlier ``run()`` before calling it.
        """
        if not self.produces_embeddings:
            raise ValueError(
                f"{self.__class__.__name__} was not asked for embeddings or cannot "
                "produce them; check produces_embeddings before calling embed()."
            )
        self.reset(detections)
        self.dataset = self.get_dataset()
        self.dataloader = self.get_dataloader()
        with self.embedding_only():
            self.run()
        return self.detections

    @contextlib.contextmanager
    def embedding_only(self):
        """Within this block, batches yield embeddings and no classifications.

        ``post_process_batch`` then skips the class scores and ``save_results``
        attaches embeddings only. The worker drives those methods directly.
        """
        self._embedding_only = True
        try:
            yield self
        finally:
            self._embedding_only = False

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


class MothClassifierTuringKenyaUganda(
    APIMothClassifier, TuringKenyaUgandaSpeciesClassifier
):
    pass


class MothClassifierGlobal(APIMothClassifier, GlobalMothSpeciesClassifier):
    pass


class InsectOrderClassifier(APIMothClassifier, InsectOrderClassifier2025):
    pass
