import datetime
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
    TuringCostaRicaSpeciesClassifier,
    UKDenmarkMothSpeciesClassifierMixedResolution,
)

from ..datasets import ClassificationImageDataset
from ..schemas import Classification, Detection, SourceImage
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
        self.detections = list(detections)
        self.results: list[Detection] = []
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
    ) -> list[Detection]:
        image_ids = metadata[0]
        detection_idxes = metadata[1]
        for image_id, detection_idx, predictions in zip(
            image_ids, detection_idxes, batch_output
        ):
            detection = self.detections[detection_idx]
            assert detection.source_image_id == image_id
            classification = Classification(
                classification=predictions[0][0],
                labels=[label for (label, _) in list(predictions)],
                scores=[score for (_, score) in list(predictions)],
                inference_time=seconds_per_item,
                algorithm=self.name,
                timestamp=datetime.datetime.now(),
            )
            detection.classifications.append(classification)
            print(detection)
        self.results.extend(self.detections)
        logger.info(f"Saving {len(self.results)} detections with classifications")
        return self.results

    def run(self) -> list[Detection]:
        super().run()
        return self.results


class MothClassifierBinary(MothClassifier, MothNonMothClassifier):
    def __init__(self, filter_results: bool = False, *args, **kwargs):
        self.filter_results = filter_results
        super().__init__(*args, **kwargs)

    def save_results(
        self, metadata, batch_output, seconds_per_item, *args, **kwargs
    ) -> list[Detection]:
        """
        Override the base class method to save only the results that have the
        label we are interested in.
        """
        image_ids = metadata[0]
        detection_idxes = metadata[1]
        for image_id, detection_idx, predictions in zip(
            image_ids, detection_idxes, batch_output
        ):
            detection = self.detections[detection_idx]
            assert detection.source_image_id == image_id
            classification = Classification(
                classification=predictions[0][0],
                labels=[label for (label, _) in list(predictions)],
                scores=[score for (_, score) in list(predictions)],
                inference_time=seconds_per_item,
                algorithm=self.name,
                timestamp=datetime.datetime.now(),
                # Specific to binary classification / the filter model
                terminal=False,
            )
            print(detection)
            if self.filter_results:
                if classification.classification == self.positive_binary_label:
                    detection.classifications.append(classification)
            else:
                detection.classifications.append(classification)

        self.results.extend(self.detections)
        logger.info(f"Saving {len(self.results)} detections with classifications")
        return self.results


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


class MothClassifierTuringCostaRica(MothClassifier, TuringCostaRicaSpeciesClassifier):
    pass
