import typing

import torch

from trapdata.common.logs import logger
from trapdata.ml.models.bioclip import BioCLIP25FeatureExtractor

from ..datasets import ClassificationImageDataset
from ..schemas import (
    AlgorithmReference,
    DetectionResponse,
    EmbeddingResponse,
    SourceImage,
)
from .base import APIInferenceBaseClass


def attach_embedding(
    detection: DetectionResponse,
    features: list[float],
    algorithm: AlgorithmReference,
) -> None:
    """Set this algorithm's vector on a detection, replacing any earlier one from it."""
    others = [e for e in detection.embeddings or [] if e.algorithm.key != algorithm.key]
    embedding = EmbeddingResponse(features=features, algorithm=algorithm)
    detection.embeddings = others + [embedding]


class APIFeatureExtractor(APIInferenceBaseClass):
    """
    Attach an embedding to each detection it is given, and nothing else.

    Mixed into a feature extractor from ``trapdata.ml.models``. Every detection gets a
    vector, whatever any filter or classifier made of it, and none gets a classification.
    """

    task_type = "embedding"

    def __init__(
        self,
        source_images: typing.Iterable[SourceImage],
        detections: typing.Iterable[DetectionResponse] = (),
        *args,
        **kwargs,
    ):
        self.source_images = list(source_images)
        self.detections = list(detections)
        self.results: list[DetectionResponse] = []
        super().__init__(*args, **kwargs)

    def get_dataset(self):
        return ClassificationImageDataset(
            source_images=self.source_images,
            detections=self.detections,
            image_transforms=self.get_transforms(),
            batch_size=self.batch_size,
        )

    @torch.no_grad()
    def predict_batch(self, batch: torch.Tensor) -> torch.Tensor:
        return self.model(batch.to(self.device, non_blocking=True))

    def post_process_batch(self, batch_output: torch.Tensor) -> list[list[float]]:
        return batch_output.cpu().tolist()

    def save_results(self, metadata, batch_output, *args, **kwargs):
        image_ids, detection_idxes = metadata
        algorithm = AlgorithmReference(name=self.name, key=self.get_key())
        for image_id, detection_idx, features in zip(
            image_ids, detection_idxes, batch_output
        ):
            detection = self.detections[int(detection_idx)]
            if detection.source_image_id != image_id:
                raise ValueError(
                    f"Detection index {detection_idx} has mismatched image_id: "
                    f"expected '{image_id}', got '{detection.source_image_id}'"
                )
            attach_embedding(detection, features, algorithm)
        self.results = self.detections

    def embed(
        self,
        detections: typing.Iterable[DetectionResponse],
        source_images: typing.Iterable[SourceImage] | None = None,
    ) -> list[DetectionResponse]:
        """Attach an embedding to each of these detections, in place, and return them."""
        self.detections = list(detections)
        if source_images is not None:
            self.source_images = list(source_images)
        self.results = []
        if not self.detections:
            return self.detections
        self.dataset = self.get_dataset()
        self.dataloader = self.get_dataloader()
        self.run()
        logger.info(f"{self.name} embedded {len(self.detections)} detections")
        return self.detections


class APIBioCLIP25FeatureExtractor(APIFeatureExtractor, BioCLIP25FeatureExtractor):
    pass
