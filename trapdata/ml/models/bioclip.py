"""
Feature extractors built on a frozen BioCLIP image encoder.

An embedding from a frozen backbone never changes for a given crop, so it can be stored
once and compared later, for example to tell whether two detections in consecutive
captures show the same individual. These extractors have no classification head and
never label a detection.
"""

import threading

import torch
import torchvision

from trapdata import logger

from .base import InferenceBaseClass


class NormalizedImageEncoder(torch.nn.Module):
    """
    Wrap an open_clip model so a forward pass returns L2-normalised image embeddings.

    Normalising here means cosine similarity between two stored vectors is their dot
    product, and vectors from any batch or device are on the same scale.
    """

    def __init__(self, clip_model: torch.nn.Module):
        super().__init__()
        self.clip_model = clip_model

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.clip_model.encode_image(images).float()
        return features / features.norm(dim=-1, keepdim=True)


class BioCLIPFeatureExtractor(InferenceBaseClass):
    """
    Load a BioCLIP backbone through open_clip and return one embedding per image.

    The backbone is large (a ViT-H/14 is about 2.5 GB in float32), so the loaded model
    is kept for the life of the process and shared by every instance on the same
    device, rather than reloaded for each request.
    """

    backbone_name: str = "hf-hub:imageomics/bioclip-2.5-vith14"
    embedding_dim: int = 1024
    model_uri: str | None = None
    type = "feature_extractor"
    stage = 4
    default_taxon_rank = ""

    _loaded: dict[tuple[str, str], tuple[torch.nn.Module, object]] = {}
    _load_lock = threading.Lock()

    def get_weights(self, weights_path):
        # open_clip downloads and caches the backbone itself, from the Hugging Face Hub.
        return None

    def get_labels(self, labels_path) -> dict[int, str]:
        return {}

    def get_queue(self):
        # Only the API uses this extractor; it has no desktop-app queue.
        return None

    def _load(self) -> tuple[torch.nn.Module, object]:
        cache_key = (self.backbone_name, str(self.device))
        with self._load_lock:
            if cache_key not in self._loaded:
                self._loaded[cache_key] = self.load_backbone()
            return self._loaded[cache_key]

    def load_backbone(self) -> tuple[torch.nn.Module, object]:
        """Return ``(encoder, preprocess)`` for this backbone, loaded onto the device."""
        import open_clip

        logger.info(f"Loading {self.backbone_name} onto {self.device}")
        clip_model, _, preprocess = open_clip.create_model_and_transforms(
            self.backbone_name
        )
        output_dim = clip_model.visual.output_dim
        if output_dim != self.embedding_dim:
            raise ValueError(
                f"{self.backbone_name} produces {output_dim}-dim embeddings, but "
                f"{self.name} declares {self.embedding_dim}."
            )
        encoder = NormalizedImageEncoder(clip_model).to(self.device)
        encoder.eval()
        return encoder, preprocess

    def get_transforms(self) -> torchvision.transforms.Compose:
        # The backbone ships the resize and normalisation it was trained with; a
        # hand-written transform would silently shift every embedding.
        return self._load()[1]

    def get_model(self) -> torch.nn.Module:
        return self._load()[0]


class BioCLIP25FeatureExtractor(BioCLIPFeatureExtractor):
    name = "BioCLIP 2.5 ViT-H/14 embeddings"
    key = "bioclip_2_5_embeddings"
    description = (
        "Image embeddings from the frozen BioCLIP 2.5 ViT-H/14 encoder, 1024 floats, "
        "L2-normalised. Adds no classification."
    )
    backbone_name = "hf-hub:imageomics/bioclip-2.5-vith14"
    embedding_dim = 1024
    model_uri = "https://huggingface.co/imageomics/bioclip-2.5-vith14"
