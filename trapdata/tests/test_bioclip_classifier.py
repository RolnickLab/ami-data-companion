"""
The BioCLIP 2.5 + LogReg classifiers are offered as pipelines, and their forward pass
returns the embedding alongside the logits so a classification carries the vector that
produced it. These tests check the wiring without downloading the backbone or a head.
"""

import pytest
import torch

from trapdata.api.api import CLASSIFIER_CHOICES
from trapdata.api.models.classification import (
    APIMothClassifier,
    MothClassifierBioCLIP25Newfoundland,
    MothClassifierBioCLIP25Panama,
)
from trapdata.api.schemas import ClassificationResponse
from trapdata.ml.models.base import ClassifierResult
from trapdata.ml.models.bioclip import (
    BioCLIP25NewfoundlandClassifier,
    BioCLIP25PanamaClassifier,
    BioCLIPWithLinearHead,
)


def test_bioclip_classifiers_are_registered_pipelines():
    assert CLASSIFIER_CHOICES["bioclip_2_5_newfoundland"] is (
        MothClassifierBioCLIP25Newfoundland
    )
    assert CLASSIFIER_CHOICES["bioclip_2_5_panama"] is MothClassifierBioCLIP25Panama


def test_bioclip_classifiers_extend_the_api_classifier():
    for cls in (MothClassifierBioCLIP25Newfoundland, MothClassifierBioCLIP25Panama):
        assert issubclass(cls, APIMothClassifier)

    assert issubclass(
        MothClassifierBioCLIP25Newfoundland, BioCLIP25NewfoundlandClassifier
    )
    assert issubclass(MothClassifierBioCLIP25Panama, BioCLIP25PanamaClassifier)


def test_bioclip_classifiers_declare_a_backbone_and_head():
    assert (
        BioCLIP25NewfoundlandClassifier.backbone_name
        == "hf-hub:imageomics/bioclip-2.5-vith14"
    )
    assert BioCLIP25NewfoundlandClassifier.head_repo_id
    assert BioCLIP25PanamaClassifier.head_filename == "head_combined.npz"


def test_forward_returns_logits_and_the_embedding_it_consumed():
    """
    The head must see an L2-normalised embedding, and that same embedding must come back
    so it can be stored. A stub encoder lets us check both without the real backbone.
    """

    class StubEncoder(torch.nn.Module):
        def encode_image(self, images):
            return torch.tensor([[3.0, 4.0]])  # norm 5, so normalises to (0.6, 0.8)

    head = torch.nn.Linear(2, 3)
    model = BioCLIPWithLinearHead(StubEncoder(), head)

    logits, features = model(torch.zeros(1, 3, 2, 2))

    assert torch.allclose(features.norm(dim=-1), torch.ones(1), atol=1e-6)
    assert torch.allclose(logits, head(features))


def test_post_process_batch_keeps_the_features_on_the_result():
    """A model that returns ``(logits, features)`` must carry the embedding through."""
    classifier = MothClassifierBioCLIP25Newfoundland.__new__(
        MothClassifierBioCLIP25Newfoundland
    )
    classifier.category_map = {0: "Species a", 1: "Species b"}

    logits = torch.tensor([[2.0, 1.0]])
    features = torch.tensor([[0.6, 0.8]])
    results = classifier.post_process_batch((logits, features))

    assert len(results) == 1
    assert isinstance(results[0], ClassifierResult)
    assert results[0].features == pytest.approx([0.6, 0.8], abs=1e-6)


def test_classification_response_carries_features():
    field = ClassificationResponse.model_fields["features"]
    assert field.default is None
