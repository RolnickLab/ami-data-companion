"""
Loading classifiers trained with mini_trainer, such as Nemo (MAMBO v3).

The checkpoint carries the backbone name, embedding width and input size, and must
load strictly into a torchvision backbone with the cosine head. The label map decides
which of the checkpoint's classes are offered. A checkpoint that does not match the
model class or its label map is refused rather than loaded.
"""

import unittest.mock

import pytest
import torch
import torchvision

from trapdata.ml.models.base import InferenceBaseClass
from trapdata.ml.models.classification import (
    CosineClassifierHead,
    MiniTrainerClassifier,
)

# A small torchvision backbone keeps the test fast; Nemo uses efficientnet_v2_s
BACKBONE = "mobilenet_v3_small"
NUM_CLASSES = 5


def save_checkpoint(path, **metadata_overrides):
    """Save a checkpoint laid out the way mini_trainer saves one."""
    model = getattr(torchvision.models, BACKBONE)(weights=None)
    in_features = model.classifier[0].in_features
    model.classifier = CosineClassifierHead(in_features, NUM_CLASSES)
    state_dict = model.state_dict()
    # Hierarchy masks and metadata are stored next to the head's weights
    state_dict["classifier.mask_0"] = torch.zeros(NUM_CLASSES, dtype=torch.long)
    state_dict["classifier._extra_state"] = {
        "mini_trainer_version": 1,
        "backbone_class": BACKBONE,
        "backbone_output_name": "classifier",
        "in_features": in_features,
        "out_features": NUM_CLASSES,
        "hidden": True,
        "normalized": True,
        "preferred_size": MiniTrainerClassifier.input_size,
        **metadata_overrides,
    }
    torch.save(state_dict, path)
    return path


def make_classifier(weights_path, label_map=None):
    """
    The model class without its data pipeline, which loading does not need.

    label_map is the {output index: name} map read from the labels file; by default
    it names every class in the checkpoint.
    """
    if label_map is None:
        label_map = {index: f"Species {index}" for index in range(NUM_CLASSES)}
    classifier = MiniTrainerClassifier.__new__(MiniTrainerClassifier)
    classifier.name = "Test mini_trainer classifier"
    classifier.weights = weights_path
    classifier.device = "cpu"
    with unittest.mock.patch.object(
        InferenceBaseClass, "get_labels", return_value=label_map
    ):
        classifier.category_map = classifier.get_labels("labels.json")
    return classifier


def test_checkpoint_loads_and_scores_every_class(tmp_path):
    classifier = make_classifier(save_checkpoint(tmp_path / "weights.pt"))
    model = classifier.get_model()

    with torch.no_grad():
        logits = model(torch.rand(2, 3, 64, 64))

    assert logits.shape == (2, NUM_CLASSES)
    assert torch.isfinite(logits).all()


def test_label_map_selects_the_classes_offered(tmp_path):
    """
    Classes left out of the label map are dropped, and the rest keep the scores the
    full model gives them, under labels renumbered from zero.
    """
    weights = save_checkpoint(tmp_path / "weights.pt")
    images = torch.rand(2, 3, 64, 64)
    full_model = make_classifier(weights).get_model()
    classifier = make_classifier(weights, label_map={4: "Species 4", 1: "Species 1"})

    with torch.no_grad():
        logits = classifier.get_model()(images)
        full_logits = full_model(images)

    assert classifier.category_map == {0: "Species 1", 1: "Species 4"}
    assert torch.allclose(logits, full_logits[:, [1, 4]], atol=1e-5)


def test_head_maps_cosine_to_a_z_score():
    """
    A class vector along the embedding scores the largest z-score, pi/2 * sqrt(D - 2),
    one pointing away scores its negative, as mini_trainer's cosine_to_zscore does.
    """
    width = 8
    head = CosineClassifierHead(in_features=width, num_classes=2)
    features = torch.rand(1, width) + 0.1  # positive, so leaky_relu passes it through
    with torch.no_grad():
        head.hidden.weight.copy_(torch.eye(width))
        head.hidden.bias.zero_()
        head.linear.bias.zero_()
        head.linear.parametrizations.weight.original1.copy_(
            torch.cat([features, -features])
        )
        logits = head(features)

    largest = torch.pi / 2 * (width - 2) ** 0.5
    assert logits[0, 0].item() == pytest.approx(largest, rel=1e-3)
    assert logits[0, 1].item() == pytest.approx(-largest, rel=1e-3)


@pytest.mark.parametrize(
    "overrides",
    [{"preferred_size": 224}, {"normalized": False}],
    ids=["input-size", "unnormalized-head"],
)
def test_mismatched_checkpoint_is_refused(tmp_path, overrides):
    classifier = make_classifier(save_checkpoint(tmp_path / "weights.pt", **overrides))

    with pytest.raises(ValueError, match="does not match the model settings"):
        classifier.get_model()


def test_label_map_naming_a_class_the_checkpoint_lacks_is_refused(tmp_path):
    label_map = {index: f"Species {index}" for index in range(NUM_CLASSES + 1)}
    classifier = make_classifier(save_checkpoint(tmp_path / "weights.pt"), label_map)

    with pytest.raises(ValueError, match="output indexes the checkpoint does not have"):
        classifier.get_model()


def test_checkpoint_without_mini_trainer_metadata_is_refused(tmp_path):
    path = tmp_path / "weights.pt"
    torch.save({"weight": torch.zeros(3)}, path)

    with pytest.raises(ValueError, match="no mini_trainer metadata"):
        make_classifier(path).get_model()
