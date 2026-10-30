"""A classifier that returns its backbone embedding also puts it on the detection.

Antenna stores vectors from ``DetectionResponse.embeddings`` (RolnickLab/antenna#1462),
and a retrain learns from those stored vectors. The model is built without loading
weights: only the response wiring is under test.
"""

import datetime

from trapdata.api.api import make_algorithm_response
from trapdata.api.models.classification import MothClassifierBioCLIP25Newfoundland
from trapdata.api.schemas import AlgorithmReference, BoundingBox, DetectionResponse
from trapdata.ml.models.base import ClassifierResult


def _classifier_with_one_detection() -> MothClassifierBioCLIP25Newfoundland:
    classifier = object.__new__(MothClassifierBioCLIP25Newfoundland)
    classifier.terminal = True
    classifier.detections = [
        DetectionResponse(
            source_image_id="img-1",
            bbox=BoundingBox(x1=0, y1=0, x2=10, y2=10),
            algorithm=AlgorithmReference(name="detector", key="detector"),
            timestamp=datetime.datetime.now(),
        )
    ]
    return classifier


def _predictions(features):
    return ClassifierResult(
        labels=["a", "b"], logit=[0.1, 0.9], scores=[0.4, 0.6], features=features
    )


def test_the_embedding_lands_on_the_detection_under_the_classifiers_key():
    classifier = _classifier_with_one_detection()

    detection = classifier.update_detection_classification(
        0.1, "img-1", 0, _predictions([0.5] * 4)
    )

    assert len(detection.embeddings) == 1
    assert detection.embeddings[0].features == [0.5] * 4
    assert detection.embeddings[0].algorithm.key == classifier.get_key()


def test_classifying_again_replaces_the_vector_rather_than_adding_one():
    classifier = _classifier_with_one_detection()

    classifier.update_detection_classification(0.1, "img-1", 0, _predictions([0.5] * 4))
    detection = classifier.update_detection_classification(
        0.1, "img-1", 0, _predictions([0.7] * 4)
    )

    assert [e.features for e in detection.embeddings] == [[0.7] * 4]


def test_a_classifier_without_features_leaves_the_detection_without_embeddings():
    classifier = _classifier_with_one_detection()

    detection = classifier.update_detection_classification(
        0.1, "img-1", 0, _predictions(None)
    )

    assert detection.embeddings is None


def test_info_declares_the_embedding_length_antenna_checks_vectors_against():
    classifier = object.__new__(MothClassifierBioCLIP25Newfoundland)
    classifier.category_map = None
    classifier.weights_path = None

    assert make_algorithm_response(classifier).embedding_dimensions == 1024
