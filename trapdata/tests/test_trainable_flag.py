"""
A processing service tells Antenna which of its algorithms can be retrained.

Only a classifier head over a frozen backbone is cheap enough to retrain, so the flag is
per algorithm rather than per service.
"""

from trapdata.api.api import make_algorithm_config_response, make_algorithm_response
from trapdata.api.schemas import AlgorithmConfigResponse
from trapdata.ml.models.base import InferenceBaseClass


class _FakeModel:
    name = "Fake classifier"
    task_type = "classification"
    description = "For testing the trainable flag."
    weights_path = None
    category_map = {}

    def get_key(self):
        return "fake-classifier"


def test_models_are_not_trainable_by_default():
    assert InferenceBaseClass.trainable is False


def test_the_response_defaults_to_not_trainable():
    assert AlgorithmConfigResponse(name="x", key="x").trainable is False


def test_a_model_that_declares_trainable_is_reported_as_such(monkeypatch):
    """Antenna reads this to decide whether retraining can be offered."""
    model = _FakeModel()
    monkeypatch.setattr(model, "trainable", True, raising=False)
    monkeypatch.setattr("trapdata.api.api.make_category_map_response", lambda m: None)

    assert make_algorithm_response(model).trainable is True
    assert make_algorithm_config_response(model).trainable is True


def test_a_model_without_the_attribute_is_not_trainable():
    """Older models predate the flag and must not be treated as trainable."""
    monkey = _FakeModel()
    assert getattr(monkey, "trainable", False) is False
