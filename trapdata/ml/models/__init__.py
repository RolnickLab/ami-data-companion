from enum import Enum

from .localization import ObjectDetector
from .classification import (
    BinaryClassifier,
    SpeciesClassifier,
)
from .tracking import FeatureExtractor


def get_default_model(choices: Enum) -> str:
    """
    Return the first model from the dictionary of choices.
    Order is not guaranteed, different models may be returned.

    Raises an exception if no models are available.
    """
    choice_list = list(choices)  # type: ignore
    if not choice_list:
        raise Exception("Not choices available for model type")
    return choice_list[0].value


object_detectors = {Model.name: Model for Model in ObjectDetector.__subclasses__()}
ObjectDetectorChoice = Enum(
    "ObjectDetectorChoice",
    {Model.get_key(): Model.name for Model in ObjectDetector.__subclasses__()},
)
DEFAULT_OBJECT_DETECTOR = get_default_model(ObjectDetectorChoice)


binary_classifiers = {Model.name: Model for Model in BinaryClassifier.__subclasses__()}
BinaryClassifierChoice = Enum(
    "BinaryClassifierChoice",
    {Model.get_key(): Model.name for Model in BinaryClassifier.__subclasses__()},
)
DEFAULT_BINARY_CLASSIFIER = get_default_model(BinaryClassifierChoice)


species_classifiers = {
    Model.name: Model for Model in SpeciesClassifier.__subclasses__()
}
SpeciesClassifierChoice = Enum(
    "SpeciesClassifierChoice",
    {Model.get_key(): Model.name for Model in SpeciesClassifier.__subclasses__()},
)
DEFAULT_SPECIES_CLASSIFIER = get_default_model(SpeciesClassifierChoice)


feature_extractors = {Model.name: Model for Model in FeatureExtractor.__subclasses__()}
FeatureExtractorChoice = Enum(
    "TrackingAlgorithm",
    {Model.get_key(): Model.name for Model in FeatureExtractor.__subclasses__()},
)
DEFAULT_FEATURE_EXTRACTOR = get_default_model(FeatureExtractorChoice)
