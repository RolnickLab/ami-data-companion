from enum import Enum

from .localization import ObjectDetector
from .classification import (
    BinaryClassifier,
    SpeciesClassifier,
)
from .tracking import FeatureExtractor

object_detectors = {Model.name: Model for Model in ObjectDetector.__subclasses__()}
ObjectDetectorChoice = Enum(
    "ObjectDetectorChoice",
    {Model.get_key(): Model.name for Model in ObjectDetector.__subclasses__()},
)


binary_classifiers = {Model.name: Model for Model in BinaryClassifier.__subclasses__()}
BinaryClassifierChoice = Enum(
    "BinaryClassifierChoice",
    {Model.get_key(): Model.name for Model in BinaryClassifier.__subclasses__()},
)

species_classifiers = {
    Model.name: Model for Model in SpeciesClassifier.__subclasses__()
}
SpeciesClassifierChoice = Enum(
    "SpeciesClassifierChoice",
    {Model.get_key(): Model.name for Model in SpeciesClassifier.__subclasses__()},
)

feature_extractors = {Model.name: Model for Model in FeatureExtractor.__subclasses__()}
FeatureExtractorChoice = Enum(
    "TrackingAlgorithm",
    {Model.get_key(): Model.name for Model in FeatureExtractor.__subclasses__()},
)
