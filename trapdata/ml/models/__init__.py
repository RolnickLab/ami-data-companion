from .localization import ObjectDetector
from .classification import (
    BinaryClassifier,
    SpeciesClassifier,
)

object_detectors = ObjectDetector.__subclasses__()
binary_classifiers = BinaryClassifier.__subclasses__()
species_classifiers = SpeciesClassifier.__subclasses__()
