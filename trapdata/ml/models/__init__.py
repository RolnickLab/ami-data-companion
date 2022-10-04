from .localization import ObjectDetector
from .classification import (
    BinaryClassifier,
    SpeciesClassifier,
)

object_detectors = {Model.name: Model for Model in ObjectDetector.__subclasses__()}
binary_classifiers = {Model.name: Model for Model in BinaryClassifier.__subclasses__()}
species_classifiers = {
    Model.name: Model for Model in SpeciesClassifier.__subclasses__()
}
