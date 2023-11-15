from trapdata.ml.models.classification import SpeciesClassifier

from .base import APIInferenceBaseClass


class APISpeciesClassifier(APIInferenceBaseClass, SpeciesClassifier):
    pass
