from .quebec_vermont_species import get_model, get_transforms, postprocess
from .quebec_vermont_species import predict as _predict

WEIGHTS = [
    "https://object-arbutus.cloud.computecanada.ca/ami-models/moths/classification/uk-denmark-moth-model_v01_efficientnetv2-b3_2022-09-08-12-54.pt"
]
LABELS = [
    "https://object-arbutus.cloud.computecanada.ca/ami-models/moths/classification/uk-denmark-moth_category-map_13Sep2022.json"
]


def predict(*args, **kwargs):
    kwargs["weights_path"] = WEIGHTS[0]
    kwargs["labels_path"] = LABELS[0]
    _predict(*args, **kwargs)


__all__ = [get_model, get_transforms, postprocess, predict]
