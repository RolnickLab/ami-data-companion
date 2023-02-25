import torch
import torchvision
import timm
from PIL import Image

from trapdata import logger
from trapdata import constants
from trapdata import db
from trapdata.db import models
from trapdata.db.models.queue import DetectedObjectQueue, UnclassifiedObjectQueue
from trapdata.db.models.detections import save_classified_objects

from .base import InferenceBaseClass


class ClassificationIterableDatabaseDataset(torch.utils.data.IterableDataset):
    def __init__(self, queue, image_transforms, batch_size=4):
        super().__init__()
        self.queue = queue
        self.image_transforms = image_transforms
        self.batch_size = batch_size

    def __len__(self):
        return self.queue.queue_count()

    def __iter__(self):
        while len(self):
            worker_info = torch.utils.data.get_worker_info()
            logger.info(f"Using worker: {worker_info}")

            records = self.queue.pull_n_from_queue(self.batch_size)
            if records:
                item_ids = torch.utils.data.default_collate(
                    [record.id for record in records]
                )
                batch_data = torch.utils.data.default_collate(
                    [self.transform(record.cropped_image_data()) for record in records]
                )
                yield (item_ids, batch_data)

    def transform(self, cropped_image):
        return self.image_transforms(cropped_image)


class EfficientNetClassifier(InferenceBaseClass):
    input_size = 300

    def get_model(self):
        num_classes = len(self.category_map)
        model = timm.create_model(
            "tf_efficientnetv2_b3",
            num_classes=num_classes,
            weights=None,
        )
        model = model.to(self.device)
        # state_dict = torch.hub.load_state_dict_from_url(weights_url)
        checkpoint = torch.load(self.weights, map_location=self.device)
        # The model state dict is nested in some checkpoints, and not in others
        state_dict = checkpoint.get("model_state_dict") or checkpoint
        model.load_state_dict(state_dict)
        model.eval()
        return model

    def get_transforms(self):
        mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]

        return torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((self.input_size, self.input_size)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean, std),
            ]
        )

    def post_process_batch(self, output):
        predictions = torch.nn.functional.softmax(output, dim=1)
        predictions = predictions.cpu().numpy()

        categories = predictions.argmax(axis=1)
        labels = [self.category_map[cat] for cat in categories]
        scores = predictions.max(axis=1).astype(float)

        result = list(zip(labels, scores))
        logger.debug(f"Post-processing result batch: {result}")
        return result


class Resnet50(torch.nn.Module):
    def __init__(self, num_classes):
        """
        Args:
            config: provides parameters for model generation
        """
        super(Resnet50, self).__init__()
        self.num_classes = num_classes
        self.backbone = torchvision.models.resnet50(weights="DEFAULT")
        out_dim = self.backbone.fc.in_features

        self.backbone = torch.nn.Sequential(*list(self.backbone.children())[:-2])
        self.avgpool = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.classifier = torch.nn.Linear(out_dim, self.num_classes, bias=False)

    def forward(self, x):
        x = self.backbone(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x


class Resnet50Classifier(InferenceBaseClass):
    input_size = 300

    def get_model(self):
        num_classes = len(self.category_map)
        model = Resnet50(num_classes=num_classes)
        model = model.to(self.device)
        # state_dict = torch.hub.load_state_dict_from_url(weights_url)
        checkpoint = torch.load(self.weights, map_location=self.device)
        # The model state dict is nested in some checkpoints, and not in others
        state_dict = checkpoint.get("model_state_dict") or checkpoint
        model.load_state_dict(state_dict)
        model.eval()
        return model

    def get_transforms(self):
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        return torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((self.input_size, self.input_size)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean, std),
            ]
        )

    def post_process_batch(self, output):
        predictions = torch.nn.functional.softmax(output, dim=1)
        predictions = predictions.cpu().numpy()

        categories = predictions.argmax(axis=1)
        labels = [self.category_map[cat] for cat in categories]
        scores = predictions.max(axis=1).astype(float)

        result = list(zip(labels, scores))
        logger.debug(f"Post-processing result batch: {result}")
        return result


class Resnet50ClassifierLowRes(Resnet50Classifier):
    input_size = 128

    def get_model(self):
        num_classes = len(self.category_map)
        model = torchvision.models.resnet50(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, num_classes)
        model = model.to(self.device)
        checkpoint = torch.load(self.weights, map_location=self.device)
        state_dict = checkpoint.get("model_state_dict") or checkpoint
        model.load_state_dict(state_dict)
        model.eval()
        return model


class BinaryClassifier(EfficientNetClassifier):
    stage = 2
    type = "binary_classification"
    positive_binary_label = None
    positive_negative_label = None

    def get_dataset(self):
        dataset = ClassificationIterableDatabaseDataset(
            queue=DetectedObjectQueue(self.db_path),
            image_transforms=self.get_transforms(),
            batch_size=self.batch_size,
        )
        return dataset

    def save_results(self, object_ids, batch_output):
        # Here we are saving the moth/non-moth labels
        classified_objects_data = [
            {
                "binary_label": str(label),
                "binary_label_score": float(score),
                "in_queue": True if label == constants.POSITIVE_BINARY_LABEL else False,
                "model_name": self.name,
            }
            for label, score in batch_output
        ]
        save_classified_objects(self.db_path, object_ids, classified_objects_data)


class MothNonMothClassifier(BinaryClassifier):
    name = "Moth / Non-Moth Classifier"
    description = "Trained on May 6, 2022"
    weights_path = "https://object-arbutus.cloud.computecanada.ca/ami-models/moths/classification/moth-nonmoth-effv2b3_20220506_061527_30.pth"
    labels_path = "https://object-arbutus.cloud.computecanada.ca/ami-models/moths/classification/05-moth-nonmoth_category_map.json"
    positive_binary_label = "moth"
    positive_negative_label = "nonmoth"


class SpeciesClassifier:
    stage = 3
    type = "fine_grained_classifier"

    def get_dataset(self):
        dataset = ClassificationIterableDatabaseDataset(
            queue=UnclassifiedObjectQueue(self.db_path),
            image_transforms=self.get_transforms(),
            batch_size=self.batch_size,
        )
        return dataset

    def save_results(self, object_ids, batch_output):
        # Here we are saving the moth/non-moth labels
        classified_objects_data = [
            {
                "specific_label": label,
                "specific_label_score": score,
                "model_name": self.name,
            }
            for label, score in batch_output
        ]
        save_classified_objects(self.db_path, object_ids, classified_objects_data)


class QuebecVermontMothSpeciesClassifierMixedResolution(
    SpeciesClassifier, Resnet50Classifier
):
    name = "Quebec & Vermont Species Classifier - Mixed Resolution"
    description = "Trained on December 22, 2022 using lower resolution images"
    weights_path = (
        "https://object-arbutus.cloud.computecanada.ca/ami-models/moths/classification/"
        "quebec-vermont-moth-model_v07_resnet50_2022-12-22-07-54.pt"
    )
    labels_path = (
        "https://object-arbutus.cloud.computecanada.ca/ami-models/moths/classification/"
        "quebec-vermont_moth-category-map_19Jan2023.json"
    )


class QuebecVermontMothSpeciesClassifierLowResolution(
    SpeciesClassifier, Resnet50ClassifierLowRes
):
    name = "Quebec & Vermont Species Classifier - Low Resolution"
    description = "Trained on February 24, 2022 using lower resolution images"
    weights_path = (
        "https://object-arbutus.cloud.computecanada.ca/ami-models/moths/classification/"
        "moths_quebecvermont_resnet50_randaug_mixres_128_fev24.pth"
    )
    labels_path = (
        "https://object-arbutus.cloud.computecanada.ca/ami-models/moths/classification/"
        "quebec-vermont_moth-category-map_19Jan2023.json"
    )


class PanamaMothSpeciesClassifierMixedResolution(SpeciesClassifier, Resnet50Classifier):
    name = "Panama Species Classifier - Mixed Resolution"
    description = "Trained on December 22, 2022 using lower resolution images"
    weights_path = (
        "https://object-arbutus.cloud.computecanada.ca/ami-models/moths/classification/"
        "panama_moth-model_v01_resnet50_2023-01-24-09-51.pt"
    )
    labels_path = (
        "https://object-arbutus.cloud.computecanada.ca/ami-models/moths/classification/"
        "panama_moth-category-map_24Jan2023.json"
    )


class QuebecVermontMothSpeciesClassifier(SpeciesClassifier, EfficientNetClassifier):
    name = "Quebec & Vermont Species Classifier"
    description = "Trained on September 8, 2022 using local species checklist from GBIF"
    weights_path = (
        "https://object-arbutus.cloud.computecanada.ca/ami-models/moths/classification/"
        "quebec-vermont-moth-model_v02_efficientnetv2-b3_2022-09-08-15-44.pt"
    )
    labels_path = (
        "https://object-arbutus.cloud.computecanada.ca/ami-models/moths/classification/"
        "quebec-vermont-moth_category-map_4Aug2022.json"
    )


class UKDenmarkMothSpeciesClassifier(SpeciesClassifier, EfficientNetClassifier):
    name = "UK & Denmark Species Classifier"
    description = (
        "Trained on September 8, 2022 using local species checklist from GBIF."
    )
    weights_path = (
        "https://object-arbutus.cloud.computecanada.ca/ami-models/moths/classification/"
        "uk-denmark-moth-model_v01_efficientnetv2-b3_2022-09-08-12-54.pt"
    )
    labels_path = (
        "https://object-arbutus.cloud.computecanada.ca/ami-models/moths/classification/"
        "uk-denmark-moth_category-map_13Sep2022.json"
    )
