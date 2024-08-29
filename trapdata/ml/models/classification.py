import timm
import torch
import torch.utils.data
import torchvision

from trapdata import constants, logger
from trapdata.db.models.detections import save_classified_objects
from trapdata.db.models.queue import DetectedObjectQueue, UnclassifiedObjectQueue

from .base import InferenceBaseClass, imagenet_normalization


class ClassificationIterableDatabaseDataset(torch.utils.data.IterableDataset):
    def __init__(self, queue, image_transforms, batch_size=4):
        super().__init__()
        self.queue = queue
        self.image_transforms = image_transforms
        self.batch_size = batch_size

    def __len__(self):
        queue_count = self.queue.queue_count()
        if queue_count > 0:
            logger.info(f"Current queue count: {queue_count}")
        return queue_count

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
        return torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((self.input_size, self.input_size)),
                torchvision.transforms.ToTensor(),
                self.normalization,
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


class Resnet50Classifier_Turing(InferenceBaseClass):
    # function to run the Turing models
    logger.info("KG: Resnet50Classifier_Turing")
    input_size = 300

    def get_model(self):
        num_classes = len(self.category_map)
        model = Resnet50(num_classes=num_classes)
        model = model.to(self.device)
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

    def _pad_to_square(self):
        """Padding transformation to make the image square"""
        width, height = self.input_size, self.input_size
        if height < width:
            return torchvision.transforms.Pad(padding=[0, 0, 0, width - height])
        elif height > width:
            return torchvision.transforms.Pad(padding=[0, 0, height - width, 0])
        else:
            return torchvision.transforms.Pad(padding=[0, 0, 0, 0])

    def get_transforms(self):
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        return torchvision.transforms.Compose(
            [
                self._pad_to_square(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Resize(
                    (self.input_size, self.input_size), antialias=True  # type: ignore
                ),
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
        model = torchvision.models.resnet50(weights=None)
        num_ftrs = model.fc.in_features
        assert (
            self.num_classes
        ), f"Number of classes could not be determined for for {self.name}"
        model.fc = torch.nn.Linear(num_ftrs, self.num_classes)
        model = model.to(self.device)
        assert self.weights, f"No weights path configured for {self.name}"
        checkpoint = torch.load(self.weights, map_location=self.device)
        state_dict = checkpoint.get("model_state_dict") or checkpoint
        model.load_state_dict(state_dict)
        model.eval()
        return model


class Resnet50TimmClassifier(Resnet50Classifier):
    def get_model(self):
        model = timm.create_model(
            "resnet50",
            pretrained=False,
            num_classes=self.num_classes,
        )
        assert self.weights, f"No weights path configured for {self.name}"
        checkpoint = torch.load(self.weights, map_location=self.device)
        state_dict = checkpoint.get("model_state_dict") or checkpoint
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        return model


class BinaryClassifier(Resnet50ClassifierLowRes):
    stage = 2
    type = "binary_classification"
    positive_binary_label: str = constants.POSITIVE_BINARY_LABEL
    negative_binary_label: str = constants.NEGATIVE_BINARY_LABEL

    def get_queue(self) -> DetectedObjectQueue:
        return DetectedObjectQueue(self.db_path, self.image_base_path)

    def get_dataset(self):
        dataset = ClassificationIterableDatabaseDataset(
            queue=self.queue,
            image_transforms=self.get_transforms(),
            batch_size=self.batch_size,
        )
        return dataset

    def save_results(self, object_ids, batch_output, *args, **kwargs):
        # Here we are saving the moth/non-moth labels
        classified_objects_data = [
            {
                "binary_label": str(label),
                "binary_label_score": float(score),
                "in_queue": True if label == self.positive_binary_label else False,
                "model_name": self.name,
            }
            for label, score in batch_output
        ]
        save_classified_objects(self.db_path, object_ids, classified_objects_data)


class MothNonMothClassifier2022(EfficientNetClassifier, BinaryClassifier):
    name = "Moth / Non-Moth Classifier"
    description = "Trained on May 6, 2022"
    weights_path = "https://object-arbutus.cloud.computecanada.ca/ami-models/moths/classification/moth-nonmoth-effv2b3_20220506_061527_30.pth"
    labels_path = "https://object-arbutus.cloud.computecanada.ca/ami-models/moths/classification/05-moth-nonmoth_category_map.json"
    positive_binary_label = "moth"
    negative_binary_label = "nonmoth"


class MothNonMothClassifier(BinaryClassifier):
    name = "Moth / Non-Moth Classifier"
    description = "Trained on April 17, 2024"
    weights_path = (
        "https://object-arbutus.cloud.computecanada.ca/ami-models/moths/classification/"
        "moth-nonmoth_resnet50_20240417_b4fe3efe.pth"
    )
    labels_path = (
        "https://object-arbutus.cloud.computecanada.ca/ami-models/moths/classification/"
        "05-moth-nonmoth_category_map.json"
    )
    positive_binary_label = "moth"
    negative_binary_label = "nonmoth"


class SpeciesClassifier(InferenceBaseClass):
    stage = 4
    type = "fine_grained_classifier"

    def get_queue(self) -> UnclassifiedObjectQueue:
        return UnclassifiedObjectQueue(self.db_path, self.image_base_path)

    def get_dataset(self):
        dataset = ClassificationIterableDatabaseDataset(
            queue=self.queue,
            image_transforms=self.get_transforms(),
            batch_size=self.batch_size,
        )
        return dataset

    def save_results(self, object_ids, batch_output, *args, **kwargs):
        # Here we are saving the specific taxon labels
        classified_objects_data = [
            {
                "specific_label": label,
                "specific_label_score": score,
                "model_name": self.name,
                "in_queue": True,  # Put back in queue for the feature extractor & tracking
            }
            for label, score in batch_output
        ]
        save_classified_objects(self.db_path, object_ids, classified_objects_data)


class QuebecVermontMothSpeciesClassifierMixedResolution(
    SpeciesClassifier, Resnet50ClassifierLowRes
):
    name = "Quebec & Vermont Species Classifier"
    description = (
        "Trained on February 24, 2022 using mix of low & med resolution images"
    )
    weights_path = (
        "https://object-arbutus.cloud.computecanada.ca/ami-models/moths/classification/"
        "moths_quebecvermont_resnet50_randaug_mixres_128_fev24.pth"
    )
    labels_path = (
        "https://object-arbutus.cloud.computecanada.ca/ami-models/moths/classification/"
        "quebec-vermont_moth-category-map_19Jan2023.json"
    )


class TuringCostaRicaSpeciesClassifier(SpeciesClassifier, Resnet50Classifier_Turing):
    name = "Turing Costa Rica Species Classifier"
    description = "Trained on 4th June 2024 by Turing team using Resnet50 model."
    weights_path = (
        "https://object-arbutus.cloud.computecanada.ca/ami-models/moths/classification/"
        "turing-costarica_v03_resnet50_2024-06-04-16-17_state.pt"
    )
    labels_path = (
        "https://object-arbutus.cloud.computecanada.ca/ami-models/moths/classification/"
        "03_costarica_data_category_map.json"
    )


class TuringUKSpeciesClassifier(SpeciesClassifier, Resnet50Classifier_Turing):
    name = "Turing UK Species Classifier"
    description = "Trained on 13th May 2024 by Turing team using Resnet50 model."
    weights_path = (
        "https://object-arbutus.cloud.computecanada.ca/ami-models/moths/classification/"
        "turing-uk_v03_resnet50_2024-05-13-10-03_state.pt"
    )
    labels_path = (
        "https://object-arbutus.cloud.computecanada.ca/ami-models/moths/classification/"
        "03_uk_data_category_map.json"
    )


class UKDenmarkMothSpeciesClassifierMixedResolution(
    SpeciesClassifier, Resnet50ClassifierLowRes
):
    """
    Training log and weights can be found here:
    https://wandb.ai/moth-ai/uk-denmark/artifacts/model/model/v0/overview

    Species checklist used for training:
    https://github.com/adityajain07/mothAI/blob/main/species_lists/UK-Denmark-Moth-List_11July2022.csv
    """

    name = "UK & Denmark Species Classifier"
    description = "Trained on April 3, 2023 using mix of low & med resolution images."
    weights_path = (
        "https://object-arbutus.cloud.computecanada.ca/ami-models/moths/classification/"
        "uk-denmark-moths-mixedres-20230403_140131_30.pth"
    )
    labels_path = (
        "https://object-arbutus.cloud.computecanada.ca/ami-models/moths/classification/"
        "01-moths-ukdenmark_v2_category_map_species_names.json"
    )


class PanamaMothSpeciesClassifierMixedResolution(SpeciesClassifier, Resnet50Classifier):
    name = "Panama Species Classifier"
    description = "Trained on December 22, 2022 using a mix of low & med resolution images. 148 species."
    weights_path = (
        "https://object-arbutus.cloud.computecanada.ca/ami-models/moths/classification/"
        "panama_moth-model_v01_resnet50_2023-01-24-09-51.pt"
    )
    labels_path = (
        "https://object-arbutus.cloud.computecanada.ca/ami-models/moths/classification/"
        "panama_moth-category-map_24Jan2023.json"
    )


class PanamaMothSpeciesClassifierMixedResolution2023(
    SpeciesClassifier, Resnet50ClassifierLowRes
):
    name = "Panama Species Classifier 2023"
    lookup_gbif_names = True
    normalization = imagenet_normalization

    description = "Trained on Novempber 11th, 2023 using a corrected species list of 1060 classes."
    weights_path = (
        "https://object-arbutus.cloud.computecanada.ca/ami-models/moths/classification/"
        "panama_resetnet50_best_5aeb515a.pth"
    )
    labels_path = (
        "https://object-arbutus.cloud.computecanada.ca/ami-models/moths/classification/"
        "03_moths_centralAmerica_category_map-202311110-with-names.json"
    )


class GlobalMothSpeciesClassifier(SpeciesClassifier, Resnet50ClassifierLowRes):
    name = "Global Species Classifier 2024"
    lookup_gbif_names = False
    normalization = imagenet_normalization

    description = (
        "Trained on August 28th, 2024 on 30k species. "
        "https://wandb.ai/moth-ai/global-moth-classifier/runs/h0cuqrbc/overview"
    )
    weights_path = (
        "https://object-arbutus.cloud.computecanada.ca/ami-models/moths/classification/"
        "global_resnet50_checkpoint_20240828.pth"
    )
    labels_path = (
        "https://object-arbutus.cloud.computecanada.ca/ami-models/moths/classification/"
        "global_category_map_with_names_20240828.json"
    )
