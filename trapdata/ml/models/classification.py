import timm
import torch
import torch.utils.data
import torchvision

from trapdata import constants, logger
from trapdata.db.models.detections import save_classified_objects
from trapdata.db.models.queue import DetectedObjectQueue, UnclassifiedObjectQueue
from trapdata.ml.utils import load_model_checkpoint

from .base import InferenceBaseClass, imagenet_normalization


class MaybeTensor:
    """Convert PIL Image to tensor if the input is not already a tensor.

    Allows classification transforms to be used with both PIL images
    (e.g. ``ami api`` / database paths) and tensors (e.g. GPU pipeline in
    antenna/worker.py) without a redundant GPU->CPU->GPU round-trip.
    """

    _to_tensor = torchvision.transforms.ToTensor()

    def __call__(self, x):
        if isinstance(x, torch.Tensor):
            return x
        return self._to_tensor(x)

    def __repr__(self):
        return f"{self.__class__.__name__}()"


class QuantizeToUint8Levels:
    """Round a [0, 1] image tensor to the 256 levels an 8-bit image can hold."""

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return torch.round(x * 255) / 255

    def __repr__(self):
        return f"{self.__class__.__name__}()"


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
        logger.debug("Checking for records to classify...")
        are_records = len(self) > 0
        while are_records:
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
            else:
                are_records = False
        logger.debug("No records left to classify")

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
        checkpoint = load_model_checkpoint(self.weights, self.device)
        # The model state dict is nested in some checkpoints, and not in others
        state_dict = checkpoint.get("model_state_dict") or checkpoint
        model.load_state_dict(state_dict)
        model.eval()
        return model

    def get_transforms(self):
        return torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((self.input_size, self.input_size)),
                MaybeTensor(),
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


class ConvNeXtOrderClassifier(InferenceBaseClass):
    """ConvNeXt based insect order classifier"""

    input_size = 128

    def get_model(self):
        num_classes = len(self.category_map)
        model = timm.create_model(
            "convnext_tiny_in22k",
            weights=None,
            num_classes=num_classes,
        )
        model = model.to(self.device)
        checkpoint = load_model_checkpoint(self.weights, self.device)
        # The model state dict is nested in some checkpoints, and not in others
        state_dict = checkpoint.get("model_state_dict") or checkpoint

        model.load_state_dict(state_dict)
        model.eval()
        return model

    def _pad_to_square(self):
        """Padding transformation to make the image square"""

        width, height = self.image.size
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
                # self._pad_to_square(),
                torchvision.transforms.Resize((self.input_size, self.input_size)),
                MaybeTensor(),
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


class Resnet50Classifier_Turing(InferenceBaseClass):
    # function to run the Turing models
    input_size = 300

    def get_model(self):
        num_classes = len(self.category_map)
        model = Resnet50(num_classes=num_classes)
        model = model.to(self.device)
        checkpoint = load_model_checkpoint(self.weights, self.device)
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
                MaybeTensor(),
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
        checkpoint = load_model_checkpoint(self.weights, self.device)
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
                MaybeTensor(),
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
        checkpoint = load_model_checkpoint(self.weights, self.device)
        state_dict = checkpoint.get("model_state_dict") or checkpoint
        model.load_state_dict(state_dict)
        model.eval()
        logger.debug(f"Loaded model ({self.name}) onto device {self.device}")
        return model


class Resnet50TimmClassifier(Resnet50Classifier):
    def get_model(self):
        model = timm.create_model(
            "resnet50",
            weights=None,
            num_classes=self.num_classes,
        )
        assert self.weights, f"No weights path configured for {self.name}"
        checkpoint = load_model_checkpoint(self.weights, self.device)
        state_dict = checkpoint.get("model_state_dict") or checkpoint
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        return model


class CosineClassifierHead(torch.nn.Module):
    """
    Classification head of models trained with mini_trainer
    (https://github.com/asgersvenning/mini_trainer), such as Nemo (MAMBO v3).

    A class scores by the cosine between the normalized embedding and its normalized
    weight vector, rescaled to a z-score, plus a bias. Module names match the
    checkpoint keys so its state dict loads strictly.
    """

    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        self.hidden = torch.nn.Linear(in_features, in_features)
        # Present in the checkpoint; a normalized head does not use it
        self.batch_norm = torch.nn.BatchNorm1d(in_features)
        self.linear = torch.nn.utils.parametrizations.weight_norm(
            torch.nn.Linear(in_features, num_classes), name="weight", dim=0
        )
        # Unit-length class vectors, so the product with the embedding is a cosine
        gain = self.linear.parametrizations.weight.original0
        torch.nn.init.ones_(gain)
        gain.requires_grad_(False)

    def keep_classes(self, indexes: list[int]) -> None:
        """Score only these classes, in this order."""
        index = torch.as_tensor(indexes, device=self.linear.bias.device)
        with torch.no_grad():
            weight = self.linear.weight.index_select(0, index)
            bias = self.linear.bias.index_select(0, index)
        self.linear = torch.nn.Linear(weight.shape[1], weight.shape[0])
        with torch.no_grad():
            self.linear.weight.copy_(weight)
            self.linear.bias.copy_(bias)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.leaky_relu(self.hidden(x))
        return torch.nn.functional.normalize(x, p=2, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cosine = torch.nn.functional.linear(self.embed(x), self.linear.weight)
        eps = 1e-7
        angle = torch.acos(-cosine.clamp(-1 + eps, 1 - eps))
        zscore = (angle - torch.pi / 2) * (self.hidden.out_features - 2) ** 0.5
        return zscore + self.linear.bias


class MiniTrainerClassifier(InferenceBaseClass):
    """
    A torchvision backbone with the mini_trainer cosine head.

    The backbone and embedding width come from the metadata mini_trainer stores in the
    checkpoint, and a checkpoint whose input size or head type does not match this
    class is refused. The label map decides which classes are offered: its indexes are
    the checkpoint's output indexes, and outputs it leaves out are dropped. Only the
    leaf (species) scores are used; Antenna derives higher ranks from the taxonomy.
    """

    input_size = 384
    # Preprocessing of the released model: squash to a square, upscale, center crop
    resize_size = 438
    normalization = imagenet_normalization

    def get_labels(self, labels_path) -> dict[int, str]:
        labels = super().get_labels(labels_path)
        # Checkpoint output index of each class offered, in the order offered
        self.output_indexes = sorted(labels)
        return {new: labels[old] for new, old in enumerate(self.output_indexes)}

    def get_model(self):
        checkpoint = load_model_checkpoint(self.weights, self.device)
        state_dict = checkpoint.get("model_state_dict") or checkpoint
        head_name, metadata = self._checkpoint_metadata(state_dict)
        self._check_metadata(metadata)

        model = getattr(torchvision.models, metadata["backbone_class"])(weights=None)
        head = CosineClassifierHead(metadata["in_features"], metadata["out_features"])
        setattr(model, head_name, head)

        # Hierarchy masks map species to genus and family; ranks above species are not
        # scored here
        state_dict = {
            key: value
            for key, value in state_dict.items()
            if not key.startswith((f"{head_name}.mask_", f"{head_name}._extra_state"))
        }
        model.load_state_dict(state_dict)
        if len(self.output_indexes) < metadata["out_features"]:
            head.keep_classes(self.output_indexes)
        model = model.to(self.device)
        model.eval()
        return model

    @staticmethod
    def _checkpoint_metadata(state_dict: dict) -> tuple[str, dict]:
        for key, value in state_dict.items():
            if key.endswith("._extra_state") and isinstance(value, dict):
                if "mini_trainer_version" in value:
                    return value["backbone_output_name"], value
        raise ValueError("Checkpoint has no mini_trainer metadata")

    def _check_metadata(self, metadata: dict) -> None:
        expected = {
            "normalized": True,
            "hidden": True,
            "preferred_size": self.input_size,
        }
        mismatched = {
            key: (metadata.get(key), value)
            for key, value in expected.items()
            if metadata.get(key) != value
        }
        if mismatched:
            raise ValueError(
                f"Checkpoint for {self.name} does not match the model settings "
                f"(checkpoint value, expected value): {mismatched}"
            )
        outside = [i for i in self.output_indexes if i >= metadata["out_features"]]
        if outside:
            raise ValueError(
                f"Label map for {self.name} names output indexes the checkpoint does "
                f"not have ({metadata['out_features']} outputs): {outside[:10]}"
            )

    def get_transforms(self):
        # Resize tensors, not PIL images: tensor nearest-neighbour sampling picks the
        # same source pixels as the released preprocessing, PIL's does not
        return torchvision.transforms.Compose(
            [
                MaybeTensor(),
                torchvision.transforms.Resize(
                    (self.input_size, self.input_size),
                    interpolation=torchvision.transforms.InterpolationMode.NEAREST,
                ),
                torchvision.transforms.Resize(self.resize_size, antialias=False),
                QuantizeToUint8Levels(),
                torchvision.transforms.CenterCrop(self.input_size),
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


class BinaryClassifier(Resnet50ClassifierLowRes):
    stage = 2
    type = "binary_classification"
    positive_binary_label: str = constants.POSITIVE_BINARY_LABEL
    negative_binary_label: str = constants.NEGATIVE_BINARY_LABEL
    default_taxon_rank = "SUPERFAMILY"

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
    weights_path = "moths/classification/moth-nonmoth-effv2b3_20220506_061527_30.pth"
    labels_path = "moths/classification/05-moth-nonmoth_category_map.json"
    positive_binary_label = "moth"
    negative_binary_label = "nonmoth"


class MothNonMothClassifier(BinaryClassifier):
    name = "Moth / Non-Moth Classifier"
    description = "Trained on April 17, 2024"
    weights_path = "moths/classification/" "moth-nonmoth_resnet50_20240417_b4fe3efe.pth"
    labels_path = "moths/classification/" "05-moth-nonmoth_category_map.json"
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
        "moths/classification/"
        "moths_quebecvermont_resnet50_randaug_mixres_128_fev24.pth"
    )
    labels_path = (
        "moths/classification/" "quebec-vermont_moth-category-map_19Jan2023.json"
    )


class TuringCostaRicaSpeciesClassifier(SpeciesClassifier, Resnet50Classifier_Turing):
    name = "Turing Costa Rica Species Classifier"
    description = "Trained on 4th June 2024 by Turing team using Resnet50 model."
    weights_path = (
        "moths/classification/"
        "turing-costarica_v03_resnet50_2024-06-04-16-17_state.pt"
    )
    labels_path = "moths/classification/" "03_costarica_data_category_map.json"


class TuringAnguillaSpeciesClassifier(SpeciesClassifier, Resnet50Classifier_Turing):
    name = "Turing Anguilla Species Classifier"
    description = "Trained on 28th June 2024 by Turing team using Resnet50 model."
    weights_path = (
        "moths/classification/" "turing-anguilla_v01_resnet50_2024-06-28-17-01_state.pt"
    )
    labels_path = "moths/classification/" "01_anguilla_data_category_map.json"


class TuringKenyaUgandaSpeciesClassifier(SpeciesClassifier, Resnet50Classifier_Turing):
    name = "Turing Kenya and Uganda Species Classifier"
    description = "Trained on 19th November 2024 by Turing team using Resnet50 model."
    weights_path = (
        "moths/classification/"
        "turing-kenya-uganda_v01_resnet50_2024-11-19-18-44_state.pt"
    )
    labels_path = "moths/classification/" "01_kenya-uganda_data_category_map.json"


class TuringUKSpeciesClassifier(SpeciesClassifier, Resnet50Classifier_Turing):
    name = "Turing UK Species Classifier"
    description = "Trained on 13th May 2024 by Turing team using Resnet50 model."
    weights_path = (
        "moths/classification/" "turing-uk_v03_resnet50_2024-05-13-10-03_state.pt"
    )
    labels_path = "moths/classification/" "03_uk_data_category_map.json"


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
        "moths/classification/" "uk-denmark-moths-mixedres-20230403_140131_30.pth"
    )
    labels_path = (
        "moths/classification/" "01-moths-ukdenmark_v2_category_map_species_names.json"
    )


class PanamaMothSpeciesClassifierMixedResolution(SpeciesClassifier, Resnet50Classifier):
    name = "Panama Species Classifier"
    description = "Trained on December 22, 2022 using a mix of low & med resolution images. 148 species."
    weights_path = (
        "moths/classification/" "panama_moth-model_v01_resnet50_2023-01-24-09-51.pt"
    )
    labels_path = "moths/classification/" "panama_moth-category-map_24Jan2023.json"


class PanamaMothSpeciesClassifierMixedResolution2023(
    SpeciesClassifier, Resnet50ClassifierLowRes
):
    name = "Panama Species Classifier 2023"
    lookup_gbif_names = True
    normalization = imagenet_normalization

    description = (
        "Trained on November 11th, 2023 using a corrected species list of 1060 classes."
    )
    weights_path = "moths/classification/" "panama_resetnet50_best_5aeb515a.pth"
    labels_path = (
        "moths/classification/"
        "03_moths_centralAmerica_category_map-202311110-with-names.json"
    )


class GlobalMothSpeciesClassifier(SpeciesClassifier, Resnet50TimmClassifier):
    input_size = 128
    normalization = imagenet_normalization
    lookup_gbif_names = False

    name = "Global Species Classifier - Aug 2024"
    description = (
        "Trained on August 28th, 2024 for 29,176 species. "
        "https://wandb.ai/moth-ai/global-moth-classifier/runs/h0cuqrbc/overview"
    )
    weights_path = "moths/classification/" "global_resnet50_20240828_b06d3b3a.pth"
    labels_path = "moths/classification/" "global_category_map_with_names_20240828.json"


class QuebecVermontMothSpeciesClassifier2024(SpeciesClassifier, Resnet50TimmClassifier):
    input_size = 128
    normalization = imagenet_normalization
    lookup_gbif_names = False

    name = "Quebec & Vermont Species Classifier - Apr 2024"
    description = (
        "Trained on April 17, 2024 for 2,497 species. "
        "https://wandb.ai/moth-ai/ami-gbif-fine-grained/runs/1x53zmp2/overview"
    )
    weights_path = (
        "moths/classification/" "quebec-vermont_resnet50_baseline_20240417_950de764.pth"
    )
    labels_path = (
        "moths/classification/"
        "01_ami-gbif_fine-grained_ne-america_category_map-with_names.json"
    )


class UKDenmarkMothSpeciesClassifier2024(SpeciesClassifier, Resnet50TimmClassifier):
    input_size = 128
    normalization = imagenet_normalization
    lookup_gbif_names = False

    name = "UK & Denmark Species Classifier - Apr 2024"
    description = (
        "Trained on April 17, 2024 for 2,603 species. "
        "https://wandb.ai/moth-ai/ami-gbif-fine-grained/runs/x5u7jcbf/overview"
    )
    weights_path = (
        "moths/classification/" "uk-denmark_resnet50_baseline_20240417_55250a8b.pth"
    )
    labels_path = (
        "moths/classification/"
        "02_ami-gbif_fine-grained_w-europe_category_map-with_names.json"
    )


class PanamaMothSpeciesClassifier2024(SpeciesClassifier, Resnet50TimmClassifier):
    input_size = 128
    normalization = imagenet_normalization
    lookup_gbif_names = False

    name = "Panama Species Classifier - Apr 2024"
    description = (
        "Trained on April 17, 2024 for 636 species. "
        "https://wandb.ai/moth-ai/ami-gbif-fine-grained/runs/1sekgkan/overview"
    )

    weights_path = (
        "moths/classification/" "panama_resnet50_baseline_20240417_edbb46dd.pth"
    )
    labels_path = (
        "moths/classification/"
        "03_ami-gbif_fine-grained_c-america_category_map-with_names.json"
    )


class GlobalMothSpeciesClassifierNemo2026(SpeciesClassifier, MiniTrainerClassifier):
    lookup_gbif_names = False

    name = "Global Species Classifier - Nemo (MAMBO v3)"
    description = (
        "Nemo (MAMBO v3) by Asger Svenning, Aarhus University. EfficientNetV2-S "
        "trained on September 11, 2026 on 5.06M GBIF images of 12,632 moth and "
        "butterfly species. Weights licensed CC BY-NC-SA 4.0. "
        "https://huggingface.co/asgersvenning/MAMBO-v3"
    )
    # Copy of bundle/models/pytorch/best.pt from the Hugging Face repo at revision
    # a70e530 (sha256 174b9214...5e8a), renamed so it cannot collide in the model cache
    weights_path = (
        "moths/classification/" "global_nemo-mambo-v3_efficientnet_v2_s_20260911.pt"
    )
    labels_path = (
        "moths/classification/" "global_nemo-mambo-v3_category_map_with_names.json"
    )


class InsectOrderClassifier2025(SpeciesClassifier, ConvNeXtOrderClassifier):
    name = "Insect Order Classifier"
    description = "ConvNeXt-T based insect order classifier for 16 classes trained by Mila in January 2025"
    weights_path = (
        "insect_orders/convnext_tiny_in22k_worder0.5_wbinary0.5_run2_checkpoint.pt"
    )
    labels_path = "insect_orders/insect_order_category_map.json"
    default_taxon_rank = "ORDER"
