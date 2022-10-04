import torch
import torchvision
import timm
from PIL import Image

from trapdata import logger
from trapdata import constants
from trapdata import db
from trapdata import models
from trapdata.models.detections import save_classified_objects

from .base import InferenceBaseClass


class BinaryClassificationDatabaseDataset(torch.utils.data.Dataset):
    def __init__(self, db_path, image_transforms):
        super().__init__()

        self.db_path = db_path
        self.transform = image_transforms
        self.query_args = {
            "in_queue": True,
            "binary_label": None,
        }

    def __len__(self):
        with db.get_session(self.db_path) as sess:
            count = (
                sess.query(models.DetectedObject)
                .filter(models.DetectedObject.bbox.is_not(None))
                .filter_by(**self.query_args)
                .count()
            )
            logger.info(f"Detected Objects found in queue: {count}")
            return count

    def __getitem__(self, idx):
        with db.get_session(self.db_path) as sess:
            next_obj = (
                sess.query(models.DetectedObject)
                .filter(models.DetectedObject.bbox.is_not(None))
                .filter_by(**self.query_args)
                .options(db.orm.joinedload(models.DetectedObject.image))
                .first()
            )
            if next_obj:
                # @TODO improve. Can't the main transforms chain do this?
                # if we pass the bbox to get_transforms?
                img = Image.open(next_obj.image.absolute_path)
                img = torchvision.transforms.ToTensor()(img)
                x1, y1, x2, y2 = next_obj.bbox
                cropped_image = img[
                    :,
                    int(y1) : int(y2),
                    int(x1) : int(x2),
                ]
                cropped_image = torchvision.transforms.ToPILImage()(cropped_image)
                next_obj.in_queue = False
                item = (next_obj.id, self.transform(cropped_image))
                sess.add(next_obj)
                sess.commit()
                return item


class SpeciesClassificationDatabaseDataset(torch.utils.data.Dataset):
    def __init__(self, db_path, image_transforms):
        super().__init__()

        self.db_path = db_path
        self.transform = image_transforms
        self.query_args = {
            "in_queue": True,
            "specific_label": None,
            "binary_label": constants.POSITIVE_BINARY_LABEL,
        }

    def __len__(self):
        with db.get_session(self.db_path) as sess:
            count = (
                sess.query(models.DetectedObject)
                .filter(models.DetectedObject.bbox.is_not(None))
                .filter_by(**self.query_args)
                .count()
            )
            logger.info(f"Objects for species classification found in queue: {count}")
            return count

    def __getitem__(self, idx):
        with db.get_session(self.db_path) as sess:
            next_obj = (
                sess.query(models.DetectedObject)
                .filter(models.DetectedObject.bbox.is_not(None))
                .filter_by(**self.query_args)
                .options(db.orm.joinedload(models.DetectedObject.image))
                .first()
            )
            if next_obj:
                # @TODO improve. Can't the main transforms chain do this?
                # if we pass the bbox to get_transforms?
                img = Image.open(next_obj.image.absolute_path)
                img = torchvision.transforms.ToTensor()(img)
                x1, y1, x2, y2 = next_obj.bbox
                cropped_image = img[
                    :,
                    int(y1) : int(y2),
                    int(x1) : int(x2),
                ]
                cropped_image = torchvision.transforms.ToPILImage()(cropped_image)
                next_obj.in_queue = False
                item = (next_obj.id, self.transform(cropped_image))
                sess.add(next_obj)
                sess.commit()
                return item


class EfficientNetClassifier(InferenceBaseClass):
    type = "classification"
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


class MothNonMothClassifier(EfficientNetClassifier):
    name = "Moth / Non-Moth Classifier"
    weights_path = "https://object-arbutus.cloud.computecanada.ca/ami-models/moths/classification/moth-nonmoth-effv2b3_20220506_061527_30.pth"
    labels_path = "https://object-arbutus.cloud.computecanada.ca/ami-models/moths/classification/05-moth-nonmoth_category_map.json"
    stage = 2

    def get_dataset(self):
        dataset = BinaryClassificationDatabaseDataset(
            db_path=self.db_path,
            image_transforms=self.get_transforms(),
        )
        return dataset

    def save_results(self, object_ids, batch_output):
        # Here we are saving the moth/non-moth labels
        classified_objects_data = [
            {
                "binary_label": str(label),
                "binary_label_score": float(score),
                "in_queue": True if label == constants.POSITIVE_BINARY_LABEL else False,
            }
            for label, score in batch_output
        ]
        save_classified_objects(self.db_path, object_ids, classified_objects_data)


class SpeciesClassifier(EfficientNetClassifier):
    stage = 3

    def get_dataset(self):
        dataset = SpeciesClassificationDatabaseDataset(
            db_path=self.db_path,
            image_transforms=self.get_transforms(),
        )
        return dataset

    def save_results(self, object_ids, batch_output):
        # Here we are saving the moth/non-moth labels
        classified_objects_data = [
            {
                "specific_label": label,
                "specific_label_score": score,
            }
            for label, score in batch_output
        ]
        save_classified_objects(self.db_path, object_ids, classified_objects_data)


class QuebecVermontMothSpeciesClassifier(SpeciesClassifier):
    name = "Quebec & Vermont Species Classifier"
    weights_path = "https://object-arbutus.cloud.computecanada.ca/ami-models/moths/classification/quebec-vermont-moth-model_v02_efficientnetv2-b3_2022-09-08-15-44.pt"
    labels_path = "https://object-arbutus.cloud.computecanada.ca/ami-models/moths/classification/quebec-vermont-moth_category-map_4Aug2022.json"


class UKDenmarkMothSpeciesClassifier(SpeciesClassifier):
    name = "Quebec & Vermont Species Classifier"
    weights_path = "https://object-arbutus.cloud.computecanada.ca/ami-models/moths/classification/uk-denmark-moth-model_v01_efficientnetv2-b3_2022-09-08-12-54.pt"
    labels_path = "https://object-arbutus.cloud.computecanada.ca/ami-models/moths/classification/uk-denmark-moth_category-map_13Sep2022.json"
