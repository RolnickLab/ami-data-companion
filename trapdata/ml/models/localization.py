import pathlib

import torch
import torchvision
import PIL.Image

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from trapdata import db
from trapdata import logger
from trapdata import models

from trapdata.models.detections import save_detected_objects
from trapdata.ml.models.base import InferenceModel


class ObjectDetector(InferenceModel):
    pass


class LocalizationDatabaseDataset(torch.utils.data.Dataset):
    def __init__(self, db_path, image_transforms):
        super().__init__()

        self.db_path = db_path
        self.transform = image_transforms
        self.query_args = {"in_queue": True}

    def __len__(self):
        with db.get_session(self.db_path) as sess:
            count = sess.query(models.Image).filter_by(**self.query_args).count()
            logger.info(f"Images found in queue: {count}")
            return count

    def __getitem__(self, idx):
        # @TODO this exits with an exception if there are no
        # images in the queue.
        # @TODO use a custom sampler instead to query all images in the batch
        # from the DB at one, rather than one by one.
        with db.get_session(self.db_path) as sess:
            next_image = sess.query(models.Image).filter_by(**self.query_args).first()
            if next_image:
                img_path = next_image.absolute_path
                pil_image = PIL.Image.open(img_path)
                next_image.in_queue = False
                item = (next_image.id, self.transform(pil_image))
                sess.add(next_image)
                sess.commit()
                return item


class LocalizationFilesystemDataset(torch.utils.data.Dataset):
    def __init__(self, directory, image_names):
        super().__init__()

        self.directory = pathlib.Path(directory)
        self.image_names = image_names
        self.transform = self.get_transforms()

    def __len__(self):
        return len(self.image_names)

    def get_transforms(self):
        transform_list = [torchvision.transforms.ToTensor()]
        return torchvision.transforms.Compose(transform_list)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = self.directory / img_name
        pil_image = PIL.Image.open(img_path)
        return str(img_path), self.transform(pil_image)


class MothFasterRCNNObjectDetector(ObjectDetector):
    name = "FasterRCNN for AMI Traps 2021"
    weights_path = "https://object-arbutus.cloud.computecanada.ca/ami-models/moths/localization/v1_localizmodel_2021-08-17-12-06.pt"
    model_type = "object_detection"
    stage = 1
    description = (
        "Model trained on moth trap data in 2021. "
        "Accurate but can be slow on a machine without GPU."
    )
    bbox_score_threshold = 0.99

    def get_model(self):
        logger.info(
            f'Loading "{self.name}" "{self.model_type}" model with weights {self.weights}'
        )
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
        num_classes = 2  # 1 class (object) + background
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        checkpoint = torch.load(self.weights, map_location=self.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(self.device)
        model.eval()
        self.model = model
        return self.model

    def get_dataset(self):
        dataset = LocalizationDatabaseDataset(
            db_path=self.db_path,
            image_transforms=self.get_transforms(),
        )
        return dataset

    def post_process_single(self, output):
        # This model does not use the labels from the object detection model
        _ = output["labels"]
        assert all([label == 1 for label in output["labels"]])

        # Filter out objects if their score is under score threshold
        bboxes = output["boxes"][output["scores"] > self.bbox_score_threshold]

        logger.info(
            f"Keeping {len(bboxes)} out of {len(output['boxes'])} objects found (threshold: {self.bbox_score_threshold})"
        )

        bboxes = bboxes.cpu().numpy().astype(int).tolist()
        return bboxes

    def save_results(self, item_ids, batch_output):
        # Format data to be saved in DB
        # Here we are just saving the bboxes of detected objects
        detected_objects_data = []
        for image_output in batch_output:
            detected_objects = [{"bbox": bbox} for bbox in image_output]
            detected_objects_data.append(detected_objects)

        save_detected_objects(self.db_path, item_ids, detected_objects_data)
