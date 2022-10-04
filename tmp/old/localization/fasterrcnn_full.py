import time

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from trapdata import logger
from trapdata.ml.utils import get_device, synchronize_clocks, get_or_download_file
from trapdata.db.models.detections import save_detected_objects

from .. import InferenceModel
from .dataloaders import LocalizationDatabaseDataset

LOCALIZATION_SCORE_THRESHOLD = 0.99


class FasterRCNN_ResNet50_FPN(InferenceModel):
    title = ("FasterRCNN for AMI Traps 2021",)
    weights = "https://object-arbutus.cloud.computecanada.ca/ami-models/moths/localization/v1_localizmodel_2021-08-17-12-06.pt"
    type = "object_detection"
    stage = 1
    description = (
        "Model trained on moth trap data in 2021. "
        "Accurate but can be slow on a machine without GPU."
    )

    def get_model(self, weights):
        logger.info(
            f'Loading "fasterrcnn_full" localization model with weights: {weights}'
        )
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
        num_classes = 2  # 1 class (object) + background
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        checkpoint = torch.load(weights, map_location=self.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(self.device)
        model.eval()
        return model

    def get_dataset(self):
        dataset = LocalizationDatabaseDataset(
            base_directory=self.db_path,
            image_transforms=self.get_transforms(),
        )
        return dataset

    def post_process(img_path, output):
        # This model does not use the labels from the object detection model
        _ = output["labels"]
        assert all([label == 1 for label in output["labels"]])

        # Filter out objects if their score is under score threshold
        bboxes = output["boxes"][output["scores"] > LOCALIZATION_SCORE_THRESHOLD]

        logger.info(
            f"Keeping {len(bboxes)} out of {len(output['boxes'])} objects found (threshold: {LOCALIZATION_SCORE_THRESHOLD})"
        )

        bboxes = bboxes.cpu().numpy().astype(int).tolist()
        return bboxes

    def format_results(self, output):
        # Format data to be saved in DB
        # Here we are just saving the bboxes of detected objects
        detected_objects_data = []
        for image_output in output:
            detected_objects = [{"bbox": bbox} for bbox in image_output]
            detected_objects_data.append(detected_objects)

    def save_results(self, batch_output):
        results = self.format_results(batch_output)
        save_detected_objects(results)
