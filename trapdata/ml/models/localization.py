import pathlib

import torch
import torchvision
import PIL.Image

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from trapdata import db
from trapdata.db.models.queue import ImageQueue
from trapdata import TrapImage
from trapdata import logger

from trapdata.db.models.detections import save_detected_objects
from trapdata.ml.models.base import InferenceBaseClass


class LocalizationIterableDatabaseDataset(torch.utils.data.IterableDataset):
    def __init__(self, queue, image_transforms, batch_size=1):
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
                    [self.transform(record.absolute_path) for record in records]
                )

                yield (item_ids, batch_data)

    def transform(self, img_path):
        return self.image_transforms(PIL.Image.open(img_path))


class LocalizationDatabaseDataset(torch.utils.data.Dataset):
    def __init__(self, db_path, image_transforms):
        super().__init__()

        self.db_path = db_path
        self.transform = image_transforms
        self.query_args = {"in_queue": True}

    def __len__(self):
        with db.get_session(self.db_path) as sesh:
            count = sesh.query(TrapImage).filter_by(**self.query_args).count()
            logger.info(f"Images found in queue: {count}")
            return int(count)

    def __getitem__(self, idx):
        # @TODO this exits with an exception if there are no
        # images in the queue.
        # @TODO use a custom sampler instead to query all images in the batch
        # from the DB at one, rather than one by one.

        # What properties do we need while session is open?
        item_id, img_path = None, None

        with db.get_session(self.db_path) as sesh:
            next_image = sesh.query(TrapImage).filter_by(**self.query_args).first()

            if not next_image:
                return

            img_path = next_image.absolute_path
            item_id = next_image.id
            next_image.in_queue = False
            sesh.add(next_image)
            sesh.commit()

        img_path = img_path
        pil_image = PIL.Image.open(img_path)
        item = (item_id, self.transform(pil_image))

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


class ObjectDetector(InferenceBaseClass):
    title = "Unknown Object Detector"
    type = "object_detection"
    stage = 1

    def get_transforms(self):
        return torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
            ]
        )

    def get_dataset(self):
        dataset = LocalizationIterableDatabaseDataset(
            queue=ImageQueue(self.db_path, self.deployment_path),
            image_transforms=self.get_transforms(),
            batch_size=self.batch_size,
        )
        return dataset

    def save_results(self, item_ids, batch_output):
        # Format data to be saved in DB
        # Here we are just saving the bboxes of detected objects
        detected_objects_data = []
        for image_output in batch_output:
            detected_objects = [
                {
                    "bbox": bbox,
                    "model_name": self.name,
                }
                for bbox in image_output
            ]
            detected_objects_data.append(detected_objects)

        save_detected_objects(
            self.db_path, item_ids, detected_objects_data, self.user_data_path
        )


class MothObjectDetector_FasterRCNN(ObjectDetector):
    name = "FasterRCNN for AMI Moth Traps 2021"
    weights_path = "https://object-arbutus.cloud.computecanada.ca/ami-models/moths/localization/v1_localizmodel_2021-08-17-12-06.pt"
    description = (
        "Model trained on moth trap data in 2021. "
        "Accurate but can be slow on a machine without GPU."
    )
    bbox_score_threshold = 0.99

    def get_model(self):
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
        num_classes = 2  # 1 class (object) + background
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        logger.debug(f"Loading weights: {self.weights}")
        checkpoint = torch.load(self.weights, map_location=self.device)
        state_dict = checkpoint.get("model_state_dict") or checkpoint
        model.load_state_dict(state_dict)
        model = model.to(self.device)
        model.eval()
        self.model = model
        return self.model

    def post_process_single(self, output):
        # This model does not use the labels from the object detection model
        _ = output["labels"]
        assert all([label == 1 for label in output["labels"]])

        # Filter out objects if their score is under score threshold
        bboxes = output["boxes"][output["scores"] > self.bbox_score_threshold]

        logger.debug(
            f"Keeping {len(bboxes)} out of {len(output['boxes'])} objects found (threshold: {self.bbox_score_threshold})"
        )

        bboxes = bboxes.cpu().numpy().astype(int).tolist()
        return bboxes


class GenericObjectDetector_FasterRCNN_MobileNet(ObjectDetector):
    name = "Pre-trained FasterRCNN with MobileNet backend"
    description = (
        "Faster version of FasterRCNN but not trained on moth trap data. "
        "Produces multiple overlapping bounding boxes. But helpful for testing on CPU machines."
    )
    bbox_score_threshold = 0.01

    def get_model(self):
        model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(
            weights="DEFAULT"
        )
        # @TODO can I use load_state_dict here with weights="DEFAULT"?
        model = model.to(self.device)
        model.eval()
        return model

    def post_process_single(self, output):
        # This model does not use the labels from the object detection model
        _ = output["labels"]

        # Filter out objects if their score is under score threshold
        bboxes = output["boxes"][
            (output["scores"] > self.bbox_score_threshold) & (output["labels"] > 1)
        ]

        # Filter out background label, if using pretrained model only!
        bboxes = output["boxes"][output["labels"] > 1]

        logger.debug(
            f"Keeping {len(bboxes)} out of {len(output['boxes'])} objects found (threshold: {self.bbox_score_threshold})"
        )

        bboxes = bboxes.cpu().numpy().astype(int).tolist()
        return bboxes
