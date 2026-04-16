import dataclasses
import pathlib

import cv2
import numpy as np
import torch
import torchvision
import torchvision.models.detection.anchor_utils
import torchvision.models.detection.backbone_utils
import torchvision.models.detection.faster_rcnn
import torchvision.models.mobilenetv3

from trapdata import TrapImage, db, logger
from trapdata.db.models.detections import save_detected_objects
from trapdata.db.models.queue import ImageQueue
from trapdata.ml.models.base import InferenceBaseClass
from trapdata.ml.utils import open_image


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
                # Filter out None transforms
                valid_records = []
                valid_transforms = []

                for record in records:
                    transformed = self.transform(record.absolute_path)
                    if transformed is not None:
                        valid_records.append(record)
                        valid_transforms.append(transformed)

                # Only yield if we have valid images
                if valid_transforms:
                    item_ids = torch.utils.data.default_collate(
                        [record.id for record in valid_records]
                    )

                    # Try batch collation first, fall back to list if sizes differ
                    try:
                        batch_data = torch.utils.data.default_collate(valid_transforms)
                    except RuntimeError as e:
                        if "stack expects each tensor to be equal size" in str(e):
                            # Fallback: return as list for variable sizes
                            logger.info(
                                "Image sizes differ, returning as list for individual processing"
                            )
                            batch_data = valid_transforms
                        else:
                            # Re-raise if it's a different RuntimeError
                            raise

                    yield (item_ids, batch_data)

    def transform(self, img_path):
        img = open_image(img_path, raise_exception=False)
        if img is None:
            return None
        return self.image_transforms(img)


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
        pil_image = open_image(img_path, raise_exception=False)
        if pil_image is None:
            logger.warning(f"Failed to open image: {img_path}")
            return None

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
        pil_image = open_image(img_path, raise_exception=False)
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

    def get_queue(self) -> ImageQueue:
        return ImageQueue(self.db_path, self.image_base_path)

    def get_dataset(self):
        dataset = LocalizationIterableDatabaseDataset(
            queue=self.queue,
            image_transforms=self.get_transforms(),
            batch_size=self.batch_size,
        )
        return dataset

    def predict_batch(self, batch):
        """
        Override base class method to handle both batched tensors and lists of tensors.
        The dataset now handles size mismatches and provides the appropriate format.
        """
        if isinstance(batch, torch.Tensor):
            # Same-size images: use efficient batch transfer
            batch_input = batch.to(self.device, non_blocking=True)
            batch_output = self.model(batch_input)
            return batch_output
        elif isinstance(batch, list):
            # Different-size images: transfer individually
            batch_input = [img.to(self.device, non_blocking=True) for img in batch]
            batch_output = self.model(batch_input)
            return batch_output
        else:
            raise TypeError(f"Expected tensor or list of tensors, got {type(batch)}")

    def save_results(self, item_ids, batch_output, *args, **kwargs):
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


class MothObjectDetector_FasterRCNN_2021(ObjectDetector):
    name = "FasterRCNN for AMI Moth Traps 2021"
    weights_path = "https://object-arbutus.cloud.computecanada.ca/ami-models/moths/localization/v1_localizmodel_2021-08-17-12-06.pt"
    description = (
        "Model trained on moth trap data in 2021. "
        "Accurate but can be slow on a machine without GPU."
    )
    bbox_score_threshold = 0.99
    box_detections_per_img = 500

    def get_model(self):
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            weights=None,
            box_detections_per_img=self.box_detections_per_img,
        )
        num_classes = 2  # 1 class (object) + background
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = (
            torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
                in_features, num_classes
            )
        )
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
        assert all(label == 1 for label in output["labels"])

        # Filter out objects if their score is under score threshold
        bboxes = output["boxes"][output["scores"] > self.bbox_score_threshold]

        logger.debug(
            f"Keeping {len(bboxes)} out of {len(output['boxes'])} objects found (threshold: {self.bbox_score_threshold})"
        )

        bboxes = bboxes.cpu().numpy().astype(int).tolist()
        return bboxes


class MothObjectDetector_FasterRCNN_2023(ObjectDetector):
    name = "FasterRCNN for AMI Moth Traps 2023"
    weights_path = "https://object-arbutus.cloud.computecanada.ca/ami-models/moths/localization/fasterrcnn_resnet50_fpn_tz53qv9v.pt"
    description = (
        "Model trained on GBIF images and synthetic data in 2023. "
        "Accurate but can be slow on a machine without GPU."
    )
    bbox_score_threshold = 0.80
    box_detections_per_img = 500

    def get_model(self):
        num_classes = 2  # 1 class (object) + background
        logger.debug(f"Loading weights: {self.weights}")
        model = torchvision.models.get_model(
            name="fasterrcnn_resnet50_fpn",
            num_classes=num_classes,
            weights=None,
            box_detections_per_img=self.box_detections_per_img,
        )
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
        assert all(label == 1 for label in output["labels"])

        # Filter out objects if their score is under score threshold
        bboxes = output["boxes"][output["scores"] > self.bbox_score_threshold]

        logger.debug(
            f"Keeping {len(bboxes)} out of {len(output['boxes'])} objects found (threshold: {self.bbox_score_threshold})"
        )

        bboxes = bboxes.cpu().numpy().astype(int).tolist()
        return bboxes


class MothObjectDetector_FasterRCNN_MobileNet_2023(ObjectDetector):
    name = "FasterRCNN - MobileNet for AMI Moth Traps 2023"
    weights_path = "https://object-arbutus.cloud.computecanada.ca/ami-models/moths/localization/fasterrcnn_mobilenet_v3_large_fpn_uqfh7u9w.pt"
    description = (
        "Model trained on GBIF images and synthetic data in 2023. "
        "Slightly less accurate but much faster than other models."
    )
    bbox_score_threshold = 0.50
    trainable_backbone_layers = 6  # all layers are trained
    anchor_sizes = (64, 128, 256, 512)
    num_classes = 2
    box_detections_per_img = 500

    def get_model(self):
        norm_layer = torch.nn.BatchNorm2d
        backbone = torchvision.models.mobilenetv3.mobilenet_v3_large(
            weights=None, norm_layer=norm_layer
        )
        backbone = torchvision.models.detection.backbone_utils._mobilenet_extractor(
            backbone, True, self.trainable_backbone_layers
        )
        anchor_sizes = (self.anchor_sizes,) * 3
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
        model = torchvision.models.detection.faster_rcnn.FasterRCNN(
            backbone,
            self.num_classes,
            rpn_anchor_generator=torchvision.models.detection.anchor_utils.AnchorGenerator(
                anchor_sizes, aspect_ratios
            ),
            rpn_score_thresh=0.05,
            box_detections_per_img=self.box_detections_per_img,
        )
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
        assert all(label == 1 for label in output["labels"])

        # Filter out objects if their score is under score threshold
        bboxes = output["boxes"][output["scores"] > self.bbox_score_threshold]

        logger.debug(
            f"Keeping {len(bboxes)} out of {len(output['boxes'])} objects found (threshold: {self.bbox_score_threshold})"
        )

        bboxes = bboxes.cpu().numpy().astype(int).tolist()
        return bboxes


# -----------------------------------------------------------------------------
# Mothbot YOLO11m-OBB detector
#
# Single-class ("creature") detector from Digital Naturalism Laboratories'
# Mothbot_Process project. Trained at imgsz=1600, Jan 2024. Weights are hosted
# on Arbutus alongside our other models.
#
# This implementation is an independent rewrite; Mothbot's repo is unlicensed
# (see docs/superpowers/specs/2026-04-14-mothbot-detection-pipeline-design.md).
# The torch 2.6 weights_only fallback below is adapted from
# Mothbot_Process/pipeline/detect.py -- the pattern is standard ultralytics
# PyTorch 2.6 compat handling, not Mothbot-specific logic.
# -----------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class YoloDetection:
    """One detection from the YOLO-OBB post-processor.

    Fields:
        x1, y1, x2, y2: axis-aligned envelope of the rotated bounding box
            (min/max of the 4 rotated corner points).
        rotation: angle in degrees, cv2.minAreaRect convention.
        score: detection confidence, in [0, 1].
    """

    x1: float
    y1: float
    x2: float
    y2: float
    rotation: float
    score: float


def _corners_to_yolo_detection(
    corners: np.ndarray,
    score: float,
    image_shape: tuple[int, int] | None = None,
) -> YoloDetection:
    """Convert 4 rotated corner points + score into a YoloDetection.

    Args:
        corners: shape (4, 2), xy coordinates of the OBB corners.
        score: detection confidence.
        image_shape: (height, width) of the source image. When provided, the
            axis-aligned envelope is clamped to [0, width] x [0, height]. YOLO-OBB
            can emit corners outside the image when a detection touches an edge;
            negative coords in particular are dangerous because PyTorch tensor
            slicing treats negative indices as end-relative, yielding empty crops.

    Returns:
        A YoloDetection with:
          - (x1, y1, x2, y2): min/max envelope of the 4 corners (axis-aligned),
            optionally clamped to image bounds.
          - rotation: angle from cv2.minAreaRect (same convention Mothbot uses).
    """
    pts = np.asarray(corners, dtype=np.float32).reshape(-1, 2)
    x1, y1 = float(pts[:, 0].min()), float(pts[:, 1].min())
    x2, y2 = float(pts[:, 0].max()), float(pts[:, 1].max())
    if image_shape is not None:
        h, w = image_shape
        x1 = max(0.0, min(x1, float(w)))
        x2 = max(0.0, min(x2, float(w)))
        y1 = max(0.0, min(y1, float(h)))
        y2 = max(0.0, min(y2, float(h)))
    rect = cv2.minAreaRect(pts)
    angle = float(rect[2])
    return YoloDetection(x1=x1, y1=y1, x2=x2, y2=y2, rotation=angle, score=float(score))


def _load_ultralytics_yolo(weights_path):
    """Load an ultralytics YOLO model with a PyTorch 2.6 weights_only fallback.

    Newer PyTorch defaults to torch.load(..., weights_only=True), which can
    refuse to load Ultralytics checkpoints that embed custom model classes.
    For local, trusted checkpoints we retry with weights_only=False.

    Adapted from Mothbot_Process/pipeline/detect.py (unlicensed repo; pattern
    is standard ultralytics PyTorch 2.6 compat handling).
    """
    import os

    import torch as _torch
    from ultralytics import YOLO

    try:
        return YOLO(str(weights_path))
    except Exception as err:
        if "Weights only load failed" not in str(err):
            raise

        logger.info(
            "Retrying YOLO load with torch.load(weights_only=False) compatibility "
            "(trusted local checkpoint)"
        )
        original_load = _torch.load
        original_force_wo = os.environ.get("TORCH_FORCE_WEIGHTS_ONLY_LOAD")
        original_force_no_wo = os.environ.get("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD")

        def _patched_load(*args, **kwargs):
            kwargs["weights_only"] = False
            return original_load(*args, **kwargs)

        _torch.load = _patched_load
        try:
            os.environ["TORCH_FORCE_WEIGHTS_ONLY_LOAD"] = "0"
            os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
            return YOLO(str(weights_path))
        finally:
            _torch.load = original_load
            if original_force_wo is None:
                os.environ.pop("TORCH_FORCE_WEIGHTS_ONLY_LOAD", None)
            else:
                os.environ["TORCH_FORCE_WEIGHTS_ONLY_LOAD"] = original_force_wo
            if original_force_no_wo is None:
                os.environ.pop("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", None)
            else:
                os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = original_force_no_wo


class MothObjectDetector_YOLO11m_Mothbot(ObjectDetector):
    name = "Mothbot YOLO11m Creature Detector"
    weights_path = (
        "https://object-arbutus.cloud.computecanada.ca/ami-models/"
        "mothbot/detection/yolo11m_4500_imgsz1600_b1_2024-01-18.pt"
    )
    description = (
        "Single-class 'creature' detector from Digital Naturalism "
        "Laboratories' Mothbot project. YOLO11m-OBB, trained at "
        "imgsz=1600, Jan 2024."
    )
    # Overrides the base: we set the category map directly instead of
    # hosting a one-entry labels.json on the object store.
    category_map = {0: "creature"}
    imgsz = 1600
    bbox_score_threshold = 0.25
    box_detections_per_img = 500

    def get_labels(self, labels_path) -> dict[int, str]:
        # The base class __init__ calls get_labels(labels_path) and assigns the
        # return value to self.category_map, which would overwrite the class-level
        # category_map with {} when labels_path is None.  Return the class-level
        # map directly so it survives the init cycle.
        if labels_path:
            return super().get_labels(labels_path)
        return type(self).category_map

    def get_transforms(self):
        # ultralytics handles letterboxing / normalization internally; just
        # pass the PIL image through unchanged.
        return lambda pil_image: pil_image

    def get_model(self):
        logger.debug(f"Loading YOLO weights: {self.weights}")
        model = _load_ultralytics_yolo(self.weights)
        # ultralytics manages its own device placement via the device kwarg
        # passed to .predict(), so we don't .to(self.device) here.
        return model

    def get_dataloader(self):
        """PIL images can't be stacked by default_collate, so we collate as
        lists and let predict_batch hand a list of PIL images to ultralytics.
        """
        logger.info(
            f"Preparing {self.name} inference dataloader "
            f"(batch_size={self.batch_size}, single={self.single})"
        )

        def collate_as_lists(batch):
            ids = [b[0] for b in batch]
            imgs = [b[1] for b in batch]
            return ids, imgs

        dataloader_args = {
            "num_workers": 0 if self.single else self.num_workers,
            "persistent_workers": False if self.single else True,
            "shuffle": False,
            "pin_memory": False,
            "batch_size": self.batch_size,
            "collate_fn": collate_as_lists,
        }
        self.dataloader = torch.utils.data.DataLoader(self.dataset, **dataloader_args)
        return self.dataloader

    def predict_batch(self, batch):
        """Run YOLO inference. Accepts either:

        - list[PIL.Image] (from our ML-layer dataloader, which collates as lists)
        - torch.Tensor of shape (B, C, H, W) (from the antenna REST dataloader,
          which applies torchvision.transforms.ToTensor to each PIL image)
        - list[torch.Tensor] of shape (C, H, W) (REST dataloader mixed-size fallback)

        For tensor inputs we convert back to numpy HWC uint8 so ultralytics
        does its own letterboxing / normalization, matching the PIL path.
        """
        if isinstance(batch, torch.Tensor):
            # (B, C, H, W) in [0, 1] float -> list of (H, W, C) uint8 numpy
            imgs = [
                (t.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8) for t in batch
            ]
        elif isinstance(batch, list) and batch and isinstance(batch[0], torch.Tensor):
            imgs = [
                (t.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8) for t in batch
            ]
        else:
            imgs = batch
        return self.model.predict(
            imgs,
            imgsz=self.imgsz,
            conf=self.bbox_score_threshold,
            max_det=self.box_detections_per_img,
            device=self.device,
            verbose=False,
        )

    def post_process_single(self, result):
        """Flatten one ultralytics Result into a list of detection records.

        Why the OBB to axis-aligned envelope:
          YOLO11m-OBB outputs 4 rotated corner points per detection. Our
          DetectionResponse schema carries a single axis-aligned bbox, and
          the downstream InsectOrderClassifier reads an axis-aligned crop.
          We therefore take the min/max envelope of the 4 corners as the
          bbox. The rotation angle (cv2.minAreaRect convention, same as
          Mothbot) is preserved separately so a future species classifier
          can reuse Mothbot's rotated crop_rect() without re-running
          detection.

          Confidence filtering already happened inside model.predict(conf=...),
          so every record here is above bbox_score_threshold.
        """
        detections = []
        if result.obb is None:
            return detections
        corners_batch = result.obb.xyxyxyxy.cpu().numpy()  # (N, 4, 2)
        scores = result.obb.conf.cpu().numpy()  # (N,)
        # orig_shape is (height, width); present on ultralytics Result objects.
        image_shape = getattr(result, "orig_shape", None)
        for i in range(len(corners_batch)):
            det = _corners_to_yolo_detection(
                corners_batch[i], float(scores[i]), image_shape=image_shape
            )
            if det.x2 <= det.x1 or det.y2 <= det.y1:
                logger.warning(
                    f"Skipping degenerate YOLO detection (zero-area envelope "
                    f"after clamp to image {image_shape}): "
                    f"x1={det.x1:.1f} y1={det.y1:.1f} x2={det.x2:.1f} y2={det.y2:.1f}"
                )
                continue
            detections.append(det)
        return detections

    def save_results(self, item_ids, batch_output, *args, **kwargs):
        """The ML-layer base class expects a save method. The API wrapper
        overrides this, so the DB path is never hit when used via the API.
        Provide a no-op that logs, for symmetry with the FasterRCNN class.
        """
        logger.info(
            f"{self.name} ML-layer save_results called with {len(item_ids)} items "
            "(no-op; API wrapper handles persistence)"
        )
