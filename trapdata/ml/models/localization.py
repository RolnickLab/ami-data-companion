import pathlib

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


class AnyBugObjectDetector_YOLO26(ObjectDetector):
    """Ultralytics YOLO26 "any-bug" object detector.

    Emits bounding boxes as absolute raw-pixel ``xyxy`` in the ORIGINAL image
    space, matching Antenna's raw-pixel contract. Two coordinate-space hazards
    are handled explicitly:

    1. Letterboxing. Ultralytics letterboxes each image for inference and maps
       its predicted boxes back to the input array's pixel space, so the
       ``xyxy`` values read from the result are already de-letterboxed to
       original pixels.
    2. EXIF auto-transpose. Ultralytics applies ``ImageOps.exif_transpose`` to
       PIL inputs, which would rotate portrait or otherwise EXIF-tagged captures
       and misplace boxes relative to the raw-pixel contract. :meth:`predict_batch`
       always hands the model raw NumPy arrays (never PIL images), which carry no
       EXIF, so no auto-rotation is applied and boxes stay in raw-pixel space.

    Two dataloaders feed this detector with different single-image formats, and
    :meth:`predict_batch` accepts both (see :meth:`_as_hwc_uint8_rgb`): the
    FastAPI ``/process`` path uses this detector's own :meth:`get_transforms`
    (HWC uint8 RGB), while the async worker's shared dataloader hardcodes
    ``ToTensor`` (CHW float32 RGB in ``[0, 1]``) because its classifier stages
    slice CHW crops from the same tensors.

    The (AGPL-3.0) ``ultralytics`` package is a required dependency, but it is
    imported inside :meth:`get_model` rather than at module scope so the other
    detectors here do not pay its import cost.
    """

    name = "AnyBug YOLO26x Detector 2024"
    key = "anybug-yolo26x-detector-2024"
    weights_path = (
        "https://object-arbutus.cloud.computecanada.ca/ami-models/moths/"
        "localization/yolo26x-anybug-v1.pt"
    )
    description = (
        "Ultralytics YOLO26x 'any-bug' object detector: the larger, server-side "
        "flavor of the combined flat-bug + Fieldguide detector, chosen for batch "
        "GPU accuracy over edge latency. Outputs raw-pixel xyxy boxes in the "
        "original image space."
    )
    # TODO(anybug): 0.25 is a starting default; tune against validation data.
    bbox_score_threshold = 0.25

    # Pin inference resolution. The checkpoint trains at 1024 and Ultralytics
    # already restores that via _reset_ckpt_args, so this is a no-op today; it
    # guards against a future release dropping that restore and silently falling
    # back to the 640 predict default, which would cut recall on small insects.
    imgsz = 1024

    def get_transforms(self):
        # Convert the PIL image to a raw HWC RGB uint8 NumPy array. NumPy arrays
        # carry no EXIF metadata, which neutralizes Ultralytics' default
        # ImageOps.exif_transpose so predicted boxes land in raw-pixel space.
        return torchvision.transforms.Compose([np.asarray])

    def get_model(self):
        # Imported here rather than at module scope so that importing this module
        # does not pull in ultralytics and its own heavy dependency chain for the
        # detectors that never touch it.
        from ultralytics import YOLO

        logger.debug(f"Loading YOLO26 weights: {self.weights}")
        model = YOLO(self.weights)
        model.to(self.device)
        self.model = model
        return self.model

    # A float image is accepted as [0, 1] within this tolerance; anything beyond
    # it (e.g. a mean/std standardized tensor) is rejected rather than clipped.
    _FLOAT_RANGE_EPS = 1e-3

    @staticmethod
    def _as_hwc_uint8_rgb(image: "np.ndarray | torch.Tensor") -> np.ndarray:
        """Convert one image (or an NCHW/NHWC batch) to HWC uint8 RGB for Ultralytics.

        The two dataloaders feed this detector mutually exclusive formats, and the
        conversion is an ASSERTED CONTRACT rather than a dtype guess: a future
        change to a dataloader transform must fail loudly here instead of silently
        handing the model corrupted pixels.

        * FastAPI ``/process`` applies this detector's own :meth:`get_transforms`
          (``np.asarray``), yielding channels-last uint8 RGB (HWC or NHWC).
        * The async worker's shared dataloader hardcodes ``torchvision``
          ``ToTensor`` — it must, so the classifier stages can slice CHW crops
          from the same tensors — yielding channels-first float32 RGB scaled to
          ``[0, 1]`` (CHW or NCHW). Reaching Ultralytics as channels-first float,
          the RGB->BGR flip in :meth:`predict_batch` would mirror the width axis
          instead of swapping channels, feeding the model garbage.

        The contract, by dtype:

        * Floating point must be channels-first (``shape[-3] in {1, 3}``) and
          within ``[0, 1]``; it is transposed to channels-last and rescaled to
          0-255 with rounding (``np.rint``), not truncation.
        * Integer must be channels-last (``shape[-1] in {1, 3}``) and is returned
          as uint8 unchanged, so the ``/process`` path stays byte-for-byte
          identical.
        * Anything else — an out-of-range float (a standardized tensor), an
          already channels-last float, an integer channels-first array, or an
          unexpected rank — raises :class:`ValueError` naming the offending shape,
          dtype, and range.
        """
        if isinstance(image, torch.Tensor):
            # One full-image copy per call: the device->host transfer here plus
            # the rescale allocation below. Negligible for a single frame, but do
            # not fan many large images through this blindly — the cost scales
            # with the total pixels handed in per call.
            image = image.detach().cpu().numpy()
        image = np.asarray(image)

        if image.ndim not in (3, 4):
            raise ValueError(
                "Expected a 3D single image or 4D batch (CHW/HWC/NCHW/NHWC), got "
                f"shape {image.shape} dtype {image.dtype}"
            )

        if np.issubdtype(image.dtype, np.floating):
            # ToTensor contract: channels-first float in [0, 1]. The channel axis
            # is the third-from-last on both CHW and NCHW, so shape[-3] validates
            # either rank.
            low, high = float(np.min(image)), float(np.max(image))
            eps = AnyBugObjectDetector_YOLO26._FLOAT_RANGE_EPS
            if low < -eps or high > 1.0 + eps:
                raise ValueError(
                    "Floating-point image must be normalized to [0, 1] (ToTensor "
                    f"output); got value range [{low:.4g}, {high:.4g}] for shape "
                    f"{image.shape}. A standardized (mean/std) tensor is rejected "
                    "rather than silently clipped."
                )
            if image.shape[-3] not in (1, 3):
                raise ValueError(
                    "Floating-point image must be channels-first (CHW or NCHW) "
                    f"with 1 or 3 channels; got shape {image.shape}. An already "
                    "channels-last float array is rejected rather than transposed "
                    "into garbage."
                )
            axes = (0, 2, 3, 1) if image.ndim == 4 else (1, 2, 0)
            image = np.transpose(image, axes)
            return np.clip(np.rint(image * 255.0), 0, 255).astype(np.uint8)

        # Integer input: require channels-last so an integer channels-first array
        # cannot slip through in the wrong layout.
        if image.shape[-1] not in (1, 3):
            raise ValueError(
                "Integer image must be channels-last (HWC or NHWC) with 1 or 3 "
                f"channels; got shape {image.shape}. An integer channels-first "
                "array is rejected rather than passed through in the wrong layout."
            )
        return image.astype(np.uint8, copy=False)

    def predict_batch(self, batch):
        # Ultralytics performs its own letterbox + normalization internally and
        # returns one Results object per image with boxes already mapped back to
        # that image's pixel space. It expects NumPy inputs as HWC uint8 arrays
        # in BGR channel order (OpenCV convention).
        #
        # Accept either format the two dataloaders produce and normalize each
        # image to HWC uint8 RGB via the asserted _as_hwc_uint8_rgb contract, then
        # flip RGB->BGR. A 4D batch is split into single images so Ultralytics
        # receives the list of images it expects.
        if isinstance(batch, (np.ndarray, torch.Tensor)):
            raw_images = list(batch) if batch.ndim == 4 else [batch]
        else:
            raw_images = list(batch)
        images = [self._as_hwc_uint8_rgb(img) for img in raw_images]
        images = [np.ascontiguousarray(img[..., ::-1]) for img in images]
        # TODO(anybug): a mixed-resolution batch cannot be default-collated into
        # a single tensor, so such batches arrive as a list and are passed through
        # one by one; wire a list collate_fn for variable-size inputs.
        return self.model.predict(
            images,
            conf=self.bbox_score_threshold,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )

    def post_process_single(self, result):
        # result.boxes.xyxy are absolute pixel coordinates in the original image
        # space (already de-letterboxed by Ultralytics).
        bboxes = result.boxes.xyxy.cpu().numpy().astype(int).tolist()
        logger.debug(f"YOLO26 detector kept {len(bboxes)} boxes")
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
