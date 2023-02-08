from typing import Generator, Sequence, Any, Optional, Union
from collections import namedtuple

import torch
from torch import nn
import torchvision
import numpy as np
import math
import PIL.Image
from torchvision import transforms
import torch.utils.data
from sqlalchemy import orm, select, func
from rich.progress import track

from trapdata import logger
from trapdata import constants
from trapdata.common.types import BoundingBox
from trapdata.ml.utils import get_device
from trapdata.ml.models.classification import SpeciesClassifier
from trapdata.db.models.queue import UntrackedObjectsQueue
from trapdata.db.models.events import MonitoringSession
from trapdata.db.models.images import TrapImage
from trapdata.db.models.detections import DetectedObject

# from trapdata.db.models.detections import save_untracked_detection
from .base import InferenceBaseClass


def image_diagonal(width: int, height: int) -> int:
    img_diagonal = int(math.ceil(math.sqrt(width**2 + height**2)))
    return img_diagonal


ItemForTrackingCost = namedtuple(
    "ItemForTrackingCost", "image_data bbox source_image_diagonal"
)


def l1_normalize(v):
    norm = np.sum(np.array(v))
    return v / norm


def cosine_similarity(img1_ftrs: torch.Tensor, img2_ftrs: torch.Tensor) -> float:
    """
    Finds cosine similarity between a pair of cropped images.

    Uses the feature embeddings array computed from a CNN model.
    """

    cosine_sim = np.dot(img1_ftrs, img2_ftrs) / (
        np.linalg.norm(img1_ftrs) * np.linalg.norm(img2_ftrs)
    )
    assert 0 <= cosine_sim <= 1, "cosine similarity score out of bounds"

    return cosine_sim


def iou(bb1: BoundingBox, bb2: BoundingBox) -> float:
    """Finds intersection over union for a bounding box pair"""

    assert bb1[0] < bb1[2], "Issue in bounding box 1 x_annotation"
    assert bb1[1] < bb1[3], "Issue in bounding box 1 y_annotation"
    assert bb2[0] < bb2[2], "Issue in bounding box 2 x_annotation"
    assert bb2[1] < bb2[3], "Issue in bounding box 2 y_annotation"

    bb1_area = (bb1[2] - bb1[0] + 1) * (bb1[3] - bb1[1] + 1)
    bb2_area = (bb2[2] - bb2[0] + 1) * (bb2[3] - bb2[1] + 1)

    x_min = max(bb1[0], bb2[0])
    x_max = min(bb1[2], bb2[2])
    width = max(0, x_max - x_min + 1)

    y_min = max(bb1[1], bb2[1])
    y_max = min(bb1[3], bb2[3])
    height = max(0, y_max - y_min + 1)

    intersec_area = width * height
    union_area = bb1_area + bb2_area - intersec_area

    iou = np.around(intersec_area / union_area, 2)
    assert 0 <= iou <= 1, "IoU out of bounds"

    return iou


def box_ratio(bb1: BoundingBox, bb2: BoundingBox) -> float:
    """Finds the ratio of the two bounding boxes"""

    bb1_area = (bb1[2] - bb1[0] + 1) * (bb1[3] - bb1[1] + 1)
    bb2_area = (bb2[2] - bb2[0] + 1) * (bb2[3] - bb2[1] + 1)

    min_area = min(bb1_area, bb2_area)
    max_area = max(bb1_area, bb2_area)

    box_ratio = min_area / max_area
    assert 0 <= box_ratio <= 1, "box ratio out of bounds"

    return box_ratio


def distance_ratio(bb1: BoundingBox, bb2: BoundingBox, img_diag: float) -> float:
    """finds the distance between the two bounding boxes and normalizes
    by the image diagonal length
    """

    centre_x_bb1 = bb1[0] + (bb1[2] - bb1[0]) / 2
    centre_y_bb1 = bb1[1] + (bb1[3] - bb1[1]) / 2

    centre_x_bb2 = bb2[0] + (bb2[2] - bb2[0]) / 2
    centre_y_bb2 = bb2[1] + (bb2[3] - bb2[1]) / 2

    dist = math.sqrt(
        (centre_x_bb2 - centre_x_bb1) ** 2 + (centre_y_bb2 - centre_y_bb1) ** 2
    )
    max_dist = img_diag

    assert dist <= max_dist, "distance between bounding boxes more than max distance"

    return dist / max_dist


def total_cost(
    img1_features: torch.Tensor,
    img2_features: torch.Tensor,
    bb1: BoundingBox,
    bb2: BoundingBox,
    image_diagonal: float,
    w_cnn: float = 1,
    w_iou: float = 1,
    w_box: float = 1,
    w_dis: float = 1,
) -> float:
    """returns the final cost"""

    cnn_cost = 1 - cosine_similarity(img1_features, img2_features)
    iou_cost = 1 - iou(bb1, bb2)
    box_ratio_cost = 1 - box_ratio(bb1, bb2)
    dist_ratio_cost = distance_ratio(bb1, bb2, image_diagonal)

    total_cost = (
        w_cnn * cnn_cost
        + w_iou * iou_cost
        + w_box * box_ratio_cost
        + w_dis * dist_ratio_cost
    )

    return total_cost


class TrackingCostOriginal:
    def __init__(
        self,
        image1: PIL.Image.Image,
        image2: PIL.Image.Image,
        bb1: tuple[int, int, int, int],
        bb2: tuple[int, int, int, int],
        source_image_diagonal: float,
        cnn_source_model,
        cost_weights: tuple[int, int, int, int] = (1, 1, 1, 1),
        cost_threshold=1,
        img_resize=224,
        device=None,
    ):
        """
        Finds tracking cost for a pair of bounding box using cnn features, distance, iou and box ratio
        Author        : Aditya Jain
        Date created  : June 23, 2022

        Args:
        image1       : first moth image
        image2       : second moth image
        bb1          : [x1, y1, x2, y2] The origin is top-left corner; x1<x2; y1<y2; integer values in the list
        bb2          : [x1, y1, x2, y2] The origin is top-left corner; x1<x2; y1<y2; integer values in the list
        weights      : weights assigned to various cost metrics
        model        : trained moth model
        img_diagonal : diagonal length of the image in pixels

        """

        self.image1 = image1
        self.image2 = image2
        self.img_resize = img_resize
        self.device = device or get_device()
        self.total_cost = 0
        self.bb1 = bb1
        self.bb2 = bb2
        self.img_diag = source_image_diagonal
        self.w_cnn = cost_weights[0]
        self.w_iou = cost_weights[1]
        self.w_box = cost_weights[2]
        self.w_dis = cost_weights[3]
        self.model = self._load_model(cnn_source_model)

    def _load_model(self, cnn_source_model):
        # Get the last feature layer of the model
        model = nn.Sequential(*list(cnn_source_model.children())[:-3])

        return model

    def _transform_image(self, image):
        """Transforms the cropped moth images for model prediction"""

        transformer = transforms.Compose(
            [
                transforms.Resize((self.img_resize, self.img_resize)),
                transforms.ToTensor(),
            ]
        )
        image = transformer(image)

        # RGBA image; extra alpha channel
        if image.shape[0] > 3:
            image = image[0:3, :, :]

        # grayscale image; converted to 3 channels r=g=b
        if image.shape[0] == 1:
            to_pil = transforms.ToPILImage()
            to_rgb = transforms.Grayscale(num_output_channels=3)
            to_tensor = transforms.ToTensor()
            image = to_tensor(to_rgb(to_pil(image)))

        return image

    def _l1_normalize(self, v):
        norm = np.sum(np.array(v))
        return v / norm

    def _cosine_similarity(self):
        """Finds cosine similarity for a bounding box pair images"""

        img2_moth = self._transform_image(self.image2)
        img2_moth = torch.unsqueeze(img2_moth, 0).to(self.device)

        img1_moth = self._transform_image(self.image1)
        img1_moth = torch.unsqueeze(img1_moth, 0).to(self.device)

        # getting model features for each image
        with torch.no_grad():
            img2_ftrs = self.model(img2_moth)
            img2_ftrs = img2_ftrs.view(-1, img2_ftrs.size(0)).cpu()
            img2_ftrs = img2_ftrs.reshape((img2_ftrs.shape[0],))
            img2_ftrs = self._l1_normalize(img2_ftrs)

            img1_ftrs = self.model(img1_moth)
            img1_ftrs = img1_ftrs.view(-1, img1_ftrs.size(0)).cpu()
            img1_ftrs = img1_ftrs.reshape((img1_ftrs.shape[0],))
            img1_ftrs = self._l1_normalize(img1_ftrs)

        cosine_sim = np.dot(img1_ftrs, img2_ftrs) / (
            np.linalg.norm(img1_ftrs) * np.linalg.norm(img2_ftrs)
        )
        assert 0 <= cosine_sim <= 1, "cosine similarity score out of bounds"

        return cosine_sim

    def _iou(self):
        """Finds intersection over union for a bounding box pair"""

        assert self.bb1[0] < self.bb1[2], "Issue in bounding box 1 x_annotation"
        assert self.bb1[1] < self.bb1[3], "Issue in bounding box 1 y_annotation"
        assert self.bb2[0] < self.bb2[2], "Issue in bounding box 2 x_annotation"
        assert self.bb2[1] < self.bb2[3], "Issue in bounding box 2 y_annotation"

        bb1_area = (self.bb1[2] - self.bb1[0] + 1) * (self.bb1[3] - self.bb1[1] + 1)
        bb2_area = (self.bb2[2] - self.bb2[0] + 1) * (self.bb2[3] - self.bb2[1] + 1)

        x_min = max(self.bb1[0], self.bb2[0])
        x_max = min(self.bb1[2], self.bb2[2])
        width = max(0, x_max - x_min + 1)

        y_min = max(self.bb1[1], self.bb2[1])
        y_max = min(self.bb1[3], self.bb2[3])
        height = max(0, y_max - y_min + 1)

        intersec_area = width * height
        union_area = bb1_area + bb2_area - intersec_area

        iou = np.around(intersec_area / union_area, 2)
        assert 0 <= iou <= 1, "IoU out of bounds"

        return iou

    def _box_ratio(self):
        """Finds the ratio of the two bounding boxes"""

        bb1_area = (self.bb1[2] - self.bb1[0] + 1) * (self.bb1[3] - self.bb1[1] + 1)
        bb2_area = (self.bb2[2] - self.bb2[0] + 1) * (self.bb2[3] - self.bb2[1] + 1)

        min_area = min(bb1_area, bb2_area)
        max_area = max(bb1_area, bb2_area)

        box_ratio = min_area / max_area
        assert 0 <= box_ratio <= 1, "box ratio out of bounds"

        return box_ratio

    def _distance_ratio(self):
        """finds the distance between the two bounding boxes and normalizes
        by the image diagonal length
        """

        centre_x_bb1 = self.bb1[0] + (self.bb1[2] - self.bb1[0]) / 2
        centre_y_bb1 = self.bb1[1] + (self.bb1[3] - self.bb1[1]) / 2

        centre_x_bb2 = self.bb2[0] + (self.bb2[2] - self.bb2[0]) / 2
        centre_y_bb2 = self.bb2[1] + (self.bb2[3] - self.bb2[1]) / 2

        dist = math.sqrt(
            (centre_x_bb2 - centre_x_bb1) ** 2 + (centre_y_bb2 - centre_y_bb1) ** 2
        )
        max_dist = self.img_diag

        assert (
            dist <= max_dist
        ), "distance between bounding boxes more than max distance"

        return dist / max_dist

    def final_cost(self):
        """returns the final cost"""

        cnn_cost = 1 - self._cosine_similarity()
        iou_cost = 1 - self._iou()
        box_ratio_cost = 1 - self._box_ratio()
        dist_ratio_cost = self._distance_ratio()

        self.total_cost = (
            self.w_cnn * cnn_cost
            + self.w_iou * iou_cost
            + self.w_box * box_ratio_cost
            + self.w_dis * dist_ratio_cost
        )

        return self.total_cost


class TrackingCost:
    def __init__(
        self,
        image1_cnn_features: torch.Tensor,
        image2_cnn_features: torch.Tensor,
        bb1: tuple[int, int, int, int],
        bb2: tuple[int, int, int, int],
        source_image_diagonal: float,
        cost_weights: tuple[int, int, int, int] = (1, 1, 1, 1),
        cost_threshold=1,
    ):
        """
        Finds tracking cost for a pair of bounding box using cnn features, distance, iou and box ratio
        Author        : Aditya Jain
        Date created  : June 23, 2022

        Args:
        image1_cnn_features       : CNN features for first moth image
        image2_cnn_features       : CNN features for second moth image
        bb1          : [x1, y1, x2, y2] The origin is top-left corner; x1<x2; y1<y2; integer values in the list
        bb2          : [x1, y1, x2, y2] The origin is top-left corner; x1<x2; y1<y2; integer values in the list
        weights      : weights assigned to various cost metrics
        img_diagonal : diagonal length of the image in pixels

        """

        self.img1_ftrs = image1_cnn_features
        self.img2_ftrs = image2_cnn_features
        self.total_cost = 0
        self.bb1 = bb1
        self.bb2 = bb2
        self.img_diag = source_image_diagonal
        self.w_cnn = cost_weights[0]
        self.w_iou = cost_weights[1]
        self.w_box = cost_weights[2]
        self.w_dis = cost_weights[3]


class UntrackedObjectsIterableDatabaseDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        queue: UntrackedObjectsQueue,
        image_transforms: transforms.Compose,
        batch_size: int = 4,
    ):
        super().__init__()
        self.queue = queue
        self.image_transforms = image_transforms
        self.batch_size = batch_size

    def __len__(self):
        count = self.queue.queue_count()
        print("TRACKING QUEUE COUNT:", count)
        return count

    def __iter__(
        self,
    ) -> Generator[
        tuple[
            torch.Tensor,
            # tuple[
            #     tuple[
            #         torch.Tensor, torch.Tensor, tuple[int]
            #     ],  # Can we make this types? help me out here!
            #     tuple[
            #         torch.Tensor,
            #         torch.Tensor,
            #     ],
            # ],
        ],
        None,
        None,
    ]:
        while len(self):
            worker_info = torch.utils.data.get_worker_info()
            logger.info(f"Using worker: {worker_info}")

            # This should probably be one item, and then all of the objects from the previous frame
            records = self.queue.pull_n_from_queue(self.batch_size)

            # Prepare data for TrackingCost calculator exactly, return in tensor

            if records:
                item_ids = torch.utils.data.default_collate(
                    [record.id for record, _ in records]
                )

                image_pairs = []
                for record, comparisons in records:
                    for comparison in comparisons:
                        image_pairs.append(
                            (
                                self.data_for_tracking(record),
                                self.data_for_tracking(comparison),
                            )
                        )

                yield (
                    item_ids,
                    # batch_image_data,
                    # batch_comparison_image_data,
                    # batch_metadata,
                    # batch_comparison_metadata,
                )

    def transform(self, cropped_image) -> torch.Tensor:
        return self.image_transforms(cropped_image)

    def data_for_tracking(
        self, record: DetectedObject
    ) -> tuple[torch.Tensor, tuple, int]:
        image_data = self.transform(record.cropped_image_data())
        bbox = tuple(record.bbox)
        diagonal = image_diagonal(record.source_image_width, record.source_image_height)
        return image_data, bbox, diagonal

    def collate_pairs():
        pass


class TrackingClassifier(InferenceBaseClass):
    name = "Default Tracking Method"
    stage = 4
    type = "tracking"
    cnn_features_model: torch.nn.Module
    cnn_features_model_transforms: torchvision.transforms.Compose
    cnn_features_model_input_size: int

    def get_model(self):
        # Get the last feature layer of the model
        model = nn.Sequential(*list(self.cnn_features_model.children())[:-3])

        return model

    def get_dataset(self):
        dataset = UntrackedObjectsIterableDatabaseDataset(
            queue=UntrackedObjectsQueue(self.db_path),
            image_transforms=self.cnn_features_model_transforms,
            batch_size=self.batch_size,
        )
        return dataset

    def predict_batch(
        self,
        batch_a: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        batch_b: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ):
        image1_data, image1_bboxes, image1_diagonals = batch_a
        image2_data, image2_bboxes, image2_diagonals = batch_b

        print("SUPER", super())
        image1_features = self.cnn_features_model.predict_batch(batch=image1_data)
        image2_features = self.cnn_features_model.predict_batch(batch=image2_data)

        batch_costs = []
        for img1_ftrs, img2_ftrs, bbox1, bbox2, diagonal in zip(
            image1_features,
            image2_features,
            image1_bboxes,
            image2_bboxes,
            image1_diagonals,
        ):
            cost = TrackingCost(
                image1_cnn_features=img1_ftrs,
                image2_cnn_features=img2_ftrs,
                bb1=bbox1,
                bb2=bbox2,
                source_image_diagonal=diagonal,
            )
            batch_costs.append(cost.final_cost())

            # Calculate the CNN features just once!
            # Get lowest cost for each detection
            # Save that cost, and the comparison it came from

        return batch_output

        TrackingCost()
        return super().predict_batch(batch)

    def save_results(self, object_ids, batch_output):
        # Save sequence_id and frame number
        pass


def new_sequence(
    obj_current: DetectedObject,
    obj_previous: DetectedObject,
    session: Optional[orm.Session] = None,
):
    """
    Create a new sequence ID and assign it to the current & previous detections.
    """
    # obj_current.sequence_id = uuid.uuid4() # @TODO ensure this is unique, or
    sequence_id = f"{obj_previous.monitoring_session.day.strftime('%Y%m%d')}-SEQ-{obj_previous.id}"
    obj_previous.sequence_id = sequence_id
    obj_previous.sequence_frame = 0

    obj_current.sequence_id = sequence_id
    obj_current.sequence_frame = 1

    logger.info(
        f"Created new sequence beginning with obj {obj_previous.id}: {sequence_id}"
    )

    if session:
        session.add(obj_current)
        session.add(obj_previous)
        session.flush()

    return sequence_id


def assign_sequence(
    obj_current: DetectedObject,
    obj_previous: DetectedObject,
    final_cost: float,
    session: orm.Session,
):
    """
    Assign a pair of objects to the same sequence.

    Will create a new sequence if necessary. Saves their similarity and order to the database.
    """
    obj_current.sequence_previous_cost = final_cost
    obj_current.sequence_previous_id = obj_previous.id
    if obj_previous.sequence_id:
        obj_current.sequence_id = obj_previous.sequence_id
        obj_current.sequence_frame = obj_previous.sequence_frame + 1
    else:
        new_sequence(obj_current, obj_previous)
    session.add(obj_current)
    session.add(obj_previous)
    session.flush()
    session.commit()
    return obj_current.sequence_id, obj_current.sequence_frame


def compare_objects(
    image_current: TrapImage,
    cnn_model: torch.nn.Module,
    session: orm.Session,
    image_previous: Optional[TrapImage] = None,
    skip_existing: bool = True,
    device: Union[torch.device, str, None] = None,
):
    """
    Calculate the similarity (tracking cost) between all objects detected in a pair of images.

    Will assign objects to a sequence if the similarity exceeds the TRACKING_COST_THRESHOLD.
    """
    if not image_previous:
        image_previous = image_current.previous_image(session)
        assert image_previous, f"No image found before image {image_current.id}"

    logger.debug(
        f"Calculating tracking costs in image {image_current.id} vs. {image_previous.id}"
    )
    objects_current = (
        session.execute(
            select(DetectedObject)
            .filter(DetectedObject.image == image_current)
            .where(DetectedObject.binary_label == constants.POSITIVE_BINARY_LABEL)
        )
        .unique()
        .scalars()
        .all()
    )

    objects_previous = (
        session.execute(
            select(DetectedObject)
            .filter(DetectedObject.image == image_previous)
            .where(DetectedObject.binary_label == constants.POSITIVE_BINARY_LABEL)
        )
        .unique()
        .scalars()
        .all()
    )

    img_shape = PIL.Image.open(image_current.absolute_path).size

    for obj_current in objects_current:
        if skip_existing and obj_current.sequence_id:
            logger.debug(
                f"Skipping obj {obj_current.id}, already assigned to sequence {obj_current.sequence_id} as frame {obj_current.sequence_frame}"
            )
            continue

        logger.debug(f"Comparing obj {obj_current.id} to all objects in previous frame")
        costs = []
        assert cnn_model is not None
        for obj_previous in objects_previous:
            cost = TrackingCostOriginal(
                obj_current.cropped_image_data(),
                obj_previous.cropped_image_data(),
                tuple(obj_current.bbox),
                tuple(obj_previous.bbox),
                source_image_diagonal=image_diagonal(img_shape[0], img_shape[1]),
                cnn_source_model=cnn_model,
                device=device,
            )
            final_cost = cost.final_cost()
            logger.debug(
                f"\tScore for obj {obj_current.id} vs. {obj_previous.id}: {final_cost}"
            )
            costs.append((final_cost, obj_previous))

        costs.sort(key=lambda cost: cost[0])
        if not costs:
            continue
        lowest_cost, best_match = costs[0]

        if lowest_cost <= constants.TRACKING_COST_THRESHOLD:
            sequence_id, frame_num = assign_sequence(
                obj_current=obj_current,
                obj_previous=best_match,
                final_cost=lowest_cost,
                session=session,
            )
            logger.info(
                f"Assigned {obj_current.id} to sequence {sequence_id} as frame #{frame_num}. Tracking cost: {lowest_cost}"
            )


def find_all_tracks(
    monitoring_session: MonitoringSession,
    cnn_model: torch.nn.Module,
    session: orm.Session,
    device: Union[torch.device, str, None] = None,
):
    """
    Retrieve all images for an Event / Monitoring Session and find all sequential objects.
    """
    images = (
        session.execute(
            select(TrapImage)
            .filter(TrapImage.monitoring_session == monitoring_session)
            .order_by(TrapImage.timestamp)
        )
        .unique()
        .scalars()
        .all()
    )
    for i, image in track(
        enumerate(images),
        description=f"Processing {monitoring_session.num_images} images with {monitoring_session.num_detected_objects} objects from event {monitoring_session.day}",
    ):
        n_current = i
        n_previous = max(n_current - 1, 0)
        image_current = images[n_current]
        image_previous = images[n_previous]
        if image_current != image_previous:
            compare_objects(
                image_current=image_current,
                image_previous=image_previous,
                cnn_model=cnn_model,
                session=session,
                device=device,
            )


def summarize_tracks(session: orm.Session, event: Optional[MonitoringSession] = None):
    query_args = {}
    if event:
        query_args = {"monitoring_session": event}

    tracks = session.execute(
        select(
            DetectedObject.monitoring_session_id,
            DetectedObject.sequence_id,
            func.count(DetectedObject.id),
        )
        .group_by(DetectedObject.monitoring_session_id, DetectedObject.sequence_id)
        .filter_by(**query_args)
    ).all()

    sequences = {}
    for ms, sequence_id, count in tracks:
        track_objects = (
            session.execute(
                select(DetectedObject)
                .where(DetectedObject.sequence_id == sequence_id)
                .order_by(DetectedObject.sequence_frame)
            )
            .unique()
            .scalars()
            .all()
        )
        sequences[sequence_id] = [
            dict(
                event=obj.monitoring_session.day,
                sequence=sequence_id,
                frame=obj.sequence_frame,
                image=obj.image_id,
                id=obj.id,
                path=obj.path,
                specific_label=obj.specific_label,
                specific_label_score=obj.specific_label_score,
                cost=obj.sequence_previous_cost,
            )
            for obj in track_objects
        ]

    return sequences
