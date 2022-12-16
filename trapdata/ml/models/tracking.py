from typing import Generator, Sequence, Any
from collections import namedtuple

import torch
from torch import nn
import numpy as np
import math
from PIL.Image import Image
from torchvision import transforms
import torch.utils.data

from trapdata import logger
from trapdata.db.models.queue import UntrackedObjectsQueue
from trapdata.db.models.detections import DetectedObject

# from trapdata.db.models.detections import save_untracked_detection
from .base import InferenceBaseClass


def image_diagonal(width: int, height: int) -> int:
    img_diagonal = int(math.ceil(math.sqrt(width**2 + height**2)))
    return img_diagonal


ItemForTrackingCost = namedtuple(
    "ItemForTrackingCost", "image_data bbox source_image_diagonal"
)


class TrackingCostOriginal:
    def __init__(
        self,
        image1: Image,
        image2: Image,
        bb1: tuple[int, int, int, int],
        bb2: tuple[int, int, int, int],
        source_image_diagonal: float,
        cnn_source_model,
        cost_weights: tuple[int, int, int, int] = (1, 1, 1, 1),
        cost_threshold=1,
        img_resize=224,
        device="cuda",
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
        self.device = device
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

    def _l1_normalize(self, v):
        norm = np.sum(np.array(v))
        return v / norm

    def _cosine_similarity(self):
        """Finds cosine similarity for a bounding box pair images"""

        cosine_sim = np.dot(self.img1_ftrs, self.img2_ftrs) / (
            np.linalg.norm(self.img1_ftrs) * np.linalg.norm(self.img2_ftrs)
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
        return self.queue.queue_count()

    def __iter__(
        self,
    ) -> Generator[
        tuple[
            torch.Tensor,
            tuple[
                tuple[
                    torch.Tensor, torch.Tensor, tuple[int]
                ],  # Can we make this types? help me out here!
                tuple[
                    torch.Tensor,
                    torch.Tensor,
                ],
            ],
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
                    batch_image_data,
                    batch_comparision_image_data,
                    batch_metadata,
                    batch_comparisoin_metadata,
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
    stage = 3
    type = "tracking"

    def get_dataset(self):
        dataset = UntrackedObjectsIterableDatabaseDataset(
            queue=UntrackedObjectsQueue(self.db_path),
            image_transforms=self.get_transforms(),
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

        image1_features = super().predict_batch(batch=image1_data)
        image2_features = super().predict_batch(batch=image2_data)

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
