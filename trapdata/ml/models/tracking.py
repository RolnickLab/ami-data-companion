""""
Author        : Aditya Jain
Date created  : June 23, 2022
About         : Finds tracking cost for a pair of bounding box using cnn features, distance, iou and box ratio
"""

import torch
from torch import nn
import numpy as np
import math
from PIL.Image import Image
from torchvision import transforms


def image_diagonal(width: float, height: float):
    img_diagonal = math.sqrt(width**2 + height**2)
    return img_diagonal


class TrackingCost:
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
        Finds tracking using multiple factors
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
