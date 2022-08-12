#!/usr/bin/env python
# coding: utf-8


"""
Author       : Aditya Jain
Date Started : July 15, 2022
About        : This file does DL-based localization and classification on raw images and saves annotation information
"""

import torch
import torchvision.models as torchmodels
import torchvision
import os
import numpy as np
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import json
import timm
import argparse


# Model Inference Class Definition
class ModelInference:
    def __init__(self, model_path, category_map_json, device, input_size=300):
        self.device = device
        self.input_size = input_size
        self.id2categ = self._load_category_map(category_map_json)
        self.transforms = self._get_transforms()
        self.model = self._load_model(model_path, num_classes=len(self.id2categ))
        self.model.eval()

    def _load_category_map(self, category_map_json):
        with open(category_map_json, "r") as f:
            categories_map = json.load(f)

        id2categ = {categories_map[categ]: categ for categ in categories_map}

        return id2categ

    def _get_transforms(self):
        mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]

        return transforms.Compose(
            [
                transforms.Resize((self.input_size, self.input_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

    def _load_model(self, model_path, num_classes):
        model = timm.create_model(
            "tf_efficientnetv2_b3", pretrained=False, num_classes=num_classes
        )
        model = model.to(self.device)
        model.load_state_dict(
            torch.load(model_path, map_location=torch.device(self.device))
        )

        return model

    def predict(self, images, confidence=False):
        with torch.no_grad():
            # @TODO can these be done in a single batch?
            images = [self.transforms(img) for img in images]
            images = [img.to(self.device) for img in images]
            images = [img.unsqueeze_(0) for img in images]
            images = torch.cat(images, 0)

            predictions = self.model(images)
            predictions = torch.nn.functional.softmax(predictions, dim=1)
            predictions = predictions.cpu().numpy()

            categs = predictions.argmax(axis=1)
            categs = [self.id2categ[cat] for cat in categs]

            if confidence:
                return categs, predictions.max(axis=1).astype(float)
            else:
                return categs


def crop_bboxes(image, bboxes):
    cropped_images = []
    bbox_list = []

    for box_numpy in bboxes:
        # label_list.append(1)
        # insect_area = (int(box_numpy[2]) - int(box_numpy[0])) * (
        #     int(box_numpy[3]) - int(box_numpy[1])
        # )
        # area_per = int(insect_area / total_area * 100)

        cropped_image = image[
            :,
            int(box_numpy[1]) : int(box_numpy[3]),
            int(box_numpy[0]) : int(box_numpy[2]),
        ]
        transform_to_PIL = transforms.ToPILImage()
        cropped_image = transform_to_PIL(cropped_image)
        cropped_images.append(cropped_image)

    return cropped_images


def predict_batch(images, model_path, category_map_json, device):
    # prediction for moth / non-moth
    print(f"Predicting {len(images)} with binary classifier.")

    # Loading Binary Classification Model (moth / non-moth)
    model = ModelInference(model_path, category_map_json, device)

    labels, scores = model.predict(images, confidence=True)
    print(labels, scores)
    # labels, scores = [], []
    # for img in images:
    #     categ, conf = model.predict(img, confidence=True)
    #     print(categ, conf)
    #     labels.append(categ)
    #     scores.append(float(conf))
    return list(labels), list(scores)


def predict_batch_specific(images, model_path, category_map_json, device):
    # prediction of specific moth species
    print(f"Predicting {len(images)} with species classifer.")

    # Loading Moth Classification Model
    model_moth = ModelInference(model_path, category_map_json, device)

    labels, scores = [], []
    for img in images:
        categ, conf = model_moth.predict(img, confidence=True)
        print(categ, conf)
        labels.append(categ)
        scores.append(float(conf))
    return labels, scores


def prep_image(img_path):
    print("Processing image", img_path)
    transform = transforms.Compose([transforms.ToTensor()])
    raw_image = Image.open(img_path)
    image_size = np.shape(raw_image)
    total_area = image_size[0] * image_size[1]
    image = transform(raw_image)
    return image


def localization_classification(args):
    """main function for localization and classification"""

    data_dir = args.data_dir
    image_folder = args.image_folder
    data_path = data_dir + image_folder + "/"
    save_path = data_dir
    annot_file = "localize_classify_annotation-" + image_folder + ".json"

    # Get cpu or gpu device for training.
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"

    # Loading Localization Model
    model_localize = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        pretrained=False
    )
    num_classes = 2  # 1 class (person) + background
    in_features = model_localize.roi_heads.box_predictor.cls_score.in_features
    model_localize.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model_path = args.model_localize
    checkpoint = torch.load(model_path, map_location=device)
    model_localize.load_state_dict(checkpoint["model_state_dict"])
    model_localize = model_localize.to(device)
    model_localize.eval()

    # Prediction on data
    annot_data = {}
    SCORE_THR = 0.99
    image_list = [
        p
        for p in os.listdir(data_path)
        if p.lower().endswith(".jpg") and not p.startswith(".")
    ]

    images = [prep_image(data_path + img) for img in image_list]
    image_batch = [torch.unsqueeze(img, 0).to(device) for img in images]
    # @TODO need to determine batch size and not run out of memory
    image_batch = torch.cat(image_batch, 0)

    print("Finding objects in image batch")
    results = model_localize(image_batch)
    print(results)

    for img_path, img, output in zip(image_list, images, results):
        bboxes = output["boxes"][output["scores"] > SCORE_THR]
        print(f"Found {len(bboxes)} objects in", img)

        bbox_list = []
        label_list = []
        binary_predictions = []  # moth / non-moth
        specific_predictions = []  # moth species / non-moth
        conf_list = []  # confidence list
        # area_list = []  # percentage area of the sheet

        for box in bboxes:
            label_list.append(1)  # Not used?
            box_numpy = box.detach().cpu().numpy()
            bbox_list.append(
                [
                    int(box_numpy[0]),
                    int(box_numpy[1]),
                    int(box_numpy[2]),
                    int(box_numpy[3]),
                ]
            )

        img_crops = crop_bboxes(img, bbox_list)
        print(f"Predicting {len(img_crops)} with binary classifier.")
        class_list, _ = predict_batch(
            img_crops, args.model_moth_nonmoth, args.category_map_moth_nonmoth, device
        )
        print(f"Predicting {len(img_crops)} with species classifier.")
        subclass_list, conf_list = predict_batch(
            img_crops, args.model_moth, args.category_map_moth, device
        )

        annot_data[img_path] = [
            bbox_list,
            label_list,
            class_list,
            subclass_list,
            conf_list,
        ]

    print("Saving annotations")
    with open(save_path + annot_file, "w") as outfile:
        json.dump(annot_data, outfile)

    return save_path + annot_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", help="root directory containing the trap data", required=True
    )
    parser.add_argument(
        "--image_folder",
        help="date folder within root directory containing the images",
        required=True,
    )
    parser.add_argument(
        "--model_localize", help="path to the localization model", required=True
    )
    parser.add_argument(
        "--model_moth",
        help="path to the fine-grained moth classification model",
        required=True,
    )
    parser.add_argument(
        "--model_moth_nonmoth", help="path to the moth-nonmoth model", required=True
    )
    parser.add_argument(
        "--category_map_moth",
        help="path to the moth category map for converting integer labels to name labels",
        required=True,
    )
    parser.add_argument(
        "--category_map_moth_nonmoth",
        help="path to the moth-nonmoth category map for converting integer labels to name labels",
        required=True,
    )
    args = parser.parse_args()
    localization_classification(args)
