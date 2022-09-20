import pathlib
import json
import argparse
import os
import multiprocessing
import functools
import time

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.models as torchmodels
import torchvision
import numpy as np
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import timm


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

    def _load_model(self, model_path, num_classes):
        model = timm.create_model(
            "tf_efficientnetv2_b3", weights=None, num_classes=num_classes
        )
        model = model.to(self.device)
        model.load_state_dict(
            torch.load(model_path, map_location=torch.device(self.device))
        )

        return model

    def _get_transforms(self):
        mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]

        return transforms.Compose(
            [
                transforms.Resize((self.input_size, self.input_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

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
    """
    Create cropped images from regions specified in a list of bounding boxes.

    Bounding boxes are assumed to be in the format:
    [(top-left-coordinate-pair), (bottom-right-coordinate-pair)]
    or: [x1, y1, x2, y2]

    The image is assumed to be a numpy array that can be indexed using the
    coordinate pairs.
    """

    for (x1, y1, x2, y2) in bboxes:

        object_area = (int(y2) - int(y1)) * (int(x2) - int(x1))
        # area_percent = int(object_area / total_area * 100)

        cropped_image = image[
            :,
            int(y1) : int(y2),
            int(x1) : int(x2),
        ]
        transform_to_PIL = transforms.ToPILImage()
        cropped_image = transform_to_PIL(cropped_image)
        yield cropped_image


def predict_batch(images, model_path, category_map_json, device):
    # prediction for moth / non-moth
    print(f"Predicting {len(images)} with binary classifier.")

    if len(images):
        # Loading Binary Classification Model (moth / non-moth)
        model = ModelInference(model_path, category_map_json, device)

        labels, scores = model.predict(images, confidence=True)
        print(labels, scores)
    else:
        labels = []
        scores = []

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


def classify_bboxes(localization_results, device, args, results_callback):
    annotations = {}
    label_list = (
        []
    )  # Placeholder, @TODO remove from annotations. Was object detection labels.

    for image_path, bbox_list in localization_results:

        print(f"Cropping bboxes")
        img = Image.open(image_path)
        img = transforms.ToTensor()(img)
        img_crops = list(crop_bboxes(img, bbox_list))

        print(f"Predicting {len(img_crops)} with binary classifier.")
        class_list, _ = predict_batch(
            img_crops, args.model_moth_nonmoth, args.category_map_moth_nonmoth, device
        )

        print(f"Predicting {len(img_crops)} with species classifier.")
        subclass_list, conf_list = predict_batch(
            img_crops, args.model_moth, args.category_map_moth, device
        )

        if results_callback:
            print(
                "=== CALLBACK START: Saving results from classifiers for single image == "
            )
            detected_objects_data = [
                {
                    "bbox": bbox,
                    "binary_label": binary_label,
                    "specific_label": specific_label,
                    "specific_label_score": specific_label_score,
                }
                for bbox, binary_label, specific_label, specific_label_score in zip(
                    bbox_list,
                    class_list,
                    subclass_list,
                    conf_list,
                )
            ]
            results_callback([image_path], [detected_objects_data])
            print("=== CALLBACK END == ")

        image_name = pathlib.Path(image_path).name
        annotations[image_name] = [
            bbox_list,
            label_list,
            class_list,
            subclass_list,
            conf_list,
        ]
    return annotations


def classification_transforms(input_size):
    mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]

    return transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


class LocalizationDataset(torch.utils.data.Dataset):
    def __init__(self, directory, image_names, max_image_size=1333):
        super().__init__()

        self.directory = pathlib.Path(directory)
        self.image_names = image_names
        self.max_image_size = max_image_size
        self.compare_image_sizes()
        self.transform = self.get_transforms()

    def __len__(self):
        return len(self.image_names)

    def compare_image_sizes(self):
        """
        @TODO what should we do about images with different dimensions?
        The model can handle that fine, however we can't load them in the same batch.
        batch_size = 1 works

        In this _test_ we are resizing all photos to the smallest image size, proportionally.
        Another option would be to drop anything that is different, put them in different batches, or pad them?
        Might have to make a custom pytorch Sampler class
        """
        image_sizes = {
            Image.open(self.directory / img).size for img in self.image_names
        }
        min_size = self.max_image_size
        min_dims = (self.max_image_size, self.max_image_size)
        for dims in image_sizes:
            size = np.prod(dims)
            if size < min_size:
                min_size = size
                min_dims = dims

        self.max_image_dim = min(min_dims)
        self.image_sizes = image_sizes

    def get_transforms(self):
        transform_list = [transforms.ToTensor()]
        if len(self.image_sizes) > 1:
            print(
                f"Multiple image sizes detected! {self.image_sizes}. Resizing all images to match."
            )
            transform_list.insert(0, transforms.Resize(size=[self.max_image_dim]))

        return transforms.Compose(transform_list)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = self.directory / img_name
        pil_image = Image.open(img_path)
        return str(img_path), self.transform(pil_image)


def fasterrcnn_mobilenet(model_path, device):
    print(f'Loading localization model with checkpoint "{model_path}"')
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    num_classes = 2  # 1 class (object) + background
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model


def fasterrcnn_mobilenet(model_path, device):
    print(f'Loading "fasterrcnn_mobilenet" localization model with default weights')
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(
        weights="DEFAULT"
    )
    model = model.to(device)
    model.eval()
    return model


def fastercnn(model_path, device):
    print(f'Loading "fasterrcnn" localization model with checkpoint "{model_path}"')
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    num_classes = 2  # 1 class (object) + background
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model


def load_localization_model(model_path, device):
    # @TODO move to modules or something modular
    return fasterrcnn_mobilenet(model_path, device)


def process_fasterrcnn_localization_output(img_path, output):
    LOCALIZATION_SCORE_THRESHOLD = 0.99
    bboxes = output["boxes"][output["scores"] > LOCALIZATION_SCORE_THRESHOLD]
    print(
        f"Keeping {len(bboxes)} out of {len(output['boxes'])} objects found (threshold: {LOCALIZATION_SCORE_THRESHOLD})"
    )

    bbox_list = []
    label_list = output["labels"]  # Should always be 1 for "object"
    assert all([l == 1 for l in output["labels"]])

    for box in bboxes:
        box_numpy = box.detach().cpu().numpy()
        bbox_list.append(
            [
                int(box_numpy[0]),
                int(box_numpy[1]),
                int(box_numpy[2]),
                int(box_numpy[3]),
            ]
        )
    return bboxes.cpu().numpy().tolist()


def process_fasterrcnn_mobilenet_localization_output(img_path, output):
    LOCALIZATION_SCORE_THRESHOLD = 0.01
    bboxes = output["boxes"][
        (output["scores"] > LOCALIZATION_SCORE_THRESHOLD) & (output["labels"] > 1)
    ]

    # Filter out background label, if using pretrained model only!
    bboxes = output["boxes"][output["labels"] > 1]

    print(
        f"Keeping {len(bboxes)} out of {len(output['boxes'])} objects found (threshold: {LOCALIZATION_SCORE_THRESHOLD})"
    )

    bbox_percents = []
    label_list = output["labels"]

    img_width, img_height = Image.open(img_path).size
    print("IMAGE DIMENSIONS", img_width, img_height)
    for box in bboxes:
        box_numpy = box.detach().cpu().numpy()
        bbox_percents.append(
            [
                round(box_numpy[0] / img_width, 4),
                round(box_numpy[1] / img_height, 4),
                round(box_numpy[2] / img_width, 4),
                round(box_numpy[3] / img_height, 4),
            ]
        )

    bboxes = bboxes.cpu().numpy().astype(int).tolist()
    # print(list(zip(bboxes, bbox_percents)))
    return bboxes


def process_localization_output(img_path, output):
    # @TODO move to module or something
    return process_fasterrcnn_mobilenet_localization_output(img_path, output)


def synchronize_clocks():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    else:
        pass


def localization_classification(args, results_callback=None):
    """main function for localization and classification"""

    synchronize_clocks()
    start = time.time()

    image_list = args.image_list
    base_directory = pathlib.Path(args.base_directory)
    annot_file = f"localize_classify_annotation-{base_directory.name}.json"
    annotations = {}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    # device = "cpu"
    # device = "cuda"

    model = load_localization_model(args.model_localize, device)
    max_image_size = model.transform.max_size

    input_size = 300
    print(f"Preparing dataset of {len(image_list)} images")
    dataset = LocalizationDataset(
        directory=base_directory,
        image_names=image_list,
        max_image_size=max_image_size,
    )

    # @TODO need to determine batch size and not run out of memory
    # @TODO make this configurable by user in settings
    if device == "cuda":
        batch_size = 8
        num_workers = 2
    else:
        batch_size = 8
        # num_workers = m   ultiprocessing.cpu_count()
        num_workers = 4

    print(
        f"Preparing dataloader with batch size of {batch_size} and {num_workers} workers on device: {device}"
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=True,
        pin_memory=True,
    )
    results = []

    with torch.no_grad():
        for i, (img_paths, data) in enumerate(dataloader):
            print(f"Batch {i+1} out of {len(dataloader)}")
            print(f"Looking for objects in {len(img_paths)} images")
            synchronize_clocks()
            batch_start = time.time()
            data = data.to(device, non_blocking=True)
            output = model(data)
            output = [
                process_localization_output(img_path, o)
                for img_path, o in zip(img_paths, output)
            ]
            batch_results = list(zip(img_paths, output))
            results += batch_results
            synchronize_clocks()
            batch_end = time.time()
            elapsed = batch_end - batch_start
            images_per_second = len(image_list) / elapsed
            seconds_per_image = elapsed / len(image_list)
            print(
                f"Time per batch: {round(elapsed, 1)} seconds. {round(seconds_per_image, 1)} seconds per image"
            )
            if results_callback:
                print("=== CALLBACK START: Save only bboxes of detected objects == ")
                # Format data to be saved in DB
                # Here we are just saving the bboxes of detected objects
                detected_objects_data = []
                for image_output in output:
                    detected_objects = [{"bbox": bbox} for bbox in image_output]
                    detected_objects_data.append(detected_objects)
                results_callback(img_paths, detected_objects_data)
                print("=== CALLBACK END == ")

    synchronize_clocks()
    end = time.time()
    elapsed = end - start
    images_per_second = len(image_list) / elapsed
    seconds_per_image = elapsed / len(image_list)
    print(
        f"Localization time: {round(elapsed, 1)} seconds. {round(seconds_per_image, 1)} seconds per image (with startup)"
    )

    annotations = classify_bboxes(
        results,
        device,
        args,
        results_callback=results_callback,
    )

    print("Saving annotations")
    with open(base_directory / annot_file, "w") as outfile:
        json.dump(annotations, outfile)

    return base_directory / annot_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_directory",
        help="root directory containing the trap data",
        required=True,
    )
    parser.add_argument(
        "--image_list",
        help="list of images paths relative to the base directory",
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
