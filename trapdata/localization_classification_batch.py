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


LOCALIZATION_SCORE_THRESHOLD = 0.99

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


def classify_bboxes(localization_results, device, args):
    annotations = {}
    label_list = (
        []
    )  # Placeholder, @TODO remove from annotations. Was object detection labels.

    for image_path, bbox_list in localization_results:

        print(f"Cropping bboxes")
        img = Image.open(image_path)
        img = transforms.ToTensor()(img)
        img_crops = crop_bboxes(img, bbox_list)

        print(f"Predicting {len(img_crops)} with binary classifier.")
        class_list, _ = predict_batch(
            img_crops, args.model_moth_nonmoth, args.category_map_moth_nonmoth, device
        )
        print(f"Predicting {len(img_crops)} with species classifier.")
        subclass_list, conf_list = predict_batch(
            img_crops, args.model_moth, args.category_map_moth, device
        )

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


def load_localization_model(model_path, device):
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


def process_localization_output(output):
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


def localization_classification(args):
    """main function for localization and classification"""

    torch.cuda.synchronize()
    start = time.time()

    data_dir = pathlib.Path(args.data_dir)
    image_folder = pathlib.Path(args.image_folder)
    data_path = data_dir / image_folder
    save_path = data_dir
    annot_file = f"localize_classify_annotation-{image_folder.name}.json"
    annotations = {}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    # device = "cpu"
    # device = "cuda"

    image_list = [
        p
        for p in os.listdir(data_path)
        if p.lower().endswith(".jpg") and not p.startswith(".")
    ]

    model = load_localization_model(args.model_localize, device)
    max_image_size = model.transform.max_size

    input_size = 300
    print(f"Preparing dataset of {len(image_list)} images")
    dataset = LocalizationDataset(
        directory=data_path,
        image_names=image_list,
        max_image_size=max_image_size,
    )

    # @TODO need to determine batch size and not run out of memory
    if device == "cuda":
        batch_size = 8
        num_workers = 2
    else:
        batch_size = 32
        num_workers = multiprocessing.cpu_count()
        # num_workers = 4

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
            torch.cuda.synchronize()
            batch_start = time.time()
            data = data.to(device, non_blocking=True)
            output = model(data)
            output = [process_localization_output(o) for o in output]
            results += list(zip(img_paths, output))
            torch.cuda.synchronize()
            batch_end = time.time()
            elapsed = batch_end - batch_start
            images_per_second = len(image_list) / elapsed
            print(
                f"Time per batch: {round(elapsed, 1)} seconds. {round(images_per_second, 1)} images per second"
            )

    torch.cuda.synchronize()
    end = time.time()
    elapsed = end - start
    images_per_second = len(image_list) / elapsed
    print(
        f"Localization time: {round(elapsed, 1)} seconds. {round(images_per_second, 1)} images per second (with startup)"
    )

    annotations = classify_bboxes(results, device, args)

    print("Saving annotations")
    with open(save_path / annot_file, "w") as outfile:
        json.dump(annotations, outfile)

    return save_path / annot_file


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
