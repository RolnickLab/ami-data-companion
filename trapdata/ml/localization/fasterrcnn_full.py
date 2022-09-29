import time

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from ...ml.utils import get_device, synchronize_clocks, get_or_download_file
from ...utils import logger
from .dataloaders import LocalizationDatabaseDataset

LOCALIZATION_SCORE_THRESHOLD = 0.99


WEIGHTS = [
    "https://object-arbutus.cloud.computecanada.ca/ami-models/moths/localization/v1_localizmodel_2021-08-17-12-06.pt"
]

LABELS = []


def get_model(weights, device):
    logger.info(f'Loading "fasterrcnn_full" localization model with weights: {weights}')
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    num_classes = 2  # 1 class (object) + background
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    checkpoint = torch.load(weights, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model


def get_transforms():
    return torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
        ]
    )


def postprocess(img_path, output):
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


def predict(
    base_directory,
    models_dir,
    weights_path=WEIGHTS[0],
    labels_path=None,
    batch_size=4,
    num_workers=2,
    device=None,
    results_callback=None,
):

    weights_path = get_or_download_file(weights_path, destination_dir=models_dir)
    device = get_device(device)

    model = get_model(weights=weights_path, device=device)

    dataset = LocalizationDatabaseDataset(
        base_directory=base_directory,
        image_transforms=get_transforms(),
    )

    logger.info(
        f"Preparing dataloader with batch size of {batch_size} and {num_workers} workers on device: {device}"
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=True,
        pin_memory=True,
    )

    synchronize_clocks()
    start = time.time()

    results = []
    with torch.no_grad():
        for i, (img_paths, input_data_batch) in enumerate(dataloader):
            logger.debug(f"Batch {i+1} out of {len(dataloader)}")
            logger.debug(f"Looking for objects in {len(img_paths)} images")

            synchronize_clocks()
            batch_start = time.time()

            input_data_batch = input_data_batch.to(device, non_blocking=True)
            output_data_batch = model(input_data_batch)

            output = [
                postprocess(img_path, o)
                for img_path, o in zip(img_paths, output_data_batch)
            ]
            batch_results = list(zip(img_paths, output))
            results += batch_results

            synchronize_clocks()
            batch_end = time.time()

            elapsed = batch_end - batch_start
            images_per_second = len(img_paths) / elapsed

            logger.info(
                f"Time per batch: {round(elapsed, 1)} seconds. {round(images_per_second, 1)} images per second"
            )
            if results_callback:
                logger.debug(
                    "=== CALLBACK START: Save only bboxes of detected objects == "
                )
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
    images_per_second = len(results) / elapsed

    logger.info(
        f"Localization time: {round(elapsed, 1)} seconds. {round(images_per_second, 1)} images per second"
    )
