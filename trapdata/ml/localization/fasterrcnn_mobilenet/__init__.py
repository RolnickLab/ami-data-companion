import pathlib
import time

import torch
import torchvision
from torchvision import transforms
from PIL import Image

from trapdata.ml.utils import get_device, synchronize_clocks
from trapdata.utils import logger

LOCALIZATION_SCORE_THRESHOLD = 0.01


class Dataset(torch.utils.data.Dataset):
    def __init__(self, directory, image_names):
        super().__init__()

        self.directory = pathlib.Path(directory)
        self.image_names = image_names
        self.transform = self.get_transforms()

    def __len__(self):
        return len(self.image_names)

    def get_transforms(self):
        transform_list = [transforms.ToTensor()]
        return transforms.Compose(transform_list)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = self.directory / img_name
        pil_image = Image.open(img_path)
        return str(img_path), self.transform(pil_image)


class DataLoader(torch.utils.data.DataLoader):
    pass


def get_model(weights, device):
    logger.info(
        f'Loading "fasterrcnn_mobilenet" localization model with weights: {weights}'
    )
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(
        weights=weights
    )
    model = model.to(device)
    model.eval()
    return model


def postprocess(img_path, output):
    # This model does not use the labels from the object detection model
    _ = output["labels"]

    # Filter out objects if their score is under score threshold
    bboxes = output["boxes"][
        (output["scores"] > LOCALIZATION_SCORE_THRESHOLD) & (output["labels"] > 1)
    ]

    # Filter out background label, if using pretrained model only!
    bboxes = output["boxes"][output["labels"] > 1]

    logger.info(
        f"Keeping {len(bboxes)} out of {len(output['boxes'])} objects found (threshold: {LOCALIZATION_SCORE_THRESHOLD})"
    )

    bboxes = bboxes.cpu().numpy().astype(int).tolist()
    return bboxes


def predict(
    base_directory,
    image_list=list(),
    batch_size=4,
    num_workers=2,
    weights="DEFAULT",
    device=None,
    results_callback=None,
):
    """"""

    synchronize_clocks()
    start = time.time()

    device = device or get_device()
    model = get_model(weights=weights, device=device)

    logger.info(f"Preparing dataset of {len(image_list)} images")
    dataset = Dataset(
        directory=base_directory,
        image_names=image_list,
    )

    logger.info(
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
            logger.debug(f"Batch {i+1} out of {len(dataloader)}")
            logger.debug(f"Looking for objects in {len(img_paths)} images")

            synchronize_clocks()
            batch_start = time.time()

            data = data.to(device, non_blocking=True)
            output = model(data)
            output = [
                postprocess(img_path, o) for img_path, o in zip(img_paths, output)
            ]
            batch_results = list(zip(img_paths, output))
            results += batch_results

            synchronize_clocks()
            batch_end = time.time()

            elapsed = batch_end - batch_start
            images_per_second = len(image_list) / elapsed

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
    images_per_second = len(image_list) / elapsed

    logger.info(
        f"Localization time: {round(elapsed, 1)} seconds. {round(images_per_second, 1)} images per second (with startup)"
    )
