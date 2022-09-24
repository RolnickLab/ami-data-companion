import time

import torch
import torchvision
import timm

from ...ml.utils import (
    get_device,
    get_weights,
    get_category_map,
    synchronize_clocks,
)
from ...utils import logger

from .dataloaders import BinaryClassificationDatabaseDataset

WEIGHTS = ["moth-nonmoth-effv2b3_20220506_061527_30.pth"]
LABELS = ["05-moth-nonmoth_category_map.json"]


def get_model(weights, device):
    logger.info(
        f'Loading "tf_efficientnetv2_b3" binary classification model with weights: {weights}'
    )
    model = timm.create_model(
        "tf_efficientnetv2_b3",
        num_classes=2,
        weights=None,
    )
    model = model.to(device)
    # state_dict = torch.hub.load_state_dict_from_url(weights_url)
    state_dict = torch.load(weights, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def get_transforms(input_size=300):
    mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]

    return torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((input_size, input_size)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean, std),
        ]
    )


def postprocess_single(obj_id, output, category_map):
    # @TODO run this on batch rather than single objects
    predictions = torch.nn.functional.softmax(output, dim=0)
    predictions = predictions.cpu().numpy()

    cat = predictions.argmax(axis=0)
    label = category_map[cat]
    score = predictions.max(axis=0).astype(float)

    result = [int(obj_id), label, score]
    logger.debug(f"Postprocess result single: {result}")
    return result


def postprocess(object_ids, output, category_map):
    predictions = torch.nn.functional.softmax(output, dim=1)
    predictions = predictions.cpu().numpy()

    categs = predictions.argmax(axis=1)
    labels = [category_map[cat] for cat in categs]
    scores = predictions.max(axis=1).astype(float)

    result = list(zip(object_ids.cpu().numpy().astype(int), labels, scores))
    logger.debug(f"Postprocess result batch: {result}")
    return result


def predict(
    base_directory,
    weights_path=WEIGHTS[0],
    labels_path=LABELS[0],
    batch_size=4,
    num_workers=2,
    device=None,
    results_callback=None,
):

    weights = get_weights(weights_path)
    device = get_device(device)
    category_map = get_category_map(labels_path)

    model = get_model(weights=weights, device=device)

    dataset = BinaryClassificationDatabaseDataset(
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
        for i, (object_ids, input_data_batch) in enumerate(dataloader):
            logger.debug(f"Batch {i+1} out of {len(dataloader)}")
            logger.debug(f"Binary classifying {len(object_ids)} detected objects")

            synchronize_clocks()
            batch_start = time.time()

            input_data_batch = input_data_batch.to(device, non_blocking=True)
            output_data_batch = model(input_data_batch)

            batch_results = postprocess(object_ids, output_data_batch, category_map)
            results += batch_results

            synchronize_clocks()
            batch_end = time.time()

            elapsed = batch_end - batch_start
            images_per_second = len(object_ids) / elapsed

            logger.info(
                f"Time per batch: {round(elapsed, 1)} seconds. {round(images_per_second, 1)} images per second"
            )
            # @TODO this doesn't need to be a callback anymore, or at least simplify it
            if results_callback:
                logger.debug(
                    "=== CALLBACK START: Save binary labels of classified objects == "
                )
                # Here we are saving the moth/non-moth labels
                classified_objects_data = [
                    {
                        # "id": object_id,
                        "binary_label": label,
                        "binary_label_score": score,
                    }
                    for object_id, label, score in batch_results
                ]
                results_callback(object_ids, classified_objects_data)
                print("=== CALLBACK END == ")

    synchronize_clocks()
    end = time.time()

    elapsed = end - start
    images_per_second = len(results) / elapsed

    logger.info(
        f"Binary classification time: {round(elapsed, 1)} seconds. {round(images_per_second, 1)} images per second"
    )
