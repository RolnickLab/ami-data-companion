import time

import torch
import torchvision
import timm

from ..utils import (
    get_device,
    get_or_download_file,
    get_category_map,
    synchronize_clocks,
)
from trapdata import logger

from .dataloaders import SpeciesClassificationDatabaseDataset

WEIGHTS = [
    "https://object-arbutus.cloud.computecanada.ca/ami-models/moths/classification/quebec-vermont-moth-model_v02_efficientnetv2-b3_2022-09-08-15-44.pt"
]
LABELS = [
    "https://object-arbutus.cloud.computecanada.ca/ami-models/moths/classification/quebec-vermont-moth_category-map_4Aug2022.json"
]


def get_model(weights, num_classes, device):
    logger.info(
        f'Loading "tf_efficientnetv2_b3" species classification model with weights: {weights}'
    )
    model = timm.create_model(
        "tf_efficientnetv2_b3",
        num_classes=num_classes,
        weights=None,
    )
    model = model.to(device)
    # state_dict = torch.hub.load_state_dict_from_url(weights_url)
    state_dict = torch.load(weights, map_location=device)
    state_dict = state_dict["model_state_dict"]
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
    models_dir,
    weights_path=WEIGHTS[0],
    labels_path=LABELS[0],
    batch_size=4,
    num_workers=2,
    device=None,
    results_callback=None,
):

    weights_path = get_or_download_file(weights_path, destination_dir=models_dir)
    labels_path = get_or_download_file(labels_path, destination_dir=models_dir)
    device = get_device(device)
    category_map = get_category_map(labels_path)

    model = get_model(
        weights=weights_path,
        num_classes=len(category_map),
        device=device,
    )

    dataset = SpeciesClassificationDatabaseDataset(
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
            logger.debug(f"Classifying {len(object_ids)} detected objects")

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
                    "=== CALLBACK START: Save species labels of classified objects == "
                )
                # Here we are saving the moth/non-moth labels
                classified_objects_data = [
                    {
                        # "id": object_id,
                        "specific_label": label,
                        "specific_label_score": score,
                    }
                    for object_id, label, score in batch_results
                ]
                object_ids = object_ids.tolist()
                print("OBJECT IDS", object_ids)
                results_callback(object_ids, classified_objects_data)
                print("=== CALLBACK END == ")

    synchronize_clocks()
    end = time.time()

    elapsed = end - start
    images_per_second = len(results) / elapsed

    logger.info(
        f"Species classification time: {round(elapsed, 1)} seconds. {round(images_per_second, 1)} images per second"
    )
