import typing

import torch
import torch.utils.data
import torchvision

from trapdata.common.logs import logger

from .schemas import (
    DetectionResponse,
    SourceImage,
)


class LocalizationImageDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        source_images: typing.Iterable[SourceImage],
        image_transforms: torchvision.transforms.Compose,
        batch_size: int = 1,
    ):
        super().__init__()
        self.source_images: list[SourceImage] = list(source_images)
        self.image_transforms: torchvision.transforms.Compose = image_transforms
        self.batch_size: int = batch_size

    def __len__(self):
        return len(list(self.source_images))

    def __getitem__(self, idx):
        worker_info = torch.utils.data.get_worker_info()
        logger.info(f"Using worker: {worker_info}")

        source_image: SourceImage = self.source_images[idx]
        image_data = source_image.open()
        if not image_data:
            return None

        image_data = self.image_transforms(image_data)

        # ids_batch = torch.utils.data.default_collate([source_image.id])
        # image_batch = torch.utils.data.default_collate([image_data])

        # logger.info(f"Batch data: {ids_batch}, {image_batch}")

        # return (ids_batch, image_batch)
        return source_image.id, image_data


class ClassificationImageDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        source_images: typing.Iterable[SourceImage],
        detections: typing.Iterable[DetectionResponse],
        image_transforms: torchvision.transforms.Compose,
        batch_size: int = 1,
    ):
        super().__init__()
        self.detections = list(detections)
        self.image_transforms: torchvision.transforms.Compose = image_transforms
        self.batch_size: int = batch_size
        self.source_images: dict[str, SourceImage] = {
            img.id: img for img in source_images
        }

    def __len__(self):
        # Append all detections to a single list
        return len(self.detections)

    def __getitem__(self, idx):
        worker_info = torch.utils.data.get_worker_info()
        logger.info(f"Using worker: {worker_info}")

        detection_idx = idx
        detection: DetectionResponse = self.detections[idx]
        source_image = self.source_images[detection.source_image_id]
        image_data = source_image.open()
        if not image_data:
            return None
        bbox = detection.bbox
        coords = bbox.x1, bbox.y1, bbox.x2, bbox.y2
        assert all(coord is not None for coord in coords)
        image_data = image_data.crop(coords)  # type: ignore
        image_data = self.image_transforms(image_data)

        # ids_batch = torch.utils.data.default_collate([source_image.id])
        # image_batch = torch.utils.data.default_collate([image_data])

        # logger.info(f"Batch data: {ids_batch}, {image_batch}")

        # return (ids_batch, image_batch)
        return (source_image.id, detection_idx), image_data
