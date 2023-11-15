import tempfile
import typing

import numpy as np
import PIL.Image
import torch
import torch.utils.data
import torchvision

from trapdata.common.logs import logger
from trapdata.ml.utils import get_or_download_file

from .queries import fetch_source_image_data
from .schemas import Detection, SourceImage


class LocalizationAPIDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        source_image_ids: list[int],
        image_transforms: torchvision.transforms.Compose,
        batch_size: int = 1,
    ):
        super().__init__()
        self.source_image_ids = source_image_ids
        self.image_transforms = image_transforms
        self.batch_size = batch_size

    def __len__(self):
        return len(self.source_image_ids)

    def __getitem__(self, idx):
        worker_info = torch.utils.data.get_worker_info()
        logger.info(f"Using worker: {worker_info}")

        source_image_id = self.source_image_ids[idx]
        source_image = fetch_source_image_data(source_image_id)
        image_data = self.fetch_image(source_image.url)
        if not image_data:
            return None

        image_data = self.image_transforms(image_data)

        ids_batch = torch.utils.data.default_collate([source_image.id])
        image_batch = torch.utils.data.default_collate([image_data])

        return (ids_batch, image_batch)

    def fetch_image(self, url) -> typing.Optional[PIL.Image.Image]:
        url = url + "?width=5000&redirect=False"
        logger.info(f"Fetching and transforming: {url}")
        tempdir = tempfile.TemporaryDirectory()
        img_path = get_or_download_file(url, destination_dir=tempdir.name)
        try:
            return PIL.Image.open(img_path)
        except PIL.UnidentifiedImageError:
            logger.error(f"Unidentified image: {img_path}")
            print(f"Unidentified image: {img_path}")
            return None
        except OSError:
            logger.error(f"OSError: {img_path}")
            print(f"OSError: {img_path}")
            return None


class LocalizationImageDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        source_images: list[SourceImage],
        image_transforms: torchvision.transforms.Compose,
        batch_size: int = 1,
    ):
        super().__init__()
        self.source_images: list[SourceImage] = source_images
        self.image_transforms: torchvision.transforms.Compose = image_transforms
        self.batch_size: int = batch_size

    def __len__(self):
        return len(self.source_images)

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
        detections: list[Detection],
        image_transforms: torchvision.transforms.Compose,
        batch_size: int = 1,
    ):
        super().__init__()
        self.detections = detections
        self.image_transforms: torchvision.transforms.Compose = image_transforms
        self.batch_size: int = batch_size

    def __len__(self):
        # Append all detections to a single list
        return len(self.detections)

    def __getitem__(self, idx):
        worker_info = torch.utils.data.get_worker_info()
        logger.info(f"Using worker: {worker_info}")

        detection: Detection = self.detections[idx]
        image_data = detection.source_image.open()
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
        return (detection.source_image.id, np.array(coords)), image_data
