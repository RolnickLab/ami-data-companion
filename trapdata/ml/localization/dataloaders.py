import pathlib

import torch
import torchvision
import PIL.Image

from trapdata import db
from trapdata import logger
from trapdata import models


class LocalizationDatabaseDataset(torch.utils.data.Dataset):
    def __init__(self, base_directory, image_transforms):
        super().__init__()

        self.directory = pathlib.Path(base_directory)
        self.transform = image_transforms
        self.query_args = {"in_queue": True}

    def __len__(self):
        with db.get_session(self.directory) as sess:
            count = sess.query(models.Image).filter_by(**self.query_args).count()
            logger.info(f"Images found in queue: {count}")
            return count

    def __getitem__(self, idx):
        with db.get_session(self.directory) as sess:
            next_image = sess.query(models.Image).filter_by(**self.query_args).first()
            if next_image:
                img_path = self.directory / next_image.path
                pil_image = PIL.Image.open(img_path)
                next_image.in_queue = False
                item = (str(img_path), self.transform(pil_image))
                sess.add(next_image)
                sess.commit()
                return item


class LocalizationFilesystemDataset(torch.utils.data.Dataset):
    def __init__(self, directory, image_names):
        super().__init__()

        self.directory = pathlib.Path(directory)
        self.image_names = image_names
        self.transform = self.get_transforms()

    def __len__(self):
        return len(self.image_names)

    def get_transforms(self):
        transform_list = [torchvision.transforms.ToTensor()]
        return torchvision.transforms.Compose(transform_list)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = self.directory / img_name
        pil_image = Image.open(img_path)
        return str(img_path), self.transform(pil_image)
