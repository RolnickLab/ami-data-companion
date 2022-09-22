import pathlib

import torch
import torchvision
from PIL import Image

from ... import db


class Dataset(torch.utils.data.Dataset):
    def __init__(self, base_directory, image_transforms):
        super().__init__()

        self.directory = pathlib.Path(base_directory)
        self.transforms = torchvision.transforms.Compose(image_transforms)
        self.query_args = {"in_queue": True}

    def __len__(self):
        with db.get_session(self.directory) as sess:
            sess.query(db.Image).filter_by(**self.query_args).count()

    def __getitem__(self, idx):
        with db.get_session(self.directory) as sess:
            next_image = sess.query(db.Image).filter_by(**self.query_args).one_or_none()
            if next_image:
                img_path = self.directory / next_image.path
                pil_image = Image.open(img_path)
                next_image.in_queue = False
                item = (str(img_path), self.transforms(pil_image))
                sess.add(next_image)
                sess.commit()
            else:
                item = (None, None)
        return item


class DataLoader(torch.utils.data.DataLoader):
    pass
