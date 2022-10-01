import pathlib

import torch
import torchvision
from PIL import Image

from trapdata import db
from trapdata import logger
from trapdata import constants
from trapdata import models


class BinaryClassificationDatabaseDataset(torch.utils.data.Dataset):
    def __init__(self, base_directory, image_transforms):
        super().__init__()

        self.directory = pathlib.Path(base_directory)
        self.transform = image_transforms
        self.query_args = {
            "in_queue": True,
            "binary_label": None,
        }

    def __len__(self):
        with db.get_session(self.directory) as sess:
            count = (
                sess.query(models.DetectedObject)
                .filter(models.DetectedObject.bbox.is_not(None))
                .filter_by(**self.query_args)
                .count()
            )
            logger.info(f"Images found in queue: {count}")
            return count

    def __getitem__(self, idx):
        with db.get_session(self.directory) as sess:
            next_obj = (
                sess.query(models.DetectedObject)
                .filter(models.DetectedObject.bbox.is_not(None))
                .filter_by(**self.query_args)
                .options(db.orm.joinedload(models.DetectedObject.image))
                .first()
            )
            if next_obj:
                # @TODO improve. Can't the main transforms chain do this?
                # if we pass the bbox to get_transforms?
                img = Image.open(next_obj.image.path)
                img = torchvision.transforms.ToTensor()(img)
                x1, y1, x2, y2 = next_obj.bbox
                cropped_image = img[
                    :,
                    int(y1) : int(y2),
                    int(x1) : int(x2),
                ]
                cropped_image = torchvision.transforms.ToPILImage()(cropped_image)
                next_obj.in_queue = False
                item = (next_obj.id, self.transform(cropped_image))
                sess.add(next_obj)
                sess.commit()
                return item


class SpeciesClassificationDatabaseDataset(torch.utils.data.Dataset):
    def __init__(self, base_directory, image_transforms):
        super().__init__()

        self.directory = pathlib.Path(base_directory)
        self.transform = image_transforms
        self.query_args = {
            "in_queue": True,
            "specific_label": None,
            "binary_label": constants.POSITIVE_BINARY_LABEL,
        }

    def __len__(self):
        with db.get_session(self.directory) as sess:
            count = (
                sess.query(models.DetectedObject)
                .filter(models.DetectedObject.bbox.is_not(None))
                .filter_by(**self.query_args)
                .count()
            )
            logger.info(f"Images found in queue: {count}")
            return count

    def __getitem__(self, idx):
        with db.get_session(self.directory) as sess:
            next_obj = (
                sess.query(models.DetectedObject)
                .filter(models.DetectedObject.bbox.is_not(None))
                .filter_by(**self.query_args)
                .options(db.orm.joinedload(models.DetectedObject.image))
                .first()
            )
            if next_obj:
                # @TODO improve. Can't the main transforms chain do this?
                # if we pass the bbox to get_transforms?
                img = Image.open(next_obj.image.path)
                img = torchvision.transforms.ToTensor()(img)
                x1, y1, x2, y2 = next_obj.bbox
                cropped_image = img[
                    :,
                    int(y1) : int(y2),
                    int(x1) : int(x2),
                ]
                cropped_image = torchvision.transforms.ToPILImage()(cropped_image)
                next_obj.in_queue = False
                item = (next_obj.id, self.transform(cropped_image))
                sess.add(next_obj)
                sess.commit()
            else:
                item = (None, None)
        return item
