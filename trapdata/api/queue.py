import logging
from typing import Sequence

from . import queries
from .schemas import SourceImage

logger = logging.getLogger(__name__)


class APIQueueManager:
    name: str
    project: int
    collection: int

    def __init__(self):
        pass

    def queue_count(self) -> int:
        return 0

    def unprocessed_count(self) -> int:
        return 0

    def done_count(self) -> int:
        return 0

    def add_unprocessed(self, *_):
        raise NotImplementedError

    def clear_queue(self, *_):
        raise NotImplementedError

    def status(self):
        return NotImplementedError

    def pull_n_from_queue(self, n: int):
        return NotImplementedError


class ImageAPIQueue(APIQueueManager):
    name = "Source images"
    description = "Raw images from camera needing object detection"

    def queue_count(self) -> int:
        return queries.get_totals(self.project)["source_images"]

    def unprocessed_count(self) -> int:
        return 999

    def done_count(self) -> int:
        return 999

    def pull_n_from_queue(self, n: int) -> Sequence[SourceImage]:
        logger.debug(f"Attempting to pull {n} images from queue")
        return queries.get_next_source_images(n)
