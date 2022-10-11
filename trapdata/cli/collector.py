import argparse

from trapdata import logger
from trapdata.common.filemanagement import find_images

from trapdata.ml.utils import StopWatch


parser = argparse.ArgumentParser(description="Scan and add trap images to database")
parser.add_argument(
    "directory",
    help="Path to directory of trap images to scan",
)
parser.add_argument("--max_num", type=int, nargs="+", help="Add at-most N samples")
parser.add_argument(
    "--queue",
    action="store_const",
    const=True,
    default=False,
    help="Automatically add all images to the process queue",
)
parser.add_argument(
    "--count-only",
    action="store_const",
    const=True,
    default=False,
    help="Just count the number of images found",
)
parser.add_argument(
    "--watch",
    action="store_const",
    const=True,
    default=False,
    help="Keep collector running and watch for new images",
)


def collect_images(path):
    images = []
    with StopWatch() as t:
        for i, f in enumerate(find_images(path)):
            logger.debug(f'Found {f["path"].name} from {f["timestamp"].strftime("%c")}')
            images.append(f)
    logger.info(f"Total images: {i+1}")
    logger.info(t)
    return images


def count_images(path):
    with StopWatch() as t:
        count = sum(1 for _ in find_images(path, include_timestamps=False))
    logger.info(f"Total images: {count}")
    logger.info(t)


if __name__ == "__main__":
    args = parser.parse_args()
    logger.debug(args)
    if args.count_only:
        count_images(args.directory)
    else:
        collect_images(args.directory)
