import datetime
import pathlib
import tempfile

import PIL.Image

from trapdata import logger
from trapdata.common.filemanagement import (
    EXIF_DATETIME_STR_FORMAT,
    construct_exif,
    find_images,
    get_exif,
)

TEST_IMAGES = pathlib.Path(__file__).parent / "images"


def run():
    logger.info(f"Using test images from: {TEST_IMAGES}")
    saved_images = []
    timestamp = datetime.datetime.now() - datetime.timedelta(days=365 * 100)
    description = f"Image with test EXIF tags created at {timestamp}"
    # keywords = ["Machine capture", "test"]

    for image in find_images(TEST_IMAGES):
        img = PIL.Image.open(image["path"])
        existing_exif = img.getexif()
        exif = construct_exif(
            timestamp=timestamp,
            description=description,
            existing_exif=existing_exif,
        )
        with tempfile.NamedTemporaryFile("wb", suffix=".jpg", delete=False) as f:
            logger.info(f"Writing exif to {f.name} from {image['path']}")
            img.save(f.name, exif=exif)
            saved_images.append(f.name)

    for fname in saved_images:
        logger.info(f"Testing exif of {fname}")
        exif_result = get_exif(fname)
        logger.info(exif_result)
        expected_timestamp = timestamp.strftime(EXIF_DATETIME_STR_FORMAT)
        assert exif_result["DateTime"] == expected_timestamp
        assert exif_result["DateTimeOriginal"] == expected_timestamp
        assert exif_result["DateTimeDigitized"] == expected_timestamp
        assert exif_result["ImageDescription"] == description


if __name__ == "__main__":
    run()
