import pathlib
import tempfile
import datetime
import exif

from trapdata import logger
from trapdata.common.filemanagement import find_images, get_exif, write_exif


TEST_IMAGES = pathlib.Path(__file__).parent / "images"


def test():
    saved_images = []
    date = datetime.datetime.now() - datetime.timedelta(days=365 * 100)
    description = f"Image with test EXIF tags created at {date}"
    # keywords = ["Machine capture", "test"]

    for image in find_images(TEST_IMAGES):
        img_out = write_exif(
            image["path"],
            date=date,
            description=description,
            # keywords=keywords,
        )
        with tempfile.NamedTemporaryFile("wb", suffix=".jpg", delete=False) as f:
            logger.info(f"Writing exif to {f.name} from {image['path']}")
            f.write(img_out.get_file())
            saved_images.append(f.name)

    for fname in saved_images:
        logger.debug(f"Testing exif of {fname}")
        exif_result = get_exif(fname)
        logger.debug(exif_result)
        assert exif_result["DateTime"] == date.strftime(exif.DATETIME_STR_FORMAT)
        assert exif_result["ImageDescription"] == description


if __name__ == "__main__":
    logger.info(f"Using test images from: {TEST_IMAGES}")
    test()
