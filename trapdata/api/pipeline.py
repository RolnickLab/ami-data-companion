from trapdata import logger
from trapdata.settings import Settings

from .models.localization import APIMothObjectDetector_FasterRCNN_MobileNet_2023


def start_pipeline(
    source_image_ids: list[int],
    settings: Settings,
):
    logger.info(f"Local user data path: {settings.user_data_path}")

    object_detector = APIMothObjectDetector_FasterRCNN_MobileNet_2023(
        source_image_ids=source_image_ids,
        batch_size=settings.localization_batch_size,
        num_workers=settings.num_workers,
        user_data_path=settings.user_data_path,
        single=True,
    )
    # if object_detector.queue.queue_count() > 0:
    object_detector.run()
    logger.info("Localization complete")
