import sqlalchemy as sa

from trapdata import logger
from trapdata.db.base import get_session_class
from trapdata.db.models import DetectedObject, TrapImage
from trapdata.settings import read_settings

settings = read_settings()

Session = get_session_class(settings.database_url)


def add_missing_image_data():
    with Session() as session:
        query = sa.select(TrapImage).where(
            (TrapImage.height.is_(None) | TrapImage.width.is_(None))
        )
        images = session.execute(query).unique().scalars().all()
        logger.info(f"Adding image dimensions to {len(images)} images")
        for image in images:
            image.update_source_data(session, commit=False)
        session.commit()


def add_missing_detection_data():
    with Session() as session:
        query = sa.select(DetectedObject).where(
            (
                DetectedObject.timestamp.is_(None)
                | DetectedObject.source_image_width.is_(None)
                | DetectedObject.source_image_height.is_(None)
                | DetectedObject.source_image_previous_frame.is_(None)
            )
        )
        detections = session.execute(query).unique().scalars().all()
        logger.info(f"Adding missing data to {len(detections)} detections")
        for detection in detections:
            detection.update_data_from_source_image(session, commit=False)
        session.commit()
