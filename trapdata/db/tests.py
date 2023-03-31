import datetime
import pathlib

from sqlalchemy import orm

from ..db.models import DetectedObject, MonitoringSession, TrapImage

TEST_IMAGES = pathlib.Path(__file__).parent / "images"


def create_test_data():
    """
    This method may be irrelevant, but it may be helpful for reference.
    """
    db = None
    with orm.Session(db) as session:
        new_ms = MonitoringSession(
            base_directory=TEST_IMAGES,
        )

        start_date = datetime.datetime.now()
        for i in range(3):
            img = TrapImage(
                path=f"2022-01-01/test_{i}.jpg",
                timestamp=start_date + datetime.timedelta(minutes=i * 60),
            )
            for i in range(2):
                bbox = DetectedObject(
                    bbox=[0, 0, 0, 0], specific_label="unknown", binary_label=True
                )
                img.detected_objects.append(bbox)
            new_ms.images.append(img)

        session.add(new_ms)

        new_ms.images.append(
            TrapImage(
                path="test", timestamp=start_date + datetime.timedelta(minutes=120)
            )
        )

        session.commit()

        monitoring_sessions = (
            session.query(MonitoringSession)
            .filter(MonitoringSession.base_directory == TEST_IMAGES)
            .all()
        )

        for ms in monitoring_sessions:
            print(ms)
            for img in ms.images:
                print("\t", img)
                for obj in img.detected_objects:
                    print("\t" * 2, obj)


if __name__ == "__main__":
    create_test_data()
