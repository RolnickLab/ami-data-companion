import datetime

import sqlalchemy as sa
from sqlalchemy import orm
from sqlalchemy_utils import aggregated, observes

from . import utils

Base = orm.declarative_base()


class MonitoringSession(Base):
    __tablename__ = "monitoring_sessions"

    id = sa.Column(sa.Integer, primary_key=True)
    base_directory = sa.Column(sa.String(255))
    start_time = sa.Column(sa.DateTime(timezone=True))
    end_time = sa.Column(sa.DateTime(timezone=True))
    notes = sa.Column(sa.JSON)

    @aggregated("images", sa.Column(sa.Integer))
    def num_images(self):
        return sa.func.count("1")

    @aggregated("images.detected_objects", sa.Column(sa.Integer))
    def num_detected_objects(self):
        return sa.func.count("1")

    @observes("images")
    def image_observer(self, images):
        timestamps = sorted([img.timestamp for img in images if img.timestamp])
        if timestamps:
            self.start_time = timestamps[0]
            self.end_time = timestamps[-1]

    images = orm.relationship(
        "Image",
        back_populates="monitoring_session",
        cascade="all, delete-orphan",
        order_by="Image.timestamp",
    )

    def __repr__(self):
        return (
            f"MonitoringSession("
            f"start_time={self.start_time.strftime('%c')!r}, end_time={self.end_time.strftime('%c')!r}, "
            f"num_images={self.num_images!r}, num_detected_objects={self.num_detected_objects!r})"
        )


class Image(Base):
    __tablename__ = "images"

    id = sa.Column(sa.Integer, primary_key=True)
    monitoring_session_id = sa.Column(sa.ForeignKey("monitoring_sessions.id"))
    path = sa.Column(sa.String(255))
    timestamp = sa.Column(sa.DateTime(timezone=True))
    last_read = sa.Column(sa.DateTime)
    last_processed = sa.Column(sa.DateTime)
    notes = sa.Column(sa.JSON)

    @aggregated("detected_objects", sa.Column(sa.Integer))
    def num_detected_objects(self):
        return sa.func.count("1")

    detected_objects = orm.relationship(
        "DetectedObject", back_populates="image", cascade="all, delete-orphan"
    )

    monitoring_session = orm.relationship(
        "MonitoringSession",
        back_populates="images",
    )

    def __repr__(self):
        return f"Image(path={self.path!r}, timestamp={self.timestamp.strftime('%c')!r}, num_detected_objects={self.num_detected_objects!r})"


class DetectedObject(Base):
    __tablename__ = "detections"

    id = sa.Column(sa.Integer, primary_key=True)
    image_id = sa.Column(sa.ForeignKey("images.id"))
    bbox = sa.Column(sa.JSON)
    specific_label = sa.Column(sa.String(255))
    binary_label = sa.Column(sa.Boolean)
    last_detected = sa.Column(sa.DateTime)
    notes = sa.Column(sa.JSON)

    image = orm.relationship("Image", back_populates="detected_objects")

    def __repr__(self):
        return f"DetectedObject(image={self.image.path!r}, specific_label={self.specific_label!r}, bbox={self.bbox!r})"


def init_db(location=":memory:"):
    engine = sa.create_engine(
        f"sqlite+pysqlite:///{location}",
        echo=False,
        future=True,
    )
    Base.metadata.create_all(engine)
    return engine


def test_db():
    from .utils import TEST_IMAGES_BASE_PATH

    engine = init_db(location=":memory:")
    with orm.Session(engine) as session:
        new_ms = MonitoringSession(
            base_directory=TEST_IMAGES_BASE_PATH,
        )

        start_date = datetime.datetime.now()
        for i in range(3):
            img = Image(
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
        session.commit()

        new_ms.images.append(
            Image(path="test", timestamp=start_date + datetime.timedelta(minutes=120))
        )
        session.commit()

        monitoring_sessions = (
            session.query(MonitoringSession)
            .filter(MonitoringSession.base_directory == TEST_IMAGES_BASE_PATH)
            .all()
        )

        for ms in monitoring_sessions:
            print(ms)
            for img in ms.images:
                print("\t", img)
                for obj in img.detected_objects:
                    print("\t" * 2, obj)


if __name__ == "__main__":
    test_db()
