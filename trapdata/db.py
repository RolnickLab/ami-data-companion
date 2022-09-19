import datetime
import time
import pathlib
import contextlib

import sqlalchemy as sa
from sqlalchemy import orm
from sqlalchemy_utils import aggregated, observes

from . import utils
from .utils import logger

Base = orm.declarative_base()

# Rename to CapturePeriod? shorter? less confusing with other types of Sessions. CaptureSession?
class MonitoringSession(Base):
    __tablename__ = "monitoring_sessions"

    id = sa.Column(sa.Integer, primary_key=True)
    day = sa.Column(sa.Date)
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
        # lazy="joined",
    )

    def __repr__(self):
        return (
            f"MonitoringSession("
            f"start_time={self.start_time.strftime('%c') if self.start_time else None !r}, end_time={self.end_time.strftime('%c') if self.end_time else None!r}, "
            f"num_images={self.num_images!r}, num_detected_objects={self.num_detected_objects!r})"
        )

    def update_aggregates(self):
        # Requires and active session
        logger.info(f"Updating cached values for {self}")
        images = self.images
        self.num_images = len(images)
        self.start_time = images[0].timestamp
        self.end_time = images[-1].timestamp

    def duration(self):
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        else:
            return None

    @property
    def duration_label(self):
        if self.duration():
            hours = int(round(self.duration().seconds / 60 / 60, 0))
            unit = "hour" if hours == 1 else "hours"
            duration = f"{hours} {unit}"
        else:
            duration = "Unknown duration"
        return duration


# These are aparently needed because none of the fancy stuff seems to work
def monitoring_session_images(ms):
    with get_session(ms.base_directory) as sess:
        return list(sess.query(Image).filter_by(monitoring_session_id=ms.id).all())


def monitoring_session_images_count(ms):
    with get_session(ms.base_directory) as sess:
        return int(sess.query(Image).filter_by(monitoring_session_id=ms.id).count())


def update_all_aggregates(directory):
    with get_session(directory) as sess:
        for ms in sess.query(MonitoringSession).all():
            ms.update_aggregates()
        sess.commit()


class Image(Base):
    __tablename__ = "images"

    id = sa.Column(sa.Integer, primary_key=True)
    monitoring_session_id = sa.Column(sa.ForeignKey("monitoring_sessions.id"))
    path = sa.Column(
        sa.String(255)
    )  # @TODO store these as relative paths to the directory
    timestamp = sa.Column(sa.DateTime(timezone=True))
    last_read = sa.Column(sa.DateTime)
    last_processed = sa.Column(sa.DateTime)
    notes = sa.Column(sa.JSON)

    def absolute_path(self, directory=None):
        if not directory:
            directory = self.monitoring_session.base_directory
        return pathlib.Path(directory) / self.path

    @aggregated("detected_objects", sa.Column(sa.Integer))
    def num_detected_objects(self):
        return sa.func.count("1")

    # @TODO let's keep the precious detected objects, even if the Monitoring Session or Image is deleted?
    detected_objects = orm.relationship(
        "DetectedObject",
        back_populates="image",
        cascade="all, delete-orphan",  # @TODO no! do not delete orphans? processing time is precious
        lazy="joined",
    )

    monitoring_session = orm.relationship(
        "MonitoringSession",
        back_populates="images",
        lazy="joined",
    )

    def __repr__(self):
        return f"Image(path={self.path!r}, timestamp={self.timestamp.strftime('%c') if self.timestamp else None !r}, num_detected_objects={self.num_detected_objects!r})"


class DetectedObject(Base):
    __tablename__ = "detections"

    id = sa.Column(sa.Integer, primary_key=True)
    image_id = sa.Column(sa.ForeignKey("images.id"))
    bbox = sa.Column(sa.JSON)
    area_pixels = sa.Column(sa.Integer)
    new_col = sa.Column(sa.Integer)
    specific_label = sa.Column(sa.String(255))
    specific_label_score = sa.Column(sa.Numeric)
    binary_label = sa.Column(sa.String(255))
    binary_label_score = sa.Column(sa.Numeric)
    last_detected = sa.Column(sa.DateTime)
    model_name = sa.Column(sa.String(255))
    notes = sa.Column(sa.JSON)

    image = orm.relationship("Image", back_populates="detected_objects")

    def __repr__(self):
        image = self.image.path if self.image else None
        return f"DetectedObject(image={image!r}, specific_label={self.specific_label!r}, bbox={self.bbox!r})"


def archive_file(filepath):
    """
    Rename an existing file to `<filepath>/<filename>.bak.<timestamp>`
    """
    filepath = pathlib.Path(filepath)
    if filepath.exists():
        suffix = f".{filepath.suffix}.backup.{str(int(time.time()))}"
        backup_filepath = filepath.with_suffix(suffix)
        logger.info(f"Moving existing file to {backup_filepath}")
        filepath.rename(backup_filepath)
        return backup_filepath


def get_db(directory=None, create=False):
    db_name = "trapdata.db"

    if directory:
        filepath = pathlib.Path(directory) / db_name
        if filepath.exists():
            if create:
                archive_file(filepath)
        else:
            create = True
        location = filepath

    else:
        # Only works in a scoped session. Used for tests.
        location = ":memory:"

    db = sa.create_engine(
        f"sqlite+pysqlite:///{location}",
        echo=False,
        future=True,
    )

    if create:
        Base.metadata.create_all(db)

    return db


@contextlib.contextmanager
def get_session(directory):
    """
    Convience method to start and close a database session.

    The database is a file-based sqlite database, so we store
    in the base directory of the trap images.
    All image paths in the database will be relative to the location
    of this base directory.


    SQL Alchemy also has a sessionmaker utility that could be used.
    # return orm.sessionmaker(db).begin()

    Usage:

    >>> directory = "/tmp/images"
    >>> with get_session(directory) as sess:
    >>>     num_images = sess.query(Image).filter_by().count()
    >>> num_images
    0
    """
    db = get_db(directory)
    session = orm.Session(db)

    yield session

    session.close()


def check_db(directory):
    """
    Try opening a database session.
    """
    try:
        with get_session(directory) as sess:
            # May have to check each model to detect schema changes
            # @TODO probably a better way to do this!
            sess.query(MonitoringSession).first()
            sess.query(Image).first()
            sess.query(DetectedObject).first()
    except sa.exc.OperationalError as e:
        logger.error(f"Error opening database session: {e}")
        raise


def query(directory, q, **kwargs):
    with get_session(directory) as sess:
        return list(sess.query(q, **kwargs))


def get_or_create(session, model, defaults=None, **kwargs):
    # https://stackoverflow.com/a/2587041/966058
    instance = session.query(model).filter_by(**kwargs).one_or_none()
    if instance:
        return instance, False
    else:
        kwargs |= defaults or {}
        instance = model(**kwargs)
        try:
            session.add(instance)
            session.commit()
        except Exception:  # The actual exception depends on the specific database so we catch all exceptions. This is similar to the official documentation: https://docs.sqlalchemy.org/en/latest/orm/session_transaction.html
            session.rollback()
            instance = session.query(model).filter_by(**kwargs).one()
            return instance, False
        else:
            return instance, True


def test_db():
    from .utils import TEST_IMAGES_BASE_PATH

    db = get_db()
    with orm.Session(db) as session:
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
