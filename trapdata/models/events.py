import sqlalchemy as sa
from sqlalchemy import orm
from sqlalchemy_utils import aggregated, observes

from trapdata.db import Base, get_session
from trapdata.common.logs import logger
from trapdata.models.images import Image
from trapdata.common.filemanagement import find_images, group_images_by_day


# Rename to CapturePeriod? shorter? less confusing with other types of Sessions. CaptureSession?
# Or SurveyEvent or Survey?
class MonitoringSession(Base):
    __tablename__ = "monitoring_sessions"

    id = sa.Column(sa.Integer, primary_key=True)
    day = sa.Column(sa.Date)
    base_directory = sa.Column(sa.String(255))
    start_time = sa.Column(sa.DateTime(timezone=True))
    end_time = sa.Column(sa.DateTime(timezone=True))
    # num_species = sa.Column(sa.Integer)
    notes = sa.Column(sa.JSON)

    @aggregated("images", sa.Column(sa.Integer))
    def num_images(self):
        return sa.func.count("1")

    @aggregated("detected_objects", sa.Column(sa.Integer))
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

    detected_objects = orm.relationship(
        "DetectedObject",
        back_populates="monitoring_session",
        cascade="all, delete-orphan",
        # lazy="joined",
    )

    def __repr__(self):
        return (
            f"MonitoringSession("
            f"\tstart_time={self.start_time.strftime('%c') if self.start_time else None !r}, \n"
            f"\tend_time={self.end_time.strftime('%c') if self.end_time else None!r}, \n"
            f"\tnum_images={self.num_images!r}, \n"
            f"\tnum_detected_objects={self.num_detected_objects!r})"
        )

    def update_aggregates(self):
        # Requires and active session
        logger.info(f"Updating cached values for {self}")
        self.num_images = len(self.images)
        self.num_detected_objects = len(self.detected_objects)
        self.start_time = self.images[0].timestamp
        self.end_time = self.images[-1].timestamp

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


def save_monitoring_session(base_directory, session):
    with get_session(base_directory) as sess:
        ms_kwargs = {"base_directory": str(base_directory), "day": session["day"]}
        ms = sess.query(MonitoringSession).filter_by(**ms_kwargs).one_or_none()
        if ms:
            logger.debug(f"Found existing Monitoring Session in db: {ms}")
        else:
            ms = MonitoringSession(**ms_kwargs)
            logger.debug(f"Adding new Monitoring Session to db: {ms}")
            sess.add(ms)
            sess.flush()

        num_existing_images = (
            sess.query(Image).filter_by(monitoring_session_id=ms.id).count()
        )
        if session["num_images"] > num_existing_images:
            logger.info(
                f"session images: {session['num_images']}, saved count: {num_existing_images}"
            )
            # Compare the number of images known in this session
            # Only scan & add images if there is a difference.
            # This does not delete missing images.
            ms_images = []
            for image in session["images"]:
                # path = pathlib.Path(image["path"]).relative_to(base_directory)
                img_kwargs = {
                    "monitoring_session_id": ms.id,
                    "path": str(image["path"]),  # @TODO these should be relative paths!
                    "timestamp": image["timestamp"],
                }
                db_img = sess.query(Image).filter_by(**img_kwargs).one_or_none()
                if db_img:
                    # logger.debug(f"Found existing Image in db: {img}")
                    pass
                else:
                    db_img = Image(**img_kwargs)
                    logger.debug(f"Adding new Image to db: {db_img}")
                ms_images.append(db_img)
            sess.bulk_save_objects(ms_images)

            # Manually update aggregate & cached values after bulk update
            ms.update_aggregates()

        logger.debug("Comitting changes to DB")
        sess.commit()
        logger.debug("Done committing")


def save_monitoring_sessions(base_directory, sessions):

    for session in sessions:
        save_monitoring_session(base_directory, session)

    return get_monitoring_sessions_from_db(base_directory)


def get_monitoring_sessions_from_filesystem(base_directory):
    # @TODO can we use the sqlalchemy classes for sessions & images before
    # they are saved to the DB?
    images = find_images(base_directory)
    sessions = []
    groups = group_images_by_day(images)
    for day, images in groups.items():
        sessions.append(
            {
                "base_directory": str(base_directory),
                "day": day,
                "num_images": len(images),
                "start_time": images[0]["timestamp"],
                "end_time": images[-1]["timestamp"],
                "images": images,
            }
        )
    sessions.sort(key=lambda s: s["day"])
    return sessions


def get_monitoring_sessions_from_db(base_directory):
    logger.info("Quering existing sessions in DB")
    with get_session(base_directory) as sess:
        return (
            sess.query(MonitoringSession)
            .filter_by(base_directory=str(base_directory))
            .all()
        )


def monitoring_sessions_exist(base_directory):
    with get_session(base_directory) as sess:
        return (
            sess.query(MonitoringSession)
            .filter_by(base_directory=str(base_directory))
            .count()
        )


def get_or_create_monitoring_sessions(base_directory):
    if not monitoring_sessions_exist(base_directory):
        sessions = get_monitoring_sessions_from_filesystem(base_directory)
        save_monitoring_sessions(base_directory, sessions)
    return get_monitoring_sessions_from_db(base_directory)


def get_monitoring_session_images(ms):
    # @TODO this is likely to slow things down. Some monitoring sessions have thousands of images.
    with get_session(ms.base_directory) as sess:
        images = list(sess.query(Image).filter_by(monitoring_session_id=ms.id).all())
    logger.info(f"Found {len(images)} images in Monitoring Session: {ms}")
    return images


# # These are apparently needed because none of the fancy stuff seems to work
def get_monitoring_session_image_ids(ms):
    # Get a list of image IDs in order of timestamps as quickly as possible
    # This could be in the thousands
    with get_session(ms.base_directory) as sess:
        images = list(
            sess.query(Image.id)
            .filter_by(monitoring_session_id=ms.id)
            .order_by(Image.timestamp)
            .all()
        )
    logger.info(f"Found {len(images)} images in Monitoring Session: {ms}")
    return images
