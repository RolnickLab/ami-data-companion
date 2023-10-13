import datetime
import pathlib
from typing import Any, Iterable, Optional, Union

import sqlalchemy as sa
from pydantic import BaseModel
from sqlalchemy import orm
from sqlalchemy.ext.hybrid import hybrid_method
from sqlalchemy_utils import aggregated

from trapdata.common.filemanagement import find_images, group_images_by_day, media_url
from trapdata.common.logs import logger
from trapdata.common.schemas import FilePath
from trapdata.common.utils import export_report
from trapdata.db import Base, get_session, models
from trapdata.db.models.deployments import deployment_name

# @TODO Rename to TrapEvent? CapturePeriod? less confusing with other types of Sessions. CaptureSession? Or SurveyEvent or Survey?


class MonitoringSessionNestedCapture(BaseModel):
    id: str
    path: pathlib.Path
    timestamp: datetime.datetime
    # example["source_image_path"] = media_url(
    #    example["source_image_path"], "captures/", media_url_base
    # )


class MonitoringSessionListItem(BaseModel):
    id: str
    day: datetime.date
    # image_base_path: str
    deployment: str
    num_captures: int
    num_detections: int
    num_occurrences: int
    num_species: int
    example_captures: list[MonitoringSessionNestedCapture]
    start_time: datetime.datetime
    end_time: datetime.datetime
    duration: datetime.timedelta
    duration_label: str


class MonitoringSessionDetail(MonitoringSessionListItem):
    notes: Optional[str]
    captures: list[
        MonitoringSessionNestedCapture
    ]  # Too many! @TODO include summary data to generate the timeline instead
    # @TODO add more info about the session, like the number of images, the number of detected objects, etc
    # @TODO add the number of species detected in this session


class MonitoringSession(Base):
    __tablename__ = "monitoring_sessions"

    id = sa.Column(sa.Integer, primary_key=True)
    day = sa.Column(sa.Date)
    # @TODO instead of base directory, we can now use a Trap object to group sessions
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

    def num_occurrences(self, session: orm.Session) -> int:
        return (
            session.execute(
                sa.select(
                    sa.func.count(models.DetectedObject.sequence_id.distinct())
                ).where(models.DetectedObject.monitoring_session_id == self.id)
            ).scalar()
            or 0
        )

    def num_species(self, session: orm.Session) -> int:
        return (
            session.execute(
                sa.select(
                    sa.func.count(models.DetectedObject.specific_label.distinct())
                ).where(models.DetectedObject.monitoring_session_id == self.id)
            ).scalar()
            or 0
        )

    # This runs an expensive/slow query every time an image is updated
    # @observes("images")
    # def image_observer(self, images):
    #     timestamps = sorted([img.timestamp for img in images if img.timestamp])
    #     if timestamps:
    #         self.start_time = timestamps[0]
    #         self.end_time = timestamps[-1]

    images = orm.relationship(
        "TrapImage",
        back_populates="monitoring_session",
        cascade="all, delete-orphan",  # @TODO do not automatically cascade delete!
        order_by="TrapImage.timestamp",
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
            f"MonitoringSession(\n"
            f"\tid={self.id !r}, \n"
            f"\tday={self.day.strftime('%Y-%m-%d') !r}, \n"
            f"\tstart_time={self.start_time.strftime('%c') if self.start_time else None !r}, \n"
            f"\tend_time={self.end_time.strftime('%c') if self.end_time else None!r}, \n"
            f"\tnum_images={self.num_images!r}, \n"
            f"\tnum_detected_objects={self.num_detected_objects!r})"
        )

    def update_aggregates(self, session: orm.Session, commit=True):
        # Requires and active session
        logger.info(f"Updating cached values for event {self.day}")
        self.num_images = session.execute(
            sa.select(sa.func.count(1)).where(
                models.TrapImage.monitoring_session_id == self.id
            )
        ).scalar_one()
        self.num_detected_objects = session.execute(
            sa.select(sa.func.count(1)).where(
                models.DetectedObject.monitoring_session_id == self.id
            )
        ).scalar_one()
        self.start_time: datetime.datetime = session.execute(
            sa.select(sa.func.min(models.TrapImage.timestamp)).where(
                models.TrapImage.monitoring_session_id == self.id
            )
        ).scalar_one()
        self.end_time: datetime.datetime = session.execute(
            sa.select(sa.func.max(models.TrapImage.timestamp)).where(
                models.TrapImage.monitoring_session_id == self.id
            )
        ).scalar_one()
        if commit:
            session.commit()

    @hybrid_method
    def duration(self) -> datetime.timedelta:
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        else:
            return datetime.timedelta(0)

    @property
    def duration_label(self):
        duration = self.duration()
        if duration:
            hours = int(round(duration.seconds / 60 / 60, 0))
            unit = "hour" if hours == 1 else "hours"
            duration = f"{hours} {unit}"
        else:
            duration = "Unknown duration"
        return duration

    @property
    def deployment(self) -> str:
        return models.deployments.deployment_name(str(self.base_directory))

    def report_data(self) -> dict[str, Any]:
        duration = self.duration()

        return {
            "id": self.id,
            "deployment": self.deployment,
            "event": self.day.isoformat(),
            "duration_minutes": int(round(duration.seconds / 60, 0)) if duration else 0,
            "duration_label": self.duration_label,
            "num_source_images": self.num_images,
            "num_detections": self.num_detected_objects,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
        }


def save_monitoring_session(db_path, base_directory, session):
    # @TODO find & save all images to the DB first, then
    # group by timestamp and construct monitoring sessions. window function?
    with get_session(db_path) as sesh:
        ms_kwargs = {"base_directory": str(base_directory), "day": session["day"]}
        ms = sesh.query(MonitoringSession).filter_by(**ms_kwargs).one_or_none()

        if ms:
            logger.debug(f"Found existing Monitoring Session in db: {ms}")
        else:
            ms = MonitoringSession(**ms_kwargs)
            logger.debug(f"Adding new Monitoring Session to db: {ms}")
            sesh.add(ms)
            sesh.flush()

        num_existing_images = (
            sesh.query(models.TrapImage).filter_by(monitoring_session_id=ms.id).count()
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
                path = pathlib.Path(image["path"]).relative_to(ms.base_directory)
                # absolute_path = pathlib.Path(ms.base_directory) / path
                img_kwargs = {
                    "monitoring_session_id": ms.id,
                    "base_path": ms.base_directory,
                    "path": str(path),
                    "timestamp": image["timestamp"],
                    "filesize": image["filesize"],
                    "width": image["shape"][0],
                    "height": image["shape"][1],
                    # file hash?
                }
                db_img = (
                    sesh.query(models.TrapImage).filter_by(**img_kwargs).one_or_none()
                )
                if db_img:
                    # logger.debug(f"Found existing Image in db: {img}")
                    pass
                else:
                    db_img = models.TrapImage(**img_kwargs)
                    logger.debug(f"Adding new Image to db: {db_img}")
                ms_images.append(db_img)
            logger.info(f"Bulk saving {len(ms_images)} objects")
            sesh.bulk_save_objects(ms_images)

            # Manually update aggregate & cached values after bulk update
            ms.update_aggregates(sesh)

        logger.debug("Committing changes to DB")
        sesh.commit()
        logger.debug("Done committing")


def save_monitoring_sessions(db_path, base_directory, sessions):
    for session in sessions:
        save_monitoring_session(db_path, base_directory, session)

    return get_monitoring_sessions_from_db(db_path, base_directory)


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


def get_monitoring_sessions_from_db(
    db_path: str,
    base_directory: Union[pathlib.Path, str, None] = None,
    update_aggregates: bool = True,
):
    query_kwargs = {}

    logger.info("Querying existing sessions in DB")

    if base_directory:
        query_kwargs["base_directory"] = str(base_directory)

    with get_session(db_path) as sesh:
        items = (
            sesh.query(MonitoringSession)
            .filter_by(
                **query_kwargs,
            )
            .order_by(MonitoringSession.start_time)
            .all()
        )
        if update_aggregates:
            [item.update_aggregates(sesh) for item in items]
        return items


def get_monitoring_session_by_date(
    db_path: str,
    event_dates: list[datetime.date],
    base_directory: Union[pathlib.Path, str, None] = None,
):
    query_kwargs = {}

    if base_directory:
        query_kwargs["base_directory"] = str(base_directory)

    with get_session(db_path) as sesh:
        items = (
            sesh.query(MonitoringSession)
            .where(MonitoringSession.day.in_(event_dates))
            .filter_by(
                **query_kwargs,
            )
            .all()
        )
        [item.update_aggregates(sesh) for item in items]
        return items


def monitoring_sessions_exist(db_path, base_directory):
    with get_session(db_path) as sesh:
        return (
            sesh.query(MonitoringSession)
            .filter_by(base_directory=str(base_directory))
            .count()
        )


def get_or_create_monitoring_sessions(db_path, base_directory):
    # @TODO Check if there are unprocessed images in monitoring session?
    if not monitoring_sessions_exist(db_path, base_directory):
        sessions = get_monitoring_sessions_from_filesystem(base_directory)
        save_monitoring_sessions(db_path, base_directory, sessions)
    return get_monitoring_sessions_from_db(db_path, base_directory)


def get_monitoring_session_images(db_path, ms, limit=None, offset=None):
    # @TODO this is likely to slow things down. Some monitoring sessions have thousands of images.
    with get_session(db_path) as sesh:
        images = list(
            sesh.query(models.TrapImage)
            .filter_by(monitoring_session_id=ms.id)
            .order_by(models.TrapImage.timestamp)
            .limit(limit)
            .offset(offset)
            .all()
        )
    logger.info(f"Found {len(images)} images in Monitoring Session: {ms}")
    return images


# # These queries are apparently needed because none of the fancy stuff seems to work
def get_monitoring_session_image_ids(db_path, ms):
    # Get a list of image IDs in order of timestamps as quickly as possible
    # This could be in the thousands
    with get_session(db_path) as sesh:
        images = list(
            sesh.query(models.TrapImage.id)
            .filter_by(monitoring_session_id=ms.id)
            .order_by(models.TrapImage.timestamp)
            .all()
        )
    logger.info(f"Found {len(images)} images in Monitoring Session: {ms}")
    return images


def update_all_aggregates(session: orm.Session, image_base_path: FilePath):
    for event in (
        session.execute(
            sa.select(MonitoringSession).where(
                MonitoringSession.base_directory == str(image_base_path)
            )
        )
        .unique()
        .scalars()
        .all()
    ):
        event.update_aggregates(session, commit=False)
    session.commit()


def export_monitoring_sessions(
    items: Iterable[MonitoringSession],
    directory: Union[pathlib.Path, str],
    report_name: str = "monitoring_events",
):
    records = [item.report_data() for item in items]
    return export_report(records, report_name, directory)


def event_response(
    session: orm.Session,
    event: MonitoringSession,
) -> MonitoringSessionListItem:
    """
    Reusable method to create a MonitoringSession Schema from a MonitoringSession model.

    @TODO decide if this is helpful or not to reuse in get_monitoring_sessions and get_monitoring_session_by_id
    """

    event.update_aggregates(session)
    event_response = MonitoringSessionListItem(
        id=event.id,
        day=event.day,
        deployment=deployment_name(str(event.base_directory)),
        start_time=event.start_time,
        end_time=event.end_time,
        num_captures=event.num_images,
        num_detections=event.num_detected_objects,
        num_occurrences=event.num_occurrences(session),
        num_species=event.num_species(session),
        example_captures=[],
        duration=event.duration(),
        duration_label=event.duration_label,
    )

    return event_response


def get_monitoring_session_by_id(
    session: orm.Session,
    event_id: int,
    media_url_base: str,
) -> Optional[MonitoringSessionDetail]:
    event: Optional[MonitoringSession] = session.get(MonitoringSession, event_id)
    if event:
        event.update_aggregates(session)
        captures = session.execute(
            sa.select(
                models.TrapImage.id, models.TrapImage.path, models.TrapImage.timestamp
            )
            .where(models.TrapImage.monitoring_session_id == event.id)
            .order_by(models.TrapImage.timestamp)
        ).all()
        nested_captures = [
            MonitoringSessionNestedCapture(
                id=row.id,
                path=media_url(row.path, "captures/", media_url_base),
                timestamp=row.timestamp,
            )
            for row in captures
        ]
        event_detail = MonitoringSessionDetail(
            id=event.id,
            day=event.day,
            deployment=deployment_name(str(event.base_directory)),
            start_time=event.start_time,
            end_time=event.end_time,
            num_captures=event.num_images,
            num_detections=event.num_detected_objects,
            num_occurrences=event.num_occurrences(session),
            num_species=event.num_species(session),
            example_captures=[],
            duration=event.duration(),
            duration_label=event.duration_label,
            notes=event.notes,
            captures=nested_captures,
        )
        return event_detail
    else:
        return None


def list_monitoring_sessions(
    session: orm.Session,
    image_base_path: FilePath,
    limit: Optional[int] = None,
    offset: int = 0,
    num_examples: int = 5,
    media_url_base: Optional[str] = None,
) -> list[MonitoringSessionListItem]:
    """ """

    update_all_aggregates(session, image_base_path)
    logger.info(f"Fetching monitoring events for images in {image_base_path}")
    events = (
        session.execute(
            sa.select(models.MonitoringSession)
            .where(models.MonitoringSession.base_directory == str(image_base_path))
            .order_by(models.MonitoringSession.day)
            .limit(limit)
            .offset(offset)
        )
        .unique()
        .scalars()
        .all()
    )

    list_items = []
    for event in events:
        event.update_aggregates(session)
        rows = session.execute(
            sa.select(
                models.TrapImage.id, models.TrapImage.path, models.TrapImage.timestamp
            )
            .where(models.TrapImage.monitoring_session_id == event.id)
            .order_by(models.TrapImage.filesize.desc())
            .limit(num_examples)
        ).all()
        example_captures = [
            MonitoringSessionNestedCapture(
                id=row.id,
                path=media_url(row.path, "captures/", media_url_base),
                timestamp=row.timestamp,
            )
            for row in rows
        ]
        list_items.append(
            MonitoringSessionListItem(
                id=event.id,
                day=event.day,
                deployment=deployment_name(str(event.base_directory)),
                start_time=event.start_time,
                end_time=event.end_time,
                num_captures=event.num_images,
                num_detections=event.num_detected_objects,
                num_occurrences=event.num_occurrences(session),
                num_species=event.num_species(session),
                example_captures=example_captures,
                duration=event.duration(),
                duration_label=event.duration_label,
            )
        )
    return list_items
