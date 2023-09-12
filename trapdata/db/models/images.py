import datetime
import pathlib
from typing import Generator, Optional

import sqlalchemy as sa
from pydantic import BaseModel
from sqlalchemy import orm
from sqlalchemy_utils import aggregated

from trapdata import constants
from trapdata.common.filemanagement import (
    get_image_dimensions,
    get_image_filesize,
    get_image_timestamp,
)
from trapdata.db import Base, get_session


class CaptureListItem(BaseModel):
    id: int
    timestamp: datetime.datetime
    source_image: str
    last_read: Optional[datetime.datetime]
    last_processed: Optional[datetime.datetime]
    num_detections: Optional[int]
    in_queue: bool


class CaptureDetail(CaptureListItem):
    id: int
    event: object
    notes: Optional[str]
    detections: list
    filesize: int
    width: int
    height: int


"""
label_studio_task = {
  "data": {
    "image": "/static/samples/sample.jpg"
  },

  "predictions": [{
    "model_version": "one",
    "score": 0.5,
    "result": [
      {
        "id": "result1",
        "type": "rectanglelabels",
        "from_name": "label", "to_name": "image",
        "original_width": 600, "original_height": 403,
        "image_rotation": 0,
        "value": {
          "rotation": 0,
          "x": 4.98, "y": 12.82,
          "width": 32.52, "height": 44.91,
          "rectanglelabels": ["Airplane"]
        }
      },
      {
        "id": "result2",
        "type": "rectanglelabels",
        "from_name": "label", "to_name": "image",
        "original_width": 600, "original_height": 403,
        "image_rotation": 0,
        "value": {
          "rotation": 0,
          "x": 75.47, "y": 82.33,
          "width": 5.74, "height": 7.40,
          "rectanglelabels": ["Car"]
        }
      },
      {
        "id": "result3",
        "type": "choices",
        "from_name": "choice", "to_name": "image",
        "value": {
          "choices": ["Airbus"]
      }
    }]
  }]
}
"""


class LabelStudioTaskData(BaseModel):
    image: str
    timestamp: datetime.datetime
    deployment: str


class LabelStudioTaskPredictionResultValue(BaseModel):
    rotation: int
    x: float
    y: float
    width: float
    height: float
    rectanglelabels: list[str]


class LabelStudioTaskPredictionResult(BaseModel):
    id: str
    type: str
    from_name: str
    to_name: str
    original_width: int
    original_height: int
    image_rotation: int
    value: LabelStudioTaskPredictionResultValue


class LabelStudioTaskPrediction(BaseModel):
    model_version: str
    score: float
    result: list[LabelStudioTaskPredictionResult]


class LabelStudioTask(BaseModel):
    data: LabelStudioTaskData
    predictions: list[LabelStudioTaskPrediction]


def _as_label_studio_task(image: "TrapImage", db_path: str) -> LabelStudioTask:
    # @TODO this is a hack to get the deployment name
    deployment = image.absolute_path.parent.parent.parent.name

    data = LabelStudioTaskData(
        image=f"{constants.IMAGE_BASE_URL}/{image.path}",
        timestamp=image.timestamp,
        deployment=deployment,
    )

    label_map = {
        constants.POSITIVE_BINARY_LABEL: "Moth",
        constants.NEGATIVE_BINARY_LABEL: "Non-Moth",
    }

    results = []

    from trapdata.db.models.detections import get_detections_for_image

    model_name = "unknown"
    for obj in get_detections_for_image(db_path, image.id):
        model_name = str(obj.model_name)
        if not obj.binary_label:
            print(f"Skipping {obj} because it has no binary label")
            continue
        label = label_map[obj.binary_label]
        result = LabelStudioTaskPredictionResult(
            id=str(obj.id),
            type="rectanglelabels",
            from_name="label",
            to_name="image",
            original_width=image.width,
            original_height=image.height,
            image_rotation=0,
            value=LabelStudioTaskPredictionResultValue(
                rotation=0,
                x=(obj.bbox[0] / image.width) * 100,
                y=(obj.bbox[1] / image.height) * 100,
                width=(obj.width() / image.width) * 100,
                height=(obj.height() / image.height) * 100,
                rectanglelabels=[label],
            ),
        )
        results.append(result)

    predictions = [
        LabelStudioTaskPrediction(
            model_version=model_name,
            score=0,
            result=results,
        )
    ]

    return LabelStudioTask(data=data, predictions=predictions)


class TrapImage(Base):
    __tablename__ = "images"

    id = sa.Column(sa.Integer, primary_key=True)
    monitoring_session_id = sa.Column(sa.ForeignKey("monitoring_sessions.id"))
    base_path = sa.Column(sa.String(255))
    path = sa.Column(sa.String(255))
    timestamp = sa.Column(sa.DateTime(timezone=True))
    filesize = sa.Column(sa.Integer)
    last_read = sa.Column(sa.DateTime)
    last_processed = sa.Column(sa.DateTime)
    in_queue = sa.Column(sa.Boolean, default=False)
    notes = sa.Column(sa.JSON)
    width = sa.Column(sa.Integer)
    height = sa.Column(sa.Integer)
    # position
    # diag
    # centroid
    # cnn features

    @property
    def absolute_path(self, directory: Optional[str] = None) -> pathlib.Path:
        # @TODO this directory argument can be removed once the image has the base
        # path stored in itself
        if not directory:
            directory = str(self.base_path)
        return pathlib.Path(directory) / str(self.path)

    @aggregated("detected_objects", sa.Column(sa.Integer))
    def num_detected_objects(self):
        return sa.func.count("1")

    def previous_image(self, session: orm.Session) -> Optional["TrapImage"]:
        img = session.execute(
            sa.select(TrapImage)
            .filter(TrapImage.timestamp < self.timestamp)
            .filter(TrapImage.monitoring_session_id == self.monitoring_session_id)
            .order_by(TrapImage.timestamp.desc())
            .limit(1)
        ).scalar()
        return img

    def next_image(self, session: orm.Session) -> Optional["TrapImage"]:
        img = session.execute(
            sa.select(TrapImage)
            .filter(TrapImage.timestamp > self.timestamp)
            .filter(TrapImage.monitoring_session_id == self.monitoring_session_id)
            .order_by(TrapImage.timestamp.asc())
            .limit(1)
        ).scalar()
        return img

    def update_source_data(self, session: orm.Session, commit=True):
        img_path = self.absolute_path
        self.width, self.height = get_image_dimensions(img_path)
        self.timestamp = get_image_timestamp(img_path)
        self.filesize = get_image_filesize(img_path)
        self.last_read = datetime.datetime.now()
        session.add(self)
        if commit:
            session.flush()
            session.commit()

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

    @property
    def classified(self):
        """Have all detected objects been classified"""

    def report_data(self) -> CaptureListItem:
        return CaptureListItem(
            id=self.id,
            source_image=f"{constants.IMAGE_BASE_URL}/{self.path}",
            timestamp=self.timestamp,
            last_read=self.last_read,
            last_processed=self.last_processed,
            in_queue=self.in_queue,
            num_detections=self.num_detected_objects,
        )

    def report_detail(self) -> CaptureDetail:
        return CaptureDetail(
            **self.report_data().dict(),
            event=self.monitoring_session.day,
            width=self.width,
            height=self.height,
            filesize=self.filesize,
            detections=[
                obj.report_data_simple().dict() for obj in self.detected_objects
            ],
            notes=self.notes,
        )

    def to_label_studio_task(self, db_path) -> LabelStudioTask:
        return _as_label_studio_task(self, db_path)

    def __repr__(self):
        return (
            f"Image(path={self.path!r}, \n"
            f"\ttimestamp={self.timestamp.strftime('%c') if self.timestamp else None !r}, \n"
            f"\tnum_detected_objects={self.num_detected_objects!r})"
        )


def get_image_with_objects(db_path, image_id):
    with get_session(db_path) as sesh:
        image_kwargs = {
            "id": image_id,
            # "path": str(image_path),
            # "monitoring_session_id": monitoring_session.id,
        }
        image = (
            sesh.query(TrapImage)
            .filter_by(**image_kwargs)
            .options(orm.joinedload(TrapImage.detected_objects))
            .one_or_none()
        )
        # logger.debug(
        #     f"Found image {image} with {len(image.detected_objects)} detected objects"
        # )
        return image


def completely_classified(db_path, image_id):
    from trapdata.db.models.detections import DetectedObject

    with get_session(db_path) as sesh:
        img = sesh.query(TrapImage).get(image_id)
        if not img or not img.last_processed or img.in_queue:
            return False

        else:
            classified_objs = (
                sesh.query(DetectedObject)
                .filter_by(
                    image_id=image_id, binary_label=constants.POSITIVE_BINARY_LABEL
                )
                .filter(
                    DetectedObject.specific_label.is_not(None),
                )
                .count()
            )
            detections = (
                sesh.query(DetectedObject)
                .filter_by(
                    image_id=image_id, binary_label=constants.POSITIVE_BINARY_LABEL
                )
                .count()
            )
            if int(classified_objs) == int(detections):
                return True
            else:
                return False


def get_images_for_labeling(
    db_path, deployment, limit=100, sampling_interval_minutes=5
) -> Generator[TrapImage, None, None]:
    with get_session(db_path) as sesh:
        images = (
            sesh.query(TrapImage)
            # .filter(TrapImage.monitoring_session(deployment=deployment))
            .filter(TrapImage.num_detected_objects > 0)
            .filter(TrapImage.in_queue.is_(False))
            .order_by(TrapImage.timestamp.asc())
            .limit(limit)
            .all()
        )

        # Filter images with python, only select images that have timestamps
        # at least 5 minutes apart (sampling_interval_minutes)
        # Do not consider last_processed time
        # Compare the time diff between the last image and the current image
        # If the time diff is greater than the sampling interval, then select
        # the image.
        last_image = None
        for image in images:
            if not last_image:
                # selected_images.append(image)
                yield image
                last_image = image
            else:
                time_diff = image.timestamp - last_image.timestamp
                if time_diff.total_seconds() / 60 >= sampling_interval_minutes:
                    # selected_images.append(image)
                    yield image
                    last_image = image

        # Same query using select and where clauses (sqlalchemy 1.4 syntax)
        # images = sa.select(TrapImage).where(
        #     TrapImage.monitoring_session.has(deployment=deployment),
        #     TrapImage.num_detected_objects > 0,
        #     TrapImage.in_queue == False,
        #     sa.or_(
        #         TrapImage.last_processed.is_(None),
        #         sa.func.timestampdiff(
        #             sa.text("MINUTE"),
        #             TrapImage.last_processed,
        #             sa.func.now(),
        #         )
        #         > sampling_interval_minutes,
        #     ),
        # )

        return images
