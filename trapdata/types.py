import os
import uuid
import datetime
import pathlib
from typing import Optional, Literal
import enum

from pydantic import BaseModel


# These are types! Some will be one-to-one with tables, others not
# The table versions can be normalized or denormalized for performance

# This follows Camera Trap DP data standard where possible
# https://tdwg.github.io/camtrap-dp/data/


class Taxon(BaseModel):
    uuid: uuid.UUID
    gbif: int
    feature_store_id: int
    species: str
    genus: str
    family: str


class QueueStatus(enum.Enum):
    none = None
    waiting = "waiting"
    in_process = "in_process"
    done = "done"


class ClassificationMethod(BaseModel):
    name: str
    version: float
    category: Literal["machine", "human", "truth"]


class TableBase(BaseModel):
    id: int
    created: datetime.datetime
    modified: datetime.datetime


class Location(BaseModel):
    name: str
    lat: float = None
    lon: float = None


class DeviceDeployment(BaseModel):
    name: str
    location: Location
    device: int = None
    events: list["CaptureSequence"] = []


class Capturer:
    # human or camera trap
    pass


class Survey:
    # walking survey, nightly capture session, etc
    pass


class CaptureSequence(BaseModel):
    # Night of capture / Monitoring Session
    name: str
    order: int = 0
    # date_start: datetime.date
    # date_end: datetime.date
    deployment: DeviceDeployment
    images: list["SourceImage"] = []

    def date_start(self):
        if self.images:
            return self.images[0].timestamp

    def date_end(self):
        if self.images:
            return self.images[-1].timestamp


class SourceImage(BaseModel):
    # Original images from camera trap
    event: CaptureSequence
    deployment: DeviceDeployment
    path: pathlib.Path
    timestamp: datetime.datetime = None
    # filename: str  # Can get this witr

    notes: str = None
    # features: object


class BoundingBox(BaseModel):
    top_left: float
    top_right: float
    bottom_left: float
    bottom_right: float
    method: ClassificationMethod


class Occurrence(BaseModel):
    # Every bounding box image
    source: SourceImage = None
    bbox: BoundingBox = None
    path: pathlib.Path = None
    classifications: list["Classification"] = []
    individual: "IndividualOrganism" = None
    timestamp: datetime.datetime = None


class IndividualOrganism(BaseModel):
    uuid: uuid.UUID
    occurrence: list[Occurrence]
    method: ClassificationMethod  # Some tracks will be human drawn / ground truths
    taxon: Taxon


class Classification(BaseModel):
    taxon: Taxon
    timestamp: datetime.datetime
    method: ClassificationMethod
    # category: Literal["detection", "binary_classification", "taxon_classification"]
    score: float


backyard = Location(name="Michael's Backyard", lon=-123, lat=45)
trap = DeviceDeployment(name="Mihow 1", location=backyard)

monday = CaptureSequence(name="Monday", deployment=trap)
tuesday = CaptureSequence(name="Tuesday", deployment=trap)

monday_images = [
    SourceImage(
        path=pathlib.Path(f"./img{n}.jpg"),
        deployment=monday.deployment,
        event=monday,
    )
    for n in range(4)
]


print(trap)
print(monday)
