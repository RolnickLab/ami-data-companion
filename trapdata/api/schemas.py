# Can these be imported from the OpenAPI spec yaml?
import datetime
import pathlib
import typing

import PIL.Image
import pydantic

from trapdata.common.logs import logger
from trapdata.ml.utils import get_image


class BoundingBox(pydantic.BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float

    @classmethod
    def from_coords(cls, coords: list[float]):
        return cls(x1=coords[0], y1=coords[1], x2=coords[2], y2=coords[3])

    def to_string(self):
        return f"{self.x1},{self.y1},{self.x2},{self.y2}"

    def to_path(self):
        return "-".join([str(int(x)) for x in [self.x1, self.y1, self.x2, self.y2]])

    def to_tuple(self):
        return (self.x1, self.y1, self.x2, self.y2)


class SourceImage(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="ignore", arbitrary_types_allowed=True)

    id: str
    url: str | None = None
    b64: str | None = None
    filepath: str | pathlib.Path | None = None
    _pil: PIL.Image.Image | None = None
    width: int | None = None
    height: int | None = None
    timestamp: datetime.datetime | None = None

    # Validate that there is at least one of the following fields
    @pydantic.model_validator(mode="after")
    def validate_source(self):
        if not any([self.url, self.b64, self.filepath, self._pil]):
            raise ValueError(
                "At least one of the following fields must be provided: "
                "url, b64, filepath, pil"
            )
        return self

    def open(self, raise_exception=False) -> PIL.Image.Image | None:
        if not self._pil:
            logger.warn(f"Opening image {self.id} for the first time")
            self._pil = get_image(
                url=self.url,
                b64=self.b64,
                filepath=self.filepath,
                raise_exception=raise_exception,
            )
        else:
            logger.info(f"Using already loaded image {self.id}")
        if self._pil:
            self.width, self.height = self._pil.size
        return self._pil


class AlgorithmReference(pydantic.BaseModel):
    name: str
    key: str


class ClassificationResponse(pydantic.BaseModel):
    classification: str
    labels: list[str] | None = pydantic.Field(
        default=None,
        description=(
            "A list of all possible labels for the model, in the correct order. "
            "Omitted if the model has too many labels to include for each "
            "classification in the response. Use the category map from the algorithm "
            "to get the full list of labels and metadata."
        ),
    )
    scores: list[float] = pydantic.Field(
        default_factory=list,
        description=(
            "The calibrated probabilities for each class label, most commonly "
            "the softmax output."
        ),
    )
    logits: list[float] = pydantic.Field(
        default_factory=list,
        description=(
            "The raw logits output by the model, before any calibration or "
            "normalization."
        ),
    )
    inference_time: float | None = None
    algorithm: AlgorithmReference
    terminal: bool = True
    timestamp: datetime.datetime


class DetectionResponse(pydantic.BaseModel):
    source_image_id: str
    bbox: BoundingBox
    inference_time: float | None = None
    algorithm: AlgorithmReference
    timestamp: datetime.datetime
    crop_image_url: str | None = None
    classifications: list[ClassificationResponse] = []


class SourceImageRequest(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="ignore")

    id: str
    url: str
    # b64: str | None = None
    # @TODO bring over new SourceImage & b64 validation from the lepsAI repo


class SourceImageResponse(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="ignore")

    id: str
    url: str


class AlgorithmCategoryMap(pydantic.BaseModel):
    data: list[dict] = pydantic.Field(
        default_factory=dict,
        description=(
            "Complete data for each label, such as id, gbif_key, explicit index, "
            "source, etc."
        ),
        examples=[
            [
                {"label": "Moth", "index": 0, "gbif_key": 1234},
                {"label": "Not a moth", "index": 1, "gbif_key": 5678},
            ]
        ],
    )
    labels: list[str] = pydantic.Field(
        default_factory=list,
        description=(
            "A simple list of string labels, in the correct index order used by "
            "the model."
        ),
        examples=[["Moth", "Not a moth"]],
    )
    version: str | None = pydantic.Field(
        default=None,
        description=(
            "The version of the category map. Can be a descriptive string or a "
            "version number."
        ),
        examples=["LepNet2021-with-2023-mods"],
    )
    description: str | None = pydantic.Field(
        default=None,
        description=(
            "A description of the category map used to train. e.g. source, "
            "purpose and modifications."
        ),
        examples=[
            "LepNet2021 with Schmidt 2023 corrections. Limited to species with > "
            "1000 observations."
        ],
    )
    url: str | None = None


class AlgorithmResponse(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="ignore")

    name: str
    key: str = pydantic.Field(
        description=(
            "A unique key for an algorithm to lookup the category map (class list) "
            "and other metadata."
        ),
    )
    description: str | None = None
    task_type: str | None = pydantic.Field(
        default=None,
        description=(
            "The type of task the model is trained for. e.g. 'detection', "
            "'classification', 'embedding', etc."
        ),
        examples=["detection", "classification", "segmentation", "embedding"],
    )
    version: int = pydantic.Field(
        default=1,
        description=(
            "A sortable version number for the model. Increment this number when "
            "the model is updated."
        ),
    )
    version_name: str | None = pydantic.Field(
        default=None,
        description="A complete version name e.g. '2021-01-01', 'LepNet2021'.",
    )
    url: str | None = None
    category_map: AlgorithmCategoryMap | None = None


PipelineChoice = typing.Literal[
    "dummy",
]  # @TODO add "random", "dummy",


class PipelineRequest(pydantic.BaseModel):
    pipeline: PipelineChoice = pydantic.Field(
        description=(
            "The pipeline to use for processing the source images, specified by key"
        ),
        examples=["vermont_quebec_moths_2023"],
    )

    source_images: list[SourceImageRequest] = pydantic.Field(
        description="A list of source image URLs to process.",
        examples=[
            [
                {
                    "id": "123",
                    "url": (
                        "https://archive.org/download/"
                        "mma_various_moths_and_butterflies_54143/54143.jpg"
                    ),
                }
            ]
        ],
    )


class PipelineResponse(pydantic.BaseModel):
    pipeline: PipelineChoice
    algorithms: dict[str, AlgorithmResponse] = pydantic.Field(
        default_factory=dict,
        description=(
            "A dictionary of all algorithms used in the pipeline, including their "
            "class list and other metadata, keyed by the algorithm key."
        ),
    )
    total_time: float
    source_images: list[SourceImageResponse]
    detections: list[DetectionResponse]


class PipelineStageParam(pydantic.BaseModel):
    """A configurable parameter of a stage of a pipeline."""

    name: str
    key: str
    category: str = "default"


class PipelineStage(pydantic.BaseModel):
    """A configurable stage of a pipeline."""

    key: str
    name: str
    params: list[PipelineStageParam] = []
    description: str | None = None


class PipelineConfig(pydantic.BaseModel):
    """A configurable pipeline."""

    name: str
    slug: str
    description: str | None = None
    stages: list[PipelineStage] = []
