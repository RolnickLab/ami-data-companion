# Can these be imported from the OpenAPI spec yaml?
import datetime
import pathlib

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
        repr=False,  # Too long to display in the repr
    )
    scores: list[float] = pydantic.Field(
        default_factory=list,
        description=(
            "The calibrated probabilities for each class label, most commonly "
            "the softmax output."
        ),
        repr=False,  # Too long to display in the repr
    )
    logits: list[float] = pydantic.Field(
        default_factory=list,
        description=(
            "The raw logits output by the model, before any calibration or "
            "normalization."
        ),
        repr=False,  # Too long to display in the repr
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

    # @TODO bring over new SourceImage & b64 validation from the lepsAI repo
    id: str = pydantic.Field(
        description=(
            "Unique identifier for the source image. This is returned in the response."
        ),
        examples=["e124f3b4"],
    )
    url: str = pydantic.Field(
        description="URL to the source image to be processed.",
        examples=[
            "https://static.dev.insectai.org/ami-trapdata/"
            "vermont/RawImages/LUNA/2022/movement/2022_06_23/20220623050407-00-235.jpg"
        ],
    )
    # b64: str | None = None


class SourceImageResponse(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="ignore")

    id: str
    url: str


class AlgorithmCategoryMapResponse(pydantic.BaseModel):
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
        repr=False,  # Too long to display in the repr
    )
    labels: list[str] = pydantic.Field(
        default_factory=list,
        description=(
            "A simple list of string labels, in the correct index order used by "
            "the model."
        ),
        examples=[["Moth", "Not a moth"]],
        repr=False,  # Too long to display in the repr
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
    uri: str | None = pydantic.Field(
        default=None,
        description="A URI to the category map file, could be a public web URL or object store path.",
    )


class AlgorithmConfigResponse(pydantic.BaseModel):
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
    uri: str | None = pydantic.Field(
        default=None,
        description="A URI to the weights or model details, could be a public web URL or object store path.",
    )
    category_map: AlgorithmCategoryMapResponse | None = None


class PipelineConfigRequest(pydantic.BaseModel):
    """
    Configuration for the processing pipeline.
    """

    example_config_param: int | None = pydantic.Field(
        default=None,
        description="Example of a configuration parameter for a pipeline.",
        examples=[3],
    )


class PipelineRequest(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(use_enum_values=True)

    pipeline: str = pydantic.Field(
        description=(
            "The pipeline to use for processing the source images, specified by key"
        ),
        examples=["vermont_quebec_moths_2023"],
    )

    source_images: list[SourceImageRequest] = pydantic.Field(
        description="A list of source image URLs to process.",
    )

    config: PipelineConfigRequest = pydantic.Field(
        default=PipelineConfigRequest(),
        examples=[PipelineConfigRequest(example_config_param=3)],
    )


class PipelineResultsResponse(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(use_enum_values=True)

    pipeline: str = pydantic.Field(
        description="The pipeline used for processing, specified by key."
    )
    algorithms: dict[str, AlgorithmConfigResponse] = pydantic.Field(
        default_factory=dict,
        description=(
            "A dictionary of all algorithms used in the pipeline, including their "
            "class list and other metadata, keyed by the algorithm key."
            "DEPRECATED: Use the algorithms list in PipelineConfigResponse instead."
        ),
        deprecated=True,
    )
    total_time: float
    source_images: list[SourceImageResponse]
    detections: list[DetectionResponse]
    config: PipelineConfigRequest = PipelineConfigRequest()


class AntennaPipelineProcessingTask(pydantic.BaseModel):
    """
    A task representing a single image or detection to be processed in an async pipeline.
    """

    id: str
    image_id: str
    image_url: str
    reply_subject: str | None = None  # The NATS subject to send the result to
    # TODO: Do we need these?
    # detections: list[DetectionRequest] | None = None
    # config: PipelineRequestConfigParameters | dict | None = None


class AntennaJobListItem(pydantic.BaseModel):
    """A single job item from the Antenna jobs list API response."""

    id: int


class AntennaJobsListResponse(pydantic.BaseModel):
    """Response from Antenna API GET /api/v2/jobs with ids_only=1."""

    results: list[AntennaJobListItem]


class AntennaTasksListResponse(pydantic.BaseModel):
    """Response from Antenna API GET /api/v2/jobs/{job_id}/tasks."""

    tasks: list[AntennaPipelineProcessingTask]


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


class PipelineConfigResponse(pydantic.BaseModel):
    """Details about a pipeline, its algorithms and category maps."""

    name: str
    slug: str
    version: int
    description: str | None = None
    algorithms: list[AlgorithmConfigResponse] = []
    stages: list[PipelineStage] = []


class AntennaTaskResultError(pydantic.BaseModel):
    """Error result for a single Antenna task that failed to process."""

    error: str
    image_id: str | None = None


class AntennaTaskResult(pydantic.BaseModel):
    """Result for a single Antenna task, either success or error."""

    reply_subject: str | None = None
    result: PipelineResultsResponse | AntennaTaskResultError


class AntennaTaskResults(pydantic.BaseModel):
    """Batch of task results to post back to Antenna API."""

    results: list[AntennaTaskResult] = pydantic.Field(default_factory=list)


class ProcessingServiceInfoResponse(pydantic.BaseModel):
    """Information about the processing service."""

    name: str = pydantic.Field(examples=["Mila Research Lab - Moth AI Services"])
    description: str | None = pydantic.Field(
        default=None,
        examples=[
            "Algorithms developed by the Mila Research Lab for analysis of moth images."
        ],
    )
    pipelines: list[PipelineConfigResponse] = pydantic.Field(
        default=list,
        examples=[
            [
                PipelineConfigResponse(
                    name="Random Pipeline", slug="random", version=1, algorithms=[]
                ),
            ]
        ],
    )


class AsyncPipelineRegistrationRequest(pydantic.BaseModel):
    """
    Request to register pipelines from an async processing service
    """

    processing_service_name: str
    pipelines: list[PipelineConfigResponse] = []
