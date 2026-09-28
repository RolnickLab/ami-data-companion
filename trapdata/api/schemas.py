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
    logits: list[float] | None = pydantic.Field(
        default=None,
        description=(
            "Raw logits (unnormalized model outputs) for each class. "
            "Omitted when include_logits=false in the pipeline config."
        ),
        repr=False,
    )
    features: list[float] | None = pydantic.Field(
        default=None,
        description=(
            "Feature vector (embedding) extracted from the model backbone before "
            "the classification head. Only included when include_features=true in "
            "the pipeline config."
        ),
        repr=False,
    )
    inference_time: float | None = None
    algorithm: AlgorithmReference
    terminal: bool = True
    timestamp: datetime.datetime


class EmbeddingResponse(pydantic.BaseModel):
    """A feature vector for one detection and the algorithm whose backbone made it.

    It sits on the detection rather than on a classification, so storing it cannot
    add a prediction. Vectors are only comparable with vectors from the same algorithm.
    """

    features: list[float] = pydantic.Field(
        description=(
            "Feature vector (embedding) from the model backbone, before the "
            "classification head."
        ),
        repr=False,
    )
    algorithm: AlgorithmReference


class DetectionResponse(pydantic.BaseModel):
    source_image_id: str
    bbox: BoundingBox
    inference_time: float | None = None
    algorithm: AlgorithmReference
    timestamp: datetime.datetime
    crop_image_url: str | None = None
    classifications: list[ClassificationResponse] = []
    embeddings: list[EmbeddingResponse] | None = pydantic.Field(
        default=None,
        description=(
            "Feature vectors for this detection, at most one per algorithm. Only "
            "included when features_for_all_detections is on, and then every "
            "detection has one, including those the moth/non-moth filter rejected."
        ),
    )


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


class DetectionRequest(pydantic.BaseModel):
    """A detection that already exists, sent back so a pipeline can reuse its box."""

    model_config = pydantic.ConfigDict(extra="ignore")

    source_image: SourceImageRequest
    bbox: BoundingBox | None = None
    crop_image_url: str | None = None
    algorithm: AlgorithmReference = pydantic.Field(
        description=(
            "The algorithm that made this detection. It is returned unchanged, so the "
            "caller can match each response detection to the one it sent."
        ),
    )


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
    include_features: bool = pydantic.Field(
        default=False,
        description=(
            "Whether to include feature vectors (embeddings) in classification "
            "responses. Feature vectors are 2048-dim floats extracted from the "
            "model backbone. Disabled by default to reduce response size."
        ),
    )
    include_logits: bool = pydantic.Field(
        default=True,
        description=(
            "Whether to include raw logits in classification responses. "
            "Logits are the unnormalized model outputs before softmax. "
            "On by default: downstream consumers re-score classifications from "
            "them. Turn it off to reduce response size."
        ),
    )
    features_for_all_detections: bool | None = pydantic.Field(
        default=None,
        description=(
            "Whether to attach a feature vector from the species classifier's "
            "backbone to every detection, as an item in `embeddings`. Detections "
            "the moth/non-moth filter rejected get the vector but no species "
            "classification, which costs one more backbone pass each. When "
            "omitted, the service's AMI_FEATURES_FOR_ALL_DETECTIONS setting applies."
        ),
    )
    embedding_extractor: str | None = pydantic.Field(
        default=None,
        description=(
            "Key of a feature extractor that attaches an embedding to every "
            "detection, as an item in `embeddings`, in addition to any other vector. "
            "It adds no classification. An empty string turns it off. When omitted, "
            "the service's AMI_EMBEDDING_EXTRACTOR setting applies. Only the "
            "extractor that the pipeline lists in /info can be requested; any other "
            "key returns HTTP 422. Feature-only pipelines ignore this field."
        ),
        examples=["bioclip_2_5_embeddings"],
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

    detections: list[DetectionRequest] | None = pydantic.Field(
        default=None,
        description=(
            "Existing detections. Only feature-only pipelines use them: each one with "
            "a bounding box is embedded as it is, and the detector does not run. "
            "Other pipelines ignore them."
        ),
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
