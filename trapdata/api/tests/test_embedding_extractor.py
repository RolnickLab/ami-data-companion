"""BioCLIP embeddings for every detection.

Tracking compares vectors from one feature extractor across captures, so every
detection needs one, whatever the moth/non-moth filter made of it, and none may gain a
classification from it.

The BioCLIP backbone is replaced by a small deterministic encoder, so these tests never
download it. The detector and classifiers are real, as in the other API tests.
"""

import pathlib
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import pytest
import torch
import torchvision
from fastapi.testclient import TestClient

from trapdata.api import api
from trapdata.api.api import (
    PipelineChoice,
    PipelineRequest,
    PipelineResponse,
    app,
    make_pipeline_config_response,
)
from trapdata.api.models.feature_extraction import APIBioCLIP25FeatureExtractor
from trapdata.api.schemas import SourceImageRequest
from trapdata.api.tests.image_server import StaticFileTestServer
from trapdata.ml.models.bioclip import BioCLIPFeatureExtractor
from trapdata.settings import Settings
from trapdata.tests import TEST_IMAGES_BASE_PATH

EXTRACTOR_KEY = "bioclip_2_5_embeddings"
DIM = APIBioCLIP25FeatureExtractor.embedding_dimensions
# The binary filter rejects some of the detections in this frame.
IMAGE = "panama/01-20231110214539-snapshot.jpg"
CLASSIFIER_PIPELINE = "quebec_vermont_moths_2023"
REAL_LOAD_BACKBONE = BioCLIPFeatureExtractor.load_backbone


class FakeEncoder(torch.nn.Module):
    """Map a crop to a unit vector that depends on its pixels, like the real encoder."""

    def __init__(self):
        super().__init__()
        generator = torch.Generator().manual_seed(0)
        self.projection = torch.randn(3 * 8 * 8, DIM, generator=generator)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        pooled = torch.nn.functional.adaptive_avg_pool2d(images, 8).flatten(1)
        features = pooled @ self.projection.to(images.device)
        return features / features.norm(dim=-1, keepdim=True)


def fake_backbone(self):
    transform = torchvision.transforms.Compose(
        [torchvision.transforms.Resize((32, 32)), torchvision.transforms.ToTensor()]
    )
    return FakeEncoder().to(self.device), transform


@pytest.fixture(autouse=True)
def no_backbone_download(monkeypatch):
    monkeypatch.setattr(BioCLIPFeatureExtractor, "load_backbone", fake_backbone)
    monkeypatch.setattr(BioCLIPFeatureExtractor, "_loaded", {})


class _APITestCase(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.file_server = StaticFileTestServer(pathlib.Path(TEST_IMAGES_BASE_PATH))
        cls.client = TestClient(app)
        cls.patches = [
            patch.object(BioCLIPFeatureExtractor, "load_backbone", fake_backbone),
            patch.object(BioCLIPFeatureExtractor, "_loaded", {}),
        ]
        for p in cls.patches:
            p.start()

    @classmethod
    def tearDownClass(cls):
        for p in cls.patches:
            p.stop()
        cls.file_server.stop()

    @classmethod
    def post(cls, request: PipelineRequest):
        with cls.file_server:
            return cls.client.post("/process", json=request.model_dump(mode="json"))

    @classmethod
    def image(cls, image_id: str = "0") -> SourceImageRequest:
        return SourceImageRequest(id=image_id, url=cls.file_server.get_url(IMAGE))


class TestEmbeddingExtractorInAClassifierPipeline(_APITestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        with patch.object(api.settings, "embedding_extractor", "none"):
            cls.off = cls._run()
        with patch.object(api.settings, "embedding_extractor", EXTRACTOR_KEY):
            cls.on = cls._run()

    @classmethod
    def _run(cls) -> PipelineResponse:
        request = PipelineRequest(
            pipeline=PipelineChoice[CLASSIFIER_PIPELINE],
            source_images=[cls.image()],
        )
        response = cls.post(request)
        assert response.status_code == 200, response.text
        return PipelineResponse(**response.json())

    def test_every_detection_including_rejected_ones_has_one_unit_vector(self):
        rejected = [
            d
            for d in self.on.detections
            if not any(c.terminal for c in d.classifications)
        ]
        self.assertGreater(len(rejected), 0, "fixture must include rejected crops")
        for detection in self.on.detections:
            vectors = [
                e
                for e in detection.embeddings or []
                if e.algorithm.key == EXTRACTOR_KEY
            ]
            self.assertEqual(len(vectors), 1)
            self.assertEqual(len(vectors[0].features), DIM)
            norm = torch.tensor(vectors[0].features).norm().item()
            self.assertAlmostEqual(norm, 1.0, places=4)

    def test_classifications_are_unchanged(self):
        def classifications(response):
            return {
                d.bbox.model_dump_json(): [
                    (c.algorithm.key, c.classification, c.terminal)
                    for c in d.classifications
                ]
                for d in response.detections
            }

        self.assertEqual(classifications(self.on), classifications(self.off))
        self.assertTrue(all(d.embeddings is None for d in self.off.detections))


def test_the_extractor_is_advertised_only_by_a_path_that_runs_it():
    Classifier = api.select_pipelines(["moth_binary"])["moth_binary"]

    keys = [a.key for a in make_pipeline_config_response(Classifier, "x").algorithms]
    assert EXTRACTOR_KEY not in keys

    algorithms = make_pipeline_config_response(
        Classifier, "x", embedding_extractor=APIBioCLIP25FeatureExtractor
    ).algorithms
    extractor = [a for a in algorithms if a.key == EXTRACTOR_KEY]
    assert len(extractor) == 1
    assert extractor[0].task_type == "embedding"
    assert extractor[0].category_map is None
    assert extractor[0].embedding_dimensions == DIM


def test_the_setting_picks_the_extractor(monkeypatch):
    monkeypatch.setattr(api.settings, "embedding_extractor", EXTRACTOR_KEY)
    assert api.resolve_embedding_extractor() is APIBioCLIP25FeatureExtractor
    for off in ("none", "None", ""):
        monkeypatch.setattr(api.settings, "embedding_extractor", off)
        assert api.resolve_embedding_extractor() is None


def test_the_api_runs_bioclip_by_default():
    """The declared default, which conftest.py overrides for the test session."""
    assert Settings.model_fields["embedding_extractor"].default == EXTRACTOR_KEY


def test_invalid_setting_is_rejected(monkeypatch):
    monkeypatch.setattr(api.settings, "embedding_extractor", "not_an_extractor")
    with pytest.raises(ValueError, match=EXTRACTOR_KEY):
        api.resolve_embedding_extractor()


def test_backbone_is_loaded_once_per_device(monkeypatch):
    calls = []

    def counting_backbone(self):
        calls.append(self.device)
        return fake_backbone(self)

    monkeypatch.setattr(BioCLIPFeatureExtractor, "load_backbone", counting_backbone)
    first = APIBioCLIP25FeatureExtractor(source_images=[], device="cpu")
    second = APIBioCLIP25FeatureExtractor(source_images=[], device="cpu")
    assert len(calls) == 1
    assert first.model is second.model


def test_backbone_with_the_wrong_dimension_is_rejected(monkeypatch):
    class Visual:
        output_dim = DIM + 1

    class Model(torch.nn.Module):
        visual = Visual()

    monkeypatch.setattr(BioCLIPFeatureExtractor, "load_backbone", REAL_LOAD_BACKBONE)
    with patch(
        "open_clip.create_model_and_transforms", return_value=(Model(), None, None)
    ):
        with pytest.raises(ValueError, match=str(DIM)):
            APIBioCLIP25FeatureExtractor(source_images=[], device="cpu")


def _register_worker_pipelines(monkeypatch, pipelines: str, include_features: bool):
    from trapdata.antenna import registration

    registered = []
    monkeypatch.setattr(
        registration,
        "register_pipelines_for_project",
        lambda **kwargs: registered.extend(kwargs["pipeline_configs"]) or (True, ""),
    )
    worker_settings = SimpleNamespace(
        antenna_api_base_url="http://antenna.test/api/v2",
        antenna_api_auth_token="token",
        pipelines=pipelines,
        include_features=include_features,
    )
    registration.register_pipelines([1], "test service", worker_settings)
    return registered


def test_worker_registration_never_advertises_the_extractor(monkeypatch):
    # The async worker does not run the extractor, so registering it would make
    # Antenna wait for output that never arrives.
    monkeypatch.setattr(api.settings, "embedding_extractor", EXTRACTOR_KEY)
    registered = _register_worker_pipelines(monkeypatch, "moth_binary", False)

    assert [p.slug for p in registered] == ["moth_binary"]
    task_types = [a.task_type for p in registered for a in p.algorithms]
    assert "embedding" not in task_types


@pytest.mark.parametrize("include_features, declared", [(True, 2048), (False, None)])
def test_worker_registration_declares_the_classifier_vectors_it_sends(
    monkeypatch, include_features, declared
):
    """The worker sends the classifier's vectors exactly when AMI_INCLUDE_FEATURES is
    on (see test_worker.py), so its registration declares a length exactly then."""
    (pipeline,) = _register_worker_pipelines(
        monkeypatch, "global_moths_2024", include_features
    )
    by_task = {a.task_type: a.embedding_dimensions for a in pipeline.algorithms}
    assert by_task["classification"] == declared
    assert by_task["localization"] is None


@pytest.mark.parametrize(
    "classifier_embeddings, declared", [(True, 2048), (False, None)]
)
def test_a_classifier_declares_its_vector_length_only_when_it_sends_vectors(
    classifier_embeddings, declared
):
    """Declared means always sent: Antenna re-sends captures whose detections lack a
    vector from an algorithm that declares a length, so none is declared when off."""
    Classifier = api.select_pipelines(["global_moths_2024"])["global_moths_2024"]
    algorithms = make_pipeline_config_response(
        Classifier, "x", classifier_embeddings=classifier_embeddings
    ).algorithms
    declared_by_name = {a.name: a.embedding_dimensions for a in algorithms}
    assert declared_by_name[Classifier.name] == declared
    localizers = [a for a in algorithms if a.task_type == "localization"]
    assert [a.embedding_dimensions for a in localizers] == [None]
