"""BioCLIP embeddings for every detection, and embedding existing detections.

Tracking compares vectors from one feature extractor across captures, so every
detection needs one, whatever the moth/non-moth filter made of it, and none may gain a
classification from it. A feature-only pipeline must also embed boxes that already
exist without re-detecting, or its vectors could not be matched back to them.

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
    classifier_pipelines,
    make_pipeline_config_response,
)
from trapdata.api.models.feature_extraction import APIBioCLIP25FeatureExtractor
from trapdata.api.schemas import (
    AlgorithmReference,
    BoundingBox,
    DetectionRequest,
    PipelineConfigRequest,
    SourceImageRequest,
)
from trapdata.api.tests.image_server import StaticFileTestServer
from trapdata.ml.models.bioclip import BioCLIPFeatureExtractor
from trapdata.tests import TEST_IMAGES_BASE_PATH

EXTRACTOR_KEY = "bioclip_2_5_embeddings"
FEATURE_PIPELINE = "bioclip_2_5_features"
DIM = APIBioCLIP25FeatureExtractor.embedding_dim
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
        # The service advertises the extractor, so a request may turn it on or off.
        cls.setting = patch.object(api.settings, "embedding_extractor", EXTRACTOR_KEY)
        cls.setting.start()
        cls.off = cls._run(PipelineConfigRequest(embedding_extractor=""))
        cls.on = cls._run(PipelineConfigRequest(embedding_extractor=EXTRACTOR_KEY))

    @classmethod
    def tearDownClass(cls):
        cls.setting.stop()
        super().tearDownClass()

    @classmethod
    def _run(cls, config: PipelineConfigRequest) -> PipelineResponse:
        request = PipelineRequest(
            pipeline=PipelineChoice[CLASSIFIER_PIPELINE],
            source_images=[cls.image()],
            config=config,
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

    def test_unknown_extractor_is_rejected(self):
        request = PipelineRequest(
            pipeline=PipelineChoice[CLASSIFIER_PIPELINE],
            source_images=[self.image()],
            config=PipelineConfigRequest(embedding_extractor="not_an_extractor"),
        )
        response = self.post(request)
        self.assertEqual(response.status_code, 422)
        self.assertIn(EXTRACTOR_KEY, response.text)


    def test_an_extractor_the_pipeline_does_not_advertise_is_rejected(self):
        request = PipelineRequest(
            pipeline=PipelineChoice[CLASSIFIER_PIPELINE],
            source_images=[self.image()],
            config=PipelineConfigRequest(embedding_extractor=EXTRACTOR_KEY),
        )
        with patch.object(api.settings, "embedding_extractor", ""):
            response = self.post(request)
        self.assertEqual(response.status_code, 422)
        self.assertIn("not offered", response.text)


class TestFeatureOnlyPipeline(_APITestCase):
    BOXES = [
        BoundingBox(x1=10, y1=20, x2=110, y2=140),
        BoundingBox(x1=300, y1=200, x2=380, y2=260),
    ]
    DETECTOR = AlgorithmReference(name="Some earlier detector", key="earlier_detector")

    def _request(self, detections, source_images=None) -> PipelineRequest:
        return PipelineRequest(
            pipeline=PipelineChoice[FEATURE_PIPELINE],
            source_images=source_images or [self.image()],
            detections=detections,
        )

    def _detection(self, bbox, image_id="0") -> DetectionRequest:
        return DetectionRequest(
            source_image=self.image(image_id), bbox=bbox, algorithm=self.DETECTOR
        )

    def test_embeds_the_given_boxes_without_detecting_or_classifying(self):
        detections = [self._detection(bbox) for bbox in self.BOXES]
        detections.append(
            DetectionRequest(source_image=self.image(), algorithm=self.DETECTOR)
        )
        with patch.object(
            api.APIMothDetector, "run", side_effect=AssertionError("detector ran")
        ):
            response = self.post(self._request(detections))

        self.assertEqual(response.status_code, 200, response.text)
        result = PipelineResponse(**response.json())
        # The box-less detection cannot be embedded and is left out.
        self.assertEqual([d.bbox for d in result.detections], self.BOXES)
        for detection in result.detections:
            self.assertEqual(detection.algorithm, self.DETECTOR)
            self.assertEqual(detection.classifications, [])
            self.assertEqual(len(detection.embeddings), 1)
            self.assertEqual(detection.embeddings[0].algorithm.key, EXTRACTOR_KEY)
            self.assertEqual(len(detection.embeddings[0].features), DIM)
        first, second = (d.embeddings[0].features for d in result.detections)
        self.assertNotEqual(first, second)

    def test_skips_boxes_with_no_area(self):
        degenerate = [
            BoundingBox(x1=50, y1=20, x2=50, y2=140),
            BoundingBox(x1=10, y1=140, x2=110, y2=20),
        ]
        detections = [self._detection(b) for b in degenerate + self.BOXES[:1]]
        response = self.post(self._request(detections))

        self.assertEqual(response.status_code, 200, response.text)
        result = PipelineResponse(**response.json())
        self.assertEqual([d.bbox for d in result.detections], self.BOXES[:1])

    def test_adds_the_image_of_a_detection_missing_from_source_images(self):
        request = self._request(
            [self._detection(self.BOXES[0], image_id="other")],
            source_images=[self.image("0")],
        )
        response = self.post(request)

        self.assertEqual(response.status_code, 200, response.text)
        result = PipelineResponse(**response.json())
        self.assertEqual({i.id for i in result.source_images}, {"0", "other"})
        self.assertEqual(result.detections[0].source_image_id, "other")
        self.assertEqual(len(result.detections[0].embeddings), 1)

    def test_without_detections_it_detects_first_and_still_does_not_classify(self):
        response = self.post(self._request(detections=None))

        self.assertEqual(response.status_code, 200, response.text)
        result = PipelineResponse(**response.json())
        self.assertGreater(len(result.detections), 0)
        detector_key = api.APIMothDetector.get_key()
        for detection in result.detections:
            self.assertEqual(detection.algorithm.key, detector_key)
            self.assertEqual(detection.classifications, [])
            self.assertEqual(len(detection.embeddings), 1)


def test_setting_advertises_the_extractor_in_classifier_pipelines(monkeypatch):
    Classifier = api.CLASSIFIER_CHOICES["moth_binary"]

    monkeypatch.setattr(api.settings, "embedding_extractor", "")
    keys = [a.key for a in make_pipeline_config_response(Classifier, "x").algorithms]
    assert EXTRACTOR_KEY not in keys

    monkeypatch.setattr(api.settings, "embedding_extractor", EXTRACTOR_KEY)
    algorithms = make_pipeline_config_response(Classifier, "x").algorithms
    extractor = [a for a in algorithms if a.key == EXTRACTOR_KEY]
    assert len(extractor) == 1
    assert extractor[0].task_type == "embedding"
    assert extractor[0].category_map is None


def test_feature_pipeline_lists_the_detector_and_extractor_only():
    config = make_pipeline_config_response(
        api.FEATURE_PIPELINE_CHOICES[FEATURE_PIPELINE], FEATURE_PIPELINE
    )
    assert [a.task_type for a in config.algorithms] == ["localization", "embedding"]
    assert config.algorithms[1].key == EXTRACTOR_KEY


def test_invalid_setting_is_rejected(monkeypatch):
    monkeypatch.setattr(api.settings, "embedding_extractor", "not_an_extractor")
    with pytest.raises(ValueError, match=EXTRACTOR_KEY):
        api.resolve_embedding_extractor()


def test_worker_skips_feature_only_pipelines():
    assert classifier_pipelines(["moth_binary", FEATURE_PIPELINE]) == ["moth_binary"]


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


def test_worker_registration_never_advertises_the_extractor(monkeypatch):
    # The async worker does not run the extractor, so registering it would make
    # Antenna wait for output that never arrives.
    from trapdata.antenna import registration

    monkeypatch.setattr(api.settings, "embedding_extractor", EXTRACTOR_KEY)
    registered = []
    monkeypatch.setattr(
        registration,
        "register_pipelines_for_project",
        lambda **kwargs: registered.extend(kwargs["pipeline_configs"]) or (True, ""),
    )
    worker_settings = SimpleNamespace(
        antenna_api_base_url="http://antenna.test/api/v2",
        antenna_api_auth_token="token",
        pipelines="moth_binary",
    )
    registration.register_pipelines([1], "test service", worker_settings)

    assert [p.slug for p in registered] == ["moth_binary"]
    task_types = [a.task_type for p in registered for a in p.algorithms]
    assert "embedding" not in task_types
