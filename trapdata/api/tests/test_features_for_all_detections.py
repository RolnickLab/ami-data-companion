"""Feature vectors on every detection (``features_for_all_detections``).

Antenna's tracking and similarity search compare vectors, and only vectors from one
feature extractor are comparable. With the setting on, every detection must carry
one vector from the species classifier's backbone, including crops the moth/non-moth
filter rejected, and those crops must gain no classification: a species label on
them would compete with the filter's label for the occurrence's determination.
"""

import os
import pathlib
from unittest import TestCase
from unittest.mock import patch

import torch
from fastapi.testclient import TestClient

from trapdata.api import api
from trapdata.api.api import (
    CLASSIFIER_CHOICES,
    PipelineChoice,
    PipelineRequest,
    PipelineResponse,
    app,
    make_pipeline_config_response,
)
from trapdata.api.models.classification import MothClassifierBinary
from trapdata.api.schemas import (
    AlgorithmReference,
    BoundingBox,
    DetectionResponse,
    PipelineConfigRequest,
    SourceImageRequest,
)
from trapdata.api.tests.image_server import StaticFileTestServer
from trapdata.api.tests.test_features_extraction import _StubAPIClassifier
from trapdata.settings import Settings
from trapdata.tests import TEST_IMAGES_BASE_PATH

TEST_PIPELINE = os.environ.get("AMI_TEST_PIPELINE", "global_moths_2024")
# The binary filter rejects 3 of the 35 detections in this frame.
IMAGE_WITH_REJECTED_CROPS = "panama/01-20231110214539-snapshot.jpg"
FEATURE_DIM = 2048


def _box(detection: DetectionResponse) -> str:
    return detection.bbox.model_dump_json()


def _is_rejected(detection: DetectionResponse) -> bool:
    return not any(c.terminal for c in detection.classifications)


def _classifications_without_timing(detection: DetectionResponse) -> list[dict]:
    return [
        c.model_dump(exclude={"inference_time", "timestamp", "features", "logits"})
        for c in detection.classifications
    ]


class TestFeaturesForAllDetectionsAPI(TestCase):
    """Runs the real pipeline once per configuration and checks the responses."""

    @classmethod
    def setUpClass(cls):
        cls.file_server = StaticFileTestServer(pathlib.Path(TEST_IMAGES_BASE_PATH))
        cls.client = TestClient(app)
        cls.species_key = CLASSIFIER_CHOICES[TEST_PIPELINE].get_key()
        cls.off = cls._run(PipelineConfigRequest())
        cls.on = cls._run(PipelineConfigRequest(features_for_all_detections=True))
        cls.on_with_features = cls._run(
            PipelineConfigRequest(
                features_for_all_detections=True, include_features=True
            )
        )
        with patch.object(api.settings, "features_for_all_detections", True):
            cls.on_from_setting = cls._run(PipelineConfigRequest())

    @classmethod
    def tearDownClass(cls):
        cls.file_server.stop()

    @classmethod
    def _run(cls, config: PipelineConfigRequest) -> PipelineResponse:
        request = PipelineRequest(
            pipeline=PipelineChoice[TEST_PIPELINE],
            source_images=[
                SourceImageRequest(
                    id="0", url=cls.file_server.get_url(IMAGE_WITH_REJECTED_CROPS)
                )
            ],
            config=config,
        )
        with cls.file_server:
            response = cls.client.post("/process", json=request.model_dump())
        assert response.status_code == 200, response.text
        return PipelineResponse(**response.json())

    def test_fixture_includes_crops_the_filter_rejected(self):
        """Guards the other tests against passing without a rejected crop to check."""
        rejected = [d for d in self.on.detections if _is_rejected(d)]
        self.assertGreater(len(rejected), 0)
        self.assertLess(len(rejected), len(self.on.detections))

    def test_every_detection_has_one_vector_from_the_species_classifier(self):
        for response in (self.on, self.on_from_setting):
            for detection in response.detections:
                self.assertIsNotNone(detection.embeddings)
                self.assertEqual(len(detection.embeddings), 1)
                embedding = detection.embeddings[0]
                self.assertEqual(embedding.algorithm.key, self.species_key)
                self.assertNotEqual(
                    embedding.algorithm.key, MothClassifierBinary.get_key()
                )
                self.assertEqual(len(embedding.features), FEATURE_DIM)

    def test_rejected_crops_gain_no_classification(self):
        """Their only classification stays the filter's non-terminal label, unchanged."""
        off_by_box = {_box(d): d for d in self.off.detections}
        for detection in self.on.detections:
            if not _is_rejected(detection):
                continue
            self.assertEqual(len(detection.classifications), 1)
            (classification,) = detection.classifications
            self.assertEqual(
                classification.algorithm.key, MothClassifierBinary.get_key()
            )
            self.assertFalse(classification.terminal)
            self.assertEqual(
                _classifications_without_timing(detection),
                _classifications_without_timing(off_by_box[_box(detection)]),
            )

    def test_vector_on_a_moth_crop_is_its_classification_feature_vector(self):
        """Both come from the same forward pass of the same model."""
        moth_crops = [
            d for d in self.on_with_features.detections if not _is_rejected(d)
        ]
        self.assertTrue(moth_crops)
        for detection in moth_crops:
            (terminal,) = [c for c in detection.classifications if c.terminal]
            self.assertEqual(
                terminal.algorithm.key, detection.embeddings[0].algorithm.key
            )
            self.assertEqual(terminal.features, detection.embeddings[0].features)

    def test_setting_off_leaves_the_response_unchanged(self):
        """The same detections and classifications, and no embeddings at all."""
        self.assertTrue(all(d.embeddings is None for d in self.off.detections))
        self.assertEqual(
            sorted(_box(d) for d in self.off.detections),
            sorted(_box(d) for d in self.on.detections),
        )
        on_by_box = {_box(d): d for d in self.on.detections}
        for detection in self.off.detections:
            twin = on_by_box[_box(detection)]
            self.assertEqual(
                [
                    (c.algorithm.key, c.classification, c.terminal)
                    for c in detection.classifications
                ],
                [
                    (c.algorithm.key, c.classification, c.terminal)
                    for c in twin.classifications
                ],
            )
            for c_off, c_on in zip(detection.classifications, twin.classifications):
                self.assertIsNone(c_on.features)
                max_diff = max(abs(a - b) for a, b in zip(c_off.scores, c_on.scores))
                self.assertLess(max_diff, 1e-4)

    def test_vector_algorithm_is_advertised_for_the_pipeline(self):
        """Antenna rejects results naming an algorithm the pipeline did not advertise."""
        config = make_pipeline_config_response(
            CLASSIFIER_CHOICES[TEST_PIPELINE], TEST_PIPELINE
        )
        advertised = {algorithm.key for algorithm in config.algorithms}
        used = {e.algorithm.key for d in self.on.detections for e in d.embeddings}
        self.assertTrue(used <= advertised, f"{used - advertised} not advertised")


class TestEmbeddingMechanics(TestCase):
    """Offline checks of how the classifier attaches embeddings (no model download)."""

    def _classifier_with_two_detections(self, **kwargs):
        classifier = _StubAPIClassifier(**kwargs)
        classifier.detections = [
            DetectionResponse(
                source_image_id="img",
                bbox=BoundingBox(x1=0, y1=0, x2=10, y2=10 + i),
                algorithm=AlgorithmReference(name="detector", key="detector"),
                timestamp="2026-01-01T00:00:00",
            )
            for i in range(2)
        ]
        batch = torch.randn(2, 3, 128, 128)
        results = classifier.post_process_batch(classifier.predict_batch(batch))
        return classifier, results

    def test_embedding_only_detection_gets_a_vector_and_no_classification(self):
        classifier, results = self._classifier_with_two_detections(
            include_embeddings=True
        )
        classified = classifier.update_detection_classification(0, "img", 0, results[0])
        embedded = classifier.update_detection_embedding("img", 1, results[1])

        self.assertEqual(len(classified.classifications), 1)
        self.assertIsNone(classified.classifications[0].features)
        self.assertEqual(len(classified.embeddings[0].features), FEATURE_DIM)

        self.assertEqual(embedded.classifications, [])
        self.assertEqual(len(embedded.embeddings), 1)
        self.assertEqual(embedded.embeddings[0].algorithm.key, classifier.get_key())

    def test_embedding_only_batches_skip_the_class_scores(self):
        """Rejected crops need only the vector; building every class score for them
        would cost far more than the backbone pass itself."""
        classifier = _StubAPIClassifier(include_embeddings=True)
        with classifier.embedding_only():
            results = classifier.post_process_batch(
                classifier.predict_batch(torch.randn(2, 3, 128, 128))
            )
        self.assertFalse(classifier._embedding_only)
        for result in results:
            self.assertEqual(result.scores, [])
            self.assertIsNone(result.logit)
            self.assertEqual(len(result.features), FEATURE_DIM)

    def test_a_second_vector_from_the_same_model_replaces_the_first(self):
        classifier, results = self._classifier_with_two_detections(
            include_embeddings=True
        )
        classifier.update_detection_embedding("img", 0, results[0])
        detection = classifier.update_detection_embedding("img", 0, results[1])
        self.assertEqual(len(detection.embeddings), 1)
        self.assertEqual(detection.embeddings[0].features, results[1].features)

    def test_no_embeddings_unless_requested(self):
        classifier, results = self._classifier_with_two_detections()
        self.assertFalse(classifier.produces_embeddings)
        detection = classifier.update_detection_classification(0, "img", 0, results[0])
        self.assertIsNone(detection.embeddings)
        with self.assertRaises(ValueError):
            classifier.update_detection_embedding("img", 1, results[1])
        with self.assertRaises(ValueError):
            classifier.embed(classifier.detections)

    def test_setting_is_off_by_default(self):
        self.assertFalse(Settings().features_for_all_detections)
        self.assertIsNone(PipelineConfigRequest().features_for_all_detections)
