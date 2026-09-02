import os
import pathlib
from unittest import TestCase

import timm
import torch
from fastapi.testclient import TestClient

from trapdata.api.api import (
    CLASSIFIER_CHOICES,
    PipelineChoice,
    PipelineRequest,
    PipelineResponse,
    app,
)
from trapdata.api.models.classification import APIMothClassifier
from trapdata.api.schemas import PipelineConfigRequest, SourceImageRequest
from trapdata.api.tests.image_server import StaticFileTestServer
from trapdata.ml.models.classification import Resnet50TimmClassifier
from trapdata.settings import Settings
from trapdata.tests import TEST_IMAGES_BASE_PATH

# Overridable so these can be run offline against whichever model is already
# cached locally; CI leaves the default.
TEST_PIPELINE = os.environ.get("AMI_TEST_PIPELINE", "global_moths_2024")


class TestFeatureAndLogitsExtractionAPI(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_images_dir = pathlib.Path(TEST_IMAGES_BASE_PATH)
        cls.file_server = StaticFileTestServer(cls.test_images_dir)
        cls.client = TestClient(app)

    @classmethod
    def tearDownClass(cls):
        cls.file_server.stop()

    def get_local_test_images(self, num=1):
        image_paths = [
            "panama/01-20231110214539-snapshot.jpg",
            "panama/01-20231111032659-snapshot.jpg",
            "panama/01-20231111015309-snapshot.jpg",
        ]
        return [
            SourceImageRequest(id=str(i), url=self.file_server.get_url(path))
            for i, path in enumerate(image_paths[:num])
        ]

    def _run_pipeline(
        self,
        include_features: bool = False,
        include_logits: bool = False,
        num_images: int = 1,
    ):
        test_images = self.get_local_test_images(num=num_images)
        config = PipelineConfigRequest(
            include_features=include_features,
            include_logits=include_logits,
        )
        pipeline_request = PipelineRequest(
            pipeline=PipelineChoice[TEST_PIPELINE],
            source_images=test_images,
            config=config,
        )
        with self.file_server:
            response = self.client.post("/process", json=pipeline_request.model_dump())
            self.assertEqual(
                response.status_code, 200, f"Request failed: {response.text}"
            )
            return PipelineResponse(**response.json())

    def test_features_included_when_enabled(self):
        """Features are present and valid when include_features=True."""
        result = self._run_pipeline(include_features=True)
        self.assertTrue(result.detections, "No detections returned")
        for detection in result.detections:
            for classification in detection.classifications:
                if classification.terminal:
                    self.assertIsNotNone(
                        classification.features,
                        "Features should not be None when enabled",
                    )
                    self.assertIsInstance(classification.features, list)
                    self.assertTrue(
                        all(isinstance(x, float) for x in classification.features)
                    )
                    self.assertEqual(len(classification.features), 2048)

    def test_features_absent_when_disabled(self):
        """Features are None when include_features=False (default)."""
        result = self._run_pipeline(include_features=False)
        self.assertTrue(result.detections, "No detections returned")
        for detection in result.detections:
            for classification in detection.classifications:
                self.assertIsNone(
                    classification.features,
                    "Features should be None when disabled",
                )

    def test_logits_included_when_enabled(self):
        """Logits are present when include_logits=True."""
        result = self._run_pipeline(include_logits=True)
        self.assertTrue(result.detections, "No detections returned")
        for detection in result.detections:
            for classification in detection.classifications:
                if classification.terminal:
                    self.assertIsNotNone(
                        classification.logits,
                        "Logits should not be None when enabled",
                    )
                    self.assertIsInstance(classification.logits, list)
                    self.assertTrue(
                        all(isinstance(x, float) for x in classification.logits)
                    )

    def test_logits_absent_when_disabled(self):
        """Logits are None when include_logits=False (default)."""
        result = self._run_pipeline(include_logits=False)
        self.assertTrue(result.detections, "No detections returned")
        for detection in result.detections:
            for classification in detection.classifications:
                self.assertIsNone(
                    classification.logits,
                    "Logits should be None when disabled",
                )

    def test_both_features_and_logits(self):
        """Both features and logits present when both flags enabled."""
        result = self._run_pipeline(include_features=True, include_logits=True)
        self.assertTrue(result.detections, "No detections returned")
        for detection in result.detections:
            for classification in detection.classifications:
                if classification.terminal:
                    self.assertIsNotNone(classification.features)
                    self.assertIsNotNone(classification.logits)

    def test_feature_vectors_are_meaningful(self):
        """Verify features are non-trivial: non-zero, varying, and deterministic."""
        result = self._run_pipeline(include_features=True)
        self.assertTrue(result.detections, "No detections returned")

        terminal_features = [
            c.features
            for d in result.detections
            for c in d.classifications
            if c.terminal and c.features is not None
        ]
        self.assertGreaterEqual(
            len(terminal_features), 1, "Need at least one feature vector"
        )

        for features in terminal_features:
            # Features should not be all zeros
            self.assertFalse(
                all(v == 0.0 for v in features),
                "Feature vector is all zeros — model may not be extracting properly",
            )
            # Features should have some variance (not a constant vector)
            unique_values = set(features)
            self.assertGreater(
                len(unique_values),
                10,
                "Feature vector has too few unique values — likely degenerate",
            )

        # If multiple detections, features should differ between them
        if len(terminal_features) >= 2:
            self.assertNotEqual(
                terminal_features[0],
                terminal_features[1],
                "Different detections produced identical features",
            )


class _RandomWeightTimmClassifier(Resnet50TimmClassifier):
    """A Resnet50TimmClassifier holding random weights, so the tests below run offline.

    Bypasses the normal constructor, which would download and load a checkpoint;
    only ``self.model`` and ``self.device`` matter to the methods under test.
    """

    name = "Random-weight Resnet50 Timm Classifier"

    def __init__(self, num_classes: int = 8):
        self.num_classes = num_classes
        self.device = "cpu"
        self.model = timm.create_model(
            "resnet50", pretrained=False, num_classes=num_classes
        )
        self.model.eval()


class _StubAPIClassifier(APIMothClassifier, _RandomWeightTimmClassifier):
    """Drives the predict/post-process seam without the API or a real checkpoint."""

    def __init__(self, num_classes: int = 8, **kwargs):
        _RandomWeightTimmClassifier.__init__(self, num_classes=num_classes)
        self.category_map = {i: f"species_{i}" for i in range(num_classes)}
        self.source_images = []
        self.detections = []
        self.terminal = True
        self.results = []
        self._last_features = None
        self.include_features = kwargs.get("include_features", False)
        self.include_logits = kwargs.get("include_logits", True)


class TestFeatureExtractionMechanics(TestCase):
    """Offline tests for how features are extracted, kept separate from the
    pipeline tests above because these need no model download."""

    def _count_backbone_passes(self, classifier, batch):
        """Run one batch, returning how many times the backbone was invoked."""
        calls = {"n": 0}
        original = classifier.model.forward_features

        def counting_forward_features(x, *args, **kwargs):
            calls["n"] += 1
            return original(x, *args, **kwargs)

        classifier.model.forward_features = counting_forward_features
        try:
            logits = classifier.predict_batch(batch)
        finally:
            classifier.model.forward_features = original
        return calls["n"], logits

    def test_backbone_runs_once_when_features_are_requested(self):
        """Features must come from the classification pass, not a second one.

        Extracting them from the input separately would run the backbone twice
        and halve throughput on exactly the configuration tracking needs.
        """
        batch = torch.randn(2, 3, 128, 128)

        with_features = _StubAPIClassifier(include_features=True)
        passes, _ = self._count_backbone_passes(with_features, batch)
        self.assertEqual(passes, 1, "Backbone ran more than once with features on")

        without_features = _StubAPIClassifier(include_features=False)
        passes, _ = self._count_backbone_passes(without_features, batch)
        self.assertEqual(passes, 1, "Backbone ran more than once with features off")

    def test_single_pass_logits_match_a_plain_forward(self):
        """Splitting the forward pass must not change the predictions."""
        classifier = _RandomWeightTimmClassifier()
        batch = torch.randn(2, 3, 128, 128)
        with torch.no_grad():
            expected = classifier.model(batch)
        actual, features = classifier.forward_with_features(batch)
        self.assertTrue(
            torch.allclose(expected, actual, atol=1e-6),
            f"Logits diverged, max abs diff {(expected - actual).abs().max().item()}",
        )
        self.assertEqual(tuple(features.shape), (2, 2048))

    def test_post_process_batch_without_a_preceding_predict(self):
        """``_last_features`` is initialised, so this order does not raise."""
        classifier = _StubAPIClassifier(include_features=True)
        results = classifier.post_process_batch(torch.randn(2, 8))
        self.assertEqual(len(results), 2)
        self.assertTrue(all(r.features is None for r in results))

    def test_features_reach_results_through_the_seam(self):
        """predict_batch hands features to post_process_batch on the instance.

        The two are called separately by the base class ``run()``, so the
        hand-off is the part that can silently break.
        """
        classifier = _StubAPIClassifier(include_features=True)
        batch = torch.randn(2, 3, 128, 128)
        results = classifier.post_process_batch(classifier.predict_batch(batch))
        self.assertEqual(len(results), 2)
        for result in results:
            self.assertIsNotNone(result.features)
            self.assertEqual(len(result.features), 2048)
        self.assertIsNone(
            classifier._last_features, "Features were not released after the batch"
        )

    def test_features_are_omitted_unless_requested(self):
        classifier = _StubAPIClassifier(include_features=False)
        batch = torch.randn(2, 3, 128, 128)
        results = classifier.post_process_batch(classifier.predict_batch(batch))
        self.assertTrue(all(r.features is None for r in results))

    def test_logits_are_omitted_only_when_switched_off(self):
        batch = torch.randn(2, 3, 128, 128)

        on = _StubAPIClassifier(include_logits=True)
        results = on.post_process_batch(on.predict_batch(batch))
        self.assertTrue(all(r.logit is not None and len(r.logit) == 8 for r in results))

        off = _StubAPIClassifier(include_logits=False)
        results = off.post_process_batch(off.predict_batch(batch))
        self.assertTrue(all(r.logit is None for r in results))

    def test_logits_stay_on_by_default(self):
        """Consumers re-score classifications from logits, so the default must
        not change without a deliberate decision."""
        self.assertTrue(PipelineConfigRequest().include_logits)
        self.assertTrue(Settings().include_logits)
        self.assertTrue(_StubAPIClassifier().include_logits)

    def test_features_stay_off_by_default(self):
        """Features are large, so they are opt-in."""
        self.assertFalse(PipelineConfigRequest().include_features)
        self.assertFalse(Settings().include_features)

    def test_supports_features_reports_which_pipelines_can_extract(self):
        """A model without a backbone hook returns no features however it is asked.

        ``supports_features()`` is what lets a caller tell that apart from
        features simply being switched off.
        """
        supported = {
            key for key, cls in CLASSIFIER_CHOICES.items() if cls.supports_features()
        }
        self.assertEqual(
            supported,
            {
                "global_moths_2024",
                "panama_moths_2024",
                "quebec_vermont_moths_2023",
                "uk_denmark_moths_2023",
            },
        )
        self.assertFalse(CLASSIFIER_CHOICES["moth_binary"].supports_features())
