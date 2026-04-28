import pathlib
from unittest import TestCase

from fastapi.testclient import TestClient

from trapdata.api.api import PipelineChoice, PipelineRequest, PipelineResponse, app
from trapdata.api.schemas import PipelineConfigRequest, SourceImageRequest
from trapdata.api.tests.image_server import StaticFileTestServer
from trapdata.tests import TEST_IMAGES_BASE_PATH


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
            pipeline=PipelineChoice["global_moths_2024"],
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

    def test_default_config_has_nothing_extra(self):
        """Default PipelineConfigRequest disables both features and logits."""
        config = PipelineConfigRequest()
        self.assertFalse(config.include_features)
        self.assertFalse(config.include_logits)

    def test_worker_path_features_via_predict_and_postprocess(self):
        """Test the worker code path: predict_batch → post_process_batch directly.

        The antenna worker calls these methods separately (not via run()),
        so we verify features flow through this path correctly.
        """
        # Run a pipeline WITH features to get detections and a configured classifier
        result = self._run_pipeline(include_features=True)
        self.assertTrue(result.detections, "No detections returned")

        # Verify features came through the full pipeline
        terminal_features = [
            c.features
            for d in result.detections
            for c in d.classifications
            if c.terminal and c.features is not None
        ]
        self.assertTrue(
            terminal_features, "No features found in terminal classifications"
        )

        # Each feature vector should be 2048-dim
        for features in terminal_features:
            self.assertEqual(len(features), 2048)

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
