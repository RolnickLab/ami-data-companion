import logging
import pathlib
from unittest import TestCase

from fastapi.testclient import TestClient

from trapdata.api.api import (
    CLASSIFIER_CHOICES,
    PipelineChoice,
    PipelineRequest,
    PipelineResponse,
    app,
    make_algorithm_response,
    make_pipeline_config_response,
)
from trapdata.api.schemas import PipelineConfigRequest, SourceImageRequest
from trapdata.api.tests.image_server import StaticFileTestServer
from trapdata.ml.models.classification import SpeciesClassifier
from trapdata.tests import TEST_IMAGES_BASE_PATH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestInferenceAPI(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_images_dir = pathlib.Path(TEST_IMAGES_BASE_PATH)
        if not cls.test_images_dir.exists():
            raise FileNotFoundError(
                f"Test images directory not found: {cls.test_images_dir}"
            )
        cls.file_server = StaticFileTestServer(cls.test_images_dir)
        cls.client = TestClient(app)

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "file_server"):
            cls.file_server.stop()

    def get_test_images(self, subdir: str = "vermont", num: int = 2):
        images_dir = self.test_images_dir / subdir
        source_image_urls = [
            self.file_server.get_url(f.relative_to(images_dir))
            for f in self.test_images_dir.glob("*.jpg")
        ][:num]
        source_images = [
            SourceImageRequest(id=str(i), url=url)
            for i, url in enumerate(source_image_urls)
        ]
        return source_images

    def get_test_pipeline(
        self, slug: str = "quebec_vermont_moths_2023"
    ) -> SpeciesClassifier:
        pipeline = CLASSIFIER_CHOICES[slug]
        return pipeline

    def test_pipeline_request(self):
        """
        Ensure that the pipeline accepts a valid request and returns a valid response.
        """
        pipeline_request = PipelineRequest(
            pipeline=PipelineChoice["quebec_vermont_moths_2023"],
            source_images=self.get_test_images(num=2),
        )
        with self.file_server:
            response = self.client.post("/process", json=pipeline_request.model_dump())
            assert response.status_code == 200
            results = PipelineResponse(**response.json())
        return results

    def test_config_num_classification_predictions(self):
        """
        Test that the pipeline respects the `max_predictions_per_classification`
        configuration.

        If the configuration is set to a number, the pipeline should return that number
        of labels/scores per prediction.
        If the configuration is set to `None`, the pipeline should return all
        labels/scores per prediction.
        """
        test_images = self.get_test_images(num=1)
        test_pipeline_slug = "quebec_vermont_moths_2023"
        terminal_classifier = self.get_test_pipeline(test_pipeline_slug)

        def _send_request(max_predictions_per_classification: int | None):
            config = PipelineConfigRequest(
                example_config_param=max_predictions_per_classification
            )
            pipeline_request = PipelineRequest(
                pipeline=PipelineChoice[test_pipeline_slug],
                source_images=test_images,
                config=config,
            )
            with self.file_server:
                response = self.client.post(
                    "/pipeline/process", json=pipeline_request.model_dump()
                )
            assert response.status_code == 200
            pipeline_response = PipelineResponse(**response.json())
            terminal_classifications = [
                classification
                for detection in pipeline_response.detections
                for classification in detection.classifications
                if classification.terminal
            ]
            for classification in terminal_classifications:
                assert classification.labels
                if max_predictions_per_classification is None:
                    # Ensure that a score is returned for every possible class
                    assert len(classification.labels) == terminal_classifier.num_classes
                    assert len(classification.scores) == terminal_classifier.num_classes
                else:
                    # Ensure that the number of predictions is limited to the number
                    # specified.
                    # There may be fewer predictions than the number specified if there
                    # are fewer classes.
                    assert (
                        len(classification.labels) <= max_predictions_per_classification
                    )
                    assert (
                        len(classification.scores) <= max_predictions_per_classification
                    )

        _send_request(max_predictions_per_classification=1)
        _send_request(max_predictions_per_classification=None)

    def test_pipeline_config_with_binary_classifier(self):
        BinaryClassifier = CLASSIFIER_CHOICES["moth_binary"]
        BinaryClassifierResponse = make_algorithm_response(BinaryClassifier)

        SpeciesClassifier = CLASSIFIER_CHOICES["quebec_vermont_moths_2023"]
        SpeciesClassifierResponse = make_algorithm_response(SpeciesClassifier)

        # Test using a pipeline that finishes with a full species classifier
        pipeline_config = make_pipeline_config_response(SpeciesClassifier)

        self.assertEqual(len(pipeline_config.algorithms), 3)
        self.assertEqual(
            pipeline_config.algorithms[-1].key, SpeciesClassifierResponse.key
        )
        self.assertEqual(
            pipeline_config.algorithms[1].key, BinaryClassifierResponse.key
        )

        # Test using a pipeline that finishes only with a binary classifier
        pipeline_config_binary_only = make_pipeline_config_response(BinaryClassifier)

        self.assertEqual(len(pipeline_config_binary_only.algorithms), 2)
        self.assertEqual(
            pipeline_config_binary_only.algorithms[-1].key, BinaryClassifierResponse.key
        )
        # self.assertTrue(pipeline_config_binary_only.algorithms[-1].terminal)

    def test_processing_with_only_binary_classifier(self):
        binary_algorithm_key = "moth_binary"
        binary_algorithm = CLASSIFIER_CHOICES[binary_algorithm_key]
        pipeline_request = PipelineRequest(
            pipeline=PipelineChoice[binary_algorithm_key],
            source_images=self.get_test_images(num=2),
        )
        with self.file_server:
            response = self.client.post("/process", json=pipeline_request.model_dump())
            assert response.status_code == 200
            results = PipelineResponse(**response.json())

        for detection in results.detections:
            for classification in detection.classifications:
                assert classification.algorithm.key == binary_algorithm_key
                assert classification.terminal
                assert classification.labels
                assert len(classification.labels) == binary_algorithm.num_classes
                assert classification.scores
                assert len(classification.scores) == binary_algorithm.num_classes
