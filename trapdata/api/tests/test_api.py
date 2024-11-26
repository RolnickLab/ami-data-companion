import logging
import pathlib
from unittest import TestCase

from fastapi.testclient import TestClient

from trapdata.api.api import (
    PIPELINE_CHOICES,
    PipelineChoice,
    PipelineConfig,
    PipelineRequest,
    PipelineResponse,
    SourceImageRequest,
    app,
)
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

    def setUp(self):
        self.file_server = StaticFileTestServer(self.test_images_dir)

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
        pipeline = PIPELINE_CHOICES[slug]
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
            response = self.client.post(
                "/pipeline/process", json=pipeline_request.dict()
            )
            assert response.status_code == 200
            PipelineResponse(**response.json())

    def test_config_num_classification_predictions(self):
        """
        Test that the pipeline respects the `max_predictions_per_classification` configuration.

        If the configuration is set to a number, the pipeline should return that number of labels/scores per prediction.
        If the configuration is set to `None`, the pipeline should return all labels/scores per prediction.
        """
        test_images = self.get_test_images(num=1)
        test_pipeline_slug = "quebec_vermont_moths_2023"
        terminal_classifier = self.get_test_pipeline(test_pipeline_slug)

        def _send_request(max_predictions_per_classification: int | None):
            config = PipelineConfig(
                example_config_param=max_predictions_per_classification
            )
            pipeline_request = PipelineRequest(
                pipeline=PipelineChoice[test_pipeline_slug],
                source_images=test_images,
                config=config,
            )
            with self.file_server:
                response = self.client.post(
                    "/pipeline/process", json=pipeline_request.dict()
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
                if max_predictions_per_classification is None:
                    # Ensure that a score is returned for every possible class
                    assert len(classification.labels) == terminal_classifier.num_classes
                    assert len(classification.scores) == terminal_classifier.num_classes
                else:
                    # Ensure that the number of predictions is limited to the number specified
                    # There may be fewer predictions than the number specified if there are fewer classes.
                    assert (
                        len(classification.labels) <= max_predictions_per_classification
                    )
                    assert (
                        len(classification.scores) <= max_predictions_per_classification
                    )

        _send_request(max_predictions_per_classification=1)
        _send_request(max_predictions_per_classification=None)
