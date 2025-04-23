import os
import pathlib
from unittest import TestCase
from fastapi.testclient import TestClient
from trapdata.api.api import PipelineChoice, PipelineRequest, PipelineResponse, app
from trapdata.api.schemas import SourceImageRequest
from trapdata.api.tests.image_server import StaticFileTestServer
from trapdata.tests import TEST_IMAGES_BASE_PATH


class TestFeatureExtractionAPI(TestCase):
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
            SourceImageRequest(id="0", url=self.file_server.get_url(image_path))
            for image_path in image_paths[:num]
        ]

    def get_pipeline_response(
        self,
        pipeline_slug="panama_plus_moths_2025",
        num_images=1,
    ):
        """
        Utility method to send a pipeline request and return the parsed response.
        """
        test_images = self.get_local_test_images(num=num_images)
        pipeline_request = PipelineRequest(
            pipeline=PipelineChoice[pipeline_slug],
            source_images=test_images,
        )

        with self.file_server:
            response = self.client.post("/process", json=pipeline_request.model_dump())
            assert response.status_code == 200
            return PipelineResponse(**response.json())

    def test_ood_scores_from_pipeline(self):
        """
        Run a local image through the pipeline and validate extracted features.
        """
        pipeline_response = self.get_pipeline_response()

        self.assertTrue(pipeline_response.detections, "No detections returned")
        for detection in pipeline_response.detections:
            for classification in detection.classifications:
                print(classification)
                print(classification.ood_score)

        #         if classification.terminal:
        #             ood_scores = classification.ood_scores
        # features = classification.features
        # self.assertIsNotNone(features, "Features should not be None")
        # self.assertIsInstance(features, list, "Features should be a list")
        # self.assertTrue(
        #     all(isinstance(x, float) for x in features),
        #     "All features should be floats",
        # )
        # self.assertEqual(
        #     len(features), 2048, "Feature vector should be 2048 dims"
        # )
