import pathlib
from unittest import TestCase

from fastapi.testclient import TestClient

from trapdata.api.api import PipelineChoice, PipelineRequest, PipelineResponse, app
from trapdata.api.schemas import SourceImageRequest
from trapdata.api.tests.image_server import StaticFileTestServer
from trapdata.tests import TEST_IMAGES_BASE_PATH


class TestTrackingAPI(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_images_dir = pathlib.Path(TEST_IMAGES_BASE_PATH)
        cls.file_server = StaticFileTestServer(cls.test_images_dir)
        cls.client = TestClient(app)

    @classmethod
    def tearDownClass(cls):
        cls.file_server.stop()

    def get_local_test_images(self, num=2):
        image_paths = [
            "panama/01-20231110214539-snapshot.jpg",
            "panama/01-20231111032659-snapshot.jpg",
        ]
        return [
            SourceImageRequest(id=str(i), url=self.file_server.get_url(image_path))
            for i, image_path in enumerate(image_paths[:num])
        ]

    def test_tracking_sequence_info_in_response(self):
        """
        Run two images through the pipeline and verify that tracking metadata
        (sequence_id, sequence_frame) is returned for detections.
        """
        test_images = self.get_local_test_images(num=2)
        pipeline_request = PipelineRequest(
            pipeline=PipelineChoice["global_moths_2024"],
            source_images=test_images,
        )

        with self.file_server:
            response = self.client.post("/process", json=pipeline_request.model_dump())
            self.assertEqual(response.status_code, 200)
            result = PipelineResponse(**response.json())

        self.assertGreater(len(result.detections), 0, "No detections returned")

        tracking_info_present = any(
            det.sequence_id is not None and det.sequence_frame is not None
            for det in result.detections
        )
        self.assertTrue(
            tracking_info_present, "Expected tracking info (sequence_id/frame) missing"
        )
