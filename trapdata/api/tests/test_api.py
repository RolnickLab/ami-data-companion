import logging
import pathlib
from unittest import TestCase

from fastapi.testclient import TestClient

from trapdata.api.api import (
    PipelineChoice,
    PipelineConfig,
    PipelineRequest,
    SourceImageRequest,
    app,
)
from trapdata.api.tests.image_server import TestImageServer
from trapdata.tests import TEST_IMAGES_BASE_PATH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestInferenceAPI(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_images_dir = pathlib.Path(TEST_IMAGES_BASE_PATH) / "vermont"
        if not cls.test_images_dir.exists():
            raise FileNotFoundError(
                f"Test images directory not found: {cls.test_images_dir}"
            )

        cls.file_server = TestImageServer(cls.test_images_dir)
        cls.client = TestClient(app)

    def setUp(self):
        self.file_server = TestImageServer(self.test_images_dir)

    def test_pipeline_request(self):
        num_images = 2
        source_image_urls = [
            self.file_server.get_url(f)
            for f in self.file_server.list_image_files()[:num_images]
        ]
        source_images = [
            SourceImageRequest(id=str(i), url=url)
            for i, url in enumerate(source_image_urls)
        ]
        pipeline_request = PipelineRequest(
            pipeline=PipelineChoice["quebec_vermont_moths_2023"],
            source_images=source_images,
            config=PipelineConfig(classification_num_predictions=1),
        )
        with self.file_server:
            response = self.client.post(
                "/pipeline/process", json=pipeline_request.dict()
            )
        assert "results" in response.json()
        assert len(response.json()["results"]) == 1
        result = response.json()["results"][0]
        assert "detections" in result
        assert "classifications" in result
