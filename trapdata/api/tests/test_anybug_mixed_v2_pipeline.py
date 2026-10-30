"""Integration test for the AnyBug Mixed v2 pipelines.

Sends one test image through ``anybug_mixed_v2_insect_orders_2025``: the
AnyBug Mixed v2 YOLO26s-OBB detector feeding the insect order classifier.

The first run downloads the detector weights (about 22 MB) and the order
classifier weights, then caches them. The test checks that the pipeline runs
end to end and that the OBB detector fills the ``rotation`` field. Accuracy is
out of scope for this suite.
"""

import pathlib
from unittest import TestCase

from fastapi.testclient import TestClient

from trapdata.api.api import PipelineChoice, PipelineRequest, PipelineResponse, app
from trapdata.api.tests.image_server import StaticFileTestServer
from trapdata.api.tests.utils import get_test_images
from trapdata.tests import TEST_IMAGES_BASE_PATH


class TestAnyBugMixedV2Pipeline(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_images_dir = pathlib.Path(TEST_IMAGES_BASE_PATH)
        cls.file_server = StaticFileTestServer(cls.test_images_dir)
        cls.client = TestClient(app)

    @classmethod
    def tearDownClass(cls):
        cls.file_server.stop()

    def test_insect_orders_pipeline_end_to_end(self):
        test_images = get_test_images(
            self.file_server, self.test_images_dir, subdir="vermont", num=1
        )
        assert test_images, "No test images found"

        pipeline_request = PipelineRequest(
            pipeline=PipelineChoice["anybug_mixed_v2_insect_orders_2025"],
            source_images=test_images,
        )
        with self.file_server:
            response = self.client.post("/process", json=pipeline_request.model_dump())
        self.assertEqual(
            response.status_code, 200, f"Unexpected status: {response.text[:500]}"
        )

        result = PipelineResponse(**response.json())
        self.assertTrue(result.detections, "pipeline returned no detections")
        self.assertTrue(
            any(d.rotation is not None for d in result.detections),
            "the OBB detector should fill the rotation field",
        )
        for detection in result.detections:
            terminal = [c for c in detection.classifications if c.terminal]
            self.assertTrue(
                terminal, f"detection {detection.bbox} has no terminal classification"
            )
            self.assertEqual(
                terminal[0].algorithm.key, "anybug_mixed_v2_insect_orders_2025"
            )
