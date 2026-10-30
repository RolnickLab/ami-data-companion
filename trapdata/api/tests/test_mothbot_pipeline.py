"""Integration tests for the Mothbot YOLO pipelines.

Covers both pipeline slugs that pair the Mothbot YOLO11m-OBB detector
with an existing terminal classifier:

  - ``mothbot_insect_orders_2025``: YOLO -> Insect Order classifier
    (binary prefilter skipped).
  - ``mothbot_panama_moths_2023``: YOLO -> binary moth/non-moth
    prefilter -> Panama 2023 moth species classifier.

Each test will download the YOLO weights (~40 MB) plus the terminal
classifier weights on first run, then cache them. Tests are loose
about content; they assert the pipeline runs and the detector populates
the new ``rotation`` field. Accuracy is out of scope for this suite.
"""

import logging
import pathlib
from unittest import TestCase

from fastapi.testclient import TestClient

from trapdata.api.api import PipelineChoice, PipelineRequest, PipelineResponse, app
from trapdata.api.tests.image_server import StaticFileTestServer
from trapdata.api.tests.utils import get_test_images
from trapdata.tests import TEST_IMAGES_BASE_PATH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestMothbotPipeline(TestCase):
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

    def test_mothbot_pipeline_end_to_end(self):
        """Send one vermont test image through the new pipeline."""
        test_images = get_test_images(
            self.file_server, self.test_images_dir, subdir="vermont", num=1
        )
        assert test_images, "No test images found"

        pipeline_request = PipelineRequest(
            pipeline=PipelineChoice["mothbot_insect_orders_2025"],
            source_images=test_images,
        )
        with self.file_server:
            response = self.client.post("/process", json=pipeline_request.model_dump())
        self.assertEqual(
            response.status_code, 200, f"Unexpected status: {response.text[:500]}"
        )

        result = PipelineResponse(**response.json())
        self.assertTrue(result.detections, "pipeline returned no detections")

        # At least one detection should carry a rotation (YOLO-OBB populates it)
        rotations = [d.rotation for d in result.detections]
        self.assertTrue(
            any(r is not None for r in rotations),
            "YOLO detector should populate the rotation field on at least one "
            "detection",
        )

        # Each detection should have an order classification from the terminal
        # classifier. (Binary prefilter is skipped for this pipeline.)
        for detection in result.detections:
            terminal = [c for c in detection.classifications if c.terminal]
            self.assertTrue(
                terminal,
                f"detection {detection.bbox} has no terminal classification",
            )
            self.assertEqual(
                terminal[0].algorithm.key,
                "mothbot_yolo_insect_orders_2025",
                f"expected order classifier, got {terminal[0].algorithm.key}",
            )

    def test_mothbot_panama_pipeline_end_to_end(self):
        """YOLO -> binary prefilter -> Panama 2023 species classifier."""
        test_images = get_test_images(
            self.file_server, self.test_images_dir, subdir="vermont", num=1
        )
        assert test_images, "No test images found"

        pipeline_request = PipelineRequest(
            pipeline=PipelineChoice["mothbot_panama_moths_2023"],
            source_images=test_images,
        )
        with self.file_server:
            response = self.client.post("/process", json=pipeline_request.model_dump())
        self.assertEqual(
            response.status_code, 200, f"Unexpected status: {response.text[:500]}"
        )

        result = PipelineResponse(**response.json())
        self.assertTrue(result.detections, "pipeline returned no detections")

        # YOLO detector still populates rotation
        rotations = [d.rotation for d in result.detections]
        self.assertTrue(
            any(r is not None for r in rotations),
            "YOLO detector should populate rotation on at least one detection",
        )

        # Each detection should carry a binary classification (prefilter is
        # terminal=False) and a Panama terminal classification *only for
        # detections the binary filter called a moth*. Detections classified
        # as non-moth short-circuit after the binary step.
        for detection in result.detections:
            algo_keys = [c.algorithm.key for c in detection.classifications]
            self.assertIn(
                "moth_nonmoth_classifier",
                algo_keys,
                f"missing binary classification on {detection.bbox}: {algo_keys}",
            )

        terminal_keys = {
            c.algorithm.key
            for d in result.detections
            for c in d.classifications
            if c.terminal
        }
        self.assertIn(
            "mothbot_yolo_panama_moths_2023",
            terminal_keys,
            f"expected Panama species classifier terminal, got {terminal_keys}",
        )
