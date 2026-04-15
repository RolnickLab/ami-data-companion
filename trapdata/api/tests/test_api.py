import logging
import pathlib
from unittest import TestCase

from fastapi.testclient import TestClient

from trapdata.api.api import (
    PIPELINE_CHOICES,
    PipelineChoice,
    PipelineRequest,
    PipelineResponse,
    app,
    make_algorithm_response,
    make_pipeline_config_response,
)
from trapdata.api.schemas import PipelineConfigRequest
from trapdata.api.tests.image_server import StaticFileTestServer
from trapdata.api.tests.utils import get_pipeline_class, get_test_images
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
        return get_test_images(self.file_server, self.test_images_dir, subdir, num)

    def get_test_pipeline(self, slug: str = "quebec_vermont_moths_2023"):
        return get_pipeline_class(slug)

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
            try:
                PipelineResponse(**response.json())
            except Exception as e:
                self.fail(f"Pipeline request did not return a valid response: {e}")

    def test_pipeline_config_with_binary_classifier(self):
        binary_classifier_pipeline_choice = "moth_binary"
        BinaryClassifier = PIPELINE_CHOICES[binary_classifier_pipeline_choice]
        binary_classifier_instance = BinaryClassifier(source_images=[], detections=[])
        BinaryClassifierResponse = make_algorithm_response(binary_classifier_instance)

        species_classifier_pipeline_choice = "quebec_vermont_moths_2023"
        SpeciesClassifier = PIPELINE_CHOICES[species_classifier_pipeline_choice]
        species_classifier_instance = SpeciesClassifier(source_images=[], detections=[])
        SpeciesClassifierResponse = make_algorithm_response(species_classifier_instance)

        # Test using a pipeline that finishes with a full species classifier
        pipeline_config = make_pipeline_config_response(
            SpeciesClassifier,
            slug=species_classifier_pipeline_choice,
        )

        self.assertEqual(len(pipeline_config.algorithms), 3)
        self.assertEqual(
            pipeline_config.algorithms[-1].key, SpeciesClassifierResponse.key
        )
        self.assertEqual(
            pipeline_config.algorithms[1].key, BinaryClassifierResponse.key
        )

        # Test using a pipeline that finishes only with a binary classifier
        pipeline_config_binary_only = make_pipeline_config_response(
            BinaryClassifier, slug=binary_classifier_pipeline_choice
        )

        self.assertEqual(len(pipeline_config_binary_only.algorithms), 2)
        self.assertEqual(
            pipeline_config_binary_only.algorithms[-1].key, BinaryClassifierResponse.key
        )
        # self.assertTrue(pipeline_config_binary_only.algorithms[-1].terminal)

    def test_processing_with_only_binary_classifier(self):
        binary_classifier_pipeline_choice = "moth_binary"
        binary_algorithm_key = "moth_nonmoth_classifier"
        BinaryAlgorithmClass = PIPELINE_CHOICES[binary_classifier_pipeline_choice]
        # Create an instance to get the num_classes
        binary_algorithm = BinaryAlgorithmClass(source_images=[], detections=[])

        pipeline_request = PipelineRequest(
            pipeline=PipelineChoice[binary_classifier_pipeline_choice],
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
                assert classification.labels is None
                assert classification.scores
                assert classification.logits
                assert len(classification.scores) == binary_algorithm.num_classes
                assert len(classification.logits) == binary_algorithm.num_classes

    def test_logits_in_classification_response(self):
        """
        Test that the logits are included in the classification response when
        requested via the pipeline configuration.
        """
        test_images = self.get_test_images(num=1)
        assert test_images, "No test images found"

        test_pipeline_slug = "insect_orders_2025"

        config = PipelineConfigRequest(
            # return_logits=True
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
        assert pipeline_response.detections, "No detections found in response"
        terminal_classifications = [
            classification
            for detection in pipeline_response.detections
            for classification in detection.classifications
            if classification.terminal
        ]
        assert terminal_classifications, "No terminal classifications found"

        # Get the expected number of classes from the model
        Classifier = self.get_test_pipeline(test_pipeline_slug)
        classifier = Classifier(source_images=[], detections=[])
        num_classes = classifier.num_classes

        for classification in terminal_classifications:
            assert classification.scores
            assert classification.logits
            num_scores = len(classification.scores)
            num_logits = len(classification.logits)
            assert num_logits == num_classes
            assert num_scores == num_classes
            assert (
                classification.logits != classification.scores
            ), "Logits and scores should not be the same"

    def test_config_num_classification_predictions(self):
        """
        Test that classification responses have no labels (from algorithm config)
        and scores/logits count equals num classes in category map.
        """
        test_images = self.get_test_images(num=1)
        assert test_images, "No test images found"

        test_pipeline_slug = "insect_orders_2025"

        config = PipelineConfigRequest()
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
        assert pipeline_response.detections, "No detections found in response"

        terminal_classifications = [
            classification
            for detection in pipeline_response.detections
            for classification in detection.classifications
            if classification.terminal
        ]
        assert terminal_classifications, "No terminal classifications found"

        # Get the expected number of classes from the model
        Classifier = self.get_test_pipeline(test_pipeline_slug)
        classifier = Classifier(source_images=[], detections=[])
        num_classes = classifier.num_classes
        assert (
            num_classes and num_classes > 2
        ), "Test requires a model with more than 2 classes"

        for classification in terminal_classifications:
            # Assert no labels are present in the classification response
            assert classification.labels is None, (
                "Labels should not be present in classification response, "
                "they should be retrieved from algorithm config"
            )

            # Assert scores and logits count equal the number of classes
            assert classification.scores, "Scores should be present"
            assert len(classification.scores) == num_classes, (
                f"Number of scores ({len(classification.scores)}) should equal "
                f"number of classes ({num_classes})"
            )

            assert classification.logits, "Logits should be present"
            assert len(classification.logits) == num_classes, (
                f"Number of logits ({len(classification.logits)}) should equal "
                f"number of classes ({num_classes})"
            )

    def test_existing_pipelines_default_to_apimothdetector(self):
        """Pre-existing pipelines must keep using APIMothDetector.

        New pipelines introduced with their own detector are exempt.
        """
        from trapdata.api.api import PIPELINE_CHOICES
        from trapdata.api.models.localization import APIMothDetector

        exempt = {"mothbot_insect_orders_2025"}
        for slug, Classifier in PIPELINE_CHOICES.items():
            if slug in exempt:
                continue
            self.assertIs(
                Classifier.detector_cls,
                APIMothDetector,
                f"{slug} should default to APIMothDetector",
            )

    def test_mothbot_pipeline_uses_yolo_detector(self):
        from trapdata.api.api import PIPELINE_CHOICES
        from trapdata.api.models.localization import APIMothDetector_YOLO11m_Mothbot

        assert "mothbot_insect_orders_2025" in PIPELINE_CHOICES
        Classifier = PIPELINE_CHOICES["mothbot_insect_orders_2025"]
        self.assertIs(Classifier.detector_cls, APIMothDetector_YOLO11m_Mothbot)

    def test_mothbot_pipeline_skips_binary_filter(self):
        from trapdata.api.api import PIPELINE_CHOICES, should_filter_detections

        Classifier = PIPELINE_CHOICES["mothbot_insect_orders_2025"]
        self.assertFalse(should_filter_detections(Classifier))

    def test_detection_response_has_optional_rotation_field(self):
        """The rotation field is opt-in for detectors that produce OBB."""
        import datetime

        from trapdata.api.schemas import (
            AlgorithmReference,
            BoundingBox,
            DetectionResponse,
        )

        # Default: rotation is None
        d = DetectionResponse(
            source_image_id="img1",
            bbox=BoundingBox(x1=0, y1=0, x2=10, y2=10),
            algorithm=AlgorithmReference(name="x", key="x"),
            timestamp=datetime.datetime.now(),
        )
        self.assertIsNone(d.rotation)

        # Accepts a float
        d2 = DetectionResponse(
            source_image_id="img1",
            bbox=BoundingBox(x1=0, y1=0, x2=10, y2=10),
            algorithm=AlgorithmReference(name="x", key="x"),
            timestamp=datetime.datetime.now(),
            rotation=-42.5,
        )
        self.assertAlmostEqual(d2.rotation, -42.5)

    def test_yolo_api_detector_instantiates(self):
        """The new YOLO detector wrapper should construct with no source images
        (matches the pattern the /info handler uses to read algorithm metadata).

        This test exercises weight download + model load — it will be slow on
        first run but cached thereafter.
        """
        from trapdata.api.models.localization import APIMothDetector_YOLO11m_Mothbot

        detector = APIMothDetector_YOLO11m_Mothbot(source_images=[])
        self.assertEqual(detector.name, "Mothbot YOLO11m Creature Detector")
        self.assertEqual(detector.category_map, {0: "creature"})
        self.assertEqual(detector.imgsz, 1600)
