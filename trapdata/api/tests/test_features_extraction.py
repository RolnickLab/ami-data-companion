import pathlib
from unittest import TestCase

from fastapi.testclient import TestClient

from trapdata.api.api import PipelineChoice, PipelineRequest, PipelineResponse, app
from trapdata.api.schemas import SourceImageRequest
from trapdata.api.tests.image_server import StaticFileTestServer
from trapdata.ml.models.tracking import cosine_similarity
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
        image_path = "panama/01-20231110214539-snapshot.jpg"
        return [SourceImageRequest(id="0", url=self.file_server.get_url(image_path))]

    def get_pipeline_response(self, pipeline_slug="global_moths_2024", num_images=1):
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

    def test_feature_extraction_from_pipeline(self):
        """
        Run a local image through the pipeline and validate extracted features.
        """
        pipeline_response = self.get_pipeline_response()

        self.assertTrue(pipeline_response.detections, "No detections returned")
        for detection in pipeline_response.detections:
            for classification in detection.classifications:
                if classification.terminal:
                    features = classification.features
                    self.assertIsNotNone(features, "Features should not be None")
                    self.assertIsInstance(features, list, "Features should be a list")
                    self.assertTrue(
                        all(isinstance(x, float) for x in features),
                        "All features should be floats",
                    )
                    self.assertEqual(
                        len(features), 2048, "Feature vector should be 2048 dims"
                    )

    def test_cosine_similarity_of_extracted_features(self):
        """
        Run the pipeline and compare features using cosine similarity to validate output.
        """
        pipeline_response = self.get_pipeline_response(num_images=1)

        # Extract all terminal classification features
        feature_vectors = []
        for detection in pipeline_response.detections:
            for classification in detection.classifications:
                if classification.terminal and classification.features:
                    feature_vectors.append(classification.features)

        self.assertGreater(
            len(feature_vectors), 1, "Need at least two features to compare"
        )

        print("Cosine similarity matrix:")
        for i, vec1 in enumerate(feature_vectors):
            sims = []
            for j, vec2 in enumerate(feature_vectors):
                sim = cosine_similarity(vec1, vec2)
                sims.append(round(sim, 4))
            print(f"Feature {i} similarities: {sims}")

        # Confirm that similarity with itself is 1.0
        for i, vec in enumerate(feature_vectors):
            self_sim = cosine_similarity(vec, vec)
            self.assertAlmostEqual(
                self_sim, 1.0, places=5, msg=f"Self similarity at index {i} not 1.0"
            )
        # Confirm that a feature is most similar to itself

        for ref_index, ref_vec in enumerate(feature_vectors):
            similarities = [
                (i, cosine_similarity(ref_vec, other_vec))
                for i, other_vec in enumerate(feature_vectors)
            ]
            similarities.sort(key=lambda x: x[1], reverse=True)
            most_similar_index = similarities[0][0]
            self.assertEqual(
                most_similar_index,
                ref_index,
                f"Expected most similar vector to be at index {ref_index}, got {most_similar_index}",
            )
