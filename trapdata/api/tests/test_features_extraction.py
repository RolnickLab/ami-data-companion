import os
import pathlib
import unittest
from unittest import TestCase

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

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
        image_paths = [
            "panama/01-20231110214539-snapshot.jpg",
            "panama/01-20231111032659-snapshot.jpg",
            "panama/01-20231111015309-snapshot.jpg",
        ]
        return [
            SourceImageRequest(id="0", url=self.file_server.get_url(image_path))
            for image_path in image_paths[:num]
        ]

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
                    assert features  # This is for type checking
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
        Run the pipeline and compare features using cosine similarity to validate
        output.
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

        for _i, vec1 in enumerate(feature_vectors):
            sims = []
            for _j, vec2 in enumerate(feature_vectors):
                sim = cosine_similarity(vec1, vec2)
                sims.append(round(sim, 4))

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
                f"Expected most similar vector to be at index {ref_index}, "
                "got {most_similar_index}",
            )

    def get_detection_crop(self, local_image_path: str, bbox) -> Image.Image | None:
        """
        Given a local image path and a bounding box, return a cropped and resized image.
        """

        try:
            if not os.path.exists(local_image_path):
                print(f"File not found: {local_image_path}")
                return None

            img = Image.open(local_image_path).convert("RGB")
            x1, y1, x2, y2 = map(int, [bbox.x1, bbox.y1, bbox.x2, bbox.y2])
            crop = img.crop((x1, y1, x2, y2)).resize((64, 64))
            return crop
        except Exception as e:
            print(f"Failed to load or crop image: {e}")
            return None

    @unittest.skip("Skipping visualization test")
    def test_feature_clustering_visualization(self):

        source_images = self.get_local_test_images(num=3)
        pipeline_response = self.get_pipeline_response(num_images=len(source_images))
        image_id_to_url = {img.id: img.url for img in source_images}

        features = []
        labels = []

        for detection in pipeline_response.detections:
            source_url = image_id_to_url.get(detection.source_image_id)
            if not source_url or not detection.bbox:
                continue

            for classification in detection.classifications:
                if classification.features:
                    features.append(classification.features)
                    print(f"Classification: {classification.classification}")

                    labels.append(classification.classification)

        if len(features) < 2:
            print("Not enough data for clustering.")
            return

        # Reduce to 3D using PCA
        features_np = np.array(features)
        reduced = PCA(n_components=3).fit_transform(features_np)
        cluster_labels = KMeans(
            n_clusters=min(8, len(features)), random_state=42
        ).fit_predict(features_np)

        import plotly.express as px  # type: ignore[import]

        fig = px.scatter_3d(
            x=reduced[:, 0],
            y=reduced[:, 1],
            z=reduced[:, 2],
            color=cluster_labels.astype(str),
            hover_name=labels,
            title="3D Clustering of Classification Feature Vectors (K-Means + PCA)",
        )

        fig.update_traces(marker={"size": 6})
        fig.write_html("feature_clustering_3d_pca.html")
