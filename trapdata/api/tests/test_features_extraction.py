"""How the species classifier produces its backbone vectors, without a model download.

The vector must come from the classification pass itself, so turning vectors on does
not run the backbone twice, and it must not change the predictions.
"""

from unittest import TestCase

import timm
import torch

from trapdata.api.api import PIPELINE_CHOICES
from trapdata.api.models.classification import APIMothClassifier
from trapdata.ml.models.classification import Resnet50TimmClassifier
from trapdata.settings import Settings


class _RandomWeightTimmClassifier(Resnet50TimmClassifier):
    """A Resnet50TimmClassifier holding random weights, so the tests below run offline.

    Bypasses the normal constructor, which would download and load a checkpoint;
    only ``self.model`` and ``self.device`` matter to the methods under test.
    """

    name = "Random-weight Resnet50 Timm Classifier"

    def __init__(self, num_classes: int = 8):
        self.num_classes = num_classes
        self.device = "cpu"
        self.model = timm.create_model(
            "resnet50", pretrained=False, num_classes=num_classes
        )
        self.model.eval()


class _StubAPIClassifier(APIMothClassifier, _RandomWeightTimmClassifier):
    """Drives the predict/post-process seam without the API or a real checkpoint."""

    def __init__(self, num_classes: int = 8, **kwargs):
        _RandomWeightTimmClassifier.__init__(self, num_classes=num_classes)
        self.category_map = {i: f"species_{i}" for i in range(num_classes)}
        self.source_images = []
        self.detections = []
        self.terminal = True
        self.results = []
        self._last_features = None
        self.include_embeddings = kwargs.get("include_embeddings", False)
        self._embedding_only = False


class TestFeatureExtractionMechanics(TestCase):
    """Offline tests for how features are extracted, kept separate from the
    pipeline tests above because these need no model download."""

    def _count_backbone_passes(self, classifier, batch):
        """Run one batch, returning how many times the backbone was invoked."""
        calls = {"n": 0}
        original = classifier.model.forward_features

        def counting_forward_features(x, *args, **kwargs):
            calls["n"] += 1
            return original(x, *args, **kwargs)

        classifier.model.forward_features = counting_forward_features
        try:
            logits = classifier.predict_batch(batch)
        finally:
            classifier.model.forward_features = original
        return calls["n"], logits

    def test_backbone_runs_once_when_features_are_requested(self):
        """Features must come from the classification pass, not a second one.

        Extracting them from the input separately would run the backbone twice
        and halve throughput on exactly the configuration tracking needs.
        """
        batch = torch.randn(2, 3, 128, 128)

        with_features = _StubAPIClassifier(include_embeddings=True)
        passes, _ = self._count_backbone_passes(with_features, batch)
        self.assertEqual(passes, 1, "Backbone ran more than once with features on")

        without_features = _StubAPIClassifier(include_embeddings=False)
        passes, _ = self._count_backbone_passes(without_features, batch)
        self.assertEqual(passes, 1, "Backbone ran more than once with features off")

    def test_single_pass_logits_match_a_plain_forward(self):
        """Splitting the forward pass must not change the predictions."""
        classifier = _RandomWeightTimmClassifier()
        batch = torch.randn(2, 3, 128, 128)
        with torch.no_grad():
            expected = classifier.model(batch)
        actual, features = classifier.forward_with_features(batch)
        self.assertTrue(
            torch.allclose(expected, actual, atol=1e-6),
            f"Logits diverged, max abs diff {(expected - actual).abs().max().item()}",
        )
        self.assertEqual(tuple(features.shape), (2, 2048))

    def test_post_process_batch_without_a_preceding_predict(self):
        """``_last_features`` is initialised, so this order does not raise."""
        classifier = _StubAPIClassifier(include_embeddings=True)
        results = classifier.post_process_batch(torch.randn(2, 8))
        self.assertEqual(len(results), 2)
        self.assertTrue(all(r.features is None for r in results))

    def test_features_reach_results_through_the_seam(self):
        """predict_batch hands features to post_process_batch on the instance.

        The two are called separately by the base class ``run()``, so the
        hand-off is the part that can silently break.
        """
        classifier = _StubAPIClassifier(include_embeddings=True)
        batch = torch.randn(2, 3, 128, 128)
        results = classifier.post_process_batch(classifier.predict_batch(batch))
        self.assertEqual(len(results), 2)
        for result in results:
            self.assertIsNotNone(result.features)
            self.assertEqual(len(result.features), 2048)
        self.assertIsNone(
            classifier._last_features, "Features were not released after the batch"
        )

    def test_features_are_omitted_unless_requested(self):
        classifier = _StubAPIClassifier(include_embeddings=False)
        batch = torch.randn(2, 3, 128, 128)
        results = classifier.post_process_batch(classifier.predict_batch(batch))
        self.assertTrue(all(r.features is None for r in results))

    def test_logits_are_always_sent(self):
        """Antenna re-scores classifications from logits, so there is no switch."""
        classifier = _StubAPIClassifier()
        results = classifier.post_process_batch(
            classifier.predict_batch(torch.randn(2, 3, 128, 128))
        )
        self.assertTrue(all(len(r.logit) == 8 for r in results))

    def test_vectors_stay_off_by_default(self):
        """Vectors are large, so they are opt-in."""
        self.assertFalse(Settings(_env_file=None).include_features)

    def test_supports_features_reports_which_pipelines_can_extract(self):
        """A model without a backbone hook returns no features however it is asked.

        ``supports_features()`` is what lets a caller tell that apart from
        features simply being switched off.
        """
        supported = {
            key for key, cls in PIPELINE_CHOICES.items() if cls.supports_features()
        }
        self.assertEqual(
            supported,
            {
                "global_moths_2024",
                "panama_moths_2024",
                "quebec_vermont_moths_2023",
                "uk_denmark_moths_2023",
            },
        )
        self.assertFalse(PIPELINE_CHOICES["moth_binary"].supports_features())
