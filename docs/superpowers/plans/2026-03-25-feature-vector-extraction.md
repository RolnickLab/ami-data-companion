# Feature Vector Extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add opt-in feature vector (embedding) extraction and opt-in logits to classification responses, toggleable via API request config and worker settings.

**Architecture:** Feature extraction hooks into `Resnet50TimmClassifier.get_features()` which calls `model.forward_features()` to get 2048-dim embeddings from the backbone before the classification head. Two new config flags — `include_features` and `include_logits` — in `PipelineConfigRequest` and `Settings` control whether these data are included in responses. The flags flow from the API request / worker settings through to `APIMothClassifier` which conditionally extracts and populates them.

**Tech Stack:** PyTorch (timm `forward_features`), Pydantic schemas, FastAPI

**Related PRs:**
- PR #77 — Mohamed's original feature extraction work (this PR, being updated)
- PR #74 — "Save logits with model predictions" (DB-layer logits, separate scope)

---

## Strategy: Merge Main into Feature Branch

The existing PR branch (`feat/add-classification-features-to-response`) diverged from main 19 commits ago and has conflicts. Rather than resetting and losing Mohamed's commit history, we will:

1. Merge `main` into the feature branch, resolving conflicts by taking main's version for structural code
2. Layer our clean implementation on top of Mohamed's foundation work
3. This preserves Mohamed's 11 commits as visible history in the PR

Mohamed's original commits established the core ideas: `get_features()` on base/classifier, features in the API response, and tests. Our work adapts these to main's current `ClassifierResult` pattern, adds opt-in config toggles (for both features AND logits), and fixes the conflicts.

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `trapdata/ml/models/base.py` | Modify | Add `features` field to `ClassifierResult`, add `get_features()` fallback to `InferenceBaseClass` |
| `trapdata/ml/models/classification.py` | Modify | Add `get_features()` to `Resnet50TimmClassifier` |
| `trapdata/api/schemas.py` | Modify | Add `features` field to `ClassificationResponse`, add `include_features` + `include_logits` to `PipelineConfigRequest` |
| `trapdata/api/models/classification.py` | Modify | Wire feature/logits extraction into `predict_batch()` and `update_detection_classification()` |
| `trapdata/api/api.py` | Modify | Pass config flags to classifier |
| `trapdata/antenna/worker.py` | Modify | Pass config flags from settings to classifier constructor |
| `trapdata/settings.py` | Modify | Add `AMI_INCLUDE_FEATURES` and `AMI_INCLUDE_LOGITS` settings |
| `trapdata/api/tests/test_features_extraction.py` | Create | Tests for feature and logits extraction via API |

---

### Task 1: Merge Main into Feature Branch

**Files:** None (git operation only)

- [ ] **Step 1: Merge main into the feature branch**

```bash
git merge main
```

This will have conflicts. For each conflicted file, resolve by taking **main's version** for the structural code (the `run()` method, `ClassifierResult`, `TuringKenyaUgandaSpeciesClassifier`, etc.), since we'll re-add feature extraction cleanly in subsequent tasks.

Key conflict resolution strategy:
- `trapdata/ml/models/base.py` — take main's version (keeps `ClassifierResult`, `reset()`, `update_detection_classification()`)
- `trapdata/ml/models/classification.py` — take main's version (keeps all classifier classes)
- `trapdata/api/models/classification.py` — take main's version (keeps `APIMothClassifier` with `ClassifierResult` pattern)
- `trapdata/api/schemas.py` — take main's version (we'll add fields in Task 3)
- Other files — take main's version, remove sklearn/plotly deps if added

- [ ] **Step 2: Remove unnecessary dependencies added by original branch**

If `pyproject.toml` was modified to add `scikit-learn` or `plotly`, revert those changes (we don't need clustering/visualization in this PR).

- [ ] **Step 3: Verify merge compiles and tests pass**

```bash
git log --oneline -3
git status
pytest trapdata/ -x -q --timeout=120 2>&1 | tail -20
```

Expected: Clean merge commit on top of Mohamed's commits + main. Tests pass.

- [ ] **Step 4: Commit merge**

The merge commit is created by `git merge`. If manual conflict resolution was needed, finalize with:
```bash
git add -A
git commit -m "merge: resolve conflicts with main, preserve Mohamed's feature extraction foundation"
```

---

### Task 2: Add `get_features()` to Base and Timm Classifier

**Files:**
- Modify: `trapdata/ml/models/base.py:340-345` (add `features` to `ClassifierResult`)
- Modify: `trapdata/ml/models/base.py:57` (add `get_features()` fallback to `InferenceBaseClass`)
- Modify: `trapdata/ml/models/classification.py:300-313` (add `get_features()` to `Resnet50TimmClassifier`)

- [ ] **Step 1: Add `features` field to `ClassifierResult`**

In `trapdata/ml/models/base.py`, the `ClassifierResult` dataclass at the bottom of the file:

```python
@dataclass
class ClassifierResult:
    labels: list[str] | None
    logit: list[float] | None
    scores: list[float]
    features: list[float] | None = None
```

- [ ] **Step 2: Add `get_features()` fallback to `InferenceBaseClass`**

In `trapdata/ml/models/base.py`, add after the `get_model()` method (around line 202):

```python
    def get_features(self, batch_input: torch.Tensor) -> torch.Tensor | None:
        """Extract feature vectors from the model backbone.

        Override in subclasses that support feature extraction.
        Returns None by default for models that don't implement it.
        """
        return None
```

- [ ] **Step 3: Add `get_features()` to `Resnet50TimmClassifier`**

In `trapdata/ml/models/classification.py`, add to the `Resnet50TimmClassifier` class after `get_model()`:

```python
    @torch.no_grad()
    def get_features(self, batch_input: torch.Tensor) -> torch.Tensor:
        """Extract 2048-dim feature vectors from the ResNet50 backbone.

        Uses timm's forward_features() which returns (B, 2048, H, W) spatial
        feature maps for ResNet50. Pooled to (B, 2048) via adaptive avg pool.
        """
        features = self.model.forward_features(batch_input)
        features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
        features = features.view(features.size(0), -1)
        return features
```

- [ ] **Step 4: Run existing tests to ensure nothing breaks**

Run: `pytest trapdata/ -x -q --timeout=120 2>&1 | tail -20`
Expected: All existing tests pass.

- [ ] **Step 5: Commit**

```bash
git add trapdata/ml/models/base.py trapdata/ml/models/classification.py
git commit -m "feat: add get_features() to InferenceBaseClass and Resnet50TimmClassifier

Adds a get_features() method to the inference base class (returns None by default)
and implements it on Resnet50TimmClassifier using timm's forward_features() to
extract 2048-dim embeddings from the ResNet50 backbone."
```

---

### Task 3: Add `include_features` and `include_logits` Config Toggles

**Files:**
- Modify: `trapdata/api/schemas.py:232-241` (`PipelineConfigRequest`)
- Modify: `trapdata/api/schemas.py:75-106` (`ClassificationResponse`)
- Modify: `trapdata/settings.py` (add `AMI_INCLUDE_FEATURES` and `AMI_INCLUDE_LOGITS` settings)

**Context:** Logits are already computed during classification and stored in `ClassifierResult.logit`. They're already in `ClassificationResponse.logits` but are always included. We make them opt-in (default off) to reduce response size, matching the same pattern as features. See PR #74 for the DB-layer logits work (separate scope).

- [ ] **Step 1: Add `include_features` and `include_logits` to `PipelineConfigRequest`**

In `trapdata/api/schemas.py`, replace the `PipelineConfigRequest` class:

```python
class PipelineConfigRequest(pydantic.BaseModel):
    """
    Configuration for the processing pipeline.
    """

    example_config_param: int | None = pydantic.Field(
        default=None,
        description="Example of a configuration parameter for a pipeline.",
        examples=[3],
    )
    include_features: bool = pydantic.Field(
        default=False,
        description=(
            "Whether to include feature vectors (embeddings) in classification "
            "responses. Feature vectors are 2048-dim floats extracted from the "
            "model backbone. Disabled by default to reduce response size."
        ),
    )
    include_logits: bool = pydantic.Field(
        default=False,
        description=(
            "Whether to include raw logits in classification responses. "
            "Logits are the unnormalized model outputs before softmax. "
            "Disabled by default to reduce response size."
        ),
    )
```

- [ ] **Step 2: Update `ClassificationResponse` fields**

In `trapdata/api/schemas.py`, the existing `logits` field should default to `None` (it may currently always be populated). Also add `features`:

```python
    logits: list[float] | None = pydantic.Field(
        default=None,
        description=(
            "Raw logits (unnormalized model outputs) for each class. "
            "Only included when include_logits=true in the pipeline config."
        ),
        repr=False,
    )
    features: list[float] | None = pydantic.Field(
        default=None,
        description=(
            "Feature vector (embedding) extracted from the model backbone before "
            "the classification head. Only included when include_features=true in "
            "the pipeline config."
        ),
        repr=False,
    )
```

- [ ] **Step 3: Add settings for both flags**

In `trapdata/settings.py`, add to the Settings class near the antenna worker settings (around `antenna_api_base_url`, `antenna_api_auth_token`, etc.):

```python
    include_features: bool = False
    include_logits: bool = False
```

This allows the worker to enable via `AMI_INCLUDE_FEATURES=true` and `AMI_INCLUDE_LOGITS=true` env vars.

- [ ] **Step 4: Commit**

```bash
git add trapdata/api/schemas.py trapdata/settings.py
git commit -m "feat: add include_features and include_logits config toggles

Adds include_features and include_logits flags to PipelineConfigRequest (API)
and Settings (worker). Adds features field to ClassificationResponse. Makes
logits field conditional (default None). Both default to off for backward
compatibility and reduced response size."
```

---

### Task 4: Wire Feature and Logits Extraction into APIMothClassifier

**Files:**
- Modify: `trapdata/api/models/classification.py:33-173`

This is the main integration. The `APIMothClassifier` needs to:
1. Accept and store `include_features` and `include_logits` flags
2. Call `get_features()` in `predict_batch()` when enabled
3. Populate `ClassifierResult.features` in `post_process_batch()`, conditionally include logits
4. Pass features and logits through to `ClassificationResponse` in `update_detection_classification()`

- [ ] **Step 1: Add config flags to `APIMothClassifier.__init__()`**

Add `include_features` and `include_logits` parameters:

```python
    def __init__(
        self,
        source_images: typing.Iterable[SourceImage],
        detections: typing.Iterable[DetectionResponse],
        terminal: bool = True,
        include_features: bool = False,
        include_logits: bool = False,
        *args,
        **kwargs,
    ):
        self.source_images = source_images
        self.detections = list(detections)
        self.terminal = terminal
        self.include_features = include_features
        self.include_logits = include_logits
        self.results: list[DetectionResponse] = []
        super().__init__(*args, **kwargs)
```

- [ ] **Step 2: Override `predict_batch()` to optionally extract features**

Add this method to `APIMothClassifier`:

```python
    def predict_batch(self, batch):
        batch_input = batch.to(self.device, non_blocking=True)
        logits = self.model(batch_input)
        features = None
        if self.include_features:
            features = self.get_features(batch_input)
        return logits, features
```

- [ ] **Step 3: Update `post_process_batch()` to handle the (logits, features) tuple**

Replace the existing `post_process_batch()`:

```python
    def post_process_batch(self, batch_output):
        """
        Return ClassifierResult objects with labels, scores, and
        optional logits and feature vectors for each image in the batch.
        """
        logits, features = batch_output
        predictions = torch.nn.functional.softmax(logits, dim=1)
        predictions = predictions.cpu().numpy()
        logits_cpu = logits.cpu()
        if features is not None:
            features = features.cpu()

        batch_results = []
        for i, pred in enumerate(predictions):
            class_indices = np.arange(len(pred))
            labels = [self.category_map[idx] for idx in class_indices]
            logit = logits_cpu[i].tolist() if self.include_logits else None
            feature_vec = features[i].tolist() if features is not None else None

            result = ClassifierResult(
                labels=labels,
                logit=logit,
                scores=pred.tolist(),
                features=feature_vec,
            )
            batch_results.append(result)

        return batch_results
```

- [ ] **Step 4: Update `update_detection_classification()` to pass features and logits**

In the existing `update_detection_classification()` method, update the `ClassificationResponse` constructor to conditionally include logits and features:

```python
        classification = ClassificationResponse(
            classification=self.get_best_label(predictions),
            scores=predictions.scores,
            logits=predictions.logit,
            features=predictions.features,
            inference_time=seconds_per_item,
            algorithm=AlgorithmReference(name=self.name, key=self.get_key()),
            timestamp=datetime.datetime.now(),
            terminal=self.terminal,
        )
```

Note: `predictions.logit` will already be `None` when `include_logits=False` (handled in `post_process_batch()`), and `predictions.features` will be `None` when `include_features=False`.

- [ ] **Step 5: Commit**

```bash
git add trapdata/api/models/classification.py
git commit -m "feat: wire feature and logits extraction into APIMothClassifier

APIMothClassifier now accepts include_features and include_logits flags.
When enabled, predict_batch() extracts features via get_features() and
post_process_batch() conditionally includes logits. Both flow through
ClassifierResult → update_detection_classification() → ClassificationResponse."
```

---

### Task 5: Pass Config Flags from API and Worker

**Files:**
- Modify: `trapdata/api/api.py:276-286` (pass config to classifier)
- Modify: `trapdata/antenna/worker.py:471` (pass flags to classifier constructor)

**Important design note:** The worker's `_process_batch()` calls `classifier.predict_batch()` and `classifier.post_process_batch()` directly. Since Task 4 already overrides `predict_batch()` to return `(logits, features)` and `post_process_batch()` to accept that tuple, the worker path works automatically — no changes needed to `_process_batch()` itself. We only need to set the flags on the classifier instance.

- [ ] **Step 1: Pass flags in `api.py` process endpoint**

In `trapdata/api/api.py`, where the classifier is instantiated (~line 276):

```python
    classifier: APIMothClassifier = Classifier(
        source_images=source_images,
        detections=detections_for_terminal_classifier,
        batch_size=settings.classification_batch_size,
        num_workers=settings.num_workers,
        single=True,
        example_config_param=data.config.example_config_param,
        terminal=True,
        include_features=data.config.include_features,
        include_logits=data.config.include_logits,
    )
```

- [ ] **Step 2: Set flags on classifier in `_process_job()`**

In `trapdata/antenna/worker.py:471`, the classifier is created:

```python
# Before:
classifier = classifier_class(source_images=[], detections=[])

# After:
classifier = classifier_class(
    source_images=[],
    detections=[],
    include_features=settings.include_features,
    include_logits=settings.include_logits,
)
```

Note: Do NOT pass these flags to the binary filter (line 476). Binary classification doesn't need features or logits. The `MothClassifierBinary` inherits the `False` defaults, which is correct.

- [ ] **Step 3: Run existing tests**

Run: `pytest trapdata/ -x -q --timeout=120 2>&1 | tail -20`
Expected: All existing tests still pass (both flags are off by default).

- [ ] **Step 4: Commit**

```bash
git add trapdata/api/api.py trapdata/antenna/worker.py
git commit -m "feat: pass include_features and include_logits from API and worker

API endpoint passes both flags from PipelineConfigRequest to classifier.
Worker passes both from Settings (AMI_INCLUDE_FEATURES, AMI_INCLUDE_LOGITS env
vars) to classifier constructor. No changes needed to _process_batch() since
the predict_batch()/post_process_batch() overrides handle the flow."
```

---

### Task 6: Write Feature and Logits Extraction Tests

**Files:**
- Create: `trapdata/api/tests/test_features_extraction.py`

- [ ] **Step 1: Write tests for feature and logits extraction via API**

```python
import pathlib
from unittest import TestCase

from fastapi.testclient import TestClient

from trapdata.api.api import PipelineChoice, PipelineRequest, PipelineResponse, app
from trapdata.api.schemas import PipelineConfigRequest, SourceImageRequest
from trapdata.api.tests.image_server import StaticFileTestServer
from trapdata.tests import TEST_IMAGES_BASE_PATH


class TestFeatureAndLogitsExtractionAPI(TestCase):
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
            SourceImageRequest(id=str(i), url=self.file_server.get_url(path))
            for i, path in enumerate(image_paths[:num])
        ]

    def _run_pipeline(
        self,
        include_features: bool = False,
        include_logits: bool = False,
        num_images: int = 1,
    ):
        test_images = self.get_local_test_images(num=num_images)
        config = PipelineConfigRequest(
            include_features=include_features,
            include_logits=include_logits,
        )
        pipeline_request = PipelineRequest(
            pipeline=PipelineChoice["global_moths_2024"],
            source_images=test_images,
            config=config,
        )
        with self.file_server:
            response = self.client.post(
                "/process", json=pipeline_request.model_dump()
            )
            assert response.status_code == 200
            return PipelineResponse(**response.json())

    def test_features_included_when_enabled(self):
        """Features are present and valid when include_features=True."""
        result = self._run_pipeline(include_features=True)
        self.assertTrue(result.detections, "No detections returned")
        for detection in result.detections:
            for classification in detection.classifications:
                if classification.terminal:
                    self.assertIsNotNone(
                        classification.features,
                        "Features should not be None when enabled",
                    )
                    self.assertIsInstance(classification.features, list)
                    self.assertTrue(
                        all(isinstance(x, float) for x in classification.features)
                    )
                    self.assertEqual(len(classification.features), 2048)

    def test_features_absent_when_disabled(self):
        """Features are None when include_features=False (default)."""
        result = self._run_pipeline(include_features=False)
        self.assertTrue(result.detections, "No detections returned")
        for detection in result.detections:
            for classification in detection.classifications:
                self.assertIsNone(
                    classification.features,
                    "Features should be None when disabled",
                )

    def test_logits_included_when_enabled(self):
        """Logits are present when include_logits=True."""
        result = self._run_pipeline(include_logits=True)
        self.assertTrue(result.detections, "No detections returned")
        for detection in result.detections:
            for classification in detection.classifications:
                if classification.terminal:
                    self.assertIsNotNone(
                        classification.logits,
                        "Logits should not be None when enabled",
                    )
                    self.assertIsInstance(classification.logits, list)
                    self.assertTrue(
                        all(isinstance(x, float) for x in classification.logits)
                    )

    def test_logits_absent_when_disabled(self):
        """Logits are None when include_logits=False (default)."""
        result = self._run_pipeline(include_logits=False)
        self.assertTrue(result.detections, "No detections returned")
        for detection in result.detections:
            for classification in detection.classifications:
                self.assertIsNone(
                    classification.logits,
                    "Logits should be None when disabled",
                )

    def test_both_features_and_logits(self):
        """Both features and logits present when both flags enabled."""
        result = self._run_pipeline(include_features=True, include_logits=True)
        self.assertTrue(result.detections, "No detections returned")
        for detection in result.detections:
            for classification in detection.classifications:
                if classification.terminal:
                    self.assertIsNotNone(classification.features)
                    self.assertIsNotNone(classification.logits)

    def test_default_config_has_nothing_extra(self):
        """Default PipelineConfigRequest disables both features and logits."""
        config = PipelineConfigRequest()
        self.assertFalse(config.include_features)
        self.assertFalse(config.include_logits)
```

- [ ] **Step 2: Run the new tests**

Run: `pytest trapdata/api/tests/test_features_extraction.py -v --timeout=300 2>&1 | tail -30`
Expected: All 6 tests pass.

- [ ] **Step 3: Run full test suite**

Run: `pytest trapdata/ -x -q --timeout=300 2>&1 | tail -20`
Expected: All tests pass.

- [ ] **Step 4: Commit**

```bash
git add trapdata/api/tests/test_features_extraction.py
git commit -m "test: add feature and logits extraction API tests

Tests that features are 2048-dim when enabled, logits present when enabled,
both absent when disabled (default), and both present when both flags set."
```

---

### Task 7: Format, Lint, and Final Verification

- [ ] **Step 1: Run formatters**

```bash
black trapdata/
isort trapdata/
```

- [ ] **Step 2: Run linter**

```bash
flake8 trapdata/ --max-line-length=120
```

Fix any issues.

- [ ] **Step 3: Run full test suite one more time**

```bash
pytest trapdata/ -x -q --timeout=300
```

- [ ] **Step 4: Commit any formatting fixes**

```bash
git add -p
git commit -m "style: format with black and isort"
```

---

### Task 8: Push to PR Branch and Update PR

- [ ] **Step 1: Push to the PR's remote branch**

Since we merged main into the feature branch (preserving Mohamed's commits), a regular push should work:

```bash
git push origin worktree-feature-vector-readiness:feat/add-classification-features-to-response
```

If the branch has diverged due to the merge, use `--force-with-lease`:

```bash
git push origin worktree-feature-vector-readiness:feat/add-classification-features-to-response --force-with-lease
```

This updates PR #77 with the new implementation on top of Mohamed's original work + main.

- [ ] **Step 2: Verify PR is no longer conflicting**

```bash
gh pr view 77 --json mergeable,mergeStateStatus
```

Expected: `"mergeable": "MERGEABLE"`

- [ ] **Step 3: Update PR description**

Update PR #77 title and body to reflect the expanded scope (features + logits, opt-in config):

```bash
gh api repos/RolnickLab/ami-data-companion/pulls/77 --method PATCH \
  --field title="feat: opt-in feature vectors and logits in classification responses" \
  --field body="$(cat <<'EOF'
## Summary

Adds opt-in support for including feature vectors (embeddings) and raw logits in classification API responses and worker output.

- **Feature vectors:** 2048-dim embeddings from the ResNet50 backbone via `model.forward_features()`. Enabled with `include_features=true`.
- **Logits:** Raw unnormalized model outputs before softmax. Enabled with `include_logits=true`.
- Both are **off by default** to keep responses compact and backward-compatible.

Built on @mohamedelabbas1996's original feature extraction work, updated to work with the current codebase and extended with opt-in config toggles.

## Configuration

**API:** Set flags in the `config` object of the request body:
```json
{
  "pipeline": "global_moths_2024",
  "source_images": [...],
  "config": {
    "include_features": true,
    "include_logits": true
  }
}
```

**Worker:** Set environment variables:
```
AMI_INCLUDE_FEATURES=true
AMI_INCLUDE_LOGITS=true
```

## Test plan

- [ ] `pytest trapdata/api/tests/test_features_extraction.py -v` — feature and logits extraction tests
- [ ] `pytest trapdata/ -x` — full test suite passes
- [ ] Verify default behavior unchanged (no features/logits in response without flags)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```
