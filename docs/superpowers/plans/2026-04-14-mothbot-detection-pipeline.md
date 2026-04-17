# Mothbot YOLO Detection Pipeline — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a new API pipeline `mothbot_insect_orders_2025` that pairs the Mothbot YOLO11m-OBB detector with the existing `InsectOrderClassifier2025`, so users can run Mothbot-style detection followed by our ConvNeXt order classifier through the existing FastAPI `/process` endpoint.

**Architecture:** Add a `detector_cls` class attribute on `APIMothClassifier` (default = existing `APIMothDetector`). The API's `/process` handler reads `Classifier.detector_cls` instead of a hardcoded reference, letting each pipeline pair a detector with a classifier. The new YOLO detector class lives alongside the existing FasterRCNN ones (ML + API split). `CLASSIFIER_CHOICES` is renamed to `PIPELINE_CHOICES` because the dict semantically maps to pipelines, not just classifiers. An optional `rotation: float | None` field is added to `DetectionResponse` to carry the YOLO OBB angle forward to a future species classifier — not used by consumers in this PR.

**Tech Stack:** Python 3.10+, FastAPI, pydantic 2, SQLAlchemy, PyTorch 2.5+, `ultralytics>=8.3` (new AGPL-3 dep), existing `InferenceBaseClass` pattern, `uv` for deps.

**Spec:** `docs/superpowers/specs/2026-04-14-mothbot-detection-pipeline-design.md`

---

## Before starting (operator step)

Upload the YOLO weights file to Arbutus. **This is a one-time operator action; no code task runs it.** Skip if the URL already resolves.

```bash
# From the worktree root:
AWS_PROFILE=ami python3 -c "
import boto3
from botocore.config import Config

s3 = boto3.client(
    's3',
    endpoint_url='https://object-arbutus.cloud.computecanada.ca',
    config=Config(request_checksum_calculation='when_required'),
)
s3.upload_file(
    'src-reference/Mothbot_Process/trained_models/yolo11m_4500_imgsz1600_b1_2024-01-18.pt',
    'ami-models',
    'mothbot/detection/yolo11m_4500_imgsz1600_b1_2024-01-18.pt',
)
print('upload done')
"

# Verify:
curl -sI "https://object-arbutus.cloud.computecanada.ca/ami-models/mothbot/detection/yolo11m_4500_imgsz1600_b1_2024-01-18.pt" | head -n 3
# Expected: HTTP/1.1 200 OK  + Content-Length around 40 MB
```

If `curl` returns 200, proceed to Task 1. If 403/404, stop and fix the upload before implementation continues.

---

## File Structure

Modified and new files produced by this plan:

| File | Change | Responsibility |
|---|---|---|
| `pyproject.toml` | modify | Add `ultralytics>=8.3` to `[project].dependencies`. |
| `uv.lock` | regen | Lockfile from `uv add`. |
| `trapdata/api/api.py` | modify | Rename dict → `PIPELINE_CHOICES`; replace hardcoded `APIMothDetector` at 2 callsites with `Classifier.detector_cls`; register new pipeline slug. |
| `trapdata/api/models/classification.py` | modify | Add `detector_cls` class attr on `APIMothClassifier` base. Add `MothbotInsectOrderClassifier` subclass. |
| `trapdata/api/models/localization.py` | modify | Add `APIMothDetector_YOLO11m_Mothbot` wrapper. |
| `trapdata/api/schemas.py` | modify | Add optional `rotation: float \| None` field to `DetectionResponse`. |
| `trapdata/api/tests/test_api.py` | modify | Rename import. |
| `trapdata/api/tests/utils.py` | modify | Rename import. |
| `trapdata/antenna/worker.py` | modify | Rename import + usage. |
| `trapdata/antenna/registration.py` | modify | Rename import + usage. |
| `trapdata/cli/worker.py` | modify | Rename import + usages. |
| `trapdata/cli/base.py` | modify | Rename import. |
| `trapdata/ml/models/localization.py` | modify | Add `YoloDetection` dataclass + `MothObjectDetector_YOLO11m_Mothbot` class. |
| `trapdata/ml/models/tests/test_mothbot_yolo.py` | create | Unit test for `_corners_to_yolo_detection` helper. |
| `trapdata/api/tests/test_mothbot_pipeline.py` | create | Integration test for the new pipeline end-to-end. |

---

## Task 1: Rebase worktree onto current `origin/main`

The worktree branch is behind main (main has the uv migration and AGPL-3 license). Implementation assumes the post-uv state.

**Files:** (none — git operation)

- [ ] **Step 1: Fetch and inspect**

```bash
cd /home/michael/Projects/AMI/ami-data-companion/.claude/worktrees/mothbot-pipeline
git fetch origin
git log --oneline HEAD..origin/main | head
```

Expected: several commits, including the uv migration merge (`029f1a8 ... feature/uv-migration` or similar) and AGPL-3 license update (`d2355c8 Update LICENSE to AGPLv3 (#137)`).

- [ ] **Step 2: Rebase**

```bash
git rebase origin/main
```

If conflicts occur, they'll almost certainly be in `poetry.lock` / `uv.lock` / `pyproject.toml` (old worktree state predates uv migration). Resolution: accept `origin/main`'s versions of `pyproject.toml` and `uv.lock`, delete `poetry.lock` if still present. The only worktree-branch content to preserve is the design doc at `docs/superpowers/specs/2026-04-14-mothbot-detection-pipeline-design.md` and this plan file.

```bash
# If conflict in lockfile / pyproject.toml:
git checkout --theirs pyproject.toml uv.lock
git rm -f poetry.lock 2>/dev/null || true
git add pyproject.toml uv.lock
git rebase --continue
```

- [ ] **Step 3: Verify state**

```bash
git log --oneline -5
ls pyproject.toml uv.lock
grep -c "ultralytics" pyproject.toml || echo "ultralytics not yet added — expected"
head -20 pyproject.toml
```

Expected: recent commits include the spec doc and this plan; `uv.lock` present, `poetry.lock` absent; `pyproject.toml` uses `[project]` syntax with a `dependencies = [...]` list.

- [ ] **Step 4: Install deps and run tests**

```bash
uv sync
uv run pytest trapdata/api/tests/test_api.py -x 2>&1 | tail -30
```

Expected: all tests pass (or the same failures as `origin/main` — record baseline).

- [ ] **Step 5: No commit needed — rebase replays existing commits.**

---

## Task 2: Rename `CLASSIFIER_CHOICES` → `PIPELINE_CHOICES`

Pure rename, no behavior change. Done as its own commit so the diff is reviewable in isolation.

**Files:**
- Modify: `trapdata/api/api.py` (definition at line 55, plus uses at lines 67–68, 219, 362)
- Modify: `trapdata/api/tests/test_api.py` (import line 8; uses at 65, 70, 102)
- Modify: `trapdata/api/tests/utils.py` (import line 10; use at 75)
- Modify: `trapdata/antenna/worker.py` (import line 17; use at 428)
- Modify: `trapdata/antenna/registration.py` (import line 10; use at 137)
- Modify: `trapdata/cli/worker.py` (import line 7; uses at 34, 36, 38, 43)
- Modify: `trapdata/cli/base.py` (import line 6)

`trapdata/api/demo.py` also has a local variable named `CLASSIFIER_CHOICES` — **do not touch it**, it's an unrelated list.

- [ ] **Step 1: Rename in `trapdata/api/api.py`**

Use sed to rename the symbol inside this one file (demo.py excluded):

```bash
sed -i 's/CLASSIFIER_CHOICES/PIPELINE_CHOICES/g' trapdata/api/api.py
```

Confirm:

```bash
grep -n "PIPELINE_CHOICES\|CLASSIFIER_CHOICES" trapdata/api/api.py
```

Expected: all occurrences are now `PIPELINE_CHOICES`; zero `CLASSIFIER_CHOICES` in this file.

- [ ] **Step 2: Rename in the six consumer files**

```bash
sed -i 's/CLASSIFIER_CHOICES/PIPELINE_CHOICES/g' \
    trapdata/api/tests/test_api.py \
    trapdata/api/tests/utils.py \
    trapdata/antenna/worker.py \
    trapdata/antenna/registration.py \
    trapdata/cli/worker.py \
    trapdata/cli/base.py
```

- [ ] **Step 3: Verify no stray references remain**

```bash
grep -rn "CLASSIFIER_CHOICES" trapdata/
```

Expected: only matches are in `trapdata/api/demo.py` (the unrelated local list) — zero matches elsewhere.

- [ ] **Step 4: Run tests**

```bash
uv run pytest trapdata/api/tests/ -x 2>&1 | tail -20
```

Expected: same pass/fail as baseline from Task 1 Step 4. No new failures.

- [ ] **Step 5: Commit**

```bash
git add \
    trapdata/api/api.py \
    trapdata/api/tests/test_api.py \
    trapdata/api/tests/utils.py \
    trapdata/antenna/worker.py \
    trapdata/antenna/registration.py \
    trapdata/cli/worker.py \
    trapdata/cli/base.py
git commit -m "refactor: rename CLASSIFIER_CHOICES to PIPELINE_CHOICES

The dict maps pipeline slug to the classifier class, but it's used
as the pipeline registry. Rename for honesty. No behavior change.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Add `detector_cls` class attribute; replace hardcoded `APIMothDetector`

Prepare the plumbing so different pipelines can use different detectors. Default remains `APIMothDetector`, so existing pipelines are unchanged.

**Files:**
- Modify: `trapdata/api/models/classification.py` (add class attr on `APIMothClassifier` around line 37)
- Modify: `trapdata/api/api.py` (swap hardcoded `APIMothDetector` at lines 140 and 221)

- [ ] **Step 1: Add import and class attribute on `APIMothClassifier`**

In `trapdata/api/models/classification.py`, find the imports block near the top:

```python
from ..datasets import ClassificationImageDataset
from ..schemas import (
    AlgorithmReference,
    ClassificationResponse,
    DetectionResponse,
    SourceImage,
)
from .base import APIInferenceBaseClass
```

Add an import from `.localization` below `.base`:

```python
from ..datasets import ClassificationImageDataset
from ..schemas import (
    AlgorithmReference,
    ClassificationResponse,
    DetectionResponse,
    SourceImage,
)
from .base import APIInferenceBaseClass
from .localization import APIMothDetector
```

(`localization.py` doesn't import from this module, so no circular import risk.)

Then find:

```python
class APIMothClassifier(
    APIInferenceBaseClass,
    InferenceBaseClass,
):
    task_type = "classification"
```

Replace with:

```python
class APIMothClassifier(
    APIInferenceBaseClass,
    InferenceBaseClass,
):
    task_type = "classification"

    # The detector class this pipeline pairs with. Subclasses override
    # to pair a specific classifier with a specific detector. Default is
    # the FasterRCNN 2023 detector that all existing pipelines use.
    detector_cls: type[APIMothDetector] = APIMothDetector
```

- [ ] **Step 2: Add test that every existing pipeline inherits the default detector**

Append to `trapdata/api/tests/test_api.py`:

```python
    def test_all_pipelines_default_to_apimothdetector(self):
        """All pre-existing pipelines must keep using APIMothDetector."""
        from trapdata.api.models.localization import APIMothDetector
        from trapdata.api.api import PIPELINE_CHOICES

        for slug, Classifier in PIPELINE_CHOICES.items():
            self.assertIs(
                Classifier.detector_cls,
                APIMothDetector,
                f"{slug} should default to APIMothDetector",
            )
```

- [ ] **Step 3: Run the new test (expect it to pass since everything inherits)**

```bash
uv run pytest trapdata/api/tests/test_api.py::TestInferenceAPI::test_all_pipelines_default_to_apimothdetector -v
```

Expected: PASS.

- [ ] **Step 4: Swap the hardcoded detector at `api.py:221`**

In `trapdata/api/api.py`, find:

```python
    Classifier = PIPELINE_CHOICES[str(data.pipeline)]

    detector = APIMothDetector(
        source_images=source_images,
        batch_size=settings.localization_batch_size,
        num_workers=settings.num_workers,
        # single=True if len(source_images) == 1 else False,
        single=True,  # @TODO solve issues with reading images in multiprocessing
    )
```

Replace `APIMothDetector(` with `Classifier.detector_cls(`:

```python
    Classifier = PIPELINE_CHOICES[str(data.pipeline)]

    detector = Classifier.detector_cls(
        source_images=source_images,
        batch_size=settings.localization_batch_size,
        num_workers=settings.num_workers,
        # single=True if len(source_images) == 1 else False,
        single=True,  # @TODO solve issues with reading images in multiprocessing
    )
```

- [ ] **Step 5: Swap the hardcoded detector in `make_pipeline_config_response` at `api.py:140`**

Find:

```python
def make_pipeline_config_response(
    Classifier: type[APIMothClassifier],
    slug: str,
) -> PipelineConfigResponse:
    """
    Create a configuration for an entire pipeline, given a species classifier class.
    """
    algorithms = []

    detector = APIMothDetector(
        source_images=[],
    )
```

Replace with:

```python
def make_pipeline_config_response(
    Classifier: type[APIMothClassifier],
    slug: str,
) -> PipelineConfigResponse:
    """
    Create a configuration for an entire pipeline, given a species classifier class.
    """
    algorithms = []

    detector = Classifier.detector_cls(
        source_images=[],
    )
```

- [ ] **Step 6: Run full API test suite**

```bash
uv run pytest trapdata/api/tests/test_api.py -x 2>&1 | tail -30
```

Expected: all tests pass; the new `test_all_pipelines_default_to_apimothdetector` passes; no previously-passing test now fails.

- [ ] **Step 7: Commit**

```bash
git add trapdata/api/models/classification.py trapdata/api/api.py trapdata/api/tests/test_api.py
git commit -m "refactor: let each pipeline specify its detector via detector_cls

Introduces a detector_cls class attribute on APIMothClassifier,
defaulting to APIMothDetector (FasterRCNN 2023). The /process and
/info handlers now read Classifier.detector_cls instead of a
hardcoded reference. No behavior change — every existing pipeline
keeps the default.

Enables pairing a non-FasterRCNN detector with a specific classifier
in a future commit.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Add optional `rotation` field to `DetectionResponse`

Forward-looking schema addition. Existing detectors leave it `None`. The YOLO detector will populate it in Task 7. No consumer reads it in this PR — the downstream classifier still crops axis-aligned.

**Files:**
- Modify: `trapdata/api/schemas.py` (`DetectionResponse` class around line 109)

- [ ] **Step 1: Write a failing test that `rotation` exists on `DetectionResponse`**

Append to `trapdata/api/tests/test_api.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest trapdata/api/tests/test_api.py::TestInferenceAPI::test_detection_response_has_optional_rotation_field -v
```

Expected: FAIL — `rotation` field rejected by pydantic (default `extra="ignore"` will silently swallow it, so the `d2.rotation` check becomes `AttributeError`). Either way, it fails until the field is added.

- [ ] **Step 3: Add the field to `DetectionResponse`**

In `trapdata/api/schemas.py`, find:

```python
class DetectionResponse(pydantic.BaseModel):
    source_image_id: str
    bbox: BoundingBox
    inference_time: float | None = None
    algorithm: AlgorithmReference
    timestamp: datetime.datetime
    crop_image_url: str | None = None
    classifications: list[ClassificationResponse] = []
```

Replace with:

```python
class DetectionResponse(pydantic.BaseModel):
    source_image_id: str
    bbox: BoundingBox
    inference_time: float | None = None
    algorithm: AlgorithmReference
    timestamp: datetime.datetime
    crop_image_url: str | None = None
    classifications: list[ClassificationResponse] = []
    rotation: float | None = pydantic.Field(
        default=None,
        description=(
            "Rotation angle in degrees (cv2.minAreaRect convention), when "
            "the detector produces oriented bounding boxes. FUTURE: "
            "downstream classifiers may use this to crop a straightened "
            "patch instead of the axis-aligned envelope. See "
            "`docs/superpowers/specs/2026-04-14-mothbot-detection-pipeline-design.md` "
            "for the proposed RotatedBoundingBox schema upgrade."
        ),
    )
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest trapdata/api/tests/test_api.py::TestInferenceAPI::test_detection_response_has_optional_rotation_field -v
uv run pytest trapdata/api/tests/ -x 2>&1 | tail -20
```

Expected: new test PASSES. All other tests still pass.

- [ ] **Step 5: Commit**

```bash
git add trapdata/api/schemas.py trapdata/api/tests/test_api.py
git commit -m "feat: add optional rotation field to DetectionResponse

Forward-looking schema addition for detectors that produce oriented
bounding boxes (first consumer: Mothbot YOLO11m-OBB in a follow-up
commit). Existing detectors leave it None. The downstream classifier
still crops axis-aligned; this field is preserved so a future species
classifier can use it for rotated crops without re-running detection.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Add `ultralytics` dependency via `uv add`

Dependency-only commit. Lockfile churn isolated from feature diffs. No code imports `ultralytics` yet.

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock`

- [ ] **Step 1: Add the dep**

```bash
uv add 'ultralytics>=8.3'
```

- [ ] **Step 2: Verify it landed in `pyproject.toml`**

```bash
grep -n "ultralytics" pyproject.toml
```

Expected: one line inside the `dependencies = [...]` array, e.g. `"ultralytics>=8.3",`.

- [ ] **Step 3: Verify import works**

```bash
uv run python -c "from ultralytics import YOLO; print(YOLO.__module__)"
```

Expected: `ultralytics.models.yolo.model` or similar — no ImportError.

- [ ] **Step 4: Run the full test suite to confirm nothing broke from version resolution**

```bash
uv run pytest trapdata/api/tests/ -x 2>&1 | tail -20
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "build: add ultralytics>=8.3 dependency

Required for the Mothbot YOLO11m detector (follow-up commit). No
code imports it in this commit.

Note: ultralytics is AGPL-3.0. This is not a license escalation —
the project is already AGPL-3 (PR #137).

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: Add ML-layer YOLO detector (`MothObjectDetector_YOLO11m_Mothbot`) + unit test

TDD: write the unit test for the corner-to-detection helper first, then implement the class.

**Files:**
- Modify: `trapdata/ml/models/localization.py` (append new class and dataclass)
- Create: `trapdata/ml/models/tests/__init__.py` (if missing)
- Create: `trapdata/ml/models/tests/test_mothbot_yolo.py`

- [ ] **Step 1: Check if `trapdata/ml/models/tests/` exists**

```bash
ls trapdata/ml/models/tests 2>&1
```

If the directory doesn't exist, create it with an empty `__init__.py`:

```bash
mkdir -p trapdata/ml/models/tests
touch trapdata/ml/models/tests/__init__.py
```

- [ ] **Step 2: Write the failing unit test**

Create `trapdata/ml/models/tests/test_mothbot_yolo.py`:

```python
"""Unit tests for the Mothbot YOLO detector's post-processing helpers.

These tests stay pure-CPU and don't load any model weights — they only
exercise the coordinate math that converts YOLO's 4 rotated corner
points into the (axis-aligned-bbox + rotation + score) shape our API
consumes. The model-loading path is covered by the integration test.
"""

import numpy as np

from trapdata.ml.models.localization import (
    YoloDetection,
    _corners_to_yolo_detection,
)


def test_corners_to_yolo_detection_axis_aligned_square():
    """A non-rotated square: envelope equals corners, rotation ~0 or ~90 (cv2 convention)."""
    corners = np.array(
        [
            [10, 10],
            [20, 10],
            [20, 20],
            [10, 20],
        ],
        dtype=np.float32,
    )
    det = _corners_to_yolo_detection(corners, score=0.9)

    assert isinstance(det, YoloDetection)
    assert det.x1 == 10 and det.y1 == 10
    assert det.x2 == 20 and det.y2 == 20
    assert det.score == 0.9
    # cv2.minAreaRect returns angle in (-90, 0] for a non-rotated square; either
    # 0 or -90 (or +90) are valid depending on corner ordering. Just assert the
    # angle is a finite float in the expected range.
    assert -90.0 <= det.rotation <= 90.0


def test_corners_to_yolo_detection_rotated_rectangle():
    """A rectangle rotated ~45°: envelope is larger than either side, rotation non-zero."""
    # 10x4 rectangle centered at (50, 50), rotated 45°.
    cx, cy = 50.0, 50.0
    half_w, half_h = 5.0, 2.0
    cos_a, sin_a = np.cos(np.pi / 4), np.sin(np.pi / 4)

    local = np.array(
        [
            [-half_w, -half_h],
            [+half_w, -half_h],
            [+half_w, +half_h],
            [-half_w, +half_h],
        ],
        dtype=np.float32,
    )
    R = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)
    corners = (local @ R.T) + np.array([cx, cy], dtype=np.float32)

    det = _corners_to_yolo_detection(corners, score=0.77)

    # Envelope must contain the rotated corners
    assert det.x1 <= corners[:, 0].min() + 1e-3
    assert det.y1 <= corners[:, 1].min() + 1e-3
    assert det.x2 >= corners[:, 0].max() - 1e-3
    assert det.y2 >= corners[:, 1].max() - 1e-3

    # Envelope for a rotated thin rectangle is strictly larger than its long side
    assert (det.x2 - det.x1) > 2 * half_w

    # Score passes through
    assert det.score == 0.77

    # Rotation is non-trivial for a visibly rotated rectangle
    assert abs(det.rotation) > 1.0


def test_yolo_detection_is_frozen_dataclass():
    """YoloDetection should be an immutable dataclass (design requirement)."""
    import dataclasses

    assert dataclasses.is_dataclass(YoloDetection)
    # frozen=True makes instances hashable
    det = YoloDetection(x1=0, y1=0, x2=1, y2=1, rotation=0.0, score=0.5)
    # Hash should not raise
    hash(det)
```

- [ ] **Step 3: Run test to verify it fails**

```bash
uv run pytest trapdata/ml/models/tests/test_mothbot_yolo.py -v 2>&1 | tail -20
```

Expected: ImportError / ModuleNotFoundError — `YoloDetection` and `_corners_to_yolo_detection` don't exist yet.

- [ ] **Step 4: Implement `YoloDetection` and the helper, and the detector class**

At the top of `trapdata/ml/models/localization.py`, inside the existing imports, add:

```python
import dataclasses
```

Add this `cv2` import at the top of the file (it's already used by Mothbot but may not be imported here yet — check first):

```bash
grep -n "^import cv2\|^from cv2" trapdata/ml/models/localization.py
```

If no match, add `import cv2` at the top of the existing imports block.

Append **to the end of `trapdata/ml/models/localization.py`**:

```python
# -----------------------------------------------------------------------------
# Mothbot YOLO11m-OBB detector
#
# Single-class ("creature") detector from Digital Naturalism Laboratories'
# Mothbot_Process project. Trained at imgsz=1600, Jan 2024. Weights are hosted
# on Arbutus alongside our other models.
#
# This implementation is an independent rewrite; Mothbot's repo is unlicensed
# (see spec). The torch 2.6 weights_only fallback below is adapted from
# Mothbot_Process/pipeline/detect.py — the pattern is standard ultralytics
# PyTorch 2.6 compat handling, not Mothbot-specific logic.
# -----------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class YoloDetection:
    """One detection from the YOLO-OBB post-processor.

    Fields:
        x1, y1, x2, y2: axis-aligned envelope of the rotated bounding box
            (min/max of the 4 rotated corner points).
        rotation: angle in degrees, cv2.minAreaRect convention.
        score: detection confidence, in [0, 1].
    """

    x1: float
    y1: float
    x2: float
    y2: float
    rotation: float
    score: float


def _corners_to_yolo_detection(corners: np.ndarray, score: float) -> YoloDetection:
    """Convert 4 rotated corner points + score into a YoloDetection.

    Args:
        corners: shape (4, 2), xy coordinates of the OBB corners.
        score: detection confidence.

    Returns:
        A YoloDetection with:
          - (x1, y1, x2, y2): min/max envelope of the 4 corners (axis-aligned).
          - rotation: angle from cv2.minAreaRect (same convention Mothbot uses).
    """
    pts = np.asarray(corners, dtype=np.float32).reshape(-1, 2)
    x1, y1 = float(pts[:, 0].min()), float(pts[:, 1].min())
    x2, y2 = float(pts[:, 0].max()), float(pts[:, 1].max())
    rect = cv2.minAreaRect(pts.astype(np.int32))
    angle = float(rect[2])
    return YoloDetection(x1=x1, y1=y1, x2=x2, y2=y2, rotation=angle, score=float(score))


def _load_ultralytics_yolo(weights_path: str):
    """Load an ultralytics YOLO model with a PyTorch 2.6 weights_only fallback.

    Newer PyTorch defaults to torch.load(..., weights_only=True), which can
    refuse to load Ultralytics checkpoints that embed custom model classes.
    For local, trusted checkpoints we retry with weights_only=False.

    Adapted from Mothbot_Process/pipeline/detect.py (unlicensed repo; pattern
    is standard ultralytics PyTorch 2.6 compat handling).
    """
    # Import lazily so the ML module doesn't pay the ultralytics import cost
    # for users who never touch this detector.
    import os

    import torch as _torch
    from ultralytics import YOLO

    try:
        return YOLO(str(weights_path))
    except Exception as err:
        if "Weights only load failed" not in str(err):
            raise

        logger.info(
            "Retrying YOLO load with torch.load(weights_only=False) compatibility "
            "(trusted local checkpoint)"
        )
        original_load = _torch.load
        original_force_wo = os.environ.get("TORCH_FORCE_WEIGHTS_ONLY_LOAD")
        original_force_no_wo = os.environ.get("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD")

        def _patched_load(*args, **kwargs):
            kwargs["weights_only"] = False
            return original_load(*args, **kwargs)

        _torch.load = _patched_load
        try:
            os.environ["TORCH_FORCE_WEIGHTS_ONLY_LOAD"] = "0"
            os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
            return YOLO(str(weights_path))
        finally:
            _torch.load = original_load
            if original_force_wo is None:
                os.environ.pop("TORCH_FORCE_WEIGHTS_ONLY_LOAD", None)
            else:
                os.environ["TORCH_FORCE_WEIGHTS_ONLY_LOAD"] = original_force_wo
            if original_force_no_wo is None:
                os.environ.pop("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", None)
            else:
                os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = original_force_no_wo


class MothObjectDetector_YOLO11m_Mothbot(ObjectDetector):
    name = "Mothbot YOLO11m Creature Detector"
    weights_path = (
        "https://object-arbutus.cloud.computecanada.ca/ami-models/"
        "mothbot/detection/yolo11m_4500_imgsz1600_b1_2024-01-18.pt"
    )
    description = (
        "Single-class 'creature' detector from Digital Naturalism "
        "Laboratories' Mothbot project. YOLO11m-OBB, trained at "
        "imgsz=1600, Jan 2024."
    )
    # Overrides the base: we set the category map directly instead of
    # hosting a one-entry labels.json on the object store.
    category_map = {0: "creature"}
    imgsz = 1600
    bbox_score_threshold = 0.25
    box_detections_per_img = 500

    def get_transforms(self):
        # ultralytics handles letterboxing / normalization internally; just
        # pass the PIL image through unchanged.
        return lambda pil_image: pil_image

    def get_model(self):
        logger.debug(f"Loading YOLO weights: {self.weights}")
        model = _load_ultralytics_yolo(self.weights)
        # ultralytics manages its own device placement via the device kwarg
        # passed to .predict(), so we don't .to(self.device) here.
        return model

    def get_dataloader(self):
        """PIL images can't be stacked by default_collate, so we collate as
        lists and let predict_batch hand a list of PIL images to ultralytics.
        """
        logger.info(
            f"Preparing {self.name} inference dataloader "
            f"(batch_size={self.batch_size}, single={self.single})"
        )

        def collate_as_lists(batch):
            ids = [b[0] for b in batch]
            imgs = [b[1] for b in batch]
            return ids, imgs

        dataloader_args = {
            "num_workers": 0 if self.single else self.num_workers,
            "persistent_workers": False if self.single else True,
            "shuffle": False,
            "pin_memory": False,
            "batch_size": self.batch_size,
            "collate_fn": collate_as_lists,
        }
        self.dataloader = torch.utils.data.DataLoader(self.dataset, **dataloader_args)
        return self.dataloader

    def predict_batch(self, batch):
        """batch is a list[PIL.Image]. Returns a list of ultralytics Results."""
        if not isinstance(batch, list):
            raise TypeError(
                f"{self.name} expects a list of PIL images from the collate fn; "
                f"got {type(batch)}"
            )
        return self.model.predict(
            batch,
            imgsz=self.imgsz,
            conf=self.bbox_score_threshold,
            max_det=self.box_detections_per_img,
            device=self.device,
            verbose=False,
        )

    def post_process_single(self, result) -> list[YoloDetection]:
        """Flatten one ultralytics Result into a list of detection records.

        Why the OBB → axis-aligned envelope:
          YOLO11m-OBB outputs 4 rotated corner points per detection. Our
          DetectionResponse schema carries a single axis-aligned bbox, and
          the downstream InsectOrderClassifier reads an axis-aligned crop.
          We therefore take the min/max envelope of the 4 corners as the
          bbox. The rotation angle (cv2.minAreaRect convention, same as
          Mothbot) is preserved separately so a future species classifier
          can reuse Mothbot's rotated crop_rect() without re-running
          detection.

          Confidence filtering already happened inside model.predict(conf=...),
          so every record here is above bbox_score_threshold.
        """
        detections: list[YoloDetection] = []
        if result.obb is None:
            return detections
        corners_batch = result.obb.xyxyxyxy.cpu().numpy()  # (N, 4, 2)
        scores = result.obb.conf.cpu().numpy()              # (N,)
        for i in range(len(corners_batch)):
            detections.append(
                _corners_to_yolo_detection(corners_batch[i], float(scores[i]))
            )
        return detections

    def save_results(self, item_ids, batch_output, *args, **kwargs):
        """The ML-layer base class expects a save method. The API wrapper
        overrides this, so the DB path is never hit when used via the API.
        Provide a no-op that logs, for symmetry with the FasterRCNN class's
        behavior.
        """
        logger.info(
            f"{self.name} ML-layer save_results called with {len(item_ids)} items "
            "(no-op; API wrapper handles persistence)"
        )
```

- [ ] **Step 5: Run the unit test**

```bash
uv run pytest trapdata/ml/models/tests/test_mothbot_yolo.py -v
```

Expected: all three tests PASS.

- [ ] **Step 6: Confirm the module imports cleanly**

```bash
uv run python -c "from trapdata.ml.models.localization import MothObjectDetector_YOLO11m_Mothbot, YoloDetection; print('ok')"
```

Expected: prints `ok`.

- [ ] **Step 7: Commit**

```bash
git add trapdata/ml/models/localization.py trapdata/ml/models/tests/__init__.py trapdata/ml/models/tests/test_mothbot_yolo.py
git commit -m "feat: add Mothbot YOLO11m-OBB detector (ML layer)

Implements MothObjectDetector_YOLO11m_Mothbot, a single-class
('creature') insect detector trained by Digital Naturalism
Laboratories. Weights hosted on Arbutus and lazily downloaded
via the existing InferenceBaseClass machinery.

Adds a YoloDetection dataclass and a _corners_to_yolo_detection
helper that converts OBB corners into an axis-aligned envelope +
rotation angle, with unit tests on the coordinate math.

The torch 2.6 weights_only fallback is adapted from
Mothbot_Process/pipeline/detect.py (unlicensed repo; pattern is
standard ultralytics PyTorch 2.6 compat handling).

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: Add API-layer YOLO detector wrapper (`APIMothDetector_YOLO11m_Mothbot`)

Wraps the ML class for the API request path: pulls `SourceImage` objects, builds `DetectionResponse`s (with populated `rotation`), skips the DB.

**Files:**
- Modify: `trapdata/api/models/localization.py` (append)

- [ ] **Step 1: Append the wrapper class**

At the **end of `trapdata/api/models/localization.py`**, add:

```python
from trapdata.ml.models.localization import (
    MothObjectDetector_YOLO11m_Mothbot,
    YoloDetection,
)


class APIMothDetector_YOLO11m_Mothbot(
    APIInferenceBaseClass, MothObjectDetector_YOLO11m_Mothbot
):
    task_type = "localization"

    def __init__(self, source_images: typing.Iterable[SourceImage], *args, **kwargs):
        self.source_images = source_images
        self.results: list[DetectionResponse] = []
        super().__init__(*args, **kwargs)

    def reset(self, source_images: typing.Iterable[SourceImage]):
        self.source_images = source_images
        self.results = []

    def get_dataset(self):
        return LocalizationImageDataset(
            self.source_images, self.get_transforms(), batch_size=self.batch_size
        )

    def get_source_image(self, source_image_id: int) -> SourceImage:
        for source_image in self.source_images:
            if source_image.id == source_image_id:
                return source_image
        raise ValueError(f"Source image with id {source_image_id} not found")

    def save_results(self, item_ids, batch_output, seconds_per_item, *args, **kwargs):
        """batch_output is a list (one per image) of list[YoloDetection]."""
        detections: list[DetectionResponse] = []
        for image_id, yolo_dets in zip(item_ids, batch_output):
            for y in yolo_dets:
                detections.append(
                    DetectionResponse(
                        source_image_id=image_id,
                        bbox=BoundingBox(x1=y.x1, y1=y.y1, x2=y.x2, y2=y.y2),
                        rotation=y.rotation,
                        inference_time=seconds_per_item,
                        algorithm=AlgorithmReference(
                            name=self.name, key=self.get_key()
                        ),
                        timestamp=datetime.datetime.now(),
                        crop_image_url=None,
                    )
                )
        self.results += detections

    def run(self) -> list[DetectionResponse]:
        super().run()
        return self.results
```

- [ ] **Step 2: Add a smoke-test that the class instantiates with no source images**

Append to `trapdata/api/tests/test_api.py`:

```python
    def test_yolo_api_detector_instantiates(self):
        """The new YOLO detector wrapper should construct with no source images
        (matches the pattern the /info handler uses to read algorithm metadata).
        The test exercises weight download + model load — it will be slow on
        first run but cached thereafter.
        """
        from trapdata.api.models.localization import (
            APIMothDetector_YOLO11m_Mothbot,
        )

        detector = APIMothDetector_YOLO11m_Mothbot(source_images=[])
        self.assertEqual(detector.name, "Mothbot YOLO11m Creature Detector")
        self.assertEqual(detector.category_map, {0: "creature"})
        self.assertEqual(detector.imgsz, 1600)
```

- [ ] **Step 3: Run tests**

```bash
uv run pytest trapdata/api/tests/test_api.py::TestInferenceAPI::test_yolo_api_detector_instantiates -v 2>&1 | tail -30
```

Expected: PASS. First run downloads ~40 MB weights from Arbutus (may take 10–30 s depending on connection).

If the test fails with a download error, verify the operator upload step completed:

```bash
curl -sI "https://object-arbutus.cloud.computecanada.ca/ami-models/mothbot/detection/yolo11m_4500_imgsz1600_b1_2024-01-18.pt" | head -n 3
```

- [ ] **Step 4: Full API test suite**

```bash
uv run pytest trapdata/api/tests/ -x 2>&1 | tail -20
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add trapdata/api/models/localization.py trapdata/api/tests/test_api.py
git commit -m "feat: add API wrapper for Mothbot YOLO11m detector

Wraps MothObjectDetector_YOLO11m_Mothbot for the /process endpoint:
consumes SourceImage objects, builds DetectionResponses with the new
rotation field populated from the YOLO-OBB angle. No pipeline uses
this detector yet — registration follows in the next commit.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: Register `mothbot_insect_orders_2025` pipeline

Pair the new detector with the existing `InsectOrderClassifier2025` as a new pipeline choice.

**Files:**
- Modify: `trapdata/api/models/classification.py`
- Modify: `trapdata/api/api.py`

- [ ] **Step 1: Add `MothbotInsectOrderClassifier` in `classification.py`**

In `trapdata/api/models/classification.py`, extend the `.localization` import added in Task 3 to also include the YOLO detector. Find:

```python
from .localization import APIMothDetector
```

Replace with:

```python
from .localization import APIMothDetector, APIMothDetector_YOLO11m_Mothbot
```

Right after the **existing** `InsectOrderClassifier` definition at the bottom of the file, add:

```python
class MothbotInsectOrderClassifier(InsectOrderClassifier):
    """Pair the Mothbot YOLO11m detector with our existing ConvNeXt order
    classifier. Overrides the default detector_cls inherited from
    APIMothClassifier.
    """

    detector_cls = APIMothDetector_YOLO11m_Mothbot
```

- [ ] **Step 2: Register in `PIPELINE_CHOICES`**

In `trapdata/api/api.py`, find:

```python
from .models.classification import (
    APIMothClassifier,
    InsectOrderClassifier,
    MothClassifierBinary,
    ...
)
```

Add `MothbotInsectOrderClassifier` to the imports.

Then find:

```python
PIPELINE_CHOICES = {
    "panama_moths_2023": MothClassifierPanama,
    ...
    "moth_binary": MothClassifierBinary,
    "insect_orders_2025": InsectOrderClassifier,
}
```

Add the new entry:

```python
PIPELINE_CHOICES = {
    "panama_moths_2023": MothClassifierPanama,
    ...
    "moth_binary": MothClassifierBinary,
    "insect_orders_2025": InsectOrderClassifier,
    "mothbot_insect_orders_2025": MothbotInsectOrderClassifier,
}
```

Update `should_filter_detections` at `api.py:76` so the new classifier also skips the binary filter (it inherits `InsectOrderClassifier`, so the `isinstance` check already covers it — but the current implementation uses `in [MothClassifierBinary, InsectOrderClassifier]` which does NOT cover subclasses. Change to `issubclass`).

Find:

```python
def should_filter_detections(Classifier: type[APIMothClassifier]) -> bool:
    if Classifier in [MothClassifierBinary, InsectOrderClassifier]:
        return False
    else:
        return True
```

Replace with:

```python
def should_filter_detections(Classifier: type[APIMothClassifier]) -> bool:
    # Classifiers that skip the binary moth/non-moth prefilter: the binary
    # classifier itself (there's nothing downstream to filter for), and any
    # order-level classifier (it already distinguishes non-moth insects,
    # so a binary prefilter would discard signal).
    if issubclass(Classifier, (MothClassifierBinary, InsectOrderClassifier)):
        return False
    return True
```

- [ ] **Step 3: Update the existing "all pipelines default to APIMothDetector" test**

The test added in Task 3 now needs to exempt `mothbot_insect_orders_2025` from the default-detector assertion. In `trapdata/api/tests/test_api.py`, find:

```python
    def test_all_pipelines_default_to_apimothdetector(self):
        """All pre-existing pipelines must keep using APIMothDetector."""
        from trapdata.api.models.localization import APIMothDetector
        from trapdata.api.api import PIPELINE_CHOICES

        for slug, Classifier in PIPELINE_CHOICES.items():
            self.assertIs(
                Classifier.detector_cls,
                APIMothDetector,
                f"{slug} should default to APIMothDetector",
            )
```

Replace with:

```python
    def test_existing_pipelines_default_to_apimothdetector(self):
        """Pre-existing pipelines must keep using APIMothDetector.

        New pipelines introduced with their own detector are exempt.
        """
        from trapdata.api.models.localization import APIMothDetector
        from trapdata.api.api import PIPELINE_CHOICES

        exempt = {"mothbot_insect_orders_2025"}
        for slug, Classifier in PIPELINE_CHOICES.items():
            if slug in exempt:
                continue
            self.assertIs(
                Classifier.detector_cls,
                APIMothDetector,
                f"{slug} should default to APIMothDetector",
            )
```

- [ ] **Step 4: Add a test that the new pipeline is registered with the YOLO detector**

Append to `trapdata/api/tests/test_api.py`:

```python
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
```

- [ ] **Step 5: Run tests**

```bash
uv run pytest trapdata/api/tests/test_api.py -v -k "mothbot or existing_pipelines_default" 2>&1 | tail -30
```

Expected: all three new/updated tests PASS.

- [ ] **Step 6: Run full API test suite**

```bash
uv run pytest trapdata/api/tests/ -x 2>&1 | tail -20
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add trapdata/api/models/classification.py trapdata/api/api.py trapdata/api/tests/test_api.py
git commit -m "feat: register mothbot_insect_orders_2025 pipeline

Pairs the Mothbot YOLO11m detector with the existing
InsectOrderClassifier2025 (ConvNeXt-T, 16 insect orders). Binary
prefilter is skipped — same policy as the existing
insect_orders_2025 pipeline, since the order classifier already
distinguishes non-moth insects.

Also tightens should_filter_detections() to use issubclass() so
subclasses of the exempt classifier set inherit the policy.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 9: End-to-end integration test for the new pipeline

Runs the new pipeline against one test image through the FastAPI test client. First run downloads the YOLO weights and the order classifier weights (~100 MB total) — cached thereafter.

**Files:**
- Create: `trapdata/api/tests/test_mothbot_pipeline.py`

- [ ] **Step 1: Write the integration test**

Create `trapdata/api/tests/test_mothbot_pipeline.py`:

```python
"""Integration test for the Mothbot YOLO + Insect Order classifier pipeline.

This test runs the full /process handler end-to-end for the new
`mothbot_insect_orders_2025` pipeline slug. It will download the YOLO
weights (~40 MB) and the ConvNeXt order classifier weights (~55 MB)
from Arbutus on first run, then cache them.

The test is intentionally loose about contents — it asserts that the
pipeline runs, returns a well-formed response, and that the YOLO
detector populates the new `rotation` field. Accuracy is out of scope
for this suite.
"""

import logging
import pathlib
from unittest import TestCase

from fastapi.testclient import TestClient

from trapdata.api.api import (
    PipelineChoice,
    PipelineRequest,
    PipelineResponse,
    app,
)
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
            response = self.client.post(
                "/process", json=pipeline_request.model_dump()
            )
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
                "insect-order-classifier",
                f"expected order classifier, got {terminal[0].algorithm.key}",
            )
```

**Note on the expected algorithm key:** the test asserts `terminal[0].algorithm.key == "insect-order-classifier"`. This comes from `slugify(InsectOrderClassifier2025.name)` → `slugify("Insect Order Classifier")`. Before running, verify the slug is correct by inspecting the key locally:

```bash
uv run python -c "
from trapdata.common.utils import slugify
from trapdata.ml.models.classification import InsectOrderClassifier2025
print(slugify(InsectOrderClassifier2025.name))
"
```

If the output is something other than `insect-order-classifier`, update the assertion to match.

- [ ] **Step 2: Run the integration test**

```bash
uv run pytest trapdata/api/tests/test_mothbot_pipeline.py -v -s 2>&1 | tail -40
```

Expected: PASS. First run may take 30–120 s while weights download; subsequent runs should finish in 10–30 s on CPU.

If the test fails:
- Empty `detections` → the test image may be a trap with nothing on it; try swapping `num=1` for `num=2` or pick a different `subdir`. Check with `ls trapdata/tests/images/`.
- `ImportError` → confirm Task 5 installed ultralytics: `uv run python -c "import ultralytics"`.
- Download error → verify weights are on Arbutus (see "Before starting" section).

- [ ] **Step 3: Full test suite green check**

```bash
uv run pytest -x 2>&1 | tail -30
```

Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
git add trapdata/api/tests/test_mothbot_pipeline.py
git commit -m "test: end-to-end integration test for mothbot pipeline

Sends one test image through the /process endpoint with the
mothbot_insect_orders_2025 slug, asserts detections are returned,
at least one has a populated rotation field, and each has an
order-level terminal classification.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## PR checklist (for the author, when opening the PR)

- [ ] Push branch: `git push -u origin worktree-mothbot-pipeline`
- [ ] Open PR with title: `feat: add Mothbot YOLO11m detection pipeline`
- [ ] PR body covers:
  - What: new `mothbot_insect_orders_2025` pipeline slug
  - Why: gives users a Mothbot-style detector paired with our existing order classifier
  - `CLASSIFIER_CHOICES` → `PIPELINE_CHOICES` rename (separate commit, reviewable alone)
  - `detector_cls` attribute on `APIMothClassifier` enables per-pipeline detectors
  - New optional `rotation` field on `DetectionResponse` — forward-looking, unused by consumers in this PR; proposed full `RotatedBoundingBox` upgrade discussed in the linked spec
  - Single-class YOLO (confirmed via checkpoint inspection: `names={0: 'creature'}`) — taxonomic labels come from the existing ConvNeXt classifier, not the detector
  - Dependency: `ultralytics>=8.3` (AGPL-3); project is already AGPL-3 so no license escalation; YOLO weights checkpoint carries embedded AGPL-3 metadata
  - Mothbot repo has no explicit license — we re-implement rather than verbatim-port; one adapted snippet (torch 2.6 weights_only fallback) attributed inline
  - Test plan: unit test on OBB → envelope math; integration test on full pipeline
  - Operator note: YOLO weights uploaded to `ami-models/mothbot/detection/` on Arbutus before merge
- [ ] Link to spec: `docs/superpowers/specs/2026-04-14-mothbot-detection-pipeline-design.md`

---

## Known follow-ups (out of scope for this PR)

- Port / reimplement Mothbot's pybioclip classifier as a second new pipeline.
- Full `RotatedBoundingBox` schema + rotated crop support in `ClassificationImageDataset`, letting a species classifier consume tighter rotated crops.
- Accuracy/latency evaluation: YOLO vs. FasterRCNN 2023 on a shared test set.
