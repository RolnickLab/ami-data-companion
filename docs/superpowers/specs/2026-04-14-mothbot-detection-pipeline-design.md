# Mothbot YOLO detection pipeline — design

**Date:** 2026-04-14
**Status:** Approved (awaiting implementation plan)
**Scope:** Add a new API pipeline `mothbot_insect_orders_2025` that pairs the Mothbot YOLO11m-OBB detector with the existing `InsectOrderClassifier2025`.

## Context

[Mothbot_Process](https://github.com/Digital-Naturalism-Laboratories/Mothbot_Process) (Digital Naturalism Labs) ships a YOLO11m-OBB insect detector trained on moth-trap imagery. Reference checkout at `src-reference/Mothbot_Process/`. Its detection stage (`pipeline/detect.py`) feeds oriented-box crops into a pybioclip+GBIF classifier (`pipeline/identify.py`) — Mothbot's classifier is **out of scope** for this task and will be considered in a follow-up.

**What we're doing:** exposing their detector as a new pipeline choice on our API, paired with our existing ConvNeXt-T order classifier.

**What we're not doing:**

- No change to the legacy CLI/desktop pipeline (`trapdata/ml/pipeline.py`). The new ML-layer detector class will still register for it automatically via `trapdata/ml/models/__init__.py:25`, so it's available there for free if someone wants it later.
- No pybioclip integration.
- No full rotated-bbox schema; see "Rotation field" below.

## Non-obvious facts verified up front

- **The YOLO model is single-class.** Inspection of `yolo11m_4500_imgsz1600_b1_2024-01-18.pt` gives `nc=1`, `names={0: 'creature'}`. Mothbot's `detect.py` hardcodes `"label": "creature"` and never reads `obb.cls`. Any taxonomic labeling must come from the classifier downstream. If a future Mothbot-trained YOLO ships with multiple classes, the detector wrapper would need to surface `obb.cls` and we'd revisit the detector→classifier contract.
- **Current `BoundingBox` schema is axis-aligned.** `trapdata/api/schemas.py:12`. The Mothbot detector produces oriented boxes. We handle this by taking the axis-aligned envelope of the 4 rotated corners and preserving the rotation angle in a new optional `rotation` field — see "Rotation field" below.
- **`origin/main` has already merged the uv migration (PR #115) and is AGPL-3 licensed (PR #137).** The worktree branch `worktree-mothbot-pipeline` is behind main — step 0 of the rollout is a rebase.

## Design

### Architecture choice: `detector_cls` attribute on the classifier

Today, `CLASSIFIER_CHOICES` in `trapdata/api/api.py:55` maps pipeline slug → `APIMothClassifier` subclass, and the detector is hardcoded to `APIMothDetector` at `api.py:221`. All 10 existing pipelines share the same detector.

**Approaches considered:**

- **1. Tuple registry** — `slug → (DetectorClass, ClassifierClass)`. Touches every existing pipeline entry.
- **2. `PipelineConfig` dataclass** — cleanest model, biggest blast radius. Overkill for one new pipeline.
- **3. `detector_cls` class attribute on the classifier** — smallest diff; only pipelines that differ from the default need to set it. **Chosen.**

The "detector on the classifier" coupling is a minor abuse (the detector is logically a sibling of the classifier, not a member), but the codebase already hangs pipeline-level concerns on classifiers (`positive_binary_label`, `get_key()`, binary-filter routing), so it's in character. If a future pipeline needs to vary more axes than just the detector, that's the moment to upgrade to Approach 2.

### Rename `CLASSIFIER_CHOICES` → `PIPELINE_CHOICES`

Honest naming — the dict maps to pipelines, not classifiers. Seven files reference the current name:

- `trapdata/api/api.py` (definition + 3 uses)
- `trapdata/api/tests/test_api.py` (4 uses)
- `trapdata/api/tests/utils.py` (2 uses)
- `trapdata/antenna/worker.py` (2 uses)
- `trapdata/antenna/registration.py` (2 uses)
- `trapdata/cli/worker.py` (4 uses)
- `trapdata/cli/base.py` (1 use)

`trapdata/api/demo.py` has an unrelated local list also named `CLASSIFIER_CHOICES` — untouched.

Done as its own commit so the rename diff is reviewable separately.

### New detector classes

Two-layer split mirrors the FasterRCNN family's ML/API split.

**ML layer — `trapdata/ml/models/localization.py`:**

```python
@dataclass(frozen=True)
class YoloDetection:
    x1: float; y1: float; x2: float; y2: float
    rotation: float   # degrees, cv2.minAreaRect convention
    score: float


class MothObjectDetector_YOLO11m_Mothbot(ObjectDetector):
    name = "Mothbot YOLO11m Creature Detector"
    description = (
        "Single-class 'creature' detector from Digital Naturalism "
        "Laboratories' Mothbot project. YOLO11m-OBB, trained at "
        "imgsz=1600, Jan 2024."
    )
    weights_path = (
        "https://object-arbutus.cloud.computecanada.ca/ami-models/"
        "mothbot/detection/yolo11m_4500_imgsz1600_b1_2024-01-18.pt"
    )
    category_map = {0: "creature"}       # overrides the base; no labels.json
    imgsz = 1600
    bbox_score_threshold = 0.25          # matches naming in FasterRCNN family
    box_detections_per_img = 500         # mirrors FasterRCNN 2023
```

**Key integration notes:**

- **Weights loading**: `InferenceBaseClass.get_weights()` already pulls from the URL via `get_or_download_file()`. `get_model()` wraps the local path in `ultralytics.YOLO(...)` with a **PyTorch 2.6 `weights_only` fallback** lifted verbatim from Mothbot's `load_yolo_model()` (`src-reference/Mothbot_Process/pipeline/detect.py:44-97`) — handles newer PyTorch refusing to load ultralytics checkpoints that embed custom classes.
- **Transforms**: `get_transforms()` returns `lambda pil: pil` (identity). YOLO does its own letterboxing/normalization.
- **Dataloader**: `get_dataloader()` overrides the base to set `collate_fn=lambda batch: (list(b[0] for b in batch), list(b[1] for b in batch))` — default collate can't stack PIL images (and source images differ in size anyway).
- **Batching**: `single=True` matches the existing API pipeline's `APIMothDetector` usage at `api.py:227`.
- **`predict_batch(batch)`** with `batch: list[PIL.Image]`: `self.model.predict(batch, imgsz=self.imgsz, conf=self.bbox_score_threshold, max_det=self.box_detections_per_img, device=self.device, verbose=False)`.
- **Confidence filtering** happens inside `model.predict(conf=...)` — unlike FasterRCNN where we filter in `post_process_single`. Asymmetric with existing code, but matches how ultralytics is meant to be used.

**Post-processing:**

```python
def post_process_single(self, result) -> list[YoloDetection]:
    """
    Flatten one ultralytics Result (an image's worth of OBB predictions)
    into a list of detection records the API layer can turn into
    DetectionResponse objects.

    Why the OBB → axis-aligned envelope:
      YOLO11m-OBB outputs 4 rotated corner points per detection. Our
      DetectionResponse schema carries a single axis-aligned bbox, and the
      downstream InsectOrderClassifier reads an axis-aligned crop. We
      therefore take the min/max envelope of the 4 corners as the bbox.
      The rotation angle (from cv2.minAreaRect, same convention Mothbot
      uses) is preserved separately so a future species classifier can
      reuse Mothbot's rotated crop_rect() without re-running detection.

    Confidence filtering already happened inside model.predict(conf=...),
    so every record here is above bbox_score_threshold.
    """
```

Implementation:

```python
pts = obb.xyxyxyxy[i].cpu().numpy().reshape(-1, 2)   # (4, 2)
score = float(obb.conf[i])
x1, y1 = pts[:, 0].min(), pts[:, 1].min()
x2, y2 = pts[:, 0].max(), pts[:, 1].max()
rect = cv2.minAreaRect(pts.astype(int))               # same as Mothbot
angle = rect[2]                                       # degrees
```

**What this detector deliberately does NOT do** (diverging from Mothbot's `detect.py`):

- No thumbnail/patch files on disk (`generateThumbnailPatches`) — API returns crops in-band via existing paths.
- No JSON sidecar output (`_botdetection.json`) — API returns results in the response.
- No human-detection-file override logic.

**API layer — `trapdata/api/models/localization.py`:**

```python
class APIMothDetector_YOLO11m_Mothbot(
    APIInferenceBaseClass, MothObjectDetector_YOLO11m_Mothbot
):
    task_type = "localization"
    # __init__, reset, get_dataset identical in shape to APIMothDetector
    def save_results(self, item_ids, batch_output, seconds_per_item, *args, **kwargs):
        # Unpack YoloDetection → DetectionResponse with rotation+score populated.
```

### `detector_cls` wiring

**`trapdata/api/models/classification.py`:**

```python
class APIMothClassifier(APIInferenceBaseClass, InferenceBaseClass):
    task_type = "classification"
    detector_cls: type["APIMothDetector"] = APIMothDetector   # default
```

**`trapdata/api/api.py`:**

- `api.py:221` — replace `APIMothDetector(...)` with `Classifier.detector_cls(...)`.
- `api.py:140` (`make_pipeline_config_response`) — same substitution.

### New classifier registration

**`trapdata/api/models/classification.py`:**

```python
class MothbotInsectOrderClassifier(InsectOrderClassifier):
    detector_cls = APIMothDetector_YOLO11m_Mothbot
```

**`trapdata/api/api.py`:** add `"mothbot_insect_orders_2025": MothbotInsectOrderClassifier` to `PIPELINE_CHOICES`.

**Binary filter**: skipped. `should_filter_detections()` at `api.py:76` already returns `False` for `InsectOrderClassifier` subclasses — `MothbotInsectOrderClassifier` inherits that behavior. Same as the existing `insect_orders_2025` pipeline. Rationale: the order classifier itself distinguishes non-moth insects; a binary prefilter would discard signal.

### Schema: `rotation` field on `DetectionResponse`

**`trapdata/api/schemas.py`:**

```python
class DetectionResponse(pydantic.BaseModel):
    # ... existing fields ...
    rotation: float | None = pydantic.Field(
        default=None,
        description=(
            "Rotation angle in degrees (cv2.minAreaRect convention), when "
            "the detector produces oriented bounding boxes. FUTURE: "
            "downstream classifiers may use this to crop a straightened "
            "patch instead of the axis-aligned envelope. See PR discussion "
            "for the proposed RotatedBoundingBox schema upgrade."
        ),
    )
```

Backwards-compatible. Existing FasterRCNN detectors leave it `None`. The Mothbot detector populates it. **This PR does not use the rotation downstream** — the order classifier still crops axis-aligned. The field is there for a future species classifier to use. The description text and PR body will both explicitly call this out as a forward-looking proposal.

### Dependency

`uv add ultralytics` — `pyproject.toml` gets `"ultralytics>=8.3"` in the `[project].dependencies` list (PEP 621), `uv.lock` updates. Separate commit so lockfile churn doesn't muddy the feature diffs.

### Licensing

| Component | License | Notes |
|---|---|---|
| `yolo11m_4500_imgsz1600_b1_2024-01-18.pt` weights | AGPL-3.0 | Tagged in checkpoint metadata: `license: AGPL-3.0 (https://ultralytics.com/license)`. |
| `ultralytics` library | AGPL-3.0 | Upstream. |
| Mothbot_Process repo code | **No explicit license** | No `LICENSE` file, no license field in `pyproject.toml`, no mention in `README.md`. Defaults to "all rights reserved" under copyright law. |
| AMI Data Companion | AGPL-3.0 | Main branch since PR #137. |

**Implication:** weights and ultralytics are cleanly compatible with our AGPL-3.0 project. Mothbot's *code* is unlicensed — we will **not** verbatim-port their files. The detection wrapper is re-implemented in our codebase. The one snippet we do adapt (the PyTorch 2.6 `weights_only_load` fallback, `Mothbot_Process/pipeline/detect.py:44-97`) is boilerplate ultralytics compatibility handling; it will be attributed in a code comment as "adapted from Mothbot_Process/pipeline/detect.py — pattern is standard ultralytics PyTorch 2.6 compat".

### Weights upload (operator step, pre-merge)

User runs (not automated by this PR):

```bash
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
"
```

Matches the pattern from the memory note on Arbutus (AWS CLI broken on this host; boto3 + `request_checksum_calculation='when_required'` needed for this endpoint).

Verify accessibility after upload:

```bash
curl -sI "https://object-arbutus.cloud.computecanada.ca/ami-models/mothbot/detection/yolo11m_4500_imgsz1600_b1_2024-01-18.pt" | head
```

## Tests

1. **Unit — `post_process_single`**: construct a fake ultralytics Result with two known OBB entries (hand-computed 4-corner points); assert `YoloDetection` fields match expected min/max envelope, rotation, and score.
2. **Integration — end-to-end pipeline**: feed one test image from `trapdata/tests/images/` through the new pipeline; assert ≥1 detection and ≥1 classification; follow existing skip patterns for tests that need downloadable weights.
3. **Rename regression**: existing tests for other pipelines continue to pass after the `CLASSIFIER_CHOICES` → `PIPELINE_CHOICES` rename (they already use the dict, just via a new name).

**Not in this PR:** YOLO accuracy eval, OBB correctness at scale, multi-image batching throughput. Those belong in a separate evaluation task.

## Rollout

0. **Rebase** `worktree-mothbot-pipeline` onto current `origin/main` (uv + AGPL-3). Resolve any conflicts before starting.
1. **Upload weights** to Arbutus (operator step, above). Verify the URL is reachable.
2. **Commit 1**: rename `CLASSIFIER_CHOICES` → `PIPELINE_CHOICES` across the 7 files. No behavior change.
3. **Commit 2**: add `detector_cls` class attribute on `APIMothClassifier` with default `APIMothDetector`; swap hardcoded `APIMothDetector` at `api.py:140` and `api.py:221` for `Classifier.detector_cls`. No behavior change for existing pipelines.
4. **Commit 3**: add `ultralytics` dep via `uv add`. Lockfile churn isolated.
5. **Commit 4**: add YOLO detector ML class + `YoloDetection` dataclass + API wrapper + unit test for `post_process_single`.
6. **Commit 5**: add `MothbotInsectOrderClassifier` + register in `PIPELINE_CHOICES` + `rotation` field on `DetectionResponse` + integration test.

Each commit must leave `pytest` passing on its own.

## PR description checklist

- **What**: new `mothbot_insect_orders_2025` pipeline.
- **Why**: users wanting a Mothbot-style detector paired with our order classifier.
- **Dependency**: ultralytics 8.3+ added (AGPL-3; project is already AGPL-3 — no escalation). YOLO weights carry embedded AGPL-3 metadata. Mothbot's code is unlicensed, so we re-implement rather than verbatim-port; one adapted snippet is attributed in a code comment.
- **Rotation field**: forward-looking addition; unused in this PR; proposed upgrade path outlined.
- **Single-class detector**: model outputs only `{0: "creature"}`; taxonomic labels come from the existing ConvNeXt classifier.
- **Rename** `CLASSIFIER_CHOICES` → `PIPELINE_CHOICES`: mechanical, one commit, reviewable alone.
- **Weights**: hosted on Arbutus at `ami-models/mothbot/detection/...`; download-on-first-run via existing `get_or_download_file()`.
- **Test plan**: unit + one integration test; accuracy eval deferred.

## Open for follow-up (not this PR)

- Mothbot pybioclip order/species classifier as a second new pipeline.
- Full `RotatedBoundingBox` schema + rotated crop support in `ClassificationImageDataset`, enabling a species classifier to use tighter rotated crops (where the rotation field would finally get read).
- YOLO detector evaluation against FasterRCNN on the same test set — accuracy, latency, GPU memory.
