"""Unit tests for the Mothbot YOLO detector's post-processing helpers.

These tests stay pure-CPU and don't load any model weights -- they only
exercise the coordinate math that converts YOLO's 4 rotated corner
points into the (axis-aligned-bbox + rotation + score) shape our API
consumes. The model-loading path is covered by the integration test.
"""

import numpy as np
import torch

from trapdata.ml.models.localization import (
    YoloDetection,
    _corners_to_yolo_detection,
    _tensor_to_bgr_numpy,
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
    """A rectangle rotated ~45 degrees: envelope is larger than either side, rotation non-zero."""
    # 10x4 rectangle centered at (50, 50), rotated 45 degrees.
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

    # Envelope for a rotated thin rectangle is strictly larger than its short side
    # (at 45 deg the envelope width = (half_w + half_h) * sqrt(2) ~ 9.9, > 2*half_h=4)
    assert (det.x2 - det.x1) > 2 * half_h

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


def test_corners_to_yolo_detection_degenerate_flat_obb():
    """All corners on same y → y1==y2 (H=0). This is the raw math output;
    callers are responsible for filtering such degenerate detections before
    passing them to a classifier crop/resize step."""
    # A perfectly horizontal line: all 4 corners share y=50
    corners = np.array(
        [[0, 50], [1045, 50], [1045, 50], [0, 50]],
        dtype=np.float32,
    )
    det = _corners_to_yolo_detection(corners, score=0.85)
    assert det.y1 == det.y2, "Expected degenerate (H=0) detection"
    assert det.x2 > det.x1, "Width should be non-zero"


def test_tensor_to_bgr_numpy_swaps_channel_order():
    """Antenna's REST dataloader feeds YOLO via ToTensor(), which produces RGB.
    Ultralytics does NOT reorder numpy inputs; it was trained on BGR (cv2.imread).
    Without the flip, detections are large / low-quality because color stats
    don't match training."""
    # Pure-red RGB tensor: R=1.0, G=0, B=0 in (C, H, W) layout
    rgb_tensor = torch.zeros((3, 4, 5))
    rgb_tensor[0] = 1.0  # R channel

    bgr_np = _tensor_to_bgr_numpy(rgb_tensor)

    assert bgr_np.shape == (4, 5, 3), f"Expected (H, W, C), got {bgr_np.shape}"
    assert bgr_np.dtype == np.uint8
    # In BGR layout the red pixel lives in the last channel (index 2)
    assert bgr_np[0, 0, 2] == 255, "Red channel should map to BGR index 2"
    assert bgr_np[0, 0, 0] == 0, "Blue channel should be empty (was G=0 in RGB)"
    assert bgr_np[0, 0, 1] == 0


def test_corners_to_yolo_detection_clamps_negative_coords():
    """YOLO-OBB can emit corners with negative coords for detections near image
    edges (observed live: y1=-274.39 on a 2464-tall image). Without clamping,
    downstream PyTorch slicing treats negatives as end-relative indices and
    produces an empty crop -- which then hard-fails the classifier's Resize."""
    corners = np.array(
        [[-50, -274], [2800, -274], [2800, 936], [-50, 936]],
        dtype=np.float32,
    )
    det = _corners_to_yolo_detection(corners, score=0.9, image_shape=(2464, 3280))
    assert det.x1 == 0.0, f"x1 should clamp to 0, got {det.x1}"
    assert det.y1 == 0.0, f"y1 should clamp to 0, got {det.y1}"
    assert det.x2 == 2800.0
    assert det.y2 == 936.0


def test_corners_to_yolo_detection_clamps_to_image_bounds():
    """Corners extending past the far edge are clamped to (width, height)."""
    corners = np.array(
        [[3500, 2000], [3500, 2600], [4000, 2600], [4000, 2000]],
        dtype=np.float32,
    )
    det = _corners_to_yolo_detection(corners, score=0.9, image_shape=(2464, 3280))
    assert det.x2 == 3280.0, f"x2 should clamp to width, got {det.x2}"
    assert det.y2 == 2464.0, f"y2 should clamp to height, got {det.y2}"
    assert det.x1 == 3280.0  # whole box lies past right edge
    assert det.y1 == 2000.0


def test_post_process_single_clamps_using_orig_shape():
    """post_process_single must pass result.orig_shape through to the clamp
    so detections with negative corners become valid crops."""
    from unittest.mock import MagicMock

    from trapdata.ml.models.localization import MothObjectDetector_YOLO11m_Mothbot

    detector = MothObjectDetector_YOLO11m_Mothbot.__new__(
        MothObjectDetector_YOLO11m_Mothbot
    )

    # Negative y1 (like we see on real Panama diopsis images with YOLO-OBB)
    neg_y_corners = np.array(
        [[1566, -274], [2793, -274], [2793, 936], [1566, 936]], dtype=np.float32
    )

    mock_obb = MagicMock()
    mock_obb.xyxyxyxy.cpu().numpy.return_value = np.stack([neg_y_corners])
    mock_obb.conf.cpu().numpy.return_value = np.array([0.9])

    mock_result = MagicMock()
    mock_result.obb = mock_obb
    mock_result.orig_shape = (2464, 3280)  # (H, W), matches ultralytics convention

    dets = detector.post_process_single(mock_result)
    assert len(dets) == 1
    # y1 must be clamped to 0 -- otherwise worker's int() preserves the sign and
    # PyTorch reads it as end-relative, yielding an empty crop.
    assert dets[0].y1 >= 0
    assert dets[0].x1 >= 0
    assert dets[0].y2 <= 2464
    assert dets[0].x2 <= 3280


def test_post_process_single_filters_degenerate_detections():
    """post_process_single must drop zero-height or zero-width detections so
    that the downstream Resize transform never receives a 0-dimension crop."""
    from unittest.mock import MagicMock

    from trapdata.ml.models.localization import MothObjectDetector_YOLO11m_Mothbot

    # Minimal detector instance (no model loaded, no weights needed)
    detector = MothObjectDetector_YOLO11m_Mothbot.__new__(
        MothObjectDetector_YOLO11m_Mothbot
    )

    # Build a mock ultralytics Result with two detections:
    #   - one valid (10x10 box)
    #   - one degenerate (H=0)
    valid_corners = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32)
    flat_corners = np.array(
        [[0, 50], [1045, 50], [1045, 50], [0, 50]], dtype=np.float32
    )

    mock_obb = MagicMock()
    mock_obb.xyxyxyxy.cpu().numpy.return_value = np.stack(
        [valid_corners, flat_corners]
    )  # shape (2, 4, 2)
    mock_obb.conf.cpu().numpy.return_value = np.array([0.9, 0.85])

    mock_result = MagicMock()
    mock_result.obb = mock_obb
    mock_result.orig_shape = (100, 1100)  # (H, W) covers both test detections

    dets = detector.post_process_single(mock_result)

    assert (
        len(dets) == 1
    ), f"Expected 1 detection (degenerate filtered), got {len(dets)}: {dets}"
    assert dets[0].x2 - dets[0].x1 > 0
    assert dets[0].y2 - dets[0].y1 > 0


def test_predict_batch_gives_each_image_the_same_letterbox_as_alone():
    """An image's detections must not depend on the other images in its batch.

    Ultralytics pads a batch of same-sized images to the smallest rectangle but
    pads a batch of mixed sizes to a full square, and the square padding finds
    fewer objects. predict_batch therefore sends each image size as its own
    call and returns the results in the original order.
    """
    from unittest.mock import MagicMock

    from trapdata.ml.models.localization import MothObjectDetector_YOLO11m_Mothbot

    detector = MothObjectDetector_YOLO11m_Mothbot.__new__(
        MothObjectDetector_YOLO11m_Mothbot
    )
    detector.imgsz = 1600
    detector.bbox_score_threshold = 0.25
    detector.nms_iou_threshold = 0.7
    detector.box_detections_per_img = 300
    detector.device = "cpu"

    calls = []

    def fake_predict(imgs, **kwargs):
        calls.append([img.shape for img in imgs])
        # Identify each image by its pixel value, so swapped results are caught.
        return [int(img[0, 0, 0]) for img in imgs]

    detector.model = MagicMock()
    detector.model.predict.side_effect = fake_predict

    # Pixel values 10, 20, 30 (scaled to uint8 by the RGB-to-BGR conversion).
    first_wide = torch.full((3, 216, 409), 10 / 255)
    small = torch.full((3, 68, 102), 20 / 255)
    second_wide = torch.full((3, 216, 409), 30 / 255)
    results = detector.predict_batch([first_wide, small, second_wide])

    assert all(len(set(shapes)) == 1 for shapes in calls)
    assert sorted(len(shapes) for shapes in calls) == [1, 2]
    assert results == [10, 20, 30]


def test_yolo_loader_refuses_a_checkpoint_with_the_wrong_checksum(tmp_path):
    """Ultralytics unpickles checkpoints, which can run code, so a weights file
    must match its pinned SHA-256 before it is loaded."""
    import pytest

    from trapdata.ml.models.localization import _load_ultralytics_yolo

    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"not the published weights")

    with pytest.raises(ValueError, match="SHA-256"):
        _load_ultralytics_yolo(checkpoint, expected_sha256="0" * 64)


def test_yolo_detectors_pin_a_checksum_for_their_weights():
    from trapdata.ml.models.localization import (
        AnyBugObjectDetector_YOLO26s_Mixed_v2,
        MothObjectDetector_YOLO11m_Mothbot,
    )

    for detector in (
        MothObjectDetector_YOLO11m_Mothbot,
        AnyBugObjectDetector_YOLO26s_Mixed_v2,
    ):
        assert len(detector.weights_sha256) == 64


def test_yolo_detectors_are_not_offered_in_the_desktop_app():
    """The desktop app reads images through a database queue that collates
    tensors, which the YOLO detectors do not accept, so they are API-only."""
    from trapdata.ml.models import ObjectDetectorChoice, object_detectors
    from trapdata.ml.models.localization import MothObjectDetector_YOLO11m_Mothbot

    assert MothObjectDetector_YOLO11m_Mothbot.name not in object_detectors
    assert MothObjectDetector_YOLO11m_Mothbot.get_key() not in {
        choice.name for choice in ObjectDetectorChoice
    }
