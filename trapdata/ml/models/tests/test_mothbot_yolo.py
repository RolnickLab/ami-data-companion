"""Unit tests for the Mothbot YOLO detector's post-processing helpers.

These tests stay pure-CPU and don't load any model weights -- they only
exercise the coordinate math that converts YOLO's 4 rotated corner
points into the (axis-aligned-bbox + rotation + score) shape our API
consumes. The model-loading path is covered by the integration test.
"""

import numpy as np

from trapdata.ml.models.localization import YoloDetection, _corners_to_yolo_detection


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

    dets = detector.post_process_single(mock_result)

    assert (
        len(dets) == 1
    ), f"Expected 1 detection (degenerate filtered), got {len(dets)}: {dets}"
    assert dets[0].x2 - dets[0].x1 > 0
    assert dets[0].y2 - dets[0].y1 > 0
