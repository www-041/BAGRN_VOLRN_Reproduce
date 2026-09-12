"""Unit tests for the opt-in KLT/TPS registration backend."""

import cv2
import numpy as np
import pytest


def test_opencv_dependency_is_available():
    assert cv2.__version__


def test_normalize_klt_image_returns_uint8_and_masks_invalid():
    from src.klt_tps_registration import normalize_klt_image

    y, x = np.mgrid[0:96, 0:96]
    image = (x + 2 * y).astype(np.float32)
    valid = np.ones_like(image, dtype=bool)
    valid[:8, :] = False
    result = normalize_klt_image(image, valid)
    assert result.dtype == np.uint8
    assert result.shape == image.shape
    assert np.isfinite(result).all()
    assert result.min() >= 0
    assert result.max() <= 255


def test_build_klt_interior_mask_erodes_boundary():
    from src.klt_tps_registration import build_klt_interior_mask

    valid = np.ones((25, 25), dtype=bool)
    mask = build_klt_interior_mask(valid, 3)
    assert mask.dtype == np.uint8
    assert mask[0, 0] == 0
    assert mask[12, 12] == 1


def _textured_scene(shape=(192, 192)):
    image = np.zeros(shape, dtype=np.float32)
    for cy in range(18, shape[0] - 10, 28):
        for cx in range(18, shape[1] - 10, 28):
            cv2.circle(image, (cx, cy), 7, 80 + ((cx + cy) % 100), -1)
            cv2.rectangle(image, (cx - 11, cy + 10), (cx + 5, cy + 17), 35, -1)
    cv2.line(image, (8, 80), (shape[1] - 8, shape[0] - 24), 210, 2)
    cv2.line(image, (12, shape[0] - 20), (shape[1] - 20, 12), 150, 2)
    return cv2.GaussianBlur(image, (3, 3), 0)


def _klt_params(**overrides):
    params = {
        "klt_tps_window": 9,
        "klt_tps_pyramid_level": 3,
        "klt_tps_max_corners": 4000,
        "klt_tps_min_corner_distance": 5.0,
        "klt_tps_fb_threshold": 0.5,
        "klt_tps_min_points": 30,
        "klt_tps_threads": 1,
    }
    params.update(overrides)
    return params


def test_bidirectional_klt_recovers_synthetic_subpixel_translation():
    from src.klt_tps_registration import match_bidirectional_klt

    reference = _textured_scene()
    expected = np.array([2.25, -1.50])
    moving = cv2.warpAffine(
        reference, np.array([[1, 0, expected[0]], [0, 1, expected[1]]], dtype=np.float32),
        (reference.shape[1], reference.shape[0]), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101,
    )
    valid = np.ones(reference.shape, dtype=bool)
    result = match_bidirectional_klt(reference, moving, valid, valid, _klt_params())
    assert result["initial_corner_count"] >= 30
    assert result["accepted_point_count"] >= 30
    median = np.median(result["displacement_xy"], axis=0)
    assert np.linalg.norm(median - expected) < 0.35
    assert np.percentile(result["forward_backward_error"], 95) < 0.5


def test_bidirectional_klt_rejects_too_few_corners():
    from src.klt_tps_registration import match_bidirectional_klt

    image = np.zeros((96, 96), dtype=np.float32)
    valid = np.ones(image.shape, dtype=bool)
    with pytest.raises(ValueError, match="insufficient"):
        match_bidirectional_klt(image, image, valid, valid, _klt_params())


def test_bidirectional_klt_rejects_collinear_controls(monkeypatch):
    from src import klt_tps_registration as module

    points = np.column_stack([np.arange(40, dtype=np.float32) + 20, np.full(40, 40, dtype=np.float32)])
    monkeypatch.setattr(module.cv2, "goodFeaturesToTrack", lambda *args, **kwargs: points.reshape(-1, 1, 2))
    monkeypatch.setattr(module.cv2, "calcOpticalFlowPyrLK", lambda *args, **kwargs: (points.reshape(-1, 1, 2), np.ones((40, 1), dtype=np.uint8), None))
    image = _textured_scene((96, 96))
    valid = np.ones(image.shape, dtype=bool)
    with pytest.raises(ValueError, match="collinear"):
        module.match_bidirectional_klt(image, image, valid, valid, _klt_params())
