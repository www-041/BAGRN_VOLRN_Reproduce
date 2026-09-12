"""Unit tests for the opt-in KLT/TPS registration backend."""

import cv2
import numpy as np
import pytest
from rasterio.transform import Affine


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


def test_map_klt_matches_identity_grid_preserves_displacement():
    from src.klt_tps_registration import map_klt_matches_to_moving_grid

    p = np.array([[10.0, 20.0], [30.0, 40.0]])
    q = p + np.array([2.5, -1.25])
    mapped = map_klt_matches_to_moving_grid(
        p, q, Affine(14, 0, 1000, 0, -14, 2000),
        Affine(14, 0, 1000, 0, -14, 2000),
    )
    np.testing.assert_allclose(mapped["control_points_xy"], p, atol=1e-9)
    np.testing.assert_allclose(mapped["source_points_xy"], q, atol=1e-9)
    np.testing.assert_allclose(mapped["displacement_xy"], q - p, atol=1e-9)


def test_map_klt_matches_translated_overlap_to_moving_native_grid():
    from src.klt_tps_registration import map_klt_matches_to_moving_grid

    overlap_transform = Affine(14, 0, 696416, 0, -14, 3375974)
    moving_transform = Affine(14, 0, 659722, 0, -14, 3371634)
    p = np.array([[20.0, 40.0], [300.0, 500.0]])
    q = p + np.array([2.5, -1.25])
    mapped = map_klt_matches_to_moving_grid(
        p, q, overlap_transform, moving_transform,
    )
    assert not np.allclose(mapped["control_points_xy"], p)
    np.testing.assert_allclose(mapped["displacement_xy"], q - p, atol=1e-9)


def test_estimate_pair_uses_existing_geographic_overlap_context(monkeypatch):
    from src import klt_tps_registration as module

    ref = np.arange(64 * 64, dtype=np.float32).reshape(64, 64)
    moving = ref.copy()
    transform = Affine(2, 0, 100, 0, -2, 200)
    overlap_transform = Affine(2, 0, 500, 0, -2, 800)
    valid = np.ones((32, 32), dtype=bool)
    monkeypatch.setattr(module, "build_pair_overlap_context", lambda *args: {
        "available": True,
        "ref_window": (10, 42, 12, 44),
        "tgt_window": (5, 37, 7, 39),
        "shape": (32, 32),
        "overlap_transform": overlap_transform,
        "ref_overlap": np.ones((32, 32), dtype=np.float32),
        "tgt_overlap": np.ones((32, 32), dtype=np.float32),
        "ref_valid": valid,
        "tgt_valid": valid,
        "resampled_target": True,
    })
    p = np.array([[4.0, 5.0], [20.0, 25.0], [25.0, 8.0]])
    q = p + np.array([1.5, -0.75])
    monkeypatch.setattr(module, "match_bidirectional_klt", lambda *args: {
        "initial_corner_count": 3,
        "accepted_point_count": 3,
        "reference_points_xy": p,
        "moving_points_xy": q,
        "displacement_xy": q - p,
        "forward_backward_error": np.zeros(3),
    })
    result = module.estimate_klt_tps_pair(
        ref, transform, moving, transform, None, None, _klt_params()
    )
    expected = module.map_klt_matches_to_moving_grid(
        p, q, overlap_transform, transform,
    )
    np.testing.assert_allclose(result["control_points_moving_xy"], expected["control_points_xy"])
    np.testing.assert_allclose(result["source_points_moving_xy"], expected["source_points_xy"])
    assert result["overlap_context"]["resampled_target"] is True
