"""Tests for the shared memory-safe affine estimator."""

from __future__ import annotations

import numpy as np
import pytest
from skimage.measure import ransac as legacy_ransac
from skimage.transform import AffineTransform

from src.registration_benchmark.geometry import estimate_affine_lstsq
from src.registration_benchmark.geometry import STATUS_OK, fit_affine_ransac
from src.registration_benchmark.models import MatchSet


def _apply(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack([points, np.ones(len(points))])
    return (matrix @ homogeneous.T).T[:, :2]


def test_exact_affine_is_recovered_from_non_collinear_points():
    matrix = np.array(
        [
            [1.002, -0.015, 12.5],
            [0.010, 0.998, -7.25],
            [0.0, 0.0, 1.0],
        ]
    )
    src = np.column_stack(
        [np.linspace(-200.0, 200.0, 100), np.sin(np.linspace(-3.0, 3.0, 100)) * 80]
    )
    dst = _apply(matrix, src)

    estimated = estimate_affine_lstsq(src, dst)

    np.testing.assert_allclose(estimated, matrix, rtol=0, atol=1e-9)


def test_noisy_affine_estimate_is_numerically_stable():
    rng = np.random.default_rng(123)
    matrix = np.array(
        [
            [1.002, -0.015, 12.5],
            [0.010, 0.998, -7.25],
            [0.0, 0.0, 1.0],
        ]
    )
    src = rng.uniform(-500.0, 500.0, size=(500, 2))
    dst = _apply(matrix, src) + rng.normal(0.0, 0.05, size=(500, 2))

    estimated = estimate_affine_lstsq(src, dst)

    np.testing.assert_allclose(estimated, matrix, rtol=0, atol=2e-3)


@pytest.mark.parametrize(
    "src,dst,match",
    [
        (np.zeros((3, 2)), np.zeros((2, 2)), "same shape"),
        (np.zeros((3, 3)), np.zeros((3, 2)), "shape (N, 2)"),
        (np.zeros((2, 2)), np.zeros((2, 2)), "at least 3"),
        (np.array([[0.0, 0.0], [1.0, 0.0], [np.nan, 1.0]]), np.zeros((3, 2)), "finite"),
        (np.zeros((3, 2)), np.array([[0.0, 0.0], [1.0, 0.0], [np.inf, 1.0]]), "finite"),
        (np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]), np.zeros((3, 2)), "rank"),
    ],
)
def test_invalid_affine_inputs_fail_with_controlled_value_error(src, dst, match):
    with pytest.raises(ValueError, match=match.replace("(", "\\(").replace(")", "\\)")):
        estimate_affine_lstsq(src, dst)


def test_shared_ransac_does_not_call_full_numpy_svd(monkeypatch):
    rng = np.random.default_rng(456)
    src = rng.uniform(-500.0, 500.0, size=(300, 2))
    matrix = np.array(
        [
            [1.002, -0.015, 12.5],
            [0.010, 0.998, -7.25],
            [0.0, 0.0, 1.0],
        ]
    )
    dst = _apply(matrix, src) + rng.normal(0.0, 0.05, size=(300, 2))
    matches = MatchSet(
        method="synthetic",
        ref_xy=dst,
        tgt_xy=src,
        confidence=np.ones(len(src)),
        runtime_sec=0.0,
        metadata={
            "coordinate_frame": "pair_common_grid",
            "confidence_semantics": "method_internal_only",
        },
    )

    def fail_full_svd(*args, **kwargs):
        raise AssertionError("full SVD must not be used by dense affine estimator")

    monkeypatch.setattr(np.linalg, "svd", fail_full_svd)

    result = fit_affine_ransac(matches, max_trials=50, random_seed=0)

    assert result.status == STATUS_OK
    assert result.inlier_ratio > 0.95


def test_lstsq_matches_legacy_affine_estimate_at_normal_scale():
    rng = np.random.default_rng(789)
    matrix = np.array(
        [
            [1.004, -0.012, 8.0],
            [0.009, 0.997, -5.0],
            [0.0, 0.0, 1.0],
        ]
    )
    src = rng.uniform(-500.0, 500.0, size=(200, 2))
    dst = _apply(matrix, src) + rng.normal(0.0, 0.15, size=(200, 2))

    old_model = AffineTransform.from_estimate(src, dst)
    new_matrix = estimate_affine_lstsq(src, dst)

    # scikit-image normalizes coordinates before its homogeneous SVD solve;
    # this introduces a measured few-e-6 difference from direct least squares.
    np.testing.assert_allclose(new_matrix, old_model.params, rtol=0, atol=1e-5)


def test_ransac_normal_scale_is_numerically_compatible_with_legacy_backend():
    rng = np.random.default_rng(790)
    matrix = np.array(
        [
            [1.004, -0.012, 8.0],
            [0.009, 0.997, -5.0],
            [0.0, 0.0, 1.0],
        ]
    )
    src_inliers = rng.uniform(-500.0, 500.0, size=(800, 2))
    dst_inliers = _apply(matrix, src_inliers) + rng.normal(
        0.0, 0.15, size=(800, 2)
    )
    src_outliers = rng.uniform(-500.0, 500.0, size=(200, 2))
    dst_outliers = rng.uniform(-500.0, 500.0, size=(200, 2))
    src = np.vstack([src_inliers, src_outliers])
    dst = np.vstack([dst_inliers, dst_outliers])

    old_model, old_mask = legacy_ransac(
        (src, dst),
        AffineTransform,
        min_samples=3,
        residual_threshold=2.0,
        max_trials=5000,
        rng=0,
    )
    matches = MatchSet(
        method="synthetic",
        ref_xy=dst,
        tgt_xy=src,
        confidence=np.ones(len(src)),
        runtime_sec=0.0,
        metadata={
            "coordinate_frame": "pair_common_grid",
            "confidence_semantics": "method_internal_only",
        },
    )
    new_result = fit_affine_ransac(matches, max_trials=5000, random_seed=0)

    assert old_model is not None
    assert old_mask is not None
    assert new_result.status == STATUS_OK
    assert abs(int(old_mask.sum()) - new_result.n_inlier) / old_mask.sum() <= 0.01

    old_residuals = old_model.residuals(src[old_mask], dst[old_mask])
    new_residuals = new_result.model.residuals(
        src[new_result.inlier_mask], dst[new_result.inlier_mask]
    )
    old_rmse = float(np.sqrt(np.mean(old_residuals**2)))
    new_rmse = float(np.sqrt(np.mean(new_residuals**2)))
    old_p95 = float(np.percentile(old_residuals, 95))
    new_p95 = float(np.percentile(new_residuals, 95))
    assert abs(old_rmse - new_rmse) <= 0.05
    assert abs(old_p95 - new_p95) <= 0.05

    controls = rng.uniform(-500.0, 500.0, size=(50, 2))
    np.testing.assert_allclose(
        new_result.model(controls), old_model(controls), rtol=0, atol=0.05
    )


def test_affine_direction_is_src_to_dst_with_scale_rotation_and_shear():
    theta = np.deg2rad(1.0)
    rotation = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )
    linear = rotation @ np.array([[1.01, 0.012], [0.0, 0.99]])
    matrix = np.array(
        [[linear[0, 0], linear[0, 1], 12.5], [linear[1, 0], linear[1, 1], -7.25], [0, 0, 1.0]]
    )
    src = np.column_stack(
        [np.linspace(-200.0, 200.0, 150), np.cos(np.linspace(-3.0, 3.0, 150)) * 100]
    )
    dst = _apply(matrix, src)
    matches = MatchSet(
        method="synthetic",
        ref_xy=dst,
        tgt_xy=src,
        confidence=np.ones(len(src)),
        runtime_sec=0.0,
        metadata={
            "coordinate_frame": "pair_common_grid",
            "confidence_semantics": "method_internal_only",
        },
    )

    result = fit_affine_ransac(matches, random_seed=0)

    assert result.status == STATUS_OK
    np.testing.assert_allclose(result.model(src), dst, rtol=0, atol=1e-8)


def _dense_stress_matches(n_points: int, seed: int) -> MatchSet:
    rng = np.random.default_rng(seed)
    matrix = np.array(
        [
            [1.002, -0.015, 12.5],
            [0.010, 0.998, -7.25],
            [0.0, 0.0, 1.0],
        ]
    )
    n_inliers = int(n_points * 0.97)
    src_inliers = rng.uniform(-5000.0, 5000.0, size=(n_inliers, 2))
    dst_inliers = _apply(matrix, src_inliers) + rng.normal(
        0.0, 0.10, size=(n_inliers, 2)
    )
    src_outliers = rng.uniform(-5000.0, 5000.0, size=(n_points - n_inliers, 2))
    dst_outliers = rng.uniform(-5000.0, 5000.0, size=(n_points - n_inliers, 2))
    src = np.vstack([src_inliers, src_outliers])
    dst = np.vstack([dst_inliers, dst_outliers])
    return MatchSet(
        method="dense_synthetic",
        ref_xy=dst,
        tgt_xy=src,
        confidence=np.ones(n_points),
        runtime_sec=0.0,
        metadata={
            "coordinate_frame": "pair_common_grid",
            "confidence_semantics": "method_internal_only",
        },
    )


@pytest.mark.parametrize(
    "n_points,seed", [(20_000, 901), (50_000, 902)], ids=["20k", "50k"]
)
def test_dense_affine_ransac_stress_is_memory_safe(n_points, seed):
    result = fit_affine_ransac(
        _dense_stress_matches(n_points, seed),
        residual_threshold=2.0,
        max_trials=5000,
        random_seed=0,
    )

    assert result.status == STATUS_OK
    assert result.model is not None
    assert np.isfinite(result.model.params).all()
    assert result.inlier_ratio > 0.94
    assert np.isfinite(result.residual_rmse)
    assert np.isfinite(result.residual_p95)
