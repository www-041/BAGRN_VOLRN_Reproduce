"""Tests for :mod:`src.registration_benchmark.geometry`."""

import numpy as np
import pytest
from skimage.transform import AffineTransform

from src.registration_benchmark.geometry import (
    STATUS_INVALID_GEOMETRY,
    STATUS_OK,
    GeometryResult,
    _decompose_affine,
    _is_geometry_valid,
    fit_affine_ransac,
    warp_target_common_grid,
)
from src.registration_benchmark.models import MatchSet


# ---------------------------------------------------------------------------
# Helper: generate synthetic matches with known ground-truth
# ---------------------------------------------------------------------------


def _make_affine_matches(
    n_inliers: int = 100,
    n_outliers: int = 20,
    tx: float = 6.5,
    ty: float = -4.0,
    rot_deg: float = 0.5,
    noise_std: float = 0.3,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, AffineTransform]:
    """Generate synthetic tie-points with a known affine transformation.

    Returns ``(ref_xy, tgt_xy, true_model)``.
    """
    rng = np.random.default_rng(seed)

    # Ground-truth affine: rotation + translation
    theta = np.radians(rot_deg)
    true_model = AffineTransform(
        matrix=[
            [np.cos(theta), -np.sin(theta), tx],
            [np.sin(theta), np.cos(theta), ty],
            [0, 0, 1],
        ]
    )

    # Generate random target points within a 512×512 image
    tgt_inliers = rng.uniform(50, 450, size=(n_inliers, 2))

    # Map to reference
    ref_inliers = true_model(tgt_inliers) + rng.normal(0, noise_std, tgt_inliers.shape)

    # Random outliers
    tgt_outliers = rng.uniform(50, 450, size=(n_outliers, 2))
    ref_outliers = rng.uniform(50, 450, size=(n_outliers, 2))

    tgt_xy = np.vstack([tgt_inliers, tgt_outliers])
    ref_xy = np.vstack([ref_inliers, ref_outliers])

    return ref_xy, tgt_xy, true_model


class TestFitAffineRansac:
    """Tests for :func:`fit_affine_ransac`."""

    def test_recovers_known_translation(self):
        """Known translation of (+6.5, -4.0) px with 0.5° rotation."""
        ref_xy, tgt_xy, true_model = _make_affine_matches()

        ms = MatchSet(
            method="synthetic",
            ref_xy=ref_xy,
            tgt_xy=tgt_xy,
            confidence=np.ones(len(ref_xy)),
            runtime_sec=0.0,
        )
        ms.validate()

        result = fit_affine_ransac(ms, residual_threshold=2.0)

        assert result.status == STATUS_OK
        assert result.n_inlier >= 90
        assert result.n_raw == 120
        assert result.inlier_ratio >= 0.80
        assert result.residual_rmse < 1.0  # Low noise → low RMSE
        assert result.model is not None

        # Check decomposition is reasonable
        assert 0.98 <= result.affine_scale_x <= 1.02
        assert 0.98 <= result.affine_scale_y <= 1.02

    def test_too_few_points(self):
        """RANSAC with fewer than 3 points."""
        ms = MatchSet(
            method="tiny",
            ref_xy=np.array([[1.0, 2.0]]),
            tgt_xy=np.array([[3.0, 4.0]]),
            confidence=np.array([1.0]),
            runtime_sec=0.0,
        )

        result = fit_affine_ransac(ms)
        assert result.status == "TOO_FEW_INLIERS"
        assert result.model is None

    def test_empty_matches(self):
        ms = MatchSet(
            method="empty",
            ref_xy=np.empty((0, 2)),
            tgt_xy=np.empty((0, 2)),
            confidence=np.empty(0),
            runtime_sec=0.0,
        )

        result = fit_affine_ransac(ms)
        assert result.status == "TOO_FEW_INLIERS"
        assert result.n_raw == 0
        assert result.n_inlier == 0

    def test_inlier_ratio_below_threshold(self):
        """Mostly outliers → should fail inlier ratio check."""
        rng = np.random.default_rng(99)
        ref_xy = rng.uniform(0, 512, size=(100, 2))
        tgt_xy = rng.uniform(0, 512, size=(100, 2))

        ms = MatchSet(
            method="noise",
            ref_xy=ref_xy,
            tgt_xy=tgt_xy,
            confidence=np.ones(100),
            runtime_sec=0.0,
        )

        result = fit_affine_ransac(ms, residual_threshold=0.5)

        # With pure random noise and very tight threshold, RANSAC should
        # find at most a handful of inliers.
        assert result.inlier_ratio < 0.30 or result.status != STATUS_OK

    def test_invalid_geometry_rejected(self):
        """A model with scale far from 1 should be rejected."""
        ref_xy, tgt_xy, _ = _make_affine_matches(
            n_inliers=80, n_outliers=5, tx=5.0, ty=3.0, rot_deg=0.0, noise_std=0.1
        )
        # Artificially scale tgt_xy to simulate extreme scale change
        tgt_xy_scaled = tgt_xy * 0.5  # Scale 0.5x → out of [0.95, 1.05]

        ms = MatchSet(
            method="scaled",
            ref_xy=ref_xy,
            tgt_xy=tgt_xy_scaled,
            confidence=np.ones(len(ref_xy)),
            runtime_sec=0.0,
        )

        result = fit_affine_ransac(ms, residual_threshold=5.0)
        # Scale should be flagged as invalid
        assert result.status == STATUS_INVALID_GEOMETRY

    def test_result_fields_populated(self):
        ref_xy, tgt_xy, _ = _make_affine_matches()

        ms = MatchSet(
            method="synthetic",
            ref_xy=ref_xy,
            tgt_xy=tgt_xy,
            confidence=np.ones(len(ref_xy)),
            runtime_sec=0.0,
        )
        ms.validate()

        result = fit_affine_ransac(ms, residual_threshold=2.0)

        assert result.status == STATUS_OK
        assert result.inlier_mask.shape == (len(ref_xy),)
        assert result.inlier_mask.dtype == bool
        assert not np.isnan(result.residual_median)
        assert not np.isnan(result.residual_rmse)
        assert not np.isnan(result.affine_scale_x)


class TestWarpTargetCommonGrid:
    """Tests for :func:`warp_target_common_grid`."""

    def test_warp_preserves_shape(self):
        """Warped output should keep the same shape as input."""
        # Create a simple identity model
        model = AffineTransform(matrix=[[1, 0, 0], [0, 1, 0], [0, 0, 1]])

        target = np.ones((128, 128), dtype=np.float64) * 100.0
        valid = np.ones((128, 128), dtype=bool)

        warped, warped_valid = warp_target_common_grid(target, valid, model)

        assert warped.shape == target.shape
        assert warped_valid.shape == target.shape

    def test_warp_identity_preserves_values(self):
        """Identity warp should preserve pixel values (within interpolation tolerance)."""
        model = AffineTransform(matrix=[[1, 0, 0], [0, 1, 0], [0, 0, 1]])

        rng = np.random.default_rng(42)
        target = rng.uniform(50, 200, size=(64, 64)).astype(np.float64)
        valid = np.ones((64, 64), dtype=bool)

        warped, warped_valid = warp_target_common_grid(target, valid, model)

        # Center region should be close (edges may differ due to interpolation)
        center = warped[16:48, 16:48]
        center_orig = target[16:48, 16:48]
        assert np.allclose(center, center_orig, rtol=0.05, atol=1.0)

    def test_warp_with_nan_invalid(self):
        """Invalid pixels become NaN after warp."""
        model = AffineTransform(matrix=[[1, 0, 5], [0, 1, -3], [0, 0, 1]])

        target = np.ones((64, 64), dtype=np.float64) * 150.0
        valid = np.ones((64, 64), dtype=bool)
        valid[:10, :] = False  # Top 10 rows invalid

        warped, warped_valid = warp_target_common_grid(target, valid, model)

        # Invalid regions should map to invalid
        assert not warped_valid.all()
        assert np.isnan(warped[~warped_valid]).all()


class TestAffineDecomposition:
    """Tests for internal decomposition helpers."""

    def test_identity_decomposition(self):
        """Identity matrix → scale=1, rotation=0, shear=0."""
        m = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        sx, sy, rot, shear = _decompose_affine(m)
        assert sx == pytest.approx(1.0)
        assert sy == pytest.approx(1.0)
        assert rot == pytest.approx(0.0, abs=1e-6)
        assert shear == pytest.approx(0.0, abs=1e-6)

    def test_pure_translation(self):
        """Translation-only matrix → scale=1, rotation=0."""
        m = np.array([[1, 0, 100], [0, 1, -50], [0, 0, 1]])
        sx, sy, rot, shear = _decompose_affine(m)
        assert sx == pytest.approx(1.0)
        assert sy == pytest.approx(1.0)
        assert rot == pytest.approx(0.0, abs=1e-6)

    def test_pure_rotation(self):
        """45° rotation → scale=1, rotation=45°."""
        theta = np.radians(45)
        m = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ])
        sx, sy, rot, shear = _decompose_affine(m)
        assert sx == pytest.approx(1.0)
        assert sy == pytest.approx(1.0)
        assert rot == pytest.approx(45.0, abs=1.0)

    def test_scale_outside_range_invalid(self):
        assert not _is_geometry_valid(0.5, 1.0, 0.0, 0.0)
        assert not _is_geometry_valid(1.0, 2.0, 0.0, 0.0)

    def test_large_rotation_invalid(self):
        assert not _is_geometry_valid(1.0, 1.0, 10.0, 0.0)

    def test_large_shear_invalid(self):
        assert not _is_geometry_valid(1.0, 1.0, 0.0, 5.0)

    def test_valid_geometry_passes(self):
        assert _is_geometry_valid(1.0, 1.0, 0.0, 0.0)
        assert _is_geometry_valid(0.98, 1.02, -1.5, 1.0)