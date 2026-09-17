"""Tests for :mod:`src.multiscene_sift.band_geometry` — pixel-to-world conversion."""

from __future__ import annotations

import numpy as np
import pytest
from rasterio.transform import from_origin

from src.multiscene_sift.band_geometry import (
    pixel_affine_to_world,
    apply_world_correction_to_transform,
)


class TestPixelAffineToWorld:
    """Tests for :func:`pixel_affine_to_world`."""

    def test_identity_stays_identity(self):
        """Identity pixel matrix should map to identity world matrix."""
        M = np.eye(3)
        T = from_origin(500000, 4000000, 30, 30)
        A = pixel_affine_to_world(M, T)
        np.testing.assert_allclose(A, np.eye(3), atol=1e-8)

    def test_pixel_translation_to_world(self):
        """A known pixel translation should convert to correct world units."""
        # pixel size = 30 m, north-up
        T = from_origin(500000, 4000000, 30, 30)
        # Translation: +10 px in X, -5 px in Y (pixel space, Y is row-down)
        M = np.array([
            [1, 0, 10],
            [0, 1, -5],
            [0, 0, 1],
        ], dtype=np.float64)
        A = pixel_affine_to_world(M, T)

        # Expect: world X += 10 * 30 = +300 m
        #         world Y += -(-5 * 30) = +150 m (because Y pixel→world is neg)
        # Actually: T @ M @ T^{-1}
        # T[:2,:2] = [[30, 0], [0, -30]]
        # T * M * T^{-1}:
        # Translation part: T[:2,:2] * M[:2,2] = [[30*10], [-30*(-5)]] = [300, 150]
        np.testing.assert_allclose(A[0, 2], 300.0, atol=1e-6)
        np.testing.assert_allclose(A[1, 2], 150.0, atol=1e-6)

    def test_rejects_non_3x3(self):
        """Non-3×3 matrices should raise ValueError."""
        T = from_origin(500000, 4000000, 30, 30)
        with pytest.raises(ValueError, match="3×3"):
            pixel_affine_to_world(np.eye(2), T)


class TestApplyWorldCorrection:
    """Tests for :func:`apply_world_correction_to_transform`."""

    def test_identity_correction_preserves_transform(self):
        """Identity G should not change the transform."""
        T_orig = from_origin(500000, 4000000, 30, 30)
        G = np.eye(3)
        T_new = apply_world_correction_to_transform(T_orig, G)
        assert T_new == pytest.approx(T_orig)

    def test_translation_correction(self):
        """G with translation should shift the transform origin."""
        T_orig = from_origin(500000, 4000000, 30, 30)
        # Shift 300 m east, 150 m north
        G = np.array([
            [1, 0, 300],
            [0, 1, 150],
            [0, 0, 1],
        ])
        T_new = apply_world_correction_to_transform(T_orig, G)
        # New origin = old origin + translation
        assert T_new.c == pytest.approx(500300.0, abs=1e-6)
        assert T_new.f == pytest.approx(4000150.0, abs=1e-6)