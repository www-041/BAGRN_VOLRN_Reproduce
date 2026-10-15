"""Tests for affine-corrected scene bounds."""

import numpy as np
import pytest
from rasterio.transform import Affine, from_origin

from src.multiscene_sift.mosaicking import raster_bounds_from_transform


class TestRasterBounds:
    def test_north_up_matches_old(self):
        """North-up transform should give same bounds as simple formula."""
        tf = from_origin(500000, 4000000, 30, 30)
        h, w = 100, 200
        l, b, r, t = raster_bounds_from_transform(tf, h, w)
        assert l == pytest.approx(500000.0)
        assert t == pytest.approx(4000000.0)
        assert r == pytest.approx(500000 + 200 * 30)
        assert b == pytest.approx(4000000 - 100 * 30)

    def test_rotated_encloses_corners(self):
        """Rotated transform should enclose all four corners."""
        # 45-degree rotation
        import math
        angle = math.radians(45)
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        tf = Affine(30 * cos_a, -30 * sin_a, 500000,
                    30 * sin_a, 30 * cos_a, 4000000)
        h, w = 100, 200
        l, b, r, t = raster_bounds_from_transform(tf, h, w)

        # Verify all four corners are within bounds
        corners = [
            tf * (0, 0), tf * (w, 0), tf * (0, h), tf * (w, h),
        ]
        for cx, cy in corners:
            assert l <= cx <= r, f"corner x={cx} outside [{l}, {r}]"
            assert b <= cy <= t, f"corner y={cy} outside [{b}, {t}]"

    def test_shear_yields_valid(self):
        """Sheared transform should produce valid finite bounds."""
        tf = Affine(30, 5, 500000, 2, -30, 4000000)
        h, w = 100, 200
        l, b, r, t = raster_bounds_from_transform(tf, h, w)
        assert np.isfinite(l) and np.isfinite(b) and np.isfinite(r) and np.isfinite(t)
        assert l < r and b < t