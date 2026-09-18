"""Tests for _pixel_window_from_bounds with affine transforms (Task 4)."""

import math
import numpy as np
import pytest
from rasterio.transform import Affine, from_origin

from src.volrn import _pixel_window_from_bounds


def _geo_bounds_for_pixel_rect(tf, shape, r0, r1, c0, c1):
    """Forward-transform a pixel rectangle to geo, return its bounding box."""
    geo_corners = [
        tf * (c0, r0), tf * (c1, r0),
        tf * (c0, r1), tf * (c1, r1),
    ]
    xs = [p[0] for p in geo_corners]
    ys = [p[1] for p in geo_corners]
    return (min(xs), min(ys), max(xs), max(ys))


def _assert_window_recovers_pixel_rect(result, geo_bounds, tf, shape,
                                       r0, r1, c0, c1):
    """Assert the returned window encloses the four pixel corners of a rect."""
    rows, cols = shape
    r_s, r_e, c_s, c_e = result
    pixel_rect_corners = [(c0, r0), (c1, r0), (c0, r1), (c1, r1)]
    for pc, pr in pixel_rect_corners:
        gx, gy = tf * (pc, pr)
        px, py = ~tf * (gx, gy)  # recovers (pc, pr) up to float tolerance
        # The geo box encloses these corners, so the window must contain them
        assert c_s - 1 <= px <= c_e + 1, (
            f"pixel ({pc},{pr}) -> geo ({gx:.1f},{gy:.1f}) ∉ cols [{c_s},{c_e}]"
        )
        assert r_s - 1 <= py <= r_e + 1, (
            f"pixel ({pc},{pr}) -> geo ({gx:.1f},{gy:.1f}) ∉ rows [{r_s},{r_e}]"
        )


class TestPixelWindowFromBounds:
    """Tests for _pixel_window_from_bounds with different transforms."""

    def test_north_up_standard(self):
        """Standard north-up transform: should match old behaviour."""
        tf = from_origin(500000, 4000000, 30, 30)
        shape = (200, 300)
        bounds = (500300, 3999500, 500900, 3999800)

        result = _pixel_window_from_bounds(bounds, tf, shape)
        assert result is not None
        r_s, r_e, c_s, c_e = result

        # bounds cover pixels cols 10-30, rows ~7-17 (north-up)
        assert c_s >= 8 and c_e <= 32
        assert r_s >= 5 and r_e <= 20

    def test_rotated_transform(self):
        """45° rotation: window must recover the full pixel rectangle."""
        angle = math.radians(45)
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        tf = Affine(30 * cos_a, -30 * sin_a, 500000,
                    30 * sin_a, 30 * cos_a, 4000000)
        shape = (500, 500)
        # Pixel rect [200..320]×[200..300], mapped to geo then back
        bounds = _geo_bounds_for_pixel_rect(tf, shape, 200, 300, 200, 320)

        result = _pixel_window_from_bounds(bounds, tf, shape)
        assert result is not None
        _assert_window_recovers_pixel_rect(result, bounds, tf, shape,
                                           200, 300, 200, 320)

    def test_shear_transform(self):
        """Shear transform: window must recover the full pixel rectangle."""
        tf = Affine(30, 5, 500000, 2, -30, 4000000)
        shape = (500, 500)
        # Pixel rect [150..250]×[100..200], mapped to geo then back
        bounds = _geo_bounds_for_pixel_rect(tf, shape, 100, 200, 150, 250)

        result = _pixel_window_from_bounds(bounds, tf, shape)
        assert result is not None
        _assert_window_recovers_pixel_rect(result, bounds, tf, shape,
                                           100, 200, 150, 250)

    def test_partial_overlap_clipped(self):
        """Geo bounds partially outside raster: window clipped to [0, shape]."""
        tf = from_origin(500000, 4000000, 30, 30)
        shape = (200, 300)
        # bounds extend outside raster on left (498000) and right (510000),
        # and above the top (4001000), but bottom 3995000 is inside
        bounds = (498000, 3995000, 510000, 4001000)

        result = _pixel_window_from_bounds(bounds, tf, shape)
        assert result is not None
        r_s, r_e, c_s, c_e = result
        # Left/right/top extend beyond raster → clipped
        assert r_s == 0
        assert c_s == 0
        assert c_e == 300
        # Bottom (y=3995000) maps to row (4000000-3995000)/30 = 166.67 → 167
        assert r_e == 167

    def test_fully_disjoint_returns_none(self):
        """Geo bounds completely outside raster: returns None."""
        tf = from_origin(500000, 4000000, 30, 30)
        shape = (200, 300)
        # bounds far from the image
        bounds = (600000, 4100000, 610000, 4110000)

        result = _pixel_window_from_bounds(bounds, tf, shape)
        assert result is None