"""Tests for :mod:`src.multiscene_sift.band_geometry` — band geometry propagation."""

from __future__ import annotations

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from src.multiscene_sift.band_geometry import (
    apply_world_correction_to_transform,
)
from tests.multiscene_sift.conftest import RESOLUTION


class TestBandGeometry:
    """Tests for band geometry propagation."""

    def test_single_g_per_scene(self, tmp_path):
        """Each scene gets one G_i, applied identically to all bands."""
        G = np.array([
            [1, 0, 300],
            [0, 1, 150],
            [0, 0, 1],
        ])

        base = from_origin(500000, 4000000, RESOLUTION, RESOLUTION)

        # Apply to B14 transform
        b14_tf = apply_world_correction_to_transform(base, G)
        # Apply to B8 transform (same original transform, same G)
        b8_tf = apply_world_correction_to_transform(base, G)

        assert b14_tf == b8_tf
        assert b14_tf.c == pytest.approx(500300.0, abs=1e-6)
        assert b14_tf.f == pytest.approx(4000150.0, abs=1e-6)

    def test_different_original_transforms(self):
        """Bands with different original transforms get same correction."""
        G = np.eye(3)
        t1 = from_origin(500000, 4000000, 30, 30)
        t2 = from_origin(500000, 4000000, 15, 15)  # different res

        r1 = apply_world_correction_to_transform(t1, G)
        r2 = apply_world_correction_to_transform(t2, G)

        assert r1 == t1
        assert r2 == t2
        assert r1 != r2  # original transforms differ