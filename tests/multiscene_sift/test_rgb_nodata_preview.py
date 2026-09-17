"""Tests for NoData-aware RGB preview."""

import json
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from src.multiscene_sift.rgb import make_rgb_preview, stack_three_band_geotiff


class TestNoDataPreview:
    def test_nodata_excluded_from_stretch(self, tmp_path):
        """NoData pixels should not affect percentile stretch."""
        tf = from_origin(500000, 4000000, 30, 30)
        nodata = 0

        # Create 3-band image where 75% is NoData=0
        rng = np.random.default_rng(42)
        # Valid area (center 32×32) has values ~10000-60000
        valid_val = 30000 + rng.uniform(-5000, 25000, size=(3, 32, 32)).astype("uint16")

        # Full 64×64 with NoData background
        data = np.zeros((3, 64, 64), dtype="uint16")
        data[:, 16:48, 16:48] = valid_val

        tif_path = tmp_path / "rgb_nodata.tif"
        with rasterio.open(
            tif_path, "w", driver="GTiff",
            height=64, width=64, count=3,
            dtype="uint16", crs="EPSG:32650", transform=tf,
            nodata=nodata,
        ) as dst:
            dst.write(data)

        png_path = tmp_path / "preview.png"
        stretch_path = tmp_path / "stretch.json"
        make_rgb_preview(tif_path, png_path, stretch_path)

        with open(stretch_path) as f:
            params = json.load(f)

        for b in range(3):
            assert params[f"band_{b}"]["valid_pixel_count"] == 32 * 32
            assert params[f"band_{b}"]["nodata"] == 0
            # P2 should be close to the valid area's P2 (around 10000-15000)
            assert params[f"band_{b}"]["percentile_2"] > 1000

    def test_no_nodata_fallback(self, tmp_path):
        """Image without nodata should work as before."""
        tf = from_origin(500000, 4000000, 30, 30)
        # No explicit nodata
        data = np.ones((3, 32, 32), dtype="uint16") * 30000

        tif_path = tmp_path / "rgb_no_nodata.tif"
        with rasterio.open(
            tif_path, "w", driver="GTiff",
            height=32, width=32, count=3,
            dtype="uint16", crs="EPSG:32650", transform=tf,
        ) as dst:
            dst.write(data)

        png_path = tmp_path / "preview.png"
        stretch_path = tmp_path / "stretch.json"
        make_rgb_preview(tif_path, png_path, stretch_path)

        with open(stretch_path) as f:
            params = json.load(f)
        # nodata should be None
        assert params["band_0"]["nodata"] is None