"""Tests for :mod:`src.multiscene_sift.rgb`."""

from __future__ import annotations

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin


class TestRgb:
    """Tests for three-band stacking and preview."""

    def test_stack_three_bands(self, tmp_path):
        """Stack three aligned bands into a 3-band GeoTIFF."""
        from src.multiscene_sift.rgb import stack_three_band_geotiff

        # Create three aligned single-band TIFFs
        tf = from_origin(500000, 4000000, 30, 30)
        band_paths = {}
        for b, name in enumerate(["R", "G", "B"]):
            path = tmp_path / f"{name}.tif"
            data = np.full((128, 128), (b + 1) * 100, dtype="uint16")
            with rasterio.open(
                path, "w", driver="GTiff", height=128, width=128,
                count=1, dtype="uint16", crs="EPSG:32650", transform=tf,
            ) as dst:
                dst.write(data, 1)
            band_paths[name] = str(path)

        out = tmp_path / "rgb.tif"
        result = stack_three_band_geotiff(band_paths, out)

        assert out.exists()
        with rasterio.open(out) as src:
            assert src.count == 3
            data = src.read()
            assert data.shape == (3, 128, 128)

    def test_grid_mismatch_raises(self, tmp_path):
        """Bands with different grid sizes should raise ValueError."""
        from src.multiscene_sift.rgb import stack_three_band_geotiff

        tf = from_origin(500000, 4000000, 30, 30)
        band_paths = {}
        # R and G are 128×128, B is 64×64
        for b, name, size in [(0, "R", 128), (1, "G", 128), (2, "B", 64)]:
            path = tmp_path / f"{name}.tif"
            data = np.full((size, size), (b + 1) * 100, dtype="uint16")
            with rasterio.open(
                path, "w", driver="GTiff", height=size, width=size,
                count=1, dtype="uint16", crs="EPSG:32650", transform=tf,
            ) as dst:
                dst.write(data, 1)
            band_paths[name] = str(path)

        out = tmp_path / "rgb.tif"
        with pytest.raises(ValueError, match="Size mismatch"):
            stack_three_band_geotiff(band_paths, out)

    def test_preview_png_created(self, tmp_path):
        """Preview PNG should be created from a 3-band TIFF."""
        from src.multiscene_sift.rgb import make_rgb_preview, stack_three_band_geotiff

        # Create aligned bands
        tf = from_origin(500000, 4000000, 30, 30)
        band_paths = {}
        for b, name in enumerate(["R", "G", "B"]):
            path = tmp_path / f"{name}.tif"
            rng = np.random.default_rng(42 + b)
            data = rng.uniform(0, 65535, size=(64, 64)).astype("uint16")
            with rasterio.open(
                path, "w", driver="GTiff", height=64, width=64,
                count=1, dtype="uint16", crs="EPSG:32650", transform=tf,
            ) as dst:
                dst.write(data, 1)
            band_paths[name] = str(path)

        tif_out = tmp_path / "rgb.tif"
        stack_three_band_geotiff(band_paths, tif_out)

        png_out = tmp_path / "preview.png"
        stretch_out = tmp_path / "stretch.json"
        make_rgb_preview(tif_out, png_out, stretch_out)

        assert png_out.exists()
        assert stretch_out.exists()