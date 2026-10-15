"""Tests for fixed output grid in create_mosaic."""

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from src.mosaic import create_mosaic


class TestFixedMosaicGrid:
    def test_legacy_call_still_works(self, tmp_path):
        """Calling without explicit grid should use auto-computed bounds."""
        data = np.ones((1, 64, 64), dtype="uint16") * 100
        tf = from_origin(500000, 4000000, 30, 30)
        out = str(tmp_path / "legacy.tif")
        result = create_mosaic(
            arrays=[data], transforms=[tf], crs="EPSG:32650",
            nodata_values=[0], output_path=out, mode="source_selection",
        )
        assert result == out
        with rasterio.open(out) as src:
            assert src.width == 64
            assert src.height == 64

    def test_explicit_grid_gives_exact_dims(self, tmp_path):
        """Explicit grid should be used directly."""
        data = np.ones((1, 64, 64), dtype="uint16") * 100
        tf = from_origin(500000, 4000000, 30, 30)
        out = str(tmp_path / "explicit.tif")
        out_tf = from_origin(500000, 4000000, 30, 30)
        result = create_mosaic(
            arrays=[data], transforms=[tf], crs="EPSG:32650",
            nodata_values=[0], output_path=out, mode="source_selection",
            output_transform=out_tf, output_width=100, output_height=80,
        )
        with rasterio.open(out) as src:
            assert src.width == 100
            assert src.height == 80
            assert src.transform == out_tf

    def test_partial_grid_raises(self, tmp_path):
        """Supplying only some grid params should raise ValueError."""
        data = np.ones((1, 64, 64), dtype="uint16") * 100
        tf = from_origin(500000, 4000000, 30, 30)
        out = str(tmp_path / "partial.tif")
        out_tf = from_origin(500000, 4000000, 30, 30)
        with pytest.raises(ValueError, match="must all be supplied"):
            create_mosaic(
                arrays=[data], transforms=[tf], crs="EPSG:32650",
                nodata_values=[0], output_path=out, mode="source_selection",
                output_transform=out_tf, output_width=100,
                # output_height missing
            )

    def test_two_mosaics_identical_metadata(self, tmp_path):
        """Two calls with same explicit grid produce identical metadata."""
        data1 = np.ones((1, 64, 64), dtype="uint16") * 100
        data2 = np.ones((1, 64, 64), dtype="uint16") * 200
        tf = from_origin(500000, 4000000, 30, 30)
        out_tf = from_origin(500000, 4000000, 30, 30)

        out1 = str(tmp_path / "band1.tif")
        out2 = str(tmp_path / "band2.tif")

        create_mosaic(
            arrays=[data1], transforms=[tf], crs="EPSG:32650",
            nodata_values=[0], output_path=out1, mode="weighted",
            output_transform=out_tf, output_width=100, output_height=80,
        )
        create_mosaic(
            arrays=[data2], transforms=[tf], crs="EPSG:32650",
            nodata_values=[0], output_path=out2, mode="weighted",
            output_transform=out_tf, output_width=100, output_height=80,
        )

        with rasterio.open(out1) as s1, rasterio.open(out2) as s2:
            assert s1.width == s2.width
            assert s1.height == s2.height
            assert s1.transform == s2.transform
            assert s1.crs == s2.crs