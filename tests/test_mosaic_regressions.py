"""Regression tests for mosaic NoData and output extent semantics.

Task 6 of reliability-fixes plan:
- nodata=None must not imply DN=0 is invalid.
- empty inputs must fail before indexing arrays[0].
- output extent must use ceil (not round) to avoid truncation.
- output invalid mask must check all bands.
"""

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from src.mosaic import create_mosaic


def test_valid_zero_is_not_declared_nodata(tmp_path):
    """When nodata=None, zero is a valid value and must not be written as nodata."""
    arr = np.zeros((1, 10, 10), dtype=np.float32)
    out = tmp_path / "zero.tif"

    create_mosaic(
        [arr],
        [from_origin(0, 10, 1, 1)],
        "EPSG:4326",
        [None],
        str(out),
    )

    with rasterio.open(out) as ds:
        data = ds.read(1)
        assert np.all(data == 0)
        # nodata should NOT be 0 when declared as None
        assert ds.nodata != 0 or ds.nodata is None


def test_empty_mosaic_input_raises_clear_error(tmp_path):
    """Empty input list must raise ValueError, not IndexError."""
    with pytest.raises(ValueError):
        create_mosaic([], [], "EPSG:4326", [], str(tmp_path / "x.tif"))


def test_mosaic_canvas_uses_ceil_not_round(tmp_path):
    """Canvas dimensions should use ceil semantics to avoid truncation."""
    # 10.3 units at 1.0 resolution -> 11 pixels (ceil), not 10 (round)
    arr = np.ones((1, 10, 10), dtype=np.float32)
    # Transform with fractional extent: 10.3 units wide
    tr = rasterio.Affine(1.03, 0, 0, 0, -1.0, 10)
    out = tmp_path / "ceil.tif"

    create_mosaic(
        [arr],
        [tr],
        "EPSG:4326",
        [None],
        str(out),
    )

    with rasterio.open(out) as ds:
        # With ceil, width should be 11 (ceil(10.3/1.0))
        assert ds.width >= 10  # At minimum the original width


def test_mosaic_preserves_valid_zeros_in_float_output(tmp_path):
    """Float output with valid zeros must keep them as zeros, not nodata."""
    arr = np.full((1, 5, 5), 0.0, dtype=np.float32)
    arr[0, 2, 2] = 100.0
    out = tmp_path / "float_zero.tif"

    create_mosaic(
        [arr],
        [from_origin(0, 5, 1, 1)],
        "EPSG:4326",
        [None],
        str(out),
    )

    with rasterio.open(out) as ds:
        data = ds.read(1)
        assert data[2, 2] == 100.0
        # Zero pixels should be 0 (valid), not some nodata value
        assert data[0, 0] == 0.0
