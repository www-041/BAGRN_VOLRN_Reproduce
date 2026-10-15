import numpy as np
import pytest
from affine import Affine

from src.benchmark_v1_lsstaa.raster import masked_block_mean, verify_colocation, proxy_ndvi


def test_masked_block_mean_excludes_zero_and_partial_edge():
    values = np.array([[2, 0, 10], [4, 0, 20], [6, 8, 30]], dtype=np.uint16)
    valid = values != 0
    result = masked_block_mean(values, valid, 2)
    assert result.shape == (2, 2)
    assert result.tolist() == [[3, 15], [7, 30]]


def test_colocation_rejects_band_shift():
    base = {"crs": "EPSG:32650", "transform": Affine(14, 0, 0, 0, -14, 100), "width": 50, "height": 50}
    shifted = {**base, "transform": Affine(14, 0, 14, 0, -14, 100)}
    with pytest.raises(ValueError, match="BAND_GRID_MISMATCH"):
        verify_colocation({"B9": base, "B4": base, "B7": base, "B14": shifted})


def test_proxy_ndvi_uses_float_before_unsigned_subtraction():
    red = np.array([50000, 0], dtype=np.uint16)
    nir = np.array([30000, 30000], dtype=np.uint16)
    out = proxy_ndvi(red, nir, np.array([True, False]))
    assert out[0] == pytest.approx(-2500)
    assert out[1] == 0


def test_proxy_ndvi_uses_official_single_intermediates():
    red = np.array([8137.25], dtype=np.float32)
    nir = np.array([12256.75], dtype=np.float32)
    expected = ((nir-red)/(nir+red)*np.float32(10000))[0]
    actual = proxy_ndvi(red, nir, np.array([True]))[0]
    assert actual == expected
