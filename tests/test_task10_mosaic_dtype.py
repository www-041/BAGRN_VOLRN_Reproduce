"""Task10C scientific mosaic dtype contract."""

import numpy as np
import rasterio
from rasterio.transform import from_origin

from src.mosaic import create_mosaic


def test_scientific_output_dtype_preserves_negative_fractional_and_large_values(tmp_path):
    first = np.array([[[-1.25, 0.5], [65536.75, 2.0]]], dtype=np.float64)
    output = tmp_path / "scientific_float.tif"

    create_mosaic(
        [first],
        [from_origin(0, 2, 1, 1)],
        "EPSG:4326",
        [None],
        str(output),
        output_dtype="float32",
    )

    with rasterio.open(output) as dataset:
        data = dataset.read(1)
        assert dataset.dtypes[0] == "float32"
        assert np.isfinite(data).all()
        assert np.allclose(data, first[0], rtol=0, atol=1e-5)


def test_default_geometry_dtype_contract_remains_unchanged(tmp_path):
    first = np.full((1, 2, 2), 100, dtype=np.uint16)
    output = tmp_path / "geometry_uint16.tif"

    create_mosaic(
        [first],
        [from_origin(0, 2, 1, 1)],
        "EPSG:4326",
        [0],
        str(output),
    )

    with rasterio.open(output) as dataset:
        assert dataset.dtypes[0] == "uint16"
