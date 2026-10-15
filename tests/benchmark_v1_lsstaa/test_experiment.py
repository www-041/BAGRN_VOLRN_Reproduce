import numpy as np
import pytest
import rasterio
from affine import Affine

from src.benchmark_v1_lsstaa.experiment import derive_model, write_corrected_scene


def test_model_rejects_unusable_three_scene_series():
    with pytest.raises(ValueError, match="EMPTY_OBSERVATIONS"):
        derive_model(np.zeros((40, 40, 3, 4), dtype=np.float32), [0, 0, 6])


def test_native_writer_keeps_grid_mask_and_float_values(tmp_path):
    profile = {"driver": "GTiff", "height": 2, "width": 3, "count": 1,
               "dtype": "float32", "crs": "EPSG:32650",
               "transform": Affine(14, 0, 100, 0, -14, 200)}
    data_path, mask_path = tmp_path / "source.tif", tmp_path / "mask.tif"
    with rasterio.open(data_path, "w", **profile) as dst:
        dst.write(np.array([[0, 60000, 2], [3, 4, 5]], np.float32), 1)
    with rasterio.open(mask_path, "w", **{**profile, "dtype": "uint8", "nodata": 0}) as dst:
        dst.write(np.array([[0, 1, 1], [1, 1, 1]], np.uint8), 1)
    out_data, out_mask = write_corrected_scene(data_path, mask_path, tmp_path / "out", 0, 1.2, 3)
    with rasterio.open(out_data) as ds, rasterio.open(out_mask) as ms:
        assert ds.transform == profile["transform"]
        assert ds.dtypes[0] == "float32"
        assert ds.read(1)[0, 1] == pytest.approx(72003)
        assert np.isnan(ds.read(1)[0, 0])
        assert ms.read(1).tolist() == [[0, 1, 1], [1, 1, 1]]
