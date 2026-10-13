import numpy as np
import rasterio
from affine import Affine
from PIL import Image
from rasterio.windows import Window

from scripts.benchmark.spectralmatch_phase2_worker import _coverage, _preview


def _write(path, data):
    profile = {
        "driver": "GTiff", "width": data.shape[1], "height": data.shape[0],
        "count": 1, "dtype": "float32", "crs": "EPSG:32650",
        "transform": Affine(14, 0, 100, 0, -14, 200), "nodata": -9999.0,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data.astype(np.float32), 1)


def test_fixed_stretch_preview_writes_two_panel_image(tmp_path):
    first, second = tmp_path / "first.tif", tmp_path / "second.tif"
    _write(first, np.arange(64, dtype=np.float32).reshape(8, 8))
    _write(second, np.arange(64, dtype=np.float32).reshape(8, 8) + 10)
    output = tmp_path / "preview.png"

    result = _preview([first, second], output, Window(1, 1, 6, 6), "synthetic")

    assert output.is_file()
    with Image.open(output) as image:
        assert image.size == (1024, 550)
    assert result["window"] == {"row_off": 1, "col_off": 1, "height": 6, "width": 6}
    assert result["stretch"]["p2"] < result["stretch"]["p98"]

    shared = {"p2": 0.0, "p98": 100.0}
    repeated = _preview([second, first], tmp_path / "shared.png", Window(1, 1, 6, 6), "shared", stretch=shared)
    assert repeated["stretch"] == shared


def test_coverage_handles_integer_pixel_shift_and_counts_uncovered_support(tmp_path):
    a_path, b_path = tmp_path / "a.tif", tmp_path / "b.tif"
    output_path, output_mask_path = tmp_path / "mosaic.tif", tmp_path / "mosaic_mask.tif"
    base = np.ones((4, 4), dtype=np.uint8)
    shifted_profile = {
        "driver": "GTiff", "width": 3, "height": 4, "count": 1, "dtype": "float32",
        "crs": "EPSG:32650", "transform": Affine(14, 0, 114, 0, -14, 200), "nodata": -9999.0,
    }
    _write(a_path, base.astype(np.float32))
    _write(b_path, np.zeros((4, 4), dtype=np.float32))
    with rasterio.open(output_path, "w", **shifted_profile) as dst:
        dst.write(np.ones((4, 3), dtype=np.float32), 1)
    mask_profile = shifted_profile | {"dtype": "uint8", "nodata": 0}
    with rasterio.open(output_mask_path, "w", **mask_profile) as dst:
        dst.write(np.ones((4, 3), dtype=np.uint8), 1)

    result = _coverage([a_path, b_path], output_path, output_mask_path)

    assert result["frozen_union"] == 16
    assert result["output_valid_inside_union"] == 12
    assert result["extra_output_outside_union"] == 0
    assert result["coverage"] == 0.75
    assert result["status"] == "COVERAGE_GAP"
