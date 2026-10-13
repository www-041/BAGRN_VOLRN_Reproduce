from types import SimpleNamespace

import numpy as np
import rasterio
from affine import Affine
from rasterio.windows import Window

from src.benchmark_v1_spectralmatch_phase2.adapter import (
    build_pair_submission,
    execute_fixed_grid_stages,
    prepare_masked_copy,
)
from scripts.benchmark.run_spectralmatch_phase2 import _schema_check


def _raster(path, values, nodata=None):
    profile = {
        "driver": "GTiff",
        "width": values.shape[1],
        "height": values.shape[0],
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:32650",
        "transform": Affine(14, 0, 633346, 0, -14, 3359874),
        "nodata": nodata,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(values.astype(np.float32), 1)


def test_masked_copy_preserves_exact_frozen_support_and_values(tmp_path):
    source = tmp_path / "source.tif"
    mask_path = tmp_path / "frozen_mask.tif"
    output = tmp_path / "spectralmatch_input.tif"
    values = np.arange(64, dtype=np.float32).reshape(8, 8)
    mask = np.ones((8, 8), dtype=np.uint8)
    mask[0, 0] = 0
    mask[3:5, 2:4] = 0
    _raster(source, values)
    _raster(mask_path, mask.astype(np.float32), nodata=0)

    result = prepare_masked_copy(source, mask_path, output, block_size=3)

    with rasterio.open(output) as dst:
        actual = dst.read(1)
        actual_valid = dst.dataset_mask() > 0
        assert dst.crs.to_string() == "EPSG:32650"
        assert dst.transform == Affine(14, 0, 633346, 0, -14, 3359874)
        assert np.isfinite(dst.nodata)
    np.testing.assert_array_equal(actual_valid, mask.astype(bool))
    np.testing.assert_array_equal(actual[actual_valid], values[mask.astype(bool)])
    assert np.all(actual[~actual_valid] == dst.nodata)
    assert result["valid_pixels"] == int(mask.sum())


def test_masked_copy_crop_uses_window_grid_and_preserves_window_support(tmp_path):
    source = tmp_path / "source.tif"
    mask_path = tmp_path / "frozen_mask.tif"
    output = tmp_path / "crop.tif"
    values = np.arange(100, dtype=np.float32).reshape(10, 10)
    mask = np.ones((10, 10), dtype=np.uint8)
    mask[4, 5] = 0
    _raster(source, values)
    _raster(mask_path, mask.astype(np.float32), nodata=0)

    result = prepare_masked_copy(source, mask_path, output, crop_window=Window(3, 2, 4, 5))

    with rasterio.open(output) as dst:
        assert (dst.width, dst.height) == (4, 5)
        assert dst.transform == Affine(14, 0, 633388, 0, -14, 3359846)
        data = dst.read(1)
        valid = dst.dataset_mask() > 0
    np.testing.assert_array_equal(valid, mask[2:7, 3:7].astype(bool))
    np.testing.assert_array_equal(data[valid], values[2:7, 3:7][valid])
    assert result["crop_window"] == {"col_off": 3, "row_off": 2, "width": 4, "height": 5}


def test_fixed_grid_workflow_skips_joint_registration_and_runs_all_radiometric_stages(tmp_path):
    calls = []

    def record(name, result):
        def call(*args, **kwargs):
            calls.append((name, args, kwargs))
            return result

        return call

    api = SimpleNamespace(
        global_regression=record("global", ["global_a.tif", "global_b.tif"]),
        local_block_adjustment=record("local", ["local_a.tif", "local_b.tif"]),
        create_footprints=record("footprints", "footprints.gpkg"),
        Seamline=SimpleNamespace(voronoi=record("seamline", "seams.gpkg")),
        mask_rasters=record("mask", ["clip_a.tif", "clip_b.tif"]),
        merge_rasters=record("merge", "mosaic.tif"),
        joint_coregistration=lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("fixed-grid experiment must not run joint coregistration")
        ),
    )

    result = execute_fixed_grid_stages(
        api,
        ["input_a.tif", "input_b.tif"],
        tmp_path,
        global_options={"pif_method": "flood_from_match_points"},
        local_options={"number_of_blocks": 100},
    )

    assert [call[0] for call in calls] == [
        "global",
        "local",
        "footprints",
        "seamline",
        "mask",
        "merge",
    ]
    assert result["scene_outputs"] == ["local_a.tif", "local_b.tif"]
    assert result["mosaic"] == "mosaic.tif"
    assert set(result["stage_wall_seconds"]) == {
        "global_regression", "local_block_adjustment", "create_footprints",
        "voronoi_seamline", "mask_rasters", "merge_rasters",
    }


def test_pair_submission_has_schema_fields_and_passes_actual_freeze_checker():
    frozen = {
        "input_freeze_sha256": "a" * 64,
        "geometry_freeze_sha256": "b" * 64,
        "band": "B9",
    }

    submission = build_pair_submission(
        pair_id="00_01",
        scene_i="scene_i",
        scene_j="scene_j",
        frozen=frozen,
        corrected_scene_paths={"scene_i": "i.tif", "scene_j": "j.tif"},
        output_mosaic_path="mosaic.tif",
        output_mosaic_mask_path="mosaic_mask.tif",
        run_id="smoke-1",
    )

    assert submission["method_kind"] == "PAIR_ONLY"
    assert submission["pixel_value_unit"] == "UNKNOWN"
    assert submission["pair_specific_outputs"]["00_01"]["scene_i"] == "scene_i"
    assert submission["status"] == "COMPLETED_DIAGNOSTIC_ONLY"
    assert submission["method_id"]
    assert submission["input_freeze_sha256"] == frozen["input_freeze_sha256"]
    assert submission["geometry_freeze_sha256"] == frozen["geometry_freeze_sha256"]
    schema = __import__("pathlib").Path("docs/benchmark/v1/unified_v1/method_submission_schema.json")
    assert _schema_check(submission, schema)["status"] == "PASS"
