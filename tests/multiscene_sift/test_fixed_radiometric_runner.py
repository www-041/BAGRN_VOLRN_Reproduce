import json

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from src.multiscene_sift.radiometric_runner import run_fixed_geometry_radiometric


def _inputs(tmp_path):
    source_root = tmp_path / "source"
    source_root.mkdir()
    records = []
    transform = from_origin(0.0, 32.0, 1.0, 1.0)
    for idx, left in enumerate((0.0, 4.0, 8.0)):
        path = source_root / f"scene_{idx}.tif"
        data = np.arange(16 * 16, dtype=np.uint16).reshape(16, 16) + idx * 100
        profile = {
            "driver": "GTiff", "height": 16, "width": 16, "count": 1,
            "dtype": "uint16", "crs": "EPSG:32650",
            "transform": from_origin(left, 32.0, 1.0, 1.0),
        }
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(data, 1)
        records.append({"scene_id": f"scene_{idx}", "b9_path": str(path)})

    source_config = tmp_path / "source.json"
    source_config.write_text(json.dumps({"band": "B9", "scenes": records}), encoding="utf-8")
    global_dir = tmp_path / "global"
    global_dir.mkdir()
    global_dir.joinpath("global_transforms.json").write_text(
        json.dumps({"transforms": [{"scene": i, "matrix": np.eye(3).tolist()} for i in range(3)]}),
        encoding="utf-8",
    )
    grid = {
        "crs": "EPSG:32650", "resolution": 1.0, "pixel_size": 1.0,
        "width": 24, "height": 16,
        "transform": [1.0, 0.0, 0.0, 0.0, -1.0, 32.0],
        "bounds": [0.0, 16.0, 24.0, 32.0],
    }
    grid_path = tmp_path / "grid.json"
    grid_path.write_text(json.dumps(grid), encoding="utf-8")
    return source_config, global_dir, grid_path


def test_raw_fixed_geometry_runner_writes_canonical_outputs(tmp_path):
    source, global_dir, grid = _inputs(tmp_path)
    result = run_fixed_geometry_radiometric(
        source, global_dir, grid, tmp_path / "run", method="RAW",
        geometry_run="sift_mst",
    )

    for name in (
        "mosaic.tif", "preview.png", "radiometric_summary.json",
        "run_config.json", "geometry_source.json", "radiometric_method.json",
        "valid_mask.tif", "contributor_count.tif", "weight_sum.tif",
    ):
        assert (tmp_path / "run" / name).exists()
    with rasterio.open(tmp_path / "run" / "mosaic.tif") as src:
        assert (src.width, src.height) == (24, 16)
        assert np.allclose(tuple(src.transform)[:6], (1.0, 0.0, 0.0, 0.0, -1.0, 32.0))
    assert result["radiometric_method"] == "RAW"
    assert result["overlap_metrics"]["summary"]["pairs"] == 3


def test_bagrn_volrn_fixed_geometry_runner_persists_parameters(tmp_path):
    source, global_dir, grid = _inputs(tmp_path)
    result = run_fixed_geometry_radiometric(
        source, global_dir, grid, tmp_path / "run", method="BAGRN_VOLRN",
        geometry_run="efficient_loftr_translation_l2", block_size_pixels=8,
        max_iter=4,
    )

    run_dir = tmp_path / "run"
    assert (run_dir / "bagrn_parameters.npz").exists()
    assert (run_dir / "volrn_parameters.npz").exists()
    assert result["radiometric_method"] == "BAGRN_VOLRN"
    assert result["convergence"]["volrn"]["all_converged"] in (True, False)


def test_bagrn_fixed_geometry_runner_stops_before_volrn(tmp_path):
    source, global_dir, grid = _inputs(tmp_path)
    result = run_fixed_geometry_radiometric(
        source, global_dir, grid, tmp_path / "run", method="BAGRN",
        geometry_run="sift_mst",
    )

    assert (tmp_path / "run" / "bagrn_parameters.npz").exists()
    assert not (tmp_path / "run" / "volrn_parameters.npz").exists()
    assert result["convergence"]["volrn"]["n_blocks"] == 0


def test_fixed_geometry_runner_rejects_nonempty_output(tmp_path):
    source, global_dir, grid = _inputs(tmp_path)
    output = tmp_path / "run"
    output.mkdir()
    (output / "partial.json").write_text("{}", encoding="utf-8")

    with pytest.raises(FileExistsError, match="non-empty"):
        run_fixed_geometry_radiometric(source, global_dir, grid, output, method="RAW")
