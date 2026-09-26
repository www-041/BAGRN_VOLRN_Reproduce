import importlib
import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import Affine

from src.mosaic import create_mosaic


runner = importlib.import_module("scripts.run_b9_weighted_mosaic")


def _write_scene(path: Path, data: np.ndarray, transform: Affine) -> None:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=data.shape[1],
        height=data.shape[0],
        count=1,
        dtype="uint16",
        crs="EPSG:32650",
        transform=transform,
        nodata=0,
    ) as dst:
        dst.write(data.astype(np.uint16), 1)


def _synthetic_inputs(tmp_path: Path):
    scene0 = tmp_path / "scene0.tif"
    scene1 = tmp_path / "scene1.tif"
    _write_scene(scene0, np.full((3, 4), 10, dtype=np.uint16), Affine(1, 0, 0, 0, -1, 4))
    _write_scene(scene1, np.full((3, 4), 20, dtype=np.uint16), Affine(1, 0, 2, 0, -1, 4))

    config = tmp_path / "config.json"
    config.write_text(json.dumps({"scenes": [{"b9_path": str(scene0)}, {"b9_path": str(scene1)}]}))

    transforms = tmp_path / "global_transforms.json"
    transforms.write_text(
        json.dumps(
            {
                "reference_index": 0,
                "transforms": [
                    {"scene": 0, "matrix": np.eye(3).tolist()},
                    {"scene": 1, "matrix": np.eye(3).tolist()},
                ],
            }
        )
    )

    grid = tmp_path / "grid.json"
    grid.write_text(
        json.dumps(
            {
                "crs": "EPSG:32650",
                "resolution": 1.0,
                "pixel_size": 1.0,
                "width": 6,
                "height": 3,
                "transform": [1.0, 0.0, 0.0, 0.0, -1.0, 4.0],
                "bounds": [0.0, 1.0, 6.0, 4.0],
            }
        )
    )
    return config, transforms, grid


def test_runner_writes_exact_grid_union_and_frozen_weighted_result(tmp_path):
    config, transforms, grid = _synthetic_inputs(tmp_path)
    output_dir = tmp_path / "run"

    result = runner.run_weighted_mosaic(config, transforms.parent, grid, output_dir)

    expected = {
        "mosaic.tif",
        "valid_mask.tif",
        "contributor_count.tif",
        "weight_sum.tif",
        "run_summary.json",
        "preview.png",
    }
    assert {path.name for path in output_dir.iterdir()} == expected

    with rasterio.open(output_dir / "mosaic.tif") as mosaic:
        assert (mosaic.width, mosaic.height) == (6, 3)
        assert mosaic.transform == Affine(1.0, 0.0, 0.0, 0.0, -1.0, 4.0)
        assert mosaic.res[0] == pytest.approx(1.0)

    with rasterio.open(output_dir / "valid_mask.tif") as valid:
        assert valid.read(1).astype(bool).all()
    with rasterio.open(output_dir / "contributor_count.tif") as contributors:
        assert set(np.unique(contributors.read(1))).issubset({0, 1, 2})
        assert np.all(contributors.read(1)[:, :2] == 1)
        assert np.all(contributors.read(1)[:, 2:4] == 2)
        assert np.all(contributors.read(1)[:, 4:] == 1)

    expected_path = tmp_path / "expected_core.tif"
    with rasterio.open(tmp_path / "scene0.tif") as first, rasterio.open(tmp_path / "scene1.tif") as second:
        create_mosaic(
            arrays=[first.read()[0:1], second.read()[0:1]],
            transforms=[first.transform, second.transform],
            crs="EPSG:32650",
            nodata_values=[first.nodata, second.nodata],
            output_path=str(expected_path),
            resolution=1.0,
            mode="weighted",
            output_transform=Affine(1.0, 0.0, 0.0, 0.0, -1.0, 4.0),
            output_width=6,
            output_height=3,
        )
    with rasterio.open(output_dir / "mosaic.tif") as actual, rasterio.open(expected_path) as expected:
        np.testing.assert_array_equal(actual.read(), expected.read())

    assert result["mosaic_path"].name == "mosaic.tif"
    summary = json.loads((output_dir / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["blend_mode"] == "weighted"
    assert summary["radiometric_normalization"] == "NONE"


def test_runner_refuses_nonempty_output_directory(tmp_path):
    config, transforms, grid = _synthetic_inputs(tmp_path)
    output_dir = tmp_path / "existing"
    output_dir.mkdir()
    (output_dir / "sentinel.txt").write_text("do not overwrite")

    with pytest.raises(FileExistsError, match="non-empty"):
        runner.run_weighted_mosaic(config, transforms.parent, grid, output_dir)
