"""Synthetic tests for B9 two-level dataset discovery."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from src.multiscene_sift.b9_dataset import discover_b9_scenes


def _write_b9_scene(
    root: Path,
    time_name: str,
    scene_name: str,
    *,
    write_b9: bool = True,
    write_mtl: bool = True,
) -> Path:
    scene_dir = root / time_name / scene_name
    scene_dir.mkdir(parents=True)
    if write_b9:
        path = scene_dir / f"{scene_name}_B9.TIF"
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=4,
            width=5,
            count=1,
            dtype="uint16",
            crs="EPSG:32650",
            transform=from_origin(500000, 4000000, 14, 14),
        ) as dst:
            dst.write(np.ones((4, 5), dtype=np.uint16), 1)
    if write_mtl:
        (scene_dir / f"{scene_name}_MTL.txt").write_text(
            "PRODUCT_ID = {0}\nDATE_ACQUIRED = 2025-11-14\n"
            "SCENE_CENTER_TIME = 02:38:47.0\nCLOUD_COVER = 0.25\n".format(scene_name),
            encoding="utf-8",
        )
    return scene_dir


def test_discovers_scenes_under_multiple_time_directories(tmp_path):
    root = tmp_path / "B9数据"
    _write_b9_scene(root, "20250101", "scene_a")
    _write_b9_scene(root, "20250102", "scene_b")

    records = discover_b9_scenes(root)

    assert [record["scene_id"] for record in records] == ["scene_a", "scene_b"]
    assert records[0]["time_dir"] == "20250101"
    assert records[0]["resolution_x_m"] == pytest.approx(14.0)
    assert records[0]["resolution_y_m"] == pytest.approx(14.0)
    assert records[0]["cloud_cover"] == pytest.approx(0.25)


def test_ignores_non_scene_directories(tmp_path):
    root = tmp_path / "root"
    (root / "20250101" / "README").mkdir(parents=True)
    _write_b9_scene(root, "20250101", "scene_a")

    records = discover_b9_scenes(root)

    assert len(records) == 1
    assert records[0]["scene_id"] == "scene_a"


def test_missing_b9_is_reported(tmp_path):
    root = tmp_path / "root"
    _write_b9_scene(root, "20250101", "scene_a", write_b9=False)

    with pytest.raises(FileNotFoundError, match="B9"):
        discover_b9_scenes(root)


def test_duplicate_scene_id_is_reported(tmp_path):
    root = tmp_path / "root"
    _write_b9_scene(root, "20250101", "same_scene")
    _write_b9_scene(root, "20250102", "same_scene")

    with pytest.raises(ValueError, match="Duplicate scene_id"):
        discover_b9_scenes(root)


def test_windows_style_chinese_directory_does_not_break_discovery(tmp_path):
    root = tmp_path / "科研" / "B9数据"
    _write_b9_scene(root, "20250101", "scene_中文")

    records = discover_b9_scenes(root)

    assert len(records) == 1
    assert Path(records[0]["b9_path"]).exists()
    assert Path(records[0]["mtl_path"]).exists()

