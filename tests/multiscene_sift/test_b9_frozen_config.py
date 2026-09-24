"""Tests for the manually frozen B9 five-scene configuration."""

from __future__ import annotations

from src.multiscene_sift.b9_frozen_config import (
    build_frozen_config,
    write_frozen_config,
)


def _record(index: int) -> dict:
    return {
        "scene_id": f"scene_{index}",
        "time_dir": f"2025010{index + 1}",
        "scene_dir": f"D:/B9/scene_{index}",
        "b9_path": f"D:/B9/scene_{index}/scene_{index}_B9.TIF",
        "mtl_path": f"D:/B9/scene_{index}/scene_{index}_MTL.txt",
        "acquisition_time": "2025-01-01T00:00:00",
        "crs": "EPSG:32650",
        "resolution_x_m": 14.0,
        "resolution_y_m": 14.0,
        "width": 100,
        "height": 100,
        "left": float(index * 100),
        "bottom": 0.0,
        "right": float(index * 100 + 100),
        "top": 100.0,
        "cloud_cover": 0.0,
    }


def _pair(i: int, j: int) -> dict:
    return {
        "idx_i": i,
        "idx_j": j,
        "scene_i": f"scene_{i}",
        "scene_j": f"scene_{j}",
        "intersection_area": 100.0,
        "overlap_area_i_ratio": 0.5,
        "overlap_area_j_ratio": 0.5,
        "symmetric_overlap_ratio": 0.5,
        "has_overlap": True,
    }


def test_freeze_config_rejects_invalid_manual_indices_without_substitution():
    records = [_record(index) for index in range(6)]
    pairs = [_pair(i, j) for i in range(6) for j in range(i + 1, 6)]

    try:
        build_frozen_config(records, pairs, [2, 3, 5, 8, 10])
    except ValueError as exc:
        assert "manifest indices" in str(exc)
    else:
        raise AssertionError("invalid manual indices must not be substituted")


def test_freeze_config_records_selected_scene_metadata_and_fixed_parameters():
    records = [_record(index) for index in range(5)]
    pairs = [_pair(i, j) for i in range(5) for j in range(i + 1, 5)]

    config = build_frozen_config(records, pairs, [0, 1, 2, 3, 4])

    assert [scene["manifest_index"] for scene in config["scenes"]] == [0, 1, 2, 3, 4]
    assert config["band"] == "B9"
    assert config["pixel_size_m"] == 14.0
    assert config["ransac"]["residual_threshold_px"] == 2.0
    assert config["ransac"]["minimum_inliers"] == 20
    assert config["matchers"] == ["sift", "loftr", "efficient_loftr", "lightglue_disk"]
    assert config["global_methods"] == ["mst", "equal_l2_translation"]


def test_write_frozen_config_preserves_manual_selection(tmp_path):
    records = [_record(index) for index in range(5)]
    pairs = [_pair(i, j) for i in range(5) for j in range(i + 1, 5)]
    config = build_frozen_config(records, pairs, [0, 1, 2, 3, 4])

    output = write_frozen_config(config, tmp_path / "04_frozen_five_scene_config.json")

    assert output.exists()
    assert output.read_text(encoding="utf-8").endswith("\n")
