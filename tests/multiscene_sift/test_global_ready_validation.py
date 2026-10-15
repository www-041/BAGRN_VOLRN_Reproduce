"""Tests for the per-matcher Global-ready and replay gate."""

from __future__ import annotations

import json

import numpy as np
import pytest
from rasterio.transform import Affine

from src.multiscene_sift.b9_runner import run_b9_registration
from src.multiscene_sift.global_ready_validation import (
    validate_global_ready_run,
)
from src.multiscene_sift.models import PairwiseRegistration


def _config():
    scenes = []
    for index in range(3):
        scenes.append({
            "scene_id": f"scene_{index}",
            "scene_dir": f"D:/B9/scene_{index}",
            "b9_path": f"D:/B9/scene_{index}/scene_{index}_B9.TIF",
            "crs": "EPSG:32650",
            "pixel_size_x_m": 14.0,
            "pixel_size_y_m": 14.0,
            "width": 100,
            "height": 100,
            "left": float(index * 50),
            "bottom": 0.0,
            "right": float(index * 50 + 100),
            "top": 100.0,
        })
    edge = {
        "intersection_area": 2500.0,
        "overlap_area_i_ratio": 0.25,
        "overlap_area_j_ratio": 0.5,
    }
    return {
        "band": "B9",
        "pixel_size_m": 14.0,
        "crs": "EPSG:32650",
        "selection": {"manifest_indices": [0, 1, 2]},
        "scenes": scenes,
        "graph_edges": [
            {"manifest_idx_i": 0, "manifest_idx_j": 1, **edge},
            {"manifest_idx_i": 0, "manifest_idx_j": 2, **edge},
            {"manifest_idx_i": 1, "manifest_idx_j": 2, **edge},
        ],
        "registration": {"match_max_side": 1024, "random_seed": 0},
        "ransac": {
            "residual_threshold_px": 2.0,
            "max_trials": 5000,
            "minimum_inliers": 20,
            "minimum_inlier_ratio": 0.3,
        },
    }


def _result(i: int, j: int) -> PairwiseRegistration:
    return PairwiseRegistration(
        idx_i=i,
        idx_j=j,
        status="OK",
        raw_matches=30,
        inliers=25,
        inlier_ratio=0.8,
        coverage=0.5,
        residual_median=0.4,
        residual_rmse=0.5,
        residual_p95=0.9,
        pair_pixel_matrix=np.eye(3).tolist(),
        pair_common_transform=Affine(14.0, 0.0, 100.0, 0.0, -14.0, 200.0),
        runtime_sec=1.0,
        matcher="sift",
        matcher_runtime_sec=0.5,
        geometry_runtime_sec=0.1,
        inlier_ref_xy=np.array([[10.0, 5.0], [20.0, 5.0]]),
        inlier_tgt_xy=np.array([[0.0, 0.0], [10.0, 0.0]]),
    )


def _write_run(tmp_path, *, graph_changed=False):
    config = _config()
    config_path = tmp_path / "04_frozen_five_scene_config_1024.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    output = tmp_path / "globalready"
    results = [_result(0, 1), _result(1, 2), _result(0, 2)]
    run_b9_registration(
        config,
        output,
        matcher="sift",
        protocol_config_path=config_path,
        pair_runner=lambda *args, **kwargs: results,
    )
    old = tmp_path / "old"
    old.mkdir()
    for name in ("pairwise_summary.json", "accepted_graph.json"):
        (old / name).write_bytes((output / name).read_bytes())
    if graph_changed:
        graph = json.loads((old / "accepted_graph.json").read_text(encoding="utf-8"))
        graph["accepted_edges"] = graph["accepted_edges"][:-1]
        graph["accepted_edge_count"] = len(graph["accepted_edges"])
        (old / "accepted_graph.json").write_text(json.dumps(graph), encoding="utf-8")
    return output, old, config, config_path


def test_global_ready_validation_reports_geometry_and_replay_pass(tmp_path):
    run, old, config, config_path = _write_run(tmp_path)

    report = validate_global_ready_run(run, old, config, config_path)

    assert report["global_ready"] is True
    assert report["replay"] == "REPLAY_CONSISTENT"
    assert report["processed_pair_count"] == 3
    assert report["geometry_bundle_count"] == 3
    assert report["total_inlier_point_count"] == 6


def test_global_ready_validation_rejects_changed_accepted_graph(tmp_path):
    run, old, config, config_path = _write_run(tmp_path, graph_changed=True)

    report = validate_global_ready_run(run, old, config, config_path)

    assert report["global_ready"] is True
    assert report["replay"] == "REPLAY_GRAPH_CHANGED"
    assert report["validation"] == "FAILED_FOR_GLOBAL"
