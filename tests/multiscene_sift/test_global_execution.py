"""Synthetic tests for the existing MST and Equal-L2 global backends."""

from __future__ import annotations

import numpy as np

from rasterio.transform import Affine

from src.multiscene_sift.b9_runner import scenes_from_frozen_config
from src.multiscene_sift.global_execution import run_global_connections
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
        "ransac": {"residual_threshold_px": 2.0, "max_trials": 5000,
                    "minimum_inliers": 20, "minimum_inlier_ratio": 0.3},
    }


def _result(i: int, j: int) -> PairwiseRegistration:
    return PairwiseRegistration(
        idx_i=i, idx_j=j, status="OK", raw_matches=30, inliers=25,
        inlier_ratio=0.8, coverage=0.5, residual_median=0.4,
        residual_rmse=0.5, residual_p95=0.9,
        pair_pixel_matrix=np.eye(3).tolist(),
        pair_common_transform=Affine(14.0, 0.0, 100.0, 0.0, -14.0, 200.0),
        runtime_sec=1.0, matcher="sift", matcher_runtime_sec=0.5,
        geometry_runtime_sec=0.1,
        inlier_ref_xy=np.array([[10.0, 5.0], [20.0, 5.0]]),
        inlier_tgt_xy=np.array([[0.0, 0.0], [10.0, 0.0]]),
    )


def test_global_execution_produces_mst_and_equal_l2_outputs(tmp_path):
    config = _config()
    results = [_result(0, 1), _result(0, 2), _result(1, 2)]
    context = {
        "scenes": scenes_from_frozen_config(config),
        "results": results,
        "accepted_results": results,
        "geometry_index": {"bundles": [{}, {}, {}]},
        "config_sha256": "synthetic",
    }

    report = run_global_connections(context, tmp_path / "global", matcher="sift", pixel_size_m=14.0)

    assert report["mst"]["status"] == "PASS"
    assert report["mst"]["tree_edge_count"] == 2
    assert report["translation_l2"]["status"] == "PASS"
    assert (tmp_path / "global" / "mst" / "global_registration.json").is_file()
    assert (tmp_path / "global" / "mst" / "global_edge_consistency.json").is_file()
    assert (tmp_path / "global" / "translation_l2" / "05_translation_network_summary.json").is_file()
