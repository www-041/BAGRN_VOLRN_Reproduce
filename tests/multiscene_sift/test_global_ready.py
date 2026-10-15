"""Synthetic tests for the strict Global-ready loader and adapters."""

from __future__ import annotations

import json

import numpy as np
import pytest
from rasterio.transform import Affine

from src.multiscene_sift.b9_runner import run_b9_registration
from src.multiscene_sift.global_geometric_adjustment import (
    build_translation_adjustment_system,
    solve_translation_adjustment,
)
from src.multiscene_sift.global_registration import (
    build_accepted_graph,
    build_spanning_tree,
)
from src.multiscene_sift.global_ready import (
    GlobalInputIncompleteError,
    ProvenanceMismatchError,
    load_global_ready_run,
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
    return {
        "band": "B9",
        "pixel_size_m": 14.0,
        "crs": "EPSG:32650",
        "selection": {"manifest_indices": [0, 1, 2]},
        "scenes": scenes,
        "graph_edges": [
            {
                "manifest_idx_i": 0,
                "manifest_idx_j": 1,
                "intersection_area": 2500.0,
                "overlap_area_i_ratio": 0.25,
                "overlap_area_j_ratio": 0.5,
            },
            {
                "manifest_idx_i": 0,
                "manifest_idx_j": 2,
                "intersection_area": 2500.0,
                "overlap_area_i_ratio": 0.25,
                "overlap_area_j_ratio": 0.5,
            },
            {
                "manifest_idx_i": 1,
                "manifest_idx_j": 2,
                "intersection_area": 5000.0,
                "overlap_area_i_ratio": 0.5,
                "overlap_area_j_ratio": 0.5,
            },
        ],
        "registration": {"match_max_side": 1024, "random_seed": 0},
        "ransac": {
            "residual_threshold_px": 2.0,
            "max_trials": 5000,
            "minimum_inliers": 20,
            "minimum_inlier_ratio": 0.3,
        },
    }


def _result(i: int, j: int, status: str = "OK") -> PairwiseRegistration:
    points_i = np.array([[10.0, 5.0], [20.0, 5.0]], dtype=np.float64)
    points_j = np.array([[0.0, 0.0], [10.0, 0.0]], dtype=np.float64)
    if status != "OK":
        points_i = np.empty((0, 2), dtype=np.float64)
        points_j = np.empty((0, 2), dtype=np.float64)
    return PairwiseRegistration(
        idx_i=i,
        idx_j=j,
        status=status,
        raw_matches=30,
        inliers=25 if status == "OK" else 0,
        inlier_ratio=0.8 if status == "OK" else 0.0,
        coverage=0.5 if status == "OK" else 0.0,
        residual_median=0.4 if status == "OK" else float("nan"),
        residual_rmse=0.5 if status == "OK" else float("nan"),
        residual_p95=0.9 if status == "OK" else float("nan"),
        pair_pixel_matrix=np.eye(3).tolist(),
        pair_common_transform=Affine(14.0, 0.0, 100.0, 0.0, -14.0, 200.0),
        runtime_sec=1.0,
        matcher="sift",
        matcher_runtime_sec=0.5,
        geometry_runtime_sec=0.1,
        inlier_ref_xy=points_i,
        inlier_tgt_xy=points_j,
    )


def _write_run(tmp_path, *, results=None):
    config = _config()
    config_path = tmp_path / "04_frozen_five_scene_config_1024.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    output = tmp_path / "globalready"
    results = results or [_result(0, 1), _result(1, 2), _result(0, 2, "TOO_FEW_INLIERS")]
    run_b9_registration(
        config,
        output,
        matcher="sift",
        protocol_config_path=config_path,
        pair_runner=lambda *args, **kwargs: results,
    )
    return output, config, config_path


def test_historical_summary_only_run_is_rejected_as_global_input_incomplete(tmp_path):
    run = tmp_path / "summary_only"
    run.mkdir()
    (run / "pairwise_summary.json").write_text(json.dumps({"results": []}), encoding="utf-8")

    with pytest.raises(GlobalInputIncompleteError, match="POINT_DATA_UNRECOVERABLE_FROM_SUMMARY"):
        load_global_ready_run(run, _config(), "04_frozen_five_scene_config_1024.json")


def test_cross_run_config_hash_is_rejected(tmp_path):
    run, config, config_path = _write_run(tmp_path)
    index_path = run / "geometry_index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["config_sha256"] = "different-run"
    index_path.write_text(json.dumps(index), encoding="utf-8")

    with pytest.raises(ProvenanceMismatchError, match="PROVENANCE_MISMATCH"):
        load_global_ready_run(run, config, config_path)


def test_mst_adapter_uses_accepted_bundles_and_excludes_rejected_edge(tmp_path):
    run, config, config_path = _write_run(tmp_path)
    context = load_global_ready_run(run, config, config_path)

    adjacency, accepted = build_accepted_graph(context["results"], matcher_name="sift")
    tree = build_spanning_tree(adjacency, accepted, ref_idx=0)

    assert {(r.idx_i, r.idx_j) for r in accepted} == {(0, 1), (1, 2)}
    assert len(tree) == 2
    assert all(r.status == "OK" for r in accepted)
    assert all(len(r.inlier_ref_xy) == 2 for r in accepted)


def test_equal_l2_adapter_receives_real_bundle_points(tmp_path):
    run, config, config_path = _write_run(
        tmp_path, results=[_result(0, 1), _result(1, 2), _result(0, 2)]
    )
    context = load_global_ready_run(run, config, config_path)
    accepted = [r for r in context["results"] if r.status == "OK"]
    observations = {
        (r.idx_i, r.idx_j): {
            "x_i": r.inlier_ref_xy,
            "x_j": r.inlier_tgt_xy,
            "pair_common_transform": np.array([
                [r.pair_common_transform.a, r.pair_common_transform.b, r.pair_common_transform.c],
                [r.pair_common_transform.d, r.pair_common_transform.e, r.pair_common_transform.f],
                [0.0, 0.0, 1.0],
            ]),
            "pixel_size": 14.0,
        }
        for r in accepted
    }
    system = build_translation_adjustment_system(
        {0: np.eye(3), 1: np.eye(3), 2: np.eye(3)},
        observations,
        reference_idx=0,
    )
    solution = solve_translation_adjustment(system)

    assert len(system["point_weights"]) == 6
    assert solution["status"] == "OK"
