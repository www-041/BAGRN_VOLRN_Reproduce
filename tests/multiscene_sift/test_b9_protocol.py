"""Task 7 provenance and completeness checks for formal B9 1024 runs."""

from __future__ import annotations

import json

from src.multiscene_sift.b9_protocol import validate_b9_1024_run


def _config():
    return {
        "selection": {"manifest_indices": [0, 1, 2]},
        "registration": {"match_max_side": 1024},
        "graph_edges": [
            {"manifest_idx_i": 0, "manifest_idx_j": 1},
            {"manifest_idx_i": 0, "manifest_idx_j": 2},
            {"manifest_idx_i": 1, "manifest_idx_j": 2},
        ],
    }


def _row(i, j, status="OK"):
    return {
        "idx_i": i,
        "idx_j": j,
        "matcher": "sift",
        "status": status,
        "raw_matches": 30,
        "inliers": 25,
        "inlier_ratio": 0.83,
        "coverage": 0.5,
        "residual_median": 0.4,
        "residual_rmse": 0.5,
        "residual_p95": 0.9,
        "runtime_sec": 1.0,
    }


def _write_run(tmp_path, *, rows=None, match_max_side=1024, protocol_path="04_frozen_five_scene_config_1024.json"):
    run = tmp_path / "run"
    run.mkdir()
    rows = rows if rows is not None else [_row(0, 1), _row(0, 2), _row(1, 2)]
    (run / "run_config.json").write_text(
        json.dumps({
            "protocol_config_path": protocol_path,
            "match_max_side": match_max_side,
            "matcher": "sift",
        }),
        encoding="utf-8",
    )
    (run / "pairwise_summary.json").write_text(
        json.dumps({"results": rows, "n_pairs": len(rows)}),
        encoding="utf-8",
    )
    (run / "runtime.json").write_text(json.dumps({"total_runtime_sec": 3.0}), encoding="utf-8")
    (run / "accepted_graph.json").write_text(
        json.dumps({
            "status": "NETWORK_DISCONNECTED",
            "connected_components": [[0, 1, 2]],
            "accepted_edge_count": 3,
        }),
        encoding="utf-8",
    )
    return run


def test_completed_run_passes_even_when_connectivity_is_only_reported(tmp_path):
    run = _write_run(tmp_path)
    result = validate_b9_1024_run(run, _config(), protocol_config_path="04_frozen_five_scene_config_1024.json")

    assert result["status"] == "PASS"
    assert result["network_status"] == "NETWORK_DISCONNECTED"
    assert result["n_pairs"] == 3


def test_wrong_protocol_is_rejected(tmp_path):
    run = _write_run(tmp_path, match_max_side=1600, protocol_path="04_frozen_five_scene_config.json")
    result = validate_b9_1024_run(run, _config(), protocol_config_path="04_frozen_five_scene_config_1024.json")

    assert result["status"] == "PROTOCOL_MISMATCH"


def test_missing_candidate_pair_is_incomplete_not_quality_failure(tmp_path):
    run = _write_run(tmp_path, rows=[_row(0, 1), _row(0, 2)])
    result = validate_b9_1024_run(run, _config(), protocol_config_path="04_frozen_five_scene_config_1024.json")

    assert result["status"] == "RUN_INCOMPLETE"
    assert result["missing_pairs"] == [[1, 2]]
