import json

import pytest


def test_missing_point_artifact_is_not_silently_rematched(tmp_path):
    """Task 0 must stop when an accepted edge has no stored point data."""
    from src.multiscene_sift.global_geometric_adjustment import (
        discover_five_scene_adjustment_inputs,
    )

    run_dir = tmp_path / "five_scene_run"
    run_dir.mkdir()
    (run_dir / "dataset_manifest.json").write_text(
        json.dumps({"scenes": [{"index": i, "name": f"scene-{i}"} for i in range(2)]}),
        encoding="utf-8",
    )
    (run_dir / "global_registration.json").write_text(
        json.dumps({"reference_index": 0}), encoding="utf-8"
    )
    (run_dir / "spanning_tree.json").write_text(
        json.dumps({"reference_index": 0, "edges": [{"parent": 0, "child": 1}]}),
        encoding="utf-8",
    )
    (run_dir / "pairwise_summary.json").write_text(
        json.dumps(
            {
                "results": [
                    {
                        "idx_i": 0,
                        "idx_j": 1,
                        "status": "OK",
                        "inliers": 4,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    manifest = discover_five_scene_adjustment_inputs(run_dir)

    assert manifest["status"] == "INSUFFICIENT_EXISTING_ARTIFACTS"
    assert manifest["accepted_edges"][0]["point_artifact_status"] == "MISSING"
    assert manifest["rematch_allowed"] is False


def test_input_manifest_writer_persists_a_machine_readable_audit(tmp_path):
    from src.multiscene_sift.global_geometric_adjustment import (
        write_adjustment_input_manifest,
    )

    path = write_adjustment_input_manifest({"status": "READY"}, tmp_path)

    assert path.name == "00_adjustment_input_manifest.json"
    assert json.loads(path.read_text(encoding="utf-8"))["status"] == "READY"


def test_historical_baselines_keep_accepted_and_rejected_edges_separate(tmp_path):
    from src.multiscene_sift.global_geometric_adjustment import (
        load_historical_pair_baselines,
    )

    (tmp_path / "pairwise_summary.json").write_text(
        json.dumps(
            {
                "results": [
                    {
                        "idx_i": 0,
                        "idx_j": 1,
                        "status": "OK",
                        "raw_matches": 10,
                        "inliers": 8,
                        "inlier_ratio": 0.8,
                        "coverage": 0.5,
                        "residual_rmse": 1.2,
                        "residual_p95": 2.3,
                        "pixel_matrix": [[1, 0, 4], [0, 1, 5], [0, 0, 1]],
                    },
                    {"idx_i": 0, "idx_j": 2, "status": "NO_OVERLAP", "inliers": 0},
                ]
            }
        ),
        encoding="utf-8",
    )

    result = load_historical_pair_baselines(tmp_path)

    assert result["accepted_edges"][0]["edge"] == [0, 1]
    assert result["accepted_edges"][0]["rmse_px"] == 1.2
    assert result["rejected_edges"][0]["edge"] == [0, 2]


def test_missing_historical_pair_summary_is_a_hard_failure(tmp_path):
    from src.multiscene_sift.global_geometric_adjustment import (
        load_historical_pair_baselines,
    )

    with pytest.raises(FileNotFoundError, match="pairwise_summary.json"):
        load_historical_pair_baselines(tmp_path)


def test_historical_baseline_writer_uses_required_artifact_name(tmp_path):
    from src.multiscene_sift.global_geometric_adjustment import (
        write_historical_pair_baselines,
    )

    path = write_historical_pair_baselines({"accepted_edges": []}, tmp_path)

    assert path.name == "00_historical_pair_baselines.json"
    assert json.loads(path.read_text(encoding="utf-8"))["accepted_edges"] == []
