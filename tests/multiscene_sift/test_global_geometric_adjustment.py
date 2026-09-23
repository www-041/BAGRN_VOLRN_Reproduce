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


def test_frozen_config_contains_registration_call_chain_values(tmp_path):
    from src.multiscene_sift.global_geometric_adjustment import (
        load_frozen_registration_config,
    )

    (tmp_path / "run_config.json").write_text(
        json.dumps(
            {
                "matcher": "sift",
                "registration_band": "B14",
                "match_max_side": 1600,
                "random_seed": 0,
                "ransac_threshold": 2.0,
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "dataset_manifest.json").write_text(
        json.dumps({"scenes": [{"bands": {"B14": {"nodata": 0.0}}}]}),
        encoding="utf-8",
    )

    config = load_frozen_registration_config(tmp_path)

    assert config["matcher"] == "SIFT"
    assert config["lowe_ratio"] == 0.75
    assert config["mutual_check"] is True
    assert config["ransac_model"] == "AffineTransform"
    assert config["ransac_max_trials"] == 5000
    assert config["minimum_inliers"] == 20
    assert config["minimum_inlier_ratio"] == 0.30
    assert config["nodata_policy"]["B14"] == [0.0]


def test_frozen_config_does_not_fill_missing_required_run_config_with_default(tmp_path):
    from src.multiscene_sift.global_geometric_adjustment import (
        load_frozen_registration_config,
    )

    (tmp_path / "run_config.json").write_text(
        json.dumps({"matcher": "sift", "registration_band": "B14"}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="FROZEN_CONFIG_INCOMPLETE"):
        load_frozen_registration_config(tmp_path)


def test_frozen_config_writer_uses_required_artifact_name(tmp_path):
    from src.multiscene_sift.global_geometric_adjustment import (
        write_frozen_registration_config,
    )

    path = write_frozen_registration_config({"matcher": "SIFT"}, tmp_path)

    assert path.name == "01_frozen_registration_config.json"


def test_replay_wrapper_calls_existing_registration_once_and_exports_inliers(tmp_path, monkeypatch):
    import numpy as np
    from src.multiscene_sift.models import PairwiseRegistration
    from src.multiscene_sift import inlier_recovery

    calls = []
    config = {
        "matcher": "SIFT",
        "registration_band": "B14",
        "match_max_side": 1600,
        "ransac_residual_threshold": 2.0,
        "seed": 0,
    }

    def fake_register(scene_i, scene_j, **kwargs):
        calls.append((scene_i, scene_j, kwargs))
        return PairwiseRegistration(
            idx_i=0,
            idx_j=1,
            status="OK",
            raw_matches=3,
            inliers=2,
            inlier_ratio=2 / 3,
            coverage=0.5,
            residual_median=0.1,
            residual_rmse=0.2,
            residual_p95=0.3,
            pair_pixel_matrix=[[1, 0, 2], [0, 1, 3], [0, 0, 1]],
            pair_common_transform=None,
            runtime_sec=0.1,
            inlier_ref_xy=np.array([[2.0, 3.0], [5.0, 6.0]]),
            inlier_tgt_xy=np.array([[0.0, 0.0], [3.0, 3.0]]),
        )

    monkeypatch.setattr(inlier_recovery, "register_pair", fake_register)
    result = inlier_recovery.replay_pair_and_capture_inliers(
        "scene-i", "scene-j", config, tmp_path
    )

    assert len(calls) == 1
    assert {key: value for key, value in calls[0][2].items() if key != "diagnostic_capture"} == {
        "band": "B14",
        "match_max_side": 1600,
        "ransac_threshold": 2.0,
        "random_seed": 0,
        "matcher": "sift",
    }
    assert callable(calls[0][2]["diagnostic_capture"])
    assert config["matcher"] == "SIFT"
    assert result["coordinate_frame"] == "pair_common_grid"
    assert result["transform_direction"] == "target_to_reference"
    rows = (tmp_path / "02_replayed_pairs" / "0_1_inliers.csv").read_text(
        encoding="utf-8"
    ).splitlines()
    assert rows[0].startswith("point_id,x_i,y_i,x_j,y_j,coordinate_frame,residual_px")
    assert len(rows) == 3


def test_replay_wrapper_captures_raw_matches_through_existing_entrypoint(tmp_path, monkeypatch):
    import numpy as np
    from src.multiscene_sift.models import PairwiseRegistration
    from src.multiscene_sift import inlier_recovery

    def fake_register(scene_i, scene_j, **kwargs):
        callback = kwargs.pop("diagnostic_capture")
        callback({
            "raw_ref_xy": np.array([[1.0, 2.0]]),
            "raw_tgt_xy": np.array([[0.0, 0.0]]),
            "inlier_mask": np.array([True]),
            "coordinate_frame": "pair_common_grid",
        })
        return PairwiseRegistration(
            idx_i=0, idx_j=1, status="OK", raw_matches=1, inliers=1,
            inlier_ratio=1.0, coverage=0.5, residual_median=0.0,
            residual_rmse=0.0, residual_p95=0.0,
            pair_pixel_matrix=[[1, 0, 1], [0, 1, 2], [0, 0, 1]],
            pair_common_transform=None, runtime_sec=0.0,
            inlier_ref_xy=np.array([[1.0, 2.0]]),
            inlier_tgt_xy=np.array([[0.0, 0.0]]),
        )

    monkeypatch.setattr(inlier_recovery, "register_pair", fake_register)
    result = inlier_recovery.replay_pair_and_capture_inliers(
        "scene-i", "scene-j",
        {"matcher": "SIFT", "registration_band": "B14", "match_max_side": 1600,
         "ransac_residual_threshold": 2.0, "seed": 0},
        tmp_path,
    )

    assert result["raw_coordinates_available"] is True
    assert result["raw_match_count"] == 1


def test_recovery_cli_replays_only_historical_accepted_edges(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from src.multiscene_sift import inlier_recovery

    monkeypatch.setattr(
        inlier_recovery,
        "load_historical_pair_baselines",
        lambda _: {
            "accepted_edges": [{"edge": [0, 1]}],
            "rejected_edges": [{"edge": [0, 2]}],
        },
    )
    monkeypatch.setattr(
        inlier_recovery,
        "load_frozen_registration_config",
        lambda _: {"matcher": "SIFT", "registration_band": "B14"},
    )
    monkeypatch.setattr(
        inlier_recovery,
        "load_scene_names",
        lambda _: ["s0", "s1", "s2"],
    )
    monkeypatch.setattr(
        inlier_recovery,
        "discover_five_scenes",
        lambda *_args, **_kwargs: ([SimpleNamespace(index=i) for i in range(3)], {}),
    )
    replayed = []
    monkeypatch.setattr(
        inlier_recovery,
        "replay_pair_and_capture_inliers",
        lambda i, j, config, output: replayed.append((i, j)) or {"edge": [0, 1]},
    )

    result = inlier_recovery.recover_accepted_pairs(tmp_path, tmp_path, tmp_path)

    assert [(i.index, j.index) for i, j in replayed] == [(0, 1)]
    assert result["replayed_edges"] == [[0, 1]]


def test_recovery_script_forwards_explicit_paths(tmp_path, monkeypatch):
    from scripts import recover_five_scene_inliers

    calls = []
    monkeypatch.setattr(
        recover_five_scene_inliers,
        "recover_accepted_pairs",
        lambda run, root, output: calls.append((run, root, output)) or {"accepted_count": 1},
    )

    assert recover_five_scene_inliers.main(
        ["--five-scene-run-dir", "run", "--input-root", "root", "--output-dir", "out"]
    ) == 0
    assert calls == [("run", "root", "out")]


def test_reproduction_gate_exact_close_and_mismatch_states():
    from src.multiscene_sift.inlier_recovery import compare_replayed_pair_to_historical

    historical = {
        "raw_matches": 100, "inliers": 80, "inlier_ratio": 0.8,
        "rmse_px": 1.0, "p95_px": 2.0,
        "affine_matrix": [[1, 0, 3], [0, 1, 4], [0, 0, 1]],
    }
    exact = {**historical, "raw_coordinates": {"tgt_xy": [[0, 0], [1, 1]]}}
    close = {**historical, "inliers": 81, "rmse_px": 1.04, "p95_px": 2.09,
             "raw_coordinates": {"tgt_xy": [[0, 0], [1, 1]]}}
    mismatch = {**close, "rmse_px": 1.2}

    assert compare_replayed_pair_to_historical(historical, exact)["state"] == "EXACT"
    assert compare_replayed_pair_to_historical(historical, close)["state"] == "CLOSE"
    assert compare_replayed_pair_to_historical(historical, mismatch)["state"] == "MISMATCH"


def test_reproduction_gate_compares_affine_pointwise_not_only_translation():
    from src.multiscene_sift.inlier_recovery import compare_replayed_pair_to_historical

    historical = {
        "raw_matches": 2, "inliers": 2, "inlier_ratio": 1.0,
        "rmse_px": 0.0, "p95_px": 0.0,
        "affine_matrix": [[1, 0, 10], [0, 1, 10], [0, 0, 1]],
    }
    replayed = {
        **historical,
        "raw_coordinates": {"tgt_xy": [[0, 0], [100, 100]]},
        "affine_matrix": [[1, 0, 10.2], [0, 1, 10], [0, 0, 1]],
    }

    result = compare_replayed_pair_to_historical(historical, replayed)

    assert result["affine_prediction_max_diff_px"] == pytest.approx(0.2)
    assert result["state"] == "CLOSE"


def test_any_required_edge_mismatch_blocks_overall_reproduction_acceptance(tmp_path):
    from src.multiscene_sift.inlier_recovery import evaluate_reproduction_gate

    historical = {
        "accepted_edges": [{
            "edge": [0, 1], "raw_matches": 1, "inliers": 1,
            "inlier_ratio": 1.0, "rmse_px": 0.0, "p95_px": 0.0,
            "affine_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        }]
    }
    replayed = [{
        "edge": [0, 1], "raw_matches": 2, "inliers": 1,
        "inlier_ratio": 0.5, "rmse_px": 0.0, "p95_px": 0.0,
        "affine_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "raw_coordinates": {"tgt_xy": [[0, 0]]},
    }]

    result = evaluate_reproduction_gate(historical, replayed, tmp_path)

    assert result["overall"] == "REPRODUCTION_FAILED"
    assert result["edges"][0]["state"] == "MISMATCH"
    assert (tmp_path / "03_reproduction_gate.csv").is_file()
    assert (tmp_path / "03_reproduction_gate.json").is_file()


def test_coordinate_validation_recomputes_rmse_and_p95(tmp_path):
    import csv
    from src.multiscene_sift.inlier_recovery import validate_recovered_pair_coordinates

    path = tmp_path / "inliers.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["point_id", "x_i", "y_i", "x_j", "y_j", "coordinate_frame", "residual_px"])
        writer.writerow([0, 1, 1, 0, 0, "pair_common_grid", 0])
        writer.writerow([1, 3, 1, 2, 0, "pair_common_grid", 0])

    result = validate_recovered_pair_coordinates({
        "edge": [0, 1],
        "inliers_csv": str(path),
        "affine_matrix": [[1, 0, 1], [0, 1, 1], [0, 0, 1]],
        "coordinate_frame": "pair_common_grid",
        "transform_direction": "target_to_reference",
        "rmse_px": 0.0,
        "p95_px": 0.0,
    })

    assert result["status"] == "PASS"
    assert result["recomputed_rmse_px"] == 0.0
    assert result["recomputed_p95_px"] == 0.0


def test_coordinate_validation_rejects_wrong_direction_and_frame(tmp_path):
    from src.multiscene_sift.inlier_recovery import validate_recovered_pair_coordinates

    base = {
        "edge": [0, 1], "inliers_csv": "missing.csv",
        "affine_matrix": [[1, 0, 1], [0, 1, 1], [0, 0, 1]],
        "rmse_px": 0.0, "p95_px": 0.0,
    }
    wrong_direction = validate_recovered_pair_coordinates({
        **base, "coordinate_frame": "pair_common_grid", "transform_direction": "reference_to_target"
    })
    wrong_frame = validate_recovered_pair_coordinates({
        **base, "coordinate_frame": "native_pixels", "transform_direction": "target_to_reference"
    })

    assert wrong_direction["status"] == "FAIL"
    assert wrong_direction["reason"] == "WRONG_TRANSFORM_DIRECTION"
    assert wrong_frame["status"] == "FAIL"
    assert wrong_frame["reason"] == "WRONG_COORDINATE_FRAME"


def test_coordinate_validation_overall_fails_when_any_edge_fails(tmp_path):
    from src.multiscene_sift.inlier_recovery import evaluate_coordinate_validation

    result = evaluate_coordinate_validation(
        [{"edge": [0, 1], "transform_direction": "wrong", "coordinate_frame": "pair_common_grid"}],
        tmp_path,
    )

    assert result["overall"] == "COORDINATE_RECOVERY_INVALID"
    assert (tmp_path / "04_inlier_coordinate_validation.csv").is_file()
    assert (tmp_path / "04_inlier_coordinate_validation.json").is_file()


def test_frozen_global_adjustment_artifact_contains_point_and_provenance_fields(tmp_path):
    import csv
    from src.multiscene_sift.inlier_recovery import freeze_global_adjustment_inliers

    csv_path = tmp_path / "edge.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["point_id", "x_i", "y_i", "x_j", "y_j", "coordinate_frame", "residual_px"])
        writer.writerow([0, 1, 2, 3, 4, "pair_common_grid", 0.5])
    replayed = [{
        "edge": [0, 1], "inliers_csv": str(csv_path),
        "coordinate_frame": "pair_common_grid", "transform_direction": "target_to_reference",
    }]
    gate = {"overall": "REPRODUCTION_ACCEPTED", "edges": [{"edge": [0, 1], "state": "EXACT"}]}
    validation = {"overall": "COORDINATE_RECOVERY_VALID", "edges": [{"edge": [0, 1], "status": "PASS"}]}

    result = freeze_global_adjustment_inliers(
        replayed, gate, validation, {"seed": 0}, reference_idx=1, output_dir=tmp_path
    )

    assert result["status"] == "READY_FOR_GLOBAL_ADJUSTMENT"
    row = (tmp_path / "05_global_adjustment_inliers.csv").read_text(encoding="utf-8").splitlines()[1]
    assert row.startswith("0,1,0,1,2,3,4,pair_common_grid")
    manifest = json.loads((tmp_path / "05_global_adjustment_inliers_manifest.json").read_text(encoding="utf-8"))
    assert manifest["reference_idx"] == 1
    assert manifest["sha256"]


def test_inlier_recovery_decision_is_ready_only_after_both_gates(tmp_path):
    from src.multiscene_sift.inlier_recovery import write_inlier_recovery_decision

    ready = write_inlier_recovery_decision(
        {"overall": "REPRODUCTION_ACCEPTED"},
        {"overall": "COORDINATE_RECOVERY_VALID"},
        {"status": "READY_FOR_GLOBAL_ADJUSTMENT"},
        {"status": "complete"},
        tmp_path,
    )
    failed = write_inlier_recovery_decision(
        {"overall": "REPRODUCTION_FAILED"},
        {"overall": "COORDINATE_RECOVERY_VALID"},
        {}, {"status": "complete"}, tmp_path
    )

    assert ready["decision"] == "READY_FOR_GLOBAL_ADJUSTMENT"
    assert failed["decision"] == "REPRODUCTION_FAILED"
    assert (tmp_path / "06_inlier_recovery_decision.json").is_file()
    assert (tmp_path / "06_inlier_recovery_decision.txt").is_file()


def test_mst_point_residuals_are_zero_for_consistent_global_transforms():
    import numpy as np
    from src.multiscene_sift.global_geometric_adjustment import evaluate_edge_point_residuals

    transforms = {0: np.eye(3), 1: np.array([[1, 0, 10], [0, 1, 0], [0, 0, 1]], float)}
    observations = {(0, 1): {
        "x_i": np.array([[1.0, 2.0], [3.0, 4.0]]),
        "x_j": np.array([[-9.0, 2.0], [-7.0, 4.0]]),
        "is_tree_edge": True,
    }}

    result = evaluate_edge_point_residuals(transforms, observations)

    assert list(result["residual_px"]) == [0.0, 0.0]
    assert result.loc[0, "is_tree_edge"] is True


def test_mst_point_residuals_report_known_ten_pixel_mismatch():
    import numpy as np
    from src.multiscene_sift.global_geometric_adjustment import evaluate_edge_point_residuals

    observations = {(0, 1): {
        "x_i": np.array([[1.0, 2.0]]), "x_j": np.array([[1.0, 2.0]]),
        "is_tree_edge": False,
    }}

    result = evaluate_edge_point_residuals({0: np.eye(3), 1: np.eye(3)}, observations)

    assert result.loc[0, "residual_px"] == 0.0
    shifted = {(0, 1): {**observations[(0, 1)], "x_j": np.array([[-9.0, 2.0]])}}
    shifted_result = evaluate_edge_point_residuals({0: np.eye(3), 1: np.eye(3)}, shifted)
    assert shifted_result.loc[0, "residual_px"] == 10.0


def test_mst_network_summary_reports_point_and_edge_balanced_metrics():
    import numpy as np
    from src.multiscene_sift.global_geometric_adjustment import (
        evaluate_edge_point_residuals,
        summarize_network_residuals,
    )

    observations = {
        (0, 1): {"x_i": np.array([[0.0, 0.0], [1.0, 0.0]]),
                 "x_j": np.array([[0.0, 0.0], [1.0, 0.0]]),
                 "is_tree_edge": False},
        (1, 2): {"x_i": np.array([[0.0, 0.0]]),
                 "x_j": np.array([[3.0, 4.0]]),
                 "is_tree_edge": True},
    }

    summary = summarize_network_residuals(
        evaluate_edge_point_residuals({0: np.eye(3), 1: np.eye(3), 2: np.eye(3)}, observations)
    )

    assert summary["zero_one"]["rmse_px"] == 0.0
    assert summary["point_weighted"]["n_points"] == 3
    assert summary["point_weighted"]["rmse_px"] == 5.0 / np.sqrt(3.0)
    assert summary["edge_balanced"]["mean_edge_rmse_px"] == 2.5
    assert summary["tree_edges"]["edge_count"] == 1
    assert summary["non_tree_edges"]["edge_count"] == 1


def test_translation_system_dimensions_and_equal_edge_weights():
    import numpy as np
    from src.multiscene_sift.global_geometric_adjustment import build_translation_adjustment_system

    observations = {
        (0, 1): {"x_i": np.array([[10.0, 5.0], [20.0, 5.0]]),
                 "x_j": np.array([[0.0, 0.0], [10.0, 0.0]])},
        (1, 2): {"x_i": np.array([[4.0, 8.0]]),
                 "x_j": np.array([[0.0, 0.0]])},
    }

    system = build_translation_adjustment_system(
        {0: np.eye(3), 1: np.eye(3), 2: np.eye(3)}, observations, reference_idx=0
    )

    assert system["unknown_scene_order"] == [1, 2]
    assert system["A"].shape == (6, 4)
    assert system["b"].shape == (6,)
    assert np.allclose(system["point_weights"], [0.5, 0.5, 1.0])
    assert system["rank_expectation"] == 4
    assert np.allclose(system["b"][:2], [-10.0, -5.0])


def test_translation_system_reversed_edge_has_same_relation():
    import numpy as np
    from src.multiscene_sift.global_geometric_adjustment import build_translation_adjustment_system

    forward = build_translation_adjustment_system(
        {0: np.eye(3), 1: np.eye(3)},
        {(0, 1): {"x_i": np.array([[10.0, 5.0]]), "x_j": np.array([[0.0, 0.0]])}},
        reference_idx=0,
    )
    reverse = build_translation_adjustment_system(
        {0: np.eye(3), 1: np.eye(3)},
        {(1, 0): {"x_i": np.array([[0.0, 0.0]]), "x_j": np.array([[10.0, 5.0]])}},
        reference_idx=0,
    )

    assert np.allclose(forward["A"].T @ forward["A"], reverse["A"].T @ reverse["A"])
    assert np.allclose(forward["A"].T @ forward["b"], reverse["A"].T @ reverse["b"])


def test_translation_system_rejects_missing_reference_and_disconnected_graph():
    import numpy as np
    import pytest
    from src.multiscene_sift.global_geometric_adjustment import build_translation_adjustment_system

    edge = {(1, 2): {"x_i": np.array([[1.0, 1.0]]), "x_j": np.array([[0.0, 0.0]])}}
    with pytest.raises(ValueError, match="reference"):
        build_translation_adjustment_system({1: np.eye(3), 2: np.eye(3)}, edge, reference_idx=0)
    with pytest.raises(ValueError, match="disconnected"):
        build_translation_adjustment_system(
            {0: np.eye(3), 1: np.eye(3), 2: np.eye(3)}, edge, reference_idx=0
        )


def test_translation_solver_recovers_exact_two_node_correction_and_gauge():
    import numpy as np
    from src.multiscene_sift.global_geometric_adjustment import (
        build_translation_adjustment_system,
        solve_translation_adjustment,
    )

    system = build_translation_adjustment_system(
        {0: np.eye(3), 1: np.eye(3)},
        {(0, 1): {"x_i": np.array([[10.0, 5.0], [20.0, 5.0]]),
                 "x_j": np.array([[0.0, 0.0], [10.0, 0.0]])}},
        reference_idx=0,
    )

    result = solve_translation_adjustment(system)

    assert result["status"] == "OK"
    assert np.allclose(result["scene_corrections_px"][0], [0.0, 0.0])
    assert np.allclose(result["scene_corrections_px"][1], [10.0, 5.0])
    assert result["objective_before"] > 0.0
    assert np.isclose(result["objective_after"], 0.0)


def test_translation_solver_redistributes_inconsistent_triangle_residuals():
    import numpy as np
    from src.multiscene_sift.global_geometric_adjustment import solve_translation_adjustment

    system = {
        "A": np.array([[1.0, 0.0, 0.0, 0.0],
                       [0.0, 1.0, 0.0, 0.0],
                       [-1.0, 0.0, 1.0, 0.0],
                       [0.0, -1.0, 0.0, 1.0],
                       [0.0, 0.0, -1.0, 0.0],
                       [0.0, 0.0, 0.0, -1.0]]),
        "b": np.array([10.0, 0.0, 0.0, 0.0, 10.0, 0.0]),
        "point_weights": np.ones(3),
        "unknown_scene_order": [1, 2],
        "reference_idx": 0,
        "rank": 4,
        "condition_number": 1.0,
    }

    result = solve_translation_adjustment(system)

    assert result["status"] == "OK"
    assert result["objective_after"] < result["objective_before"]
    assert result["objective_after"] > 0.0
    assert np.allclose(result["scene_corrections_px"][0], [0.0, 0.0])


def test_translation_solver_reports_rank_deficiency_without_fabricating_solution():
    import numpy as np
    from src.multiscene_sift.global_geometric_adjustment import solve_translation_adjustment

    result = solve_translation_adjustment({
        "A": np.zeros((2, 2)), "b": np.array([1.0, 2.0]),
        "point_weights": np.ones(1), "unknown_scene_order": [1],
        "reference_idx": 0, "rank": 0, "condition_number": float("inf"),
    })

    assert result["status"] == "RANK_DEFICIENT"
    assert result["scene_corrections_px"] == {0: [0.0, 0.0]}


def test_translation_correction_composes_in_global_frame_and_keeps_reference():
    import numpy as np
    from src.multiscene_sift.global_geometric_adjustment import apply_translation_corrections

    transforms = {
        0: np.eye(3),
        1: np.array([[1.0, 0.0, 10.0], [0.0, 1.0, 20.0], [0.0, 0.0, 1.0]]),
    }
    adjusted = apply_translation_corrections(transforms, {0: (0.0, 0.0), 1: (2.0, -3.0)})

    assert np.allclose(adjusted[0], transforms[0])
    assert np.allclose(adjusted[1], [[1.0, 0.0, 12.0], [0.0, 1.0, 17.0], [0.0, 0.0, 1.0]])


def test_translation_before_after_evaluation_uses_same_point_ids():
    import numpy as np
    from src.multiscene_sift.global_geometric_adjustment import (
        apply_translation_corrections,
        build_translation_adjustment_system,
        evaluate_edge_point_residuals,
        solve_translation_adjustment,
    )

    observations = {(0, 1): {
        "x_i": np.array([[10.0, 5.0], [20.0, 5.0]]),
        "x_j": np.array([[0.0, 0.0], [10.0, 0.0]]),
        "is_tree_edge": False,
    }}
    mst = {0: np.eye(3), 1: np.eye(3)}
    before = evaluate_edge_point_residuals(mst, observations)
    solution = solve_translation_adjustment(
        build_translation_adjustment_system(mst, observations, reference_idx=0)
    )
    after = evaluate_edge_point_residuals(
        apply_translation_corrections(mst, solution["scene_corrections_px"]), observations
    )

    assert list(before["point_id"]) == list(after["point_id"]) == [0, 1]
    assert list(before["residual_px"]) == [np.sqrt(125.0), np.sqrt(125.0)]
    assert np.allclose(after["residual_px"], [0.0, 0.0])


def test_cycle_closure_is_invariant_under_node_translation():
    import numpy as np
    from src.multiscene_sift.global_geometric_adjustment import (
        evaluate_cycle_invariance_under_node_translation,
    )

    result = evaluate_cycle_invariance_under_node_translation(
        {(0, 1): np.array([10.0, 0.0]),
         (1, 4): np.array([0.0, 20.0]),
         (0, 4): np.array([7.0, 3.0])},
        {0: (4.0, -2.0), 1: (-3.0, 6.0), 4: (0.0, 0.0)},
        [0, 1, 4],
    )

    assert result["invariant"] is True
    assert np.allclose(result["closure_before_px"], result["closure_after_px"])
    assert np.allclose(result["closure_before_px"], [3.0, 17.0])


def test_translation_diagnostic_plots_distinguish_tree_and_non_tree_edges(tmp_path):
    import numpy as np
    from src.multiscene_sift.global_geometric_adjustment import (
        plot_mst_vs_translation_network,
        plot_mst_vs_translation_residuals,
    )

    summary_before = {"per_edge": [
        {"edge_i": 0, "edge_j": 1, "is_tree_edge": False, "p95_px": 10.0},
        {"edge_i": 1, "edge_j": 2, "is_tree_edge": True, "p95_px": 2.0},
    ]}
    summary_after = {"per_edge": [
        {"edge_i": 0, "edge_j": 1, "is_tree_edge": False, "p95_px": 5.0},
        {"edge_i": 1, "edge_j": 2, "is_tree_edge": True, "p95_px": 3.0},
    ]}
    transforms = {0: np.eye(3), 1: np.array([[1, 0, 2], [0, 1, 1], [0, 0, 1.]]),
                  2: np.array([[1, 0, 4], [0, 1, 1], [0, 0, 1.]])}

    network = plot_mst_vs_translation_network(
        transforms, [(1, 2)], [(0, 1), (1, 2)], {0: (0, 0), 1: (1, 2), 2: (0, 0)},
        tmp_path / "network.png"
    )
    residuals = plot_mst_vs_translation_residuals(summary_before, summary_after, tmp_path / "residuals.png")

    assert network.is_file()
    assert residuals.is_file()


def test_translation_decision_gate_distinguishes_partial_and_invalid():
    from src.multiscene_sift.global_geometric_adjustment import decide_translation_adjustment

    before = {"edge_balanced": {"mean_edge_rmse_px": 10.0, "max_edge_p95_px": 20.0},
              "zero_one": {"p95_px": 30.0},
              "per_edge": [{"edge_i": 0, "edge_j": 1, "p95_px": 30.0}]}
    after = {"edge_balanced": {"mean_edge_rmse_px": 9.0, "max_edge_p95_px": 10.0},
             "zero_one": {"p95_px": 10.0},
             "per_edge": [{"edge_i": 0, "edge_j": 1, "p95_px": 10.0}]}
    partial = decide_translation_adjustment(before, after, {
        "status": "OK", "rank": 4, "objective_before": 100.0, "objective_after": 50.0,
        "scene_corrections_px": {0: [0.0, 0.0]},
    })
    invalid = decide_translation_adjustment(before, after, {
        "status": "RANK_DEFICIENT", "rank": 2, "objective_before": None, "objective_after": None,
        "scene_corrections_px": {0: [0.0, 0.0]},
    })

    assert partial["decision"] == "TRANSLATION_ADJUSTMENT_PARTIAL"
    assert invalid["decision"] == "TRANSLATION_SYSTEM_INVALID"
