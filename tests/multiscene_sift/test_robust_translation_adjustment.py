from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def test_loader_rejects_missing_equal_l2_baseline(tmp_path):
    from src.multiscene_sift.robust_translation_adjustment import (
        load_robust_translation_inputs,
    )

    with pytest.raises(FileNotFoundError):
        load_robust_translation_inputs(tmp_path, tmp_path, tmp_path)


def test_equal_l2_baseline_is_loaded_not_recomputed_with_new_weights():
    from src.multiscene_sift.robust_translation_adjustment import (
        load_robust_translation_inputs,
    )

    root = Path("data/output")
    inputs = load_robust_translation_inputs(
        root / "five_scene_sift_B14",
        root / "five_scene_inlier_recovery",
        root / "five_scene_global_adjustment",
    )
    assert inputs["equal_l2_summary"]["zero_one"]["p95_px"] > 0
    assert inputs["candidate_weighting_used_for_baseline"] is False
    assert inputs["equal_l2_solution"]["status"] == "OK"


def test_quality_weight_does_not_use_inlier_count():
    from src.multiscene_sift.robust_translation_adjustment import (
        compute_intrinsic_edge_quality_weights,
    )

    metrics = {
        (0, 1): {"inlier_ratio": 0.8, "coverage": 0.5, "rmse_px": 1.0, "inliers": 20},
        (0, 4): {"inlier_ratio": 0.8, "coverage": 0.5, "rmse_px": 1.0, "inliers": 5000},
    }
    weights = compute_intrinsic_edge_quality_weights(metrics)
    assert weights[(0, 1)]["weight"] == pytest.approx(weights[(0, 4)]["weight"])


def test_quality_weights_are_bounded():
    from src.multiscene_sift.robust_translation_adjustment import (
        compute_intrinsic_edge_quality_weights,
    )

    metrics = {
        (0, 1): {"inlier_ratio": 0.31, "coverage": 0.05, "rmse_px": 2.0},
        (0, 4): {"inlier_ratio": 0.99, "coverage": 0.90, "rmse_px": 0.2},
    }
    weights = compute_intrinsic_edge_quality_weights(metrics)
    assert all(0.5 <= item["weight"] <= 2.0 for item in weights.values())


def test_huber_factor_is_one_inside_delta():
    from src.multiscene_sift.robust_translation_adjustment import huber_edge_factor

    assert huber_edge_factor(1.0, 3.0) == pytest.approx(1.0)


def test_huber_factor_downweights_but_never_deletes_edge():
    from src.multiscene_sift.robust_translation_adjustment import huber_edge_factor

    assert huber_edge_factor(30.0, 3.0, min_factor=0.1) == pytest.approx(0.1)


def test_huber_delta_uses_pairwise_p95_not_global_residual():
    from src.multiscene_sift.robust_translation_adjustment import derive_huber_delta_px

    metrics = {
        (0, 1): {"p95_px": 1.5},
        (0, 4): {"p95_px": 2.0},
        (1, 4): {"p95_px": 1.0},
    }
    assert derive_huber_delta_px(metrics) == pytest.approx(3.0)


def _synthetic_translation_inputs():
    transforms = {scene: np.eye(3, dtype=np.float64) for scene in range(4)}
    observations = {}
    for edge in ((0, 1), (1, 2), (2, 3), (0, 3)):
        observations[edge] = {
            "x_i": np.array([[0.0, 0.0], [10.0, 5.0]]),
            "x_j": np.array([[0.0, 0.0], [10.0, 5.0]]),
        }
    observations[(0, 3)] = {
        "x_i": np.array([[30.0, 0.0], [40.0, 5.0]]),
        "x_j": np.array([[0.0, 0.0], [10.0, 5.0]]),
    }
    return transforms, observations


def test_irls_without_huber_and_unit_weights_matches_existing_equal_l2_solver():
    from src.multiscene_sift.global_geometric_adjustment import (
        build_translation_adjustment_system,
        solve_translation_adjustment,
    )
    from src.multiscene_sift.robust_translation_adjustment import (
        solve_edge_weighted_translation_irls,
    )

    transforms = {0: np.eye(3), 1: np.eye(3), 2: np.eye(3)}
    observations = {
        (0, 1): {"x_i": np.array([[2.0, 1.0], [4.0, 3.0]]), "x_j": np.zeros((2, 2))},
        (1, 2): {"x_i": np.array([[1.0, 0.0], [3.0, 2.0]]), "x_j": np.zeros((2, 2))},
    }
    robust = solve_edge_weighted_translation_irls(
        transforms,
        observations,
        reference_idx=0,
        prior_edge_weights={(0, 1): 1.0, (1, 2): 1.0},
        use_huber=False,
        huber_delta_px=3.0,
    )
    baseline = solve_translation_adjustment(
        build_translation_adjustment_system(
            transforms, observations, reference_idx=0, weight_mode="equal_edge"
        )
    )
    assert robust["scene_corrections_px"] == pytest.approx(baseline["scene_corrections_px"])


def test_equal_huber_keeps_conflicting_edge_active_with_bounded_factor():
    from src.multiscene_sift.robust_translation_adjustment import (
        solve_edge_weighted_translation_irls,
    )

    transforms, observations = _synthetic_translation_inputs()
    result = solve_edge_weighted_translation_irls(
        transforms,
        observations,
        reference_idx=0,
        prior_edge_weights={edge: 1.0 for edge in observations},
        use_huber=True,
        huber_delta_px=3.0,
    )
    factors = result["final_edge_factors"]
    assert result["status"] == "OK"
    assert factors[(0, 3)] < 1.0
    assert all(0.1 <= value <= 1.0 for value in factors.values())


def test_edge_balanced_irls_is_invariant_to_point_duplication():
    from src.multiscene_sift.robust_translation_adjustment import (
        solve_edge_weighted_translation_irls,
    )

    transforms, observations = _synthetic_translation_inputs()
    duplicated = {
        edge: {"x_i": np.tile(value["x_i"], (100 if edge == (1, 2) else 1, 1)),
               "x_j": np.tile(value["x_j"], (100 if edge == (1, 2) else 1, 1))}
        for edge, value in observations.items()
    }
    kwargs = {
        "reference_idx": 0,
        "prior_edge_weights": {edge: 1.0 for edge in observations},
        "use_huber": True,
        "huber_delta_px": 3.0,
    }
    original = solve_edge_weighted_translation_irls(transforms, observations, **kwargs)
    repeated = solve_edge_weighted_translation_irls(transforms, duplicated, **kwargs)
    for scene in original["scene_corrections_px"]:
        assert repeated["scene_corrections_px"][scene] == pytest.approx(
            original["scene_corrections_px"][scene], abs=1e-8
        )


def test_variant_set_is_frozen():
    from src.multiscene_sift.robust_translation_adjustment import VARIANTS

    assert list(VARIANTS) == [
        "EQUAL_L2",
        "EQUAL_HUBER",
        "QUALITY_L2",
        "QUALITY_HUBER",
    ]


def test_equal_l2_variant_reproduces_frozen_baseline(tmp_path):
    from src.multiscene_sift.robust_translation_adjustment import (
        load_robust_translation_inputs,
        run_translation_variants,
    )

    root = Path("data/output")
    inputs = load_robust_translation_inputs(
        root / "five_scene_sift_B14",
        root / "five_scene_inlier_recovery",
        root / "five_scene_global_adjustment",
    )
    result = run_translation_variants(inputs, tmp_path)
    assert result["baseline_reproduction_status"] == "PASS"
    assert result["variants"]["EQUAL_L2"]["summary"]["zero_one"]["p95_px"] == pytest.approx(
        inputs["equal_l2_summary"]["zero_one"]["p95_px"], abs=1e-8
    )


def test_variant_comparison_preserves_same_point_ids(tmp_path):
    from src.multiscene_sift.robust_translation_adjustment import (
        build_robust_translation_comparison,
        load_robust_translation_inputs,
        run_translation_variants,
    )

    root = Path("data/output")
    inputs = load_robust_translation_inputs(
        root / "five_scene_sift_B14",
        root / "five_scene_inlier_recovery",
        root / "five_scene_global_adjustment",
    )
    result = run_translation_variants(inputs, tmp_path)
    expected_ids = {
        edge: list(range(len(observation["x_i"])))
        for edge, observation in inputs["edge_observations"].items()
    }
    for variant in result["variants"].values():
        rows = variant["point_residuals"].to_dict(orient="records")
        actual_ids = {}
        for row in rows:
            edge = (int(row["edge_i"]), int(row["edge_j"]))
            actual_ids.setdefault(edge, []).append(int(row["point_id"]))
        assert actual_ids == expected_ids
    comparison = build_robust_translation_comparison(
        inputs["equal_l2_summary"],
        inputs["equal_l2_summary"],
        {name: value["summary"] for name, value in result["variants"].items()},
    )
    assert {row["method"] for row in comparison} == {
        "MST", "EQUAL_L2", "EQUAL_HUBER", "QUALITY_L2", "QUALITY_HUBER"
    }


def test_cycle_edge_list_excludes_leaf_edges():
    from src.multiscene_sift.robust_translation_adjustment import (
        CYCLE_EDGES,
        LEAF_EDGES,
    )

    assert CYCLE_EDGES == ((0, 1), (0, 4), (1, 4))
    assert set(LEAF_EDGES) == {(0, 2), (3, 4)}


def test_cycle_sensitivity_only_removes_redundant_triangle_edges():
    from src.multiscene_sift.robust_translation_adjustment import (
        CYCLE_EDGES,
        evaluate_cycle_edge_sensitivity,
    )

    transforms = {0: np.eye(3), 1: np.eye(3), 2: np.eye(3), 3: np.eye(3), 4: np.eye(3)}
    observations = {
        edge: {"x_i": np.zeros((2, 2)), "x_j": np.zeros((2, 2))}
        for edge in ((0, 1), (0, 2), (0, 4), (1, 4), (3, 4))
    }
    result = evaluate_cycle_edge_sensitivity(
        {"huber": False}, transforms, observations, 4,
        {edge: 1.0 for edge in observations}, 3.0,
    )
    assert tuple(result["removed_edges"]) == CYCLE_EDGES
    assert all(item["removed_edge"] not in {(0, 2), (3, 4)} for item in result["results"])


def test_synthetic_case_a_clean_path_drift_is_not_worsened_by_huber():
    from src.multiscene_sift.robust_translation_adjustment import (
        synthetic_robust_translation_check,
    )

    result = synthetic_robust_translation_check()
    case = result["case_a_path_drift"]
    assert case["equal_l2_endpoint_error_px"] < 1.0
    assert case["equal_huber_endpoint_error_px"] <= case["equal_l2_endpoint_error_px"] + 0.1


def test_synthetic_case_b_conflicting_edge_is_downweighted_not_deleted():
    from src.multiscene_sift.robust_translation_adjustment import (
        synthetic_robust_translation_check,
    )

    result = synthetic_robust_translation_check()
    case = result["case_b_conflicting_edge"]
    assert case["equal_huber_reliable_edge_mean_p95_px"] < case["equal_l2_reliable_edge_mean_p95_px"]
    assert case["equal_huber_conflicting_edge_factor"] >= 0.1
    assert case["equal_huber_conflicting_edge_factor"] < 1.0
