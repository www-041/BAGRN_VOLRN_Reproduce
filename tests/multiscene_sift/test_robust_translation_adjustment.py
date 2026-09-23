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
