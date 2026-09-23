from __future__ import annotations

from pathlib import Path

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
