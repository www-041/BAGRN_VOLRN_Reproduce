import numpy as np
import pytest


def _controls():
    points = np.asarray(
        [[x, y] for y in np.linspace(0, 30, 4)
         for x in np.linspace(0, 30, 4)],
        dtype=float,
    )
    return {
        "points_xy": points,
        "residual_dx": points[:, 0] / 30.0 + points[:, 1] / 60.0,
        "residual_dy": points[:, 1] / 60.0,
        "confidence": np.ones(len(points)),
        "n_valid": len(points),
    }


def test_build_local_cv_fold_plan_is_deterministic():
    from src import multiband_pipeline

    a = multiband_pipeline._build_local_cv_fold_plan(_controls()["points_xy"], {})
    b = multiband_pipeline._build_local_cv_fold_plan(_controls()["points_xy"], {})
    assert np.array_equal(a["groups"], b["groups"])
    assert [(f["group_id"], f["train_idx"].tolist(), f["test_idx"].tolist())
            for f in a["folds"]] == [
        (f["group_id"], f["train_idx"].tolist(), f["test_idx"].tolist())
        for f in b["folds"]
    ]


def test_build_local_cv_fold_plan_covers_same_controls_once():
    from src import multiband_pipeline

    result = multiband_pipeline._build_local_cv_fold_plan(
        _controls()["points_xy"], {}
    )
    validation = np.concatenate([fold["test_idx"] for fold in result["folds"]])
    assert len(np.unique(validation)) == len(validation)
    assert np.array_equal(np.sort(validation), np.arange(16))
    assert result["validation_coverage"] == pytest.approx(1.0)


def test_build_local_cv_fold_plan_rejects_too_few_folds():
    from src import multiband_pipeline

    points = np.zeros((8, 2), dtype=float)
    result = multiband_pipeline._build_local_cv_fold_plan(
        points, {"local_cv_min_folds": 3}
    )
    assert result["available"] is False
    assert result["n_folds_attempted"] == 1


def test_build_local_cv_fold_plan_rejects_too_few_validation_controls():
    from src import multiband_pipeline

    result = multiband_pipeline._build_local_cv_fold_plan(
        _controls()["points_xy"], {"local_cv_min_validation_controls": 17}
    )
    assert result["available"] is False
    assert "validation controls" in result["failure_reason"]


def test_evaluate_candidate_uses_all_planned_folds(monkeypatch):
    from src import coregistration, multiband_pipeline

    calls = []

    class ConstantRBF:
        def __call__(self, points):
            return np.zeros(len(points))

    def fake_fit(*args, **kwargs):
        calls.append(kwargs["smoothing"])
        return ConstantRBF(), ConstantRBF(), (0.0, 0.0), (30.0, 30.0)

    monkeypatch.setattr(coregistration, "fit_local_rbf", fake_fit)
    controls = _controls()
    plan = multiband_pipeline._build_local_cv_fold_plan(
        controls["points_xy"], {}
    )
    result = multiband_pipeline._evaluate_local_rbf_smoothing_candidate(
        controls, plan, 0.1, {}
    )
    assert result["available"] is True
    assert result["n_folds"] == len(plan["folds"])
    assert calls == [0.1] * len(plan["folds"])


def test_evaluate_candidate_marks_fold_failure_unavailable(monkeypatch):
    from src import coregistration, multiband_pipeline

    calls = {"count": 0}

    def fail_first(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise ValueError("synthetic fold failure")
        raise AssertionError("all planned folds must stop after one failure")

    monkeypatch.setattr(coregistration, "fit_local_rbf", fail_first)
    controls = _controls()
    plan = multiband_pipeline._build_local_cv_fold_plan(
        controls["points_xy"], {}
    )
    result = multiband_pipeline._evaluate_local_rbf_smoothing_candidate(
        controls, plan, 0.1, {}
    )
    assert result["available"] is False
    assert result["failed_group_ids"]
    assert "synthetic fold failure" in result["failure_reason"]


def test_evaluate_candidate_rejects_nonfinite_predictions(monkeypatch):
    from src import coregistration, multiband_pipeline

    class NonFiniteRBF:
        def __call__(self, points):
            return np.full(len(points), np.nan)

    monkeypatch.setattr(
        coregistration,
        "fit_local_rbf",
        lambda *args, **kwargs: (
            NonFiniteRBF(), NonFiniteRBF(), (0.0, 0.0), (30.0, 30.0)
        ),
    )
    controls = _controls()
    plan = multiband_pipeline._build_local_cv_fold_plan(
        controls["points_xy"], {}
    )
    result = multiband_pipeline._evaluate_local_rbf_smoothing_candidate(
        controls, plan, 0.1, {}
    )
    assert result["available"] is False
    assert "finite" in result["failure_reason"]


def test_evaluate_candidate_scores_after_component_clipping(monkeypatch):
    from src import coregistration, multiband_pipeline

    class ConstantRBF:
        def __call__(self, points):
            return np.full(len(points), 10.0)

    monkeypatch.setattr(
        coregistration,
        "fit_local_rbf",
        lambda *args, **kwargs: (
            ConstantRBF(), ConstantRBF(), (0.0, 0.0), (30.0, 30.0)
        ),
    )
    controls = _controls()
    controls["residual_dx"] = np.zeros(len(controls["points_xy"]))
    controls["residual_dy"] = np.zeros(len(controls["points_xy"]))
    plan = multiband_pipeline._build_local_cv_fold_plan(
        controls["points_xy"], {}
    )
    result = multiband_pipeline._evaluate_local_rbf_smoothing_candidate(
        controls, plan, 0.1, {"local_hard_max_component": 2.5}
    )
    assert result["candidate_rmse"] == pytest.approx(np.hypot(2.5, 2.5))


def test_local_holdout_cv_evaluates_every_configured_smoothing(monkeypatch):
    from src import multiband_pipeline

    seen = []

    def fake_evaluate(controls, fold_plan, smoothing, params):
        seen.append((smoothing, id(fold_plan)))
        return {
            "smoothing": smoothing, "available": True,
            "candidate_rmse": 0.5 + smoothing,
            "candidate_p95": 0.8 + smoothing,
            "n_folds": 4, "n_folds_attempted": 4,
            "n_validation_controls": 16, "validation_coverage": 1.0,
            "failed_group_ids": [], "failure_reason": None,
        }

    monkeypatch.setattr(
        multiband_pipeline, "_evaluate_local_rbf_smoothing_candidate", fake_evaluate
    )
    result = multiband_pipeline._local_holdout_cv(
        _controls(), {"local_smoothing_candidates": [0.01, 0.05, 0.1, 0.5, 1.0]}
    )
    assert [item[0] for item in seen] == [0.01, 0.05, 0.1, 0.5, 1.0]
    assert len({item[1] for item in seen}) == 1
    assert len(result["candidate_results"]) == 5


def test_local_holdout_cv_uses_same_fold_plan_for_all_candidates(monkeypatch):
    from src import multiband_pipeline

    plans = []
    monkeypatch.setattr(
        multiband_pipeline,
        "_evaluate_local_rbf_smoothing_candidate",
        lambda controls, fold_plan, smoothing, params: (
            plans.append(fold_plan) or {
                "smoothing": smoothing, "available": True,
                "candidate_rmse": 0.5, "candidate_p95": 0.8,
                "n_folds": 4, "n_folds_attempted": 4,
                "n_validation_controls": 16, "validation_coverage": 1.0,
                "failed_group_ids": [], "failure_reason": None,
            }
        ),
    )
    multiband_pipeline._local_holdout_cv(
        _controls(), {"local_smoothing_candidates": [0.1, 0.2]}
    )
    assert len(plans) == 2
    assert plans[0] is plans[1]


def _candidate(smoothing, rmse, p95, available=True):
    return {
        "smoothing": smoothing, "available": available,
        "candidate_rmse": rmse, "candidate_p95": p95,
        "failure_reason": None if available else "failed",
    }


def test_local_holdout_cv_selects_best_candidate_that_passes_both_gates():
    from src import multiband_pipeline

    result = multiband_pipeline._select_local_rbf_cv_candidate(
        [_candidate(0.1, 0.8, 1.5), _candidate(0.5, 0.7, 1.6)],
        1.0, 2.0,
        {"local_cv_min_rmse_improvement": 0.1,
         "local_cv_min_p95_improvement": 0.3},
    )
    assert result["selected_smoothing"] == 0.5
    assert result["has_passing_candidate"] is True
    assert result["selection_reason"] == "best_passing_candidate"


def test_local_holdout_cv_does_not_choose_rmse_only_candidate_when_p95_fails():
    from src import multiband_pipeline

    result = multiband_pipeline._select_local_rbf_cv_candidate(
        [_candidate(0.1, 0.7, 1.9), _candidate(0.5, 0.8, 1.5)],
        1.0, 2.0,
        {"local_cv_min_rmse_improvement": 0.1,
         "local_cv_min_p95_improvement": 0.3},
    )
    assert result["selected_smoothing"] == 0.5
    assert result["has_passing_candidate"] is True


def test_local_holdout_cv_returns_diagnostic_best_when_none_pass(monkeypatch):
    from src import multiband_pipeline

    result = multiband_pipeline._select_local_rbf_cv_candidate(
        [_candidate(0.1, 0.95, 1.9), _candidate(0.5, 0.9, 1.8)],
        1.0, 2.0,
        {"local_cv_min_rmse_improvement": 0.2,
         "local_cv_min_p95_improvement": 0.3},
    )
    assert result["selected_smoothing"] == 0.5
    assert result["has_passing_candidate"] is False
    assert result["selection_reason"] == "best_available_but_gate_failed"


def test_local_holdout_cv_continues_when_one_candidate_is_unavailable():
    from src import multiband_pipeline

    result = multiband_pipeline._select_local_rbf_cv_candidate(
        [_candidate(0.1, 0.5, 0.8, available=False),
         _candidate(0.5, 0.8, 1.5)],
        1.0, 2.0,
        {"local_cv_min_rmse_improvement": 0.1,
         "local_cv_min_p95_improvement": 0.3},
    )
    assert result["selected_smoothing"] == 0.5


def test_local_holdout_cv_tie_break_is_deterministic():
    from src import multiband_pipeline

    result = multiband_pipeline._select_local_rbf_cv_candidate(
        [_candidate(0.5, 0.8, 1.5), _candidate(0.1, 0.8, 1.5)],
        1.0, 2.0,
        {"local_cv_min_rmse_improvement": 0.1,
         "local_cv_min_p95_improvement": 0.3},
    )
    assert result["selected_smoothing"] == 0.1


def test_fit_local_rbf_field_uses_explicit_selected_smoothing(monkeypatch):
    from src import coregistration, multiband_pipeline

    seen = []

    class ConstantRBF:
        def __call__(self, points):
            return np.zeros(len(points))

    def fake_fit(*args, **kwargs):
        seen.append(kwargs["smoothing"])
        return ConstantRBF(), ConstantRBF(), (0.0, 0.0), (30.0, 30.0)

    monkeypatch.setattr(coregistration, "fit_local_rbf", fake_fit)
    monkeypatch.setattr(
        coregistration, "compute_hull_fade_mask",
        lambda points, h, w, buffer: np.ones((h, w)),
    )
    _, _, stats = multiband_pipeline._fit_local_rbf_field(
        _controls(), (8, 8), {"local_smoothing_candidates": [0.1, 0.5]},
        smoothing=0.5,
    )
    assert seen == [0.5]
    assert stats["smoothing"] == 0.5


def test_fit_local_rbf_field_rejects_implicit_first_candidate_when_multiple_values():
    from src import multiband_pipeline

    with pytest.raises(ValueError, match="selected smoothing required"):
        multiband_pipeline._fit_local_rbf_field(
            _controls(), (8, 8),
            {"local_smoothing_candidates": [0.1, 0.5]},
        )


def test_accept_local_rbf_candidate_rejects_when_no_smoothing_passes():
    from src import multiband_pipeline

    result = multiband_pipeline._accept_local_rbf_candidate(
        _controls(), {"local_min_controls": 12, "local_min_spatial_groups": 3},
        cv_result={
            "available": True, "has_passing_candidate": False,
            "baseline_rmse": 1.0, "candidate_rmse": 0.9,
            "baseline_p95": 2.0, "candidate_p95": 1.9,
        },
    )
    assert result["accepted"] is False
    assert "no smoothing candidate" in result["reason"]
