import numpy as np


def _grouped_points():
    return np.asarray([
        [0.0, 0.0], [0.5, 0.0], [0.0, 0.5], [0.5, 0.5],
        [20.0, 0.0], [20.5, 0.0], [20.0, 0.5], [20.5, 0.5],
        [0.0, 20.0], [0.5, 20.0], [0.0, 20.5], [0.5, 20.5],
        [20.0, 20.0], [20.5, 20.0], [20.0, 20.5], [20.5, 20.5],
    ])


def test_buffered_fold_plan_excludes_training_points_inside_buffer():
    from src import multiband_pipeline

    points = _grouped_points()
    result = multiband_pipeline._build_local_cv_fold_plan(
        points, {"local_min_controls": 3, "local_cv_buffer_pixels": 5}
    )

    assert result["available"] is True
    for fold in result["folds"]:
        train = points[fold["train_idx"]]
        test = points[fold["test_idx"]]
        distances = np.linalg.norm(train[:, None, :] - test[None, :, :], axis=2)
        assert np.all(distances.min(axis=1) >= 5.0)
        assert fold["n_train_after_buffer"] == len(fold["train_idx"])


def test_zero_buffer_preserves_group_only_split_behavior():
    from src import multiband_pipeline

    points = _grouped_points()
    baseline = multiband_pipeline._build_local_cv_fold_plan(
        points, {"local_min_controls": 3}
    )
    zero_buffer = multiband_pipeline._build_local_cv_fold_plan(
        points, {"local_min_controls": 3, "local_cv_buffer_pixels": 0}
    )

    assert np.array_equal(baseline["groups"], zero_buffer["groups"])
    assert [
        (f["group_id"], f["train_idx"].tolist(), f["test_idx"].tolist())
        for f in baseline["folds"]
    ] == [
        (f["group_id"], f["train_idx"].tolist(), f["test_idx"].tolist())
        for f in zero_buffer["folds"]
    ]


def test_buffered_fold_plan_drops_fold_when_train_controls_below_local_minimum():
    from src import multiband_pipeline

    points = np.asarray([
        [0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [0.1, 0.1],
        [10.0, 10.0], [10.1, 10.0], [10.0, 10.1], [10.1, 10.1],
    ])
    result = multiband_pipeline._build_local_cv_fold_plan(
        points, {"local_min_controls": 5, "local_cv_buffer_pixels": 100}
    )

    assert result["available"] is False
    assert result["dropped_folds"]
    assert all("train controls" in item["reason"] for item in result["dropped_folds"])


def test_buffered_fold_plan_reports_actual_min_train_test_distance():
    from src import multiband_pipeline

    points = np.asarray([
        [0.0, 0.0], [0.0, 0.1], [0.1, 0.0],
        [10.0, 0.0], [10.0, 0.1], [10.1, 0.0],
        [0.0, 10.0], [0.0, 10.1], [0.1, 10.0],
    ])
    result = multiband_pipeline._build_local_cv_fold_plan(
        points, {"local_min_controls": 3, "local_cv_buffer_pixels": 2}
    )

    assert result["available"] is True
    assert all(
        fold["min_train_test_distance_px"] >= 9.9
        for fold in result["folds"]
    )


def test_buffered_fold_plan_is_deterministic():
    from src import multiband_pipeline

    points = _grouped_points()
    params = {"local_min_controls": 3, "local_cv_buffer_pixels": 5}
    first = multiband_pipeline._build_local_cv_fold_plan(points, params)
    second = multiband_pipeline._build_local_cv_fold_plan(points, params)

    assert np.array_equal(first["groups"], second["groups"])
    assert first["dropped_folds"] == second["dropped_folds"]
    assert [
        (f["group_id"], f["train_idx"].tolist(), f["test_idx"].tolist(),
         f["n_train_before_buffer"], f["n_train_after_buffer"],
         f["min_train_test_distance_px"])
        for f in first["folds"]
    ] == [
        (f["group_id"], f["train_idx"].tolist(), f["test_idx"].tolist(),
         f["n_train_before_buffer"], f["n_train_after_buffer"],
         f["min_train_test_distance_px"])
        for f in second["folds"]
    ]


def test_every_retained_train_point_is_at_least_buffer_pixels_from_test_points():
    from src import multiband_pipeline

    points = _grouped_points()
    buffer_pixels = 7.5
    result = multiband_pipeline._build_local_cv_fold_plan(
        points, {"local_min_controls": 3, "local_cv_buffer_pixels": buffer_pixels}
    )

    for fold in result["folds"]:
        train = points[fold["train_idx"]]
        test = points[fold["test_idx"]]
        distances = np.linalg.norm(train[:, None, :] - test[None, :, :], axis=2)
        assert np.all(distances.min(axis=1) >= buffer_pixels)


def test_all_smoothing_candidates_receive_identical_buffered_fold_plan(monkeypatch):
    from src import multiband_pipeline

    points = _grouped_points()
    controls = {
        "points_xy": points,
        "residual_dx": np.zeros(len(points)),
        "residual_dy": np.zeros(len(points)),
        "confidence": np.ones(len(points)),
        "n_valid": len(points),
    }
    plans = []

    def fake_evaluate(controls, fold_plan, smoothing, params):
        plans.append(fold_plan)
        return {
            "smoothing": smoothing, "available": True,
            "candidate_rmse": 0.1, "candidate_p95": 0.2,
            "n_folds": fold_plan["n_folds"],
            "n_folds_attempted": fold_plan["n_folds_attempted"],
            "n_validation_controls": fold_plan["n_validation_controls"],
            "validation_coverage": fold_plan["validation_coverage"],
            "failed_group_ids": [], "failure_reason": None,
        }

    monkeypatch.setattr(
        multiband_pipeline, "_evaluate_local_rbf_smoothing_candidate", fake_evaluate
    )
    result = multiband_pipeline._local_holdout_cv(
        controls, {"local_min_controls": 3, "local_cv_buffer_pixels": 5,
                   "local_smoothing_candidates": [0.1, 0.5]}
    )

    assert len(plans) == 2
    assert plans[0] is plans[1]
    assert result["cv_strategy"] == "buffered_spatial_group"
    assert result["buffer_pixels"] == 5.0


def test_local_holdout_cv_reports_buffered_strategy_metadata():
    from src import multiband_pipeline

    points = _grouped_points()
    controls = {
        "points_xy": points,
        "residual_dx": np.zeros(len(points)),
        "residual_dy": np.zeros(len(points)),
        "confidence": np.ones(len(points)),
        "n_valid": len(points),
    }
    result = multiband_pipeline._local_holdout_cv(
        controls, {"local_min_controls": 3, "local_cv_buffer_pixels": 5,
                   "local_smoothing_candidates": []}
    )

    assert result["cv_strategy"] == "buffered_spatial_group"
    assert result["buffer_pixels"] == 5.0
    assert result["fold_diagnostics"]
    assert result["dropped_folds"] == []


def test_local_holdout_cv_unavailable_when_buffer_starves_required_folds():
    from src import multiband_pipeline

    points = _grouped_points()
    controls = {
        "points_xy": points,
        "residual_dx": np.zeros(len(points)),
        "residual_dy": np.zeros(len(points)),
        "confidence": np.ones(len(points)),
        "n_valid": len(points),
    }
    result = multiband_pipeline._local_holdout_cv(
        controls, {"local_min_controls": 3, "local_cv_buffer_pixels": 100,
                   "local_smoothing_candidates": [0.1]}
    )

    assert result["available"] is False
    assert result["dropped_folds"]
    assert "after buffer" in result["failure_reason"]
