import numpy as np

from scripts import two_image_pipeline as tip


def test_summarize_residuals_uses_vector_magnitude():
    dx = np.array([3.0, 0.0])
    dy = np.array([4.0, 0.0])
    result = tip._summarize_residuals(dx, dy)

    assert result["count"] == 2
    assert result["mean_magnitude_pixels"] == 2.5
    assert result["median_magnitude_pixels"] == 2.5
    assert np.isclose(result["rmse_pixels"], np.sqrt((25.0 + 0.0) / 2.0))
    assert result["max_magnitude_pixels"] == 5.0
    assert "mae_pixels" not in result
    assert "mean_error_pixels" not in result


def test_summarize_residuals_empty_returns_none_metrics():
    result = tip._summarize_residuals(np.array([]), np.array([]))

    assert result["count"] == 0
    assert result["rmse_pixels"] is None
    assert result["p95_magnitude_pixels"] is None


def test_spatial_coverage_is_relative_to_overlap_not_full_image():
    matches = []
    for y in [125, 375, 625, 875]:
        for x in [562, 687, 812, 937]:
            matches.append({"ref_x": x, "ref_y": y})

    result = tip._spatial_coverage(
        matches,
        reference_shape=(1000, 1000),
        overlap_window=(0, 1000, 500, 1000),
        grid_rows=4,
        grid_cols=4,
    )

    assert result["coverage_domain"] == "overlap"
    assert result["grid_cells_covered"] >= 12
    assert result["grid_coverage_ratio"] >= 0.75


def test_affine_diagnostic_does_not_control_applied_model():
    summary = {
        "translation": {"p95": 2.0},
        "affine": {"p95": 0.4},
        "rbf": {"p95": 1.0},
    }

    result = tip.select_translation_or_rbf(summary, 0.10)

    assert result["selected_model"] == "rbf"


def test_rbf_requires_minimum_p95_improvement():
    summary = {
        "translation": {"p95": 1.0},
        "affine": {"p95": 0.2},
        "rbf": {"p95": 0.95},
    }

    result = tip.select_translation_or_rbf(summary, 0.10)

    assert result["selected_model"] == "translation"


def test_global_only_and_final_warps_use_the_same_original_based_path():
    original = np.arange(24, dtype=np.float64).reshape(4, 6)
    local_dx = np.zeros_like(original)
    local_dy = np.zeros_like(original)
    calls = []

    def fake_warp(arr, global_dx, global_dy, local_dx_field, local_dy_field, nodata):
        calls.append({
            "arr": arr,
            "global_dx": global_dx,
            "global_dy": global_dy,
            "local_dx": local_dx_field.copy(),
            "local_dy": local_dy_field.copy(),
            "nodata": nodata,
        })
        return arr.copy()

    global_warp, final_warp = tip.apply_registration_warps(
        original,
        global_dx=1.5,
        global_dy=-0.5,
        local_dx_field=local_dx,
        local_dy_field=local_dy,
        nodata=0.0,
        use_local=False,
        warp_fn=fake_warp,
    )

    assert len(calls) == 1
    assert calls[0]["arr"] is original
    assert np.array_equal(calls[0]["local_dx"], np.zeros_like(original))
    assert np.array_equal(calls[0]["local_dy"], np.zeros_like(original))
    assert final_warp is global_warp


def test_post_warp_registration_reports_residual_to_zero(monkeypatch):
    matches = [
        {
            "shift_dx": 1.0, "shift_dy": 0.0, "confidence": 0.8,
            "ref_x": 100, "ref_y": 100, "tgt_x": 100, "tgt_y": 100,
        },
        {
            "shift_dx": 3.0, "shift_dy": 4.0, "confidence": 0.9,
            "ref_x": 200, "ref_y": 200, "tgt_x": 200, "tgt_y": 200,
        },
    ]

    def fake_collect(*args, **kwargs):
        return matches, {"total": 4, "accepted": len(matches)}

    monkeypatch.setattr(
        "src.coregistration.collect_block_matches", fake_collect)
    result = tip.evaluate_post_warp_registration(
        np.ones((512, 512)),
        None,
        np.ones((512, 512)),
        None,
        0.0,
        0.0,
        reference_overlap_window=(0, 512, 0, 512),
    )

    assert result["available"] is True
    assert result["metric_scope"] == (
        "same-data post-warp diagnostic; not independent holdout"
    )
    assert result["candidate_blocks"] == 4
    assert result["accepted_matches"] == 2
    assert result["residual_to_zero"]["mean_magnitude_pixels"] == 3.0
    assert result["residual_to_zero"]["max_magnitude_pixels"] == 5.0


def test_stage_comparison_positive_means_final_is_better():
    global_post = {
        "available": True,
        "residual_to_zero": {
            "median_magnitude_pixels": 1.0,
            "rmse_pixels": 2.0,
            "p95_magnitude_pixels": 3.0,
        },
    }
    final_post = {
        "available": True,
        "residual_to_zero": {
            "median_magnitude_pixels": 0.5,
            "rmse_pixels": 1.0,
            "p95_magnitude_pixels": 2.0,
        },
    }

    result = tip.compare_post_warp_stages(global_post, final_post)

    assert result["rmse_improvement_pixels"] == 1.0
    assert result["p95_improvement_pixels"] == 1.0
    assert result["median_improvement_pixels"] == 0.5
    assert result["interpretation"] == "positive means final is better"


def test_registration_schema_separates_model_fit_and_post_warp_metrics():
    initial = {
        "offset": {"dx_pixels": 2.0},
        "phase_confidence": 0.9,
        "matching": {"accepted_matches": 2},
        "residual_relative_to_global_model": {"rmse_pixels": 0.2},
        "inlier_residual_relative_to_global_model": {"rmse_pixels": 0.1},
        "spatial_coverage": {},
        "inlier_spatial_coverage": {},
        "phase_block_stats": {},
    }
    local = {"model_used": "translation"}
    global_post = {"available": True}
    final_post = {"available": True}
    comparison = {"rmse_improvement_pixels": 0.0}

    result = tip.build_registration_schema(
        initial, local, global_post, final_post, comparison)

    assert result["schema_version"] == 2
    assert result["metric_scope_note"] == (
        "post_warp metrics are same-data diagnostics, not independent HOLDOUT"
    )
    assert result["initial_model_fit"] is initial
    assert result["local"] is local
    assert result["post_warp"]["global_only"] is global_post
    assert result["post_warp"]["final"] is final_post
    assert "residual" not in result["initial_model_fit"]
    assert "inlier_residual" not in result["initial_model_fit"]


def test_post_warp_match_rows_use_residual_to_zero_fields():
    matches = [{
        "ref_x": 10, "ref_y": 20, "tgt_x": 15, "tgt_y": 18,
        "shift_dx": 1.5, "shift_dy": -2.0, "confidence": 0.88,
    }]

    rows = tip.build_post_warp_match_rows(matches)

    assert rows == [{
        "match_index": 0,
        "ref_x": 10.0,
        "ref_y": 20.0,
        "tgt_x": 15.0,
        "tgt_y": 18.0,
        "residual_dx_pixels": 1.5,
        "residual_dy_pixels": -2.0,
        "residual_magnitude_pixels": 2.5,
        "confidence": 0.88,
    }]


def test_post_warp_without_matches_is_unavailable(monkeypatch):
    monkeypatch.setattr(
        "src.coregistration.collect_block_matches",
        lambda *args, **kwargs: ([], {"total": 3, "accepted": 0}),
    )

    result = tip.evaluate_post_warp_registration(
        np.ones((512, 512)), None,
        np.ones((512, 512)), None,
        0.0, 0.0,
    )

    assert result["available"] is False
    assert result["failure_reason"] == "no accepted post-warp block matches"
    assert result["residual_to_zero"]["rmse_pixels"] is None


def test_post_warp_single_match_preserves_count_one(monkeypatch):
    match = {
        "shift_dx": 2.0, "shift_dy": 0.0, "confidence": 0.8,
        "ref_x": 100, "ref_y": 100, "tgt_x": 100, "tgt_y": 100,
    }
    monkeypatch.setattr(
        "src.coregistration.collect_block_matches",
        lambda *args, **kwargs: ([match], {"total": 1, "accepted": 1}),
    )

    result = tip.evaluate_post_warp_registration(
        np.ones((512, 512)), None,
        np.ones((512, 512)), None,
        0.0, 0.0,
    )

    assert result["available"] is True
    assert result["residual_to_zero"]["count"] == 1
    assert result["residual_to_zero"]["mean_magnitude_pixels"] == 2.0


def test_post_warp_systematic_residual_is_not_centered_to_zero(monkeypatch):
    matches = [
        {
            "shift_dx": 1.0, "shift_dy": 0.0, "confidence": 0.8,
            "ref_x": 100, "ref_y": 100, "tgt_x": 100, "tgt_y": 100,
        },
        {
            "shift_dx": 1.0, "shift_dy": 0.0, "confidence": 0.9,
            "ref_x": 200, "ref_y": 200, "tgt_x": 200, "tgt_y": 200,
        },
    ]
    monkeypatch.setattr(
        "src.coregistration.collect_block_matches",
        lambda *args, **kwargs: (matches, {"total": 2, "accepted": 2}),
    )

    result = tip.evaluate_post_warp_registration(
        np.ones((512, 512)), None,
        np.ones((512, 512)), None,
        0.0, 0.0,
    )

    assert result["residual_to_zero"]["rmse_pixels"] == 1.0
    assert result["residual_to_zero"]["mean_residual_dx_pixels"] == 1.0


def test_stage_comparison_reports_negative_improvement_when_rbf_worsens():
    global_post = {
        "available": True,
        "residual_to_zero": {
            "median_magnitude_pixels": 1.0,
            "rmse_pixels": 1.0,
            "p95_magnitude_pixels": 1.0,
        },
    }
    final_post = {
        "available": True,
        "residual_to_zero": {
            "median_magnitude_pixels": 1.5,
            "rmse_pixels": 2.0,
            "p95_magnitude_pixels": 3.0,
        },
    }

    result = tip.compare_post_warp_stages(global_post, final_post)

    assert result["median_improvement_pixels"] == -0.5
    assert result["rmse_improvement_pixels"] == -1.0
    assert result["p95_improvement_pixels"] == -2.0


def test_stage_comparison_does_not_fake_improvement_without_local_model():
    global_post = {
        "available": True,
        "residual_to_zero": {
            "median_magnitude_pixels": 1.0,
            "rmse_pixels": 2.0,
            "p95_magnitude_pixels": 3.0,
        },
    }

    result = tip.compare_post_warp_stages(global_post, global_post)

    assert result["median_improvement_pixels"] == 0.0
    assert result["rmse_improvement_pixels"] == 0.0
    assert result["p95_improvement_pixels"] == 0.0
