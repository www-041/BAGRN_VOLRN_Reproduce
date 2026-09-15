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
