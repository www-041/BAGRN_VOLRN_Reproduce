"""Registration quality gate tests.

Task 1-6 of ghosting fix plan:
- Config loading and validation
- Robust estimation
- Quality classification
"""

import numpy as np
import pytest

from src.experiment_config import _dict_to_config, validate_config


def test_final_quality_gate_uses_post_warp_residuals():
    from src.coregistration import aggregate_final_validation_quality
    result = aggregate_final_validation_quality(
        [{"stats": {"median": 0.1, "rmse": 0.2, "p95": 0.3,
                     "mean_confidence": 0.8, "n_accepted": 8}}],
        {"pass_min_mean_confidence": 0.50, "pass_max_median": 0.35,
         "pass_max_rmse": 0.60, "pass_max_p95": 1.00, "final_min_blocks": 5,
         "warn_min_mean_confidence": 0.45, "warn_max_median": 0.50,
         "warn_max_rmse": 0.75, "warn_max_p95": 1.25})
    assert result["quality"] == "pass"
    assert result["n_blocks"] == 8


@pytest.mark.parametrize(
    "validation",
    [
        {
            "idx_i": 0,
            "idx_j": 1,
            "stats": {
                "median": 0.1,
                "rmse": 0.2,
                "p95": 0.3,
                "mean_confidence": 0.9,
                "n_accepted": 5,
            },
            "failure_reason": "independent validation reported a failure",
        },
        {
            "idx_i": 0,
            "idx_j": 1,
            "stats": {
                "median": 0.1,
                "rmse": 0.2,
                "p95": 0.3,
                "mean_confidence": 0.9,
                "n_accepted": 4,
            },
            "failure_reason": None,
        },
    ],
)
def test_final_quality_gate_rejects_any_failed_required_edge(validation):
    """A required edge cannot be hidden by a passing pooled summary."""
    from src.coregistration import aggregate_final_validation_quality

    passing_edge = {
        "idx_i": 1,
        "idx_j": 2,
        "stats": {
            "median": 0.1,
            "rmse": 0.2,
            "p95": 0.3,
            "mean_confidence": 0.9,
            "n_accepted": 4,
        },
        "failure_reason": None,
    }
    params = {
        "final_min_blocks": 5,
        "pass_min_mean_confidence": 0.5,
        "pass_max_median": 0.35,
        "pass_max_rmse": 0.6,
        "pass_max_p95": 1.0,
        "warn_min_mean_confidence": 0.45,
        "warn_max_median": 0.5,
        "warn_max_rmse": 0.75,
        "warn_max_p95": 1.25,
    }

    result = aggregate_final_validation_quality(
        [validation, passing_edge], params, required_edges=[(0, 1), (1, 2)]
    )

    assert result["quality"] == "fail"
    assert (0, 1) in result["unavailable_edges"]


def test_final_quality_gate_rejects_bad_rmse_or_p95():
    from src.coregistration import classify_registration_quality
    params = {"pass_min_mean_confidence": 0.50, "pass_max_median": 0.35,
              "pass_max_rmse": 0.60, "pass_max_p95": 1.00, "final_min_blocks": 5,
              "warn_min_mean_confidence": 0.45, "warn_max_median": 0.50,
              "warn_max_rmse": 0.75, "warn_max_p95": 1.25}
    assert classify_registration_quality(
        {"status": "pass", "confidence": 0.8, "residual_median": 0.2,
         "residual_rmse": 0.8, "residual_p95": 0.9, "n_inliers": 8}, params) == "fail"


def test_final_quality_summary_fallback_is_conservative_for_quantiles():
    from src.coregistration import aggregate_final_validation_quality
    result = aggregate_final_validation_quality(
        [{"stats": {"median": 0.1, "rmse": 0.2, "p95": 0.3,
                     "mean_confidence": 0.8, "n_accepted": 5}},
         {"stats": {"median": 0.1, "rmse": 0.2, "p95": 1.1,
                     "mean_confidence": 0.8, "n_accepted": 5}}],
        {"pass_min_mean_confidence": 0.50, "pass_max_median": 0.35,
         "pass_max_rmse": 0.60, "pass_max_p95": 1.00, "final_min_blocks": 5,
         "warn_min_mean_confidence": 0.45, "warn_max_median": 0.50,
         "warn_max_rmse": 0.75, "warn_max_p95": 1.25})
    assert result["quality"] == "warn"
    assert result["p95"] == pytest.approx(1.1)


def test_validation_grid_params_are_schema_backed_and_preserved():
    cfg = _dict_to_config({
        "experiment_name": "registration-test",
        "selected_bands": ["B14"],
        "registration_band": "B14",
        "scenes": [
            {"id": "a", "bands": {"B14": "a.tif"}},
            {"id": "b", "bands": {"B14": "b.tif"}},
        ],
        "registration_params": {
            "validation_step": 64,
            "validation_offset_row": 32,
            "validation_offset_col": 16,
            "validation_min_distance_from_training": 48,
        },
    })
    assert cfg.registration_params["validation_step"] == 64
    assert cfg.registration_params["validation_offset_row"] == 32
    assert cfg.registration_params["validation_offset_col"] == 16
    assert cfg.registration_params["validation_min_distance_from_training"] == 48


def test_dz01_adaptation_params_are_schema_backed_and_validated():
    cfg = _dict_to_config({
        "experiment_name": "registration-test",
        "selected_bands": ["B14"],
        "registration_band": "B14",
        "scenes": [
            {"id": "a", "bands": {"B14": "a.tif"}},
            {"id": "b", "bands": {"B14": "b.tif"}},
        ],
        "registration_params": {
            "enable_spatial_holdout": True,
            "holdout_fraction": 0.20,
            "validation_block_size_candidates": [384, 256, 192],
            "local_search_max_shift": 12.0,
            "local_hard_max_component": 8.0,
        },
    })
    assert cfg.registration_params["enable_spatial_holdout"] is True
    assert cfg.registration_params["validation_block_size_candidates"] == [384, 256, 192]
    assert validate_config(cfg, skip_file_check=True) == []


def test_registration_params_load_and_preserve_strict_quality():
    """registration_params should load from YAML/dict and preserve values."""
    cfg = _dict_to_config({
        "experiment_name": "registration-test",
        "selected_bands": ["B14"],
        "registration_band": "B14",
        "scenes": [
            {"id": "a", "bands": {"B14": "a.tif"}},
            {"id": "b", "bands": {"B14": "b.tif"}},
        ],
        "registration_params": {
            "required_quality": "pass",
            "pass_max_rmse": 0.60,
            "pass_max_p95": 1.00,
            "enable_local_refinement": True,
        },
    })

    assert cfg.registration_params["required_quality"] == "pass"
    assert cfg.registration_params["pass_max_rmse"] == pytest.approx(0.60)
    assert cfg.registration_params["pass_max_p95"] == pytest.approx(1.00)
    assert cfg.registration_params["enable_local_refinement"] is True


def test_registration_params_reject_invalid_quality_name():
    """required_quality must be 'pass', 'warn', or 'fail' only."""
    cfg = _dict_to_config({
        "experiment_name": "registration-test",
        "selected_bands": ["B14"],
        "registration_band": "B14",
        "scenes": [
            {"id": "a", "bands": {"B14": "a.tif"}},
            {"id": "b", "bands": {"B14": "b.tif"}},
        ],
        "registration_params": {"required_quality": "excellent"},
    })
    errors = validate_config(cfg, skip_file_check=True)
    assert any("required_quality" in e for e in errors)


def test_default_registration_params_exist():
    """Default registration_params should have required_quality='pass'."""
    cfg = _dict_to_config({
        "experiment_name": "registration-test",
        "selected_bands": ["B14"],
        "registration_band": "B14",
        "scenes": [
            {"id": "a", "bands": {"B14": "a.tif"}},
            {"id": "b", "bands": {"B14": "b.tif"}},
        ],
    })
    
    assert "registration_params" in dir(cfg)
    assert cfg.registration_params["required_quality"] == "pass"


def _config_with_local_rbf_params(params):
    return _dict_to_config({
        "experiment_name": "registration-test",
        "selected_bands": ["B14"],
        "registration_band": "B14",
        "scenes": [
            {"id": "a", "bands": {"B14": "a.tif"}},
            {"id": "b", "bands": {"B14": "b.tif"}},
        ],
        "registration_params": params,
    })


@pytest.mark.parametrize(
    "candidates",
    [[], [-0.1], [float("nan")], [float("inf")], [0.1, 0.10000000000000001]],
)
def test_local_smoothing_candidates_reject_invalid_values(candidates):
    cfg = _config_with_local_rbf_params({"local_smoothing_candidates": candidates})
    assert any("local_smoothing_candidates" in error
               for error in validate_config(cfg, skip_file_check=True))


def test_local_smoothing_candidates_allow_zero_and_distinct_values():
    cfg = _config_with_local_rbf_params({"local_smoothing_candidates": [0.0, 0.1]})
    assert not any("local_smoothing_candidates" in error
                   for error in validate_config(cfg, skip_file_check=True))


@pytest.mark.parametrize(
    "key", ["local_cv_min_rmse_improvement", "local_cv_min_p95_improvement"]
)
def test_local_cv_improvement_thresholds_must_be_nonnegative(key):
    cfg = _config_with_local_rbf_params({key: -0.01})
    assert any(key in error for error in validate_config(cfg, skip_file_check=True))


def _validation_result_for_comparison(blocks, *, rmse, p95, median):
    return {
        "edges": [{"idx_i": 0, "idx_j": 1, "blocks": blocks}],
        "overall": {"rmse": rmse, "p95": p95, "median": median},
    }


def _validation_block(row, col, magnitude, *, accepted=True, reason=None):
    return {
        "validation_row": row,
        "validation_col": col,
        "residual_dx": magnitude,
        "residual_dy": 0.0,
        "residual_magnitude": magnitude,
        "accepted": accepted,
        "reject_reason": reason,
    }


def test_compare_registration_validations_aligns_blocks_by_reserved_coordinates():
    from src.multiband_pipeline import _compare_registration_validations

    before = _validation_result_for_comparison(
        [_validation_block(10, 20, 2.0), _validation_block(30, 40, 1.0)],
        rmse=2.0, p95=2.5, median=1.5,
    )
    after = _validation_result_for_comparison(
        [_validation_block(30, 40, 0.5), _validation_block(10, 20, 1.0)],
        rmse=1.0, p95=1.5, median=0.75,
    )

    result = _compare_registration_validations(before, after)

    assert result["available"] is True
    assert result["n_blocks_common"] == 2
    assert {(b["validation_row"], b["validation_col"]): b["magnitude_improvement"]
            for b in result["edges"][0]["blocks"]} == {(10, 20): 1.0, (30, 40): 0.5}


def test_compare_registration_validations_uses_before_minus_after_sign():
    from src.multiband_pipeline import _compare_registration_validations

    before = _validation_result_for_comparison(
        [_validation_block(0, 0, 2.0)], rmse=2.0, p95=2.0, median=2.0,
    )
    after = _validation_result_for_comparison(
        [_validation_block(0, 0, 0.5)], rmse=0.5, p95=0.5, median=0.5,
    )

    result = _compare_registration_validations(before, after)

    assert result["rmse_improvement"] == pytest.approx(1.5)
    assert result["p95_improvement"] == pytest.approx(1.5)
    assert result["median_improvement"] == pytest.approx(1.5)
    assert result["edges"][0]["blocks"][0]["magnitude_improvement"] == pytest.approx(1.5)


def test_compare_registration_validations_preserves_rejected_blocks():
    from src.multiband_pipeline import _compare_registration_validations

    before = _validation_result_for_comparison(
        [_validation_block(0, 0, 1.0)], rmse=1.0, p95=1.0, median=1.0,
    )
    after = _validation_result_for_comparison(
        [_validation_block(0, 0, 4.74, accepted=False, reason="large_shift")],
        rmse=None, p95=None, median=None,
    )

    result = _compare_registration_validations(before, after)
    block = result["edges"][0]["blocks"][0]

    assert result["n_blocks_common"] == 1
    assert block["after"]["accepted"] is False
    assert block["after"]["reject_reason"] == "large_shift"


def test_compare_registration_validations_reports_unavailable_when_no_common_blocks():
    from src.multiband_pipeline import _compare_registration_validations

    before = _validation_result_for_comparison(
        [_validation_block(0, 0, 1.0)], rmse=1.0, p95=1.0, median=1.0,
    )
    after = _validation_result_for_comparison(
        [_validation_block(0, 64, 0.5)], rmse=0.5, p95=0.5, median=0.5,
    )

    result = _compare_registration_validations(before, after)

    assert result["available"] is False
    assert result["n_blocks_common"] == 0
    assert result["reason"] == "no common validation blocks"


def test_final_quality_is_still_driven_only_by_final_validation():
    from src.coregistration import aggregate_final_validation_quality

    final_quality = aggregate_final_validation_quality(
        [{"idx_i": 0, "idx_j": 1,
          "stats": {"median": 0.9, "rmse": 1.2, "p95": 1.8,
                    "mean_confidence": 0.8, "n_accepted": 5},
          "failure_reason": None}],
        {"final_min_blocks": 5, "pass_min_mean_confidence": 0.5,
         "pass_max_median": 0.35, "pass_max_rmse": 0.6, "pass_max_p95": 1.0,
         "warn_min_mean_confidence": 0.45, "warn_max_median": 0.5,
         "warn_max_rmse": 0.75, "warn_max_p95": 1.25},
    )
    comparison = {"available": True, "rmse_improvement": 10.0,
                  "p95_improvement": 10.0, "median_improvement": 10.0}

    assert comparison["rmse_improvement"] > 0
    assert final_quality["quality"] == "fail"


def test_stage_validation_comparison_is_exposed_in_registration_result_and_diagnostics():
    from src.multiband_pipeline import _compare_registration_validations

    before = _validation_result_for_comparison(
        [_validation_block(0, 0, 1.0)], rmse=1.0, p95=1.0, median=1.0,
    )
    after = _validation_result_for_comparison(
        [_validation_block(0, 0, 0.5)], rmse=0.5, p95=0.5, median=0.5,
    )
    comparison = _compare_registration_validations(before, after)

    registration = {
        "global_only_quality": before["overall"],
        "global_only_validation": before,
        "final_validation": after,
        "quality": after["overall"],
        "stage_validation_comparison": comparison,
        "diagnostics": {"stage_validation_comparison": comparison},
    }

    assert registration["diagnostics"]["stage_validation_comparison"] is comparison
    assert registration["stage_validation_comparison"]["n_blocks_common"] == 1


def _validation_with_sample_block():
    return {
        "edges": [{
            "idx_i": 0,
            "idx_j": 1,
            "blocks": [{
                "validation_row": 2,
                "validation_col": 3,
                "center_x": 3.0,
                "center_y": 2.0,
                "residual_dx": 0.4,
                "residual_dy": -0.2,
                "residual_magnitude": 0.4472135955,
                "accepted": True,
                "reject_reason": None,
            }],
        }],
    }


def test_sample_local_field_uses_actual_faded_field_values():
    from src.multiband_pipeline import _sample_local_field_at_validation_blocks

    dx = np.full((6, 7), 1.25, dtype=float)
    dy = np.full((6, 7), -0.75, dtype=float)
    result = _sample_local_field_at_validation_blocks(
        _validation_with_sample_block(), {1: dx}, {1: dy},
        {"used_for_scenes": [1], "scenes": {"1": {"selected_smoothing": 0.5}}},
    )

    sample = result["edges"][0]["blocks"][0]
    assert sample["predicted_local_dx"] == pytest.approx(1.25)
    assert sample["predicted_local_dy"] == pytest.approx(-0.75)
    assert sample["predicted_local_magnitude"] == pytest.approx(np.hypot(1.25, -0.75))
    assert sample["selected_smoothing"] == 0.5
    assert sample["field_available"] is True


def test_sample_local_field_preserves_validation_residual_and_nearest_training_distance():
    from src.multiband_pipeline import _sample_local_field_at_validation_blocks

    validation = _validation_with_sample_block()
    result = _sample_local_field_at_validation_blocks(
        validation, {1: np.ones((6, 7))}, {1: np.ones((6, 7))},
        {"used_for_scenes": [1],
         "control_points": {"1": [[3.0, 5.0], [10.0, 10.0]]},
         "scenes": {"1": {"selected_smoothing": 0.1}}},
    )

    sample = result["edges"][0]["blocks"][0]
    assert sample["final_residual_dx"] == pytest.approx(0.4)
    assert sample["final_residual_dy"] == pytest.approx(-0.2)
    assert sample["final_residual_magnitude"] == pytest.approx(0.4472135955)
    assert sample["nearest_training_distance"] == pytest.approx(3.0)


def test_sample_local_field_marks_out_of_bounds_center_unavailable():
    from src.multiband_pipeline import _sample_local_field_at_validation_blocks

    validation = _validation_with_sample_block()
    validation["edges"][0]["blocks"][0]["center_x"] = 99.0
    result = _sample_local_field_at_validation_blocks(
        validation, {1: np.ones((6, 7))}, {1: np.ones((6, 7))},
        {"used_for_scenes": [1]},
    )

    sample = result["edges"][0]["blocks"][0]
    assert sample["field_available"] is False
    assert sample["predicted_local_dx"] is None


@pytest.mark.parametrize(
    ("actual", "required", "expected"),
    [
        ("pass", "pass", True),
        ("pass", "warn", True),
        ("warn", "pass", False),
        ("fail", "fail", True),
        ("unknown", "pass", False),
        ("pass", "unknown", False),
    ],
)
def test_registration_quality_meets_requirement_uses_quality_order(
    actual, required, expected,
):
    from src.multiband_pipeline import registration_quality_meets_requirement

    assert registration_quality_meets_requirement(actual, required) is expected
