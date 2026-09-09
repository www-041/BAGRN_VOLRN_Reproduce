"""Registration quality gate tests.

Task 1-6 of ghosting fix plan:
- Config loading and validation
- Robust estimation
- Quality classification
"""

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
