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


def test_final_quality_gate_rejects_bad_rmse_or_p95():
    from src.coregistration import classify_registration_quality
    params = {"pass_min_mean_confidence": 0.50, "pass_max_median": 0.35,
              "pass_max_rmse": 0.60, "pass_max_p95": 1.00, "final_min_blocks": 5,
              "warn_min_mean_confidence": 0.45, "warn_max_median": 0.50,
              "warn_max_rmse": 0.75, "warn_max_p95": 1.25}
    assert classify_registration_quality(
        {"status": "pass", "confidence": 0.8, "residual_median": 0.2,
         "residual_rmse": 0.8, "residual_p95": 0.9, "n_inliers": 8}, params) == "fail"


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
