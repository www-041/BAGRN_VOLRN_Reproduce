"""Registration quality gate tests.

Task 1-6 of ghosting fix plan:
- Config loading and validation
- Robust estimation
- Quality classification
"""

import pytest

from src.experiment_config import _dict_to_config, validate_config


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
