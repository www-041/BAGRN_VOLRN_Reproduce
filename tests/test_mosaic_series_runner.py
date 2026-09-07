"""Tests for mosaic series runner configuration and execution.

Task 7 of dz01v mosaic readiness plan:
- Robust boolean parsing for config
- enable_spectral_metrics field
- _requested_normalization_methods helper
"""

import pytest
from src.experiment_config import ExperimentConfig, _dict_to_config


def test_robust_boolean_parser():
    """Test robust boolean parsing for various input formats."""
    from src.experiment_config import _parse_bool
    
    # Test True values
    assert _parse_bool(True) is True
    assert _parse_bool("true") is True
    assert _parse_bool("TRUE") is True
    assert _parse_bool("yes") is True
    assert _parse_bool("1") is True
    assert _parse_bool(1) is True
    
    # Test False values
    assert _parse_bool(False) is False
    assert _parse_bool("false") is False
    assert _parse_bool("FALSE") is False
    assert _parse_bool("no") is False
    assert _parse_bool("0") is False
    assert _parse_bool(0) is False
    
    # Test invalid values
    with pytest.raises(ValueError):
        _parse_bool("maybe")
    with pytest.raises(ValueError):
        _parse_bool("2")


def test_enable_spectral_metrics_config():
    """Test that enable_spectral_metrics is properly loaded from config."""
    raw = {
        "experiment_name": "test",
        "enable_spectral_metrics": False,
    }
    cfg = _dict_to_config(raw)
    assert cfg.enable_spectral_metrics is False
    
    raw = {
        "experiment_name": "test",
        "enable_spectral_metrics": "false",
    }
    cfg = _dict_to_config(raw)
    assert cfg.enable_spectral_metrics is False
    
    # Default should be True
    raw = {"experiment_name": "test"}
    cfg = _dict_to_config(raw)
    assert cfg.enable_spectral_metrics is True


def test_requested_normalization_methods():
    """Test _requested_normalization_methods helper."""
    from src.multiband_pipeline import MultibandPipeline
    
    # Create a minimal pipeline instance
    cfg = ExperimentConfig()
    cfg.ablation_methods = []
    pipeline = MultibandPipeline(cfg)
    
    # With empty ablation_methods, should return base methods
    methods = pipeline._requested_normalization_methods()
    assert "original" in methods
    assert "bagrn" in methods
    assert "bagrn_volrn" in methods
    
    # With ablation methods, should include them
    cfg.ablation_methods = ["volrn_only", "histogram_matching"]
    pipeline = MultibandPipeline(cfg)
    methods = pipeline._requested_normalization_methods()
    assert "original" in methods
    assert "bagrn" in methods
    assert "bagrn_volrn" in methods
    assert "volrn_only" in methods
    assert "histogram_matching" in methods
