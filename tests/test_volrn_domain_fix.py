"""Regression tests for VOLRN domain mixing and quality check fixes.

Tests for:
A. Valid normalized zero is not treated as nodata
B. Quality check allows NaN only in reference nodata area
C. Quality check rejects NaN in reference valid area
D. Pipeline status doesn't require unrequested volrn_only
"""

import numpy as np
import pytest
from rasterio.transform import from_origin


def test_volrn_valid_normalized_zero_is_not_treated_as_nodata():
    """
    Construct nodata=0 synthetic data where valid DN=vmin becomes 0.0 
    in normalized domain. After VOLRN, these originally valid pixels 
    must still be finite.
    """
    from src.volrn import volrn_normalize
    
    # Create synthetic data where min value will normalize to 0
    # Use 32x32 images with overlap
    img_a = np.full((1, 32, 32), 100.0, dtype=np.float64)
    img_b = np.full((1, 32, 32), 110.0, dtype=np.float64)
    
    # Set nodata=0 for both (this is the critical test)
    nodata = 0.0
    
    transforms = [
        from_origin(0, 32, 1, 1),
        from_origin(16, 32, 1, 1),
    ]
    bounds = [
        (0, 0, 32, 32),
        (16, 0, 48, 32),
    ]
    
    # Run VOLRN with nodata=0
    result, coeffs = volrn_normalize(
        [img_a, img_b],
        transforms,
        bounds,
        [nodata, nodata],
        block_size_pixels=16,
        max_iter=3,
        tol=1e-3,
    )
    
    # All output pixels should be finite (no NaN propagation)
    assert np.all(np.isfinite(result[0])), "VOLRN output contains NaN for valid pixels"
    assert np.all(np.isfinite(result[1])), "VOLRN output contains NaN for valid pixels"


def test_quality_check_allows_nan_only_in_reference_nodata_area():
    """
    Quality check should allow NaN in reference nodata areas.
    """
    from src.multiband_pipeline import MultibandPipeline
    from src.experiment_config import ExperimentConfig
    
    config = ExperimentConfig()
    pipeline = MultibandPipeline(config)
    
    # Create reference with nodata
    reference = np.full((1, 10, 10), 100.0, dtype=np.float64)
    reference[0, 0, 0] = -9999.0  # nodata pixel
    nodata = -9999.0
    
    # Create output with NaN only in nodata area
    output = reference.copy()
    output[0, 0, 0] = np.nan  # NaN in nodata area - should be OK
    
    # Use check_data_quality with proper structure
    normalized_dict = {"test_method": [output]}
    scene_data = {
        "nodata_values": [nodata],
        "transforms": [from_origin(0, 10, 1, 1)],
        "crs": "EPSG:4326",
        "resolution": 1.0,
        "band_names": ["B14"],
    }
    
    quality = pipeline.check_data_quality(normalized_dict, scene_data)
    
    # Should pass because NaN is in nodata area
    assert quality["test_method"]["status"] == "pass", \
        f"Should pass but got: {quality['test_method']}"


def test_quality_check_rejects_nan_in_reference_valid_area():
    """
    Quality check should reject NaN in reference valid areas.
    """
    from src.multiband_pipeline import MultibandPipeline
    from src.experiment_config import ExperimentConfig
    
    config = ExperimentConfig()
    pipeline = MultibandPipeline(config)
    
    # Create reference - all valid
    reference = np.full((1, 10, 10), 100.0, dtype=np.float64)
    nodata = -9999.0
    
    # Create output with NaN in valid area
    output = reference.copy()
    output[0, 5, 5] = np.nan  # NaN in valid area - should fail
    
    # Must include "original" as reference for valid mask
    normalized_dict = {
        "original": [reference],
        "test_method": [output]
    }
    scene_data = {
        "nodata_values": [nodata],
        "transforms": [from_origin(0, 10, 1, 1)],
        "crs": "EPSG:4326",
        "resolution": 1.0,
        "band_names": ["B14"],
    }
    
    quality = pipeline.check_data_quality(normalized_dict, scene_data)
    
    # Should fail because NaN is in valid area
    assert quality["test_method"]["status"] == "fail", \
        f"Should fail but got: {quality['test_method']}"


def test_pipeline_status_does_not_require_unrequested_volrn_only():
    """
    Pipeline status should not require volrn_only if not requested.
    """
    from src.multiband_pipeline import MultibandPipeline
    from src.experiment_config import ExperimentConfig
    
    config = ExperimentConfig()
    config.ablation_methods = []  # No volrn_only
    pipeline = MultibandPipeline(config)
    
    requested = pipeline._requested_normalization_methods()
    
    # Should not include volrn_only
    assert "volrn_only" not in requested, f"volrn_only should not be requested: {requested}"
    
    # Should include base methods
    assert "original" in requested
    assert "bagrn" in requested
    assert "bagrn_volrn" in requested
