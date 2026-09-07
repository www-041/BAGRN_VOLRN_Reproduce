"""Tests for strict same-resolution enforcement.

Task 3 of dz01v mosaic readiness plan:
- strict mode rejects different pixel resolutions
- strict mode allows different origins with same resolution
"""

import numpy as np
import pytest
from rasterio.transform import from_origin


def test_strict_grid_allows_different_origins_with_same_resolution():
    """Scenes with same resolution but different origins should pass strict validation."""
    from src.scene_preflight import validate_strict_scene_grids
    
    # Same resolution (1.0), different origins
    transforms = [
        from_origin(0, 100, 1.0, 1.0),
        from_origin(50, 150, 1.0, 1.0),  # Different origin
    ]
    crs_list = ["EPSG:4326", "EPSG:4326"]
    scene_ids = ["scene1", "scene2"]
    
    # Should not raise
    validate_strict_scene_grids(scene_ids, transforms, crs_list)


def test_strict_grid_rejects_different_resolution():
    """Scenes with different resolutions should fail strict validation."""
    from src.scene_preflight import validate_strict_scene_grids
    
    transforms = [
        from_origin(0, 100, 1.0, 1.0),   # 1m resolution
        from_origin(0, 100, 2.0, 2.0),   # 2m resolution - different!
    ]
    crs_list = ["EPSG:4326", "EPSG:4326"]
    scene_ids = ["scene1", "scene2"]
    
    with pytest.raises(ValueError, match="resolution"):
        validate_strict_scene_grids(scene_ids, transforms, crs_list)


def test_strict_grid_rejects_different_crs():
    """Scenes with different CRS should fail strict validation."""
    from src.scene_preflight import validate_strict_scene_grids
    
    transforms = [
        from_origin(0, 100, 1.0, 1.0),
        from_origin(0, 100, 1.0, 1.0),
    ]
    crs_list = ["EPSG:4326", "EPSG:32650"]  # Different CRS
    scene_ids = ["scene1", "scene2"]
    
    with pytest.raises(ValueError, match="CRS"):
        validate_strict_scene_grids(scene_ids, transforms, crs_list)


def test_detect_overlaps_independent_masks():
    """detect_overlaps must use independent valid masks for each window.
    
    When two windows have different sizes (e.g., due to floor/ceil rounding),
    the code must not use joint masking (mi & mj) which would fail with
    broadcasting error.
    """
    import numpy as np
    from unittest.mock import patch
    from rasterio.transform import from_origin
    
    # Create a minimal MultibandPipeline instance with smoke mode
    from src.experiment_config import ExperimentConfig
    from src.multiband_pipeline import MultibandPipeline
    
    config = ExperimentConfig()
    config.smoke = True  # Reduce min_pixels threshold
    pipeline = MultibandPipeline(config)
    pipeline.n_bands = 1  # Manually set for test
    
    # Two scenes with larger arrays to pass min_pixels check
    # Scene 0: 100x100, Scene 1: 100x100
    rng = np.random.default_rng(42)
    arrays = [
        rng.random((1, 100, 100)),
        rng.random((1, 100, 100)),
    ]
    transforms = [from_origin(0, 100, 1, 1), from_origin(0, 100, 1, 1)]
    bounds = [(0, 0, 100, 100), (0, 0, 100, 100)]
    nodata_values = [None, None]
    scene_ids = ["scene0", "scene1"]
    
    scene_data = {
        "arrays": arrays,
        "transforms": transforms,
        "bounds": bounds,
        "nodata_values": nodata_values,
        "scene_ids": scene_ids,
    }
    
    # Monkeypatch get_overlap_window to return different-sized windows
    # This simulates floor/ceil rounding causing 1-pixel differences
    def mock_get_overlap_window(*args, **kwargs):
        # Scene 0 window: 50x50
        # Scene 1 window: 51x50 (1 pixel wider due to rounding)
        return ((0, 50, 0, 50), (0, 50, 0, 51))
    
    with patch('src.multiband_pipeline.get_overlap_window', side_effect=mock_get_overlap_window):
        overlaps = pipeline.detect_overlaps(scene_data)
    
    # Should not raise broadcasting error
    assert len(overlaps) > 0
    overlap = overlaps[0]
    
    # Should have finite mean/std for both scenes
    assert 'per_band_stats' in overlap
    stats = overlap['per_band_stats'][0]
    assert np.isfinite(stats['mean_i'])
    assert np.isfinite(stats['mean_j'])
    assert np.isfinite(stats['std_i'])
    assert np.isfinite(stats['std_j'])
