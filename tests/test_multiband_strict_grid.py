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
