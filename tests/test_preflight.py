"""Tests for metadata-only preflight checks.

Task 10 of dz01v mosaic readiness plan:
- Verify scene metadata before running experiments
- Check file existence, sensor, CRS, resolution, overlap connectivity
"""

import pytest
import tempfile
import os
import numpy as np
import rasterio
from rasterio.transform import from_origin


def test_inspect_scene_headers():
    """Test inspecting scene metadata headers."""
    from src.preflight import inspect_scene_headers
    
    # Create a temporary GeoTIFF
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.tif")
        
        # Create a simple 64x64 image
        data = np.ones((1, 64, 64), dtype=np.float32)
        transform = from_origin(0, 64, 1, 1)
        
        with rasterio.open(
            path, 'w',
            driver='GTiff',
            height=64, width=64,
            count=1, dtype=data.dtype,
            crs='EPSG:4326',
            transform=transform,
        ) as dst:
            dst.write(data)
        
        scene = {
            "id": "test_scene",
            "sensor": "DZ01V",
            "bands": {"B14": path}
        }
        
        info = inspect_scene_headers(scene, "B14")
        
        assert info["scene_id"] == "test_scene"
        assert info["sensor"] == "DZ01V"
        assert info["width"] == 64
        assert info["height"] == 64
        assert info["crs"] == "EPSG:4326"
        assert info["x_res"] == 1.0
        assert info["y_res"] == 1.0


def test_check_overlap():
    """Test checking geographic overlap between scenes."""
    from src.preflight import check_overlap
    
    # Create two overlapping scenes
    scene1 = {
        "scene_id": "scene1",
        "bounds": (0, 0, 10, 10),  # left, bottom, right, top
    }
    scene2 = {
        "scene_id": "scene2",
        "bounds": (5, 5, 15, 15),  # Overlaps with scene1
    }
    scene3 = {
        "scene_id": "scene3",
        "bounds": (20, 20, 30, 30),  # No overlap
    }
    
    assert check_overlap(scene1, scene2) is True
    assert check_overlap(scene1, scene3) is False
    assert check_overlap(scene2, scene3) is False


def test_build_overlap_graph():
    """Test building overlap graph."""
    from src.preflight import build_overlap_graph
    import networkx as nx
    
    scenes = [
        {"scene_id": "scene1", "bounds": (0, 0, 10, 10)},
        {"scene_id": "scene2", "bounds": (5, 5, 15, 15)},
        {"scene_id": "scene3", "bounds": (10, 10, 20, 20)},
    ]
    
    G = build_overlap_graph(scenes)
    
    # scene1 overlaps with scene2
    # scene2 overlaps with scene3
    # scene1 does NOT overlap with scene3
    assert G.has_edge("scene1", "scene2")
    assert G.has_edge("scene2", "scene3")
    assert not G.has_edge("scene1", "scene3")
    
    # Graph should be connected
    assert nx.is_connected(G)
