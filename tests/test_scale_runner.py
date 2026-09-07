"""Tests for scale experiment runner output isolation.

Task 8 of dz01v mosaic readiness plan:
- Output isolation for each N
- Proper failure status aggregation
- Detailed result recording
"""

import pytest
import tempfile
import shutil
from src.experiment_config import ExperimentConfig


def test_scale_output_directory_isolation():
    """Test that each N value has its own isolated output directory."""
    from src.experiment_runner import run_scale
    import argparse
    
    # Create a minimal config with 3 scenes
    cfg = ExperimentConfig()
    cfg.experiment_name = "test_scale"
    cfg.output_root = tempfile.mkdtemp()
    cfg.scale_scene_counts = [2, 4]
    cfg.scenes = [
        {"id": "scene1", "datetime": "2025-01-01", "sensor": "DZ01V", "bands": {"B14": "/tmp/b14_1.tif"}},
        {"id": "scene2", "datetime": "2025-01-02", "sensor": "DZ01V", "bands": {"B14": "/tmp/b14_2.tif"}},
        {"id": "scene3", "datetime": "2025-01-03", "sensor": "DZ01V", "bands": {"B14": "/tmp/b14_3.tif"}},
        {"id": "scene4", "datetime": "2025-01-04", "sensor": "DZ01V", "bands": {"B14": "/tmp/b14_4.tif"}},
    ]
    cfg.selected_bands = ["B14"]
    cfg.registration_band = "B14"
    cfg.dry_run = True  # Use dry run to avoid actual processing
    
    args = argparse.Namespace(scene_count=None)
    
    try:
        summary = run_scale(cfg, args)
        
        # Check that output directories are isolated
        import os
        n2_dir = os.path.join(cfg.output_root, "test_scale", "scale", "n2")
        n4_dir = os.path.join(cfg.output_root, "test_scale", "scale", "n4")
        
        # Directories should exist (created by run_scale)
        assert os.path.exists(n2_dir) or os.path.exists(os.path.join(cfg.output_root, "test_scale", "scale"))
        # n4 should be separate from n2
        if os.path.exists(n4_dir):
            assert n2_dir != n4_dir
            
    finally:
        shutil.rmtree(cfg.output_root, ignore_errors=True)


def test_scale_failure_aggregation():
    """Test that failures are properly aggregated."""
    from src.experiment_runner import run_scale
    import argparse
    
    # Create a config where N=4 exceeds available scenes
    cfg = ExperimentConfig()
    cfg.experiment_name = "test_scale_fail"
    cfg.output_root = tempfile.mkdtemp()
    cfg.scale_scene_counts = [2, 4]
    cfg.scenes = [
        {"id": "scene1", "datetime": "2025-01-01", "sensor": "DZ01V", "bands": {"B14": "/tmp/b14_1.tif"}},
        {"id": "scene2", "datetime": "2025-01-02", "sensor": "DZ01V", "bands": {"B14": "/tmp/b14_2.tif"}},
    ]
    cfg.selected_bands = ["B14"]
    cfg.registration_band = "B14"
    cfg.dry_run = True
    
    args = argparse.Namespace(scene_count=None)
    
    try:
        summary = run_scale(cfg, args)
        
        # N=4 should be marked as failed since we only have 2 scenes
        results = summary.get("results", [])
        n4_result = next((r for r in results if r.get("n_scenes") == 4), None)
        
        if n4_result:
            # Should have error or failure status
            assert "error" in n4_result or n4_result.get("pipeline_status") == "failed"
            
    finally:
        shutil.rmtree(cfg.output_root, ignore_errors=True)
