"""End-to-end smoke test with 6 synthetic scenes.

Task 11 of dz01v mosaic readiness plan:
Verify the entire pipeline works with synthetic data before running on real data.
"""

import pytest
import numpy as np
import tempfile
import os
from rasterio.transform import from_origin

from src.experiment_config import ExperimentConfig
from src.multiband_pipeline import MultibandPipeline


def create_synthetic_geotiff(path, width=64, height=64, value=100.0, seed=42):
    """Create a synthetic GeoTIFF with texture for testing."""
    import rasterio
    
    # Create textured image (not uniform, so registration can work)
    np.random.seed(seed)
    # Base value + random texture
    data = np.full((1, height, width), value, dtype=np.float32)
    # Add some texture (random noise + gradient)
    texture = np.random.randn(height, width).astype(np.float32) * 5.0
    # Add gradient
    y_grad = np.linspace(0, 20, height).reshape(-1, 1)
    x_grad = np.linspace(0, 20, width).reshape(1, -1)
    data[0] = data[0] + texture + y_grad + x_grad
    
    transform = from_origin(0, height, 1, 1)
    
    with rasterio.open(
        path, 'w',
        driver='GTiff',
        height=height, width=width,
        count=1, dtype=data.dtype,
        crs='EPSG:4326',
        transform=transform,
        nodata=-9999.0,
    ) as dst:
        dst.write(data)


def test_six_scene_synthetic_smoke():
    """
    End-to-end test with 6 synthetic scenes.
    
    Creates 6 scenes with overlapping regions, runs the full pipeline:
    - Registration
    - BAGRN
    - VOLRN
    - Mosaic
    
    Verifies all steps complete without error.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create 6 synthetic scenes with overlapping footprints
        scenes = []
        for i in range(6):
            scene_dir = os.path.join(tmpdir, f"scene_{i}")
            os.makedirs(scene_dir)
            
            # Create B14 band with different seed for each scene (different texture)
            b14_path = os.path.join(scene_dir, "B14.tif")
            create_synthetic_geotiff(b14_path, value=100.0 + i * 10, seed=42+i)
            
            scenes.append({
                "id": f"scene_{i}",
                "datetime": f"2025-11-{14+i:02d}",
                "sensor": "DZ01V",
                "bands": {"B14": b14_path}
            })
        
        # Create config
        config = ExperimentConfig()
        config.experiment_name = "test_six_scene"
        config.output_root = tmpdir
        config.scenes = scenes
        config.selected_bands = ["B14"]
        config.registration_band = "B14"
        config.control_scene = "scene_0"
        config.common_bands_strategy = "strict"
        config.mosaic_modes = ["weighted"]
        config.feather_widths = [10]
        config.volrn_params = {
            "block_size": 16,  # Small for testing
            "lambda": 0.1,
            "rho": 1.0,
            "max_iter": 5,  # Few iterations for speed
            "tol": 1e-3,
        }
        config.enable_spectral_metrics = False
        config.dry_run = False
        
        # Run pipeline
        pipeline = MultibandPipeline(config)
        results = pipeline.run()
        
        # Verify results
        assert results is not None
        assert "pipeline_status" in results
        assert results["pipeline_status"] == "success"
        
        # Check that normalization was performed
        # results["normalized"] is a dict with method names as keys
        assert "normalized" in results
        normalized = results["normalized"]
        assert isinstance(normalized, dict)
        # Should have at least original, bagrn, bagrn_volrn
        assert "original" in normalized or "bagrn" in normalized
        
        # Check that overlaps were detected
        assert "overlaps" in results
        assert len(results["overlaps"]) > 0
        
        # Check that metrics were computed
        assert "metrics" in results
        # Should have metrics for at least one method
        assert len(results["metrics"]) > 0
        
        print("✓ Six-scene synthetic smoke test PASSED")
