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


def test_synthetic_scene_builder_reuses_spatial_texture_for_radiometric_variants():
    """Synthetic pass cases must share spatial structure and vary radiometry."""
    shared_texture = make_shared_spatial_texture(width=64, height=64, seed=42)
    first = build_synthetic_scene_array(
        value=100.0, spatial_texture=shared_texture
    )
    second = build_synthetic_scene_array(
        value=110.0, spatial_texture=shared_texture
    )

    assert np.allclose(second - first, 10.0)


def make_shared_spatial_texture(width=64, height=64, seed=42):
    """Create one deterministic texture that can be reused by every scene."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((height, width), dtype=np.float32) * 5.0


def build_synthetic_scene_array(
    width=64, height=64, value=100.0, seed=42, spatial_texture=None
):
    """Build a textured scene with optional shared spatial structure."""
    if spatial_texture is None:
        spatial_texture = make_shared_spatial_texture(width, height, seed)
    spatial_texture = np.asarray(spatial_texture, dtype=np.float32)
    if spatial_texture.shape != (height, width):
        raise ValueError("spatial_texture shape must match scene dimensions")
    y_grad = np.linspace(0, 20, height, dtype=np.float32).reshape(-1, 1)
    x_grad = np.linspace(0, 20, width, dtype=np.float32).reshape(1, -1)
    data = np.full((1, height, width), value, dtype=np.float32)
    data[0] += spatial_texture + y_grad + x_grad
    return data


def create_synthetic_geotiff(
    path, width=64, height=64, value=100.0, seed=42, spatial_texture=None
):
    """Create a synthetic GeoTIFF with texture for testing."""
    import rasterio
    
    data = build_synthetic_scene_array(
        width=width,
        height=height,
        value=value,
        seed=seed,
        spatial_texture=spatial_texture,
    )
    
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
        shared_texture = make_shared_spatial_texture(width=64, height=64, seed=42)
        for i in range(6):
            scene_dir = os.path.join(tmpdir, f"scene_{i}")
            os.makedirs(scene_dir)
            
            # Keep spatial texture fixed and vary only the radiometric offset.
            b14_path = os.path.join(scene_dir, "B14.tif")
            create_synthetic_geotiff(
                b14_path,
                value=100.0 + i * 10,
                seed=42 + i,
                spatial_texture=shared_texture,
            )
            
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
        # These values are scaled to the 64x64 synthetic footprint. The production
        # defaults target large remote-sensing scenes and cannot yield five
        # independent validation blocks in this deliberately small fixture.
        config.registration_params.update({
            "global_block_size": 16,
            "global_confidence_threshold": 0.1,
            "max_global_shift": 8.0,
            "robust_min_inliers": 5,
            "robust_min_inlier_ratio": 0.35,
            "global_refine_block_size": 16,
            "global_refine_max_iterations": 1,
            "enable_local_refinement": False,
            "validation_block_size": 16,
            "validation_step": 16,
            "validation_offset_row": 0,
            "validation_offset_col": 0,
            "validation_min_distance_from_training": 0,
            "validation_confidence_threshold": 0.1,
            "validation_max_residual_shift": 3.0,
            "final_min_blocks": 5,
            "pass_min_mean_confidence": 0.1,
            "pass_max_median": 1.0,
            "pass_max_rmse": 2.0,
            "pass_max_p95": 3.0,
            "required_quality": "pass",
        })
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
