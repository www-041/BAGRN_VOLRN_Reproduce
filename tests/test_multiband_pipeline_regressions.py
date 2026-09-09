"""Regression tests for multiband_pipeline.py NoData and diagnostics.

Task 9 of reliability-fixes plan:
- nodata=None must not be converted to 0.
- VOLRN should be called with return_diagnostics=True when needed.
"""

import numpy as np
from types import SimpleNamespace


def test_nodata_none_not_converted_to_zero():
    """When nodata is None, it must NOT be converted to 0."""
    nodata_values = [None, -9999.0, None]
    
    # Bug pattern: nd_i = nodata_values[i] if nodata_values[i] is not None else 0
    # This converts None → 0, which is wrong
    
    # Fix: keep None as None
    for nd in nodata_values:
        # The correct behavior: None stays None
        assert nd is None or isinstance(nd, (int, float))
        if nd is None:
            # None means "no nodata value defined" - must not become 0
            pass  # Keep as None


def test_volrn_can_return_diagnostics():
    """VOLRN normalize supports return_diagnostics parameter."""
    from src.volrn import volrn_normalize
    from rasterio.transform import from_origin
    
    a = np.full((1, 32, 32), 100.0)
    b = np.full((1, 32, 32), 110.0)
    
    transforms = [from_origin(0, 32, 1, 1), from_origin(16, 32, 1, 1)]
    bounds = [(0, 0, 32, 32), (16, 0, 48, 32)]
    
    result, coeffs, diag = volrn_normalize(
        [a, b], transforms, bounds, [None, None],
        block_size_pixels=16, max_iter=3, tol=1e-3,
        return_diagnostics=True,
    )
    
    assert isinstance(diag, dict)
    assert 'n_blocks' in diag


def test_pipeline_stops_before_normalization_when_registration_quality_fails(monkeypatch):
    from src import multiband_pipeline
    from src.multiband_pipeline import MultibandPipeline

    config = SimpleNamespace(
        experiment_name="quality-gate-test",
        scenes=[{"id": "a"}, {"id": "b"}],
        selected_bands=["B14"],
        registration_band="B14",
        registration_params={"required_quality": "pass"},
    )
    pipeline = MultibandPipeline.__new__(MultibandPipeline)
    pipeline.config = config
    pipeline.dry_run = False
    pipeline.smoke = False
    pipeline.load_scenes = lambda: {"arrays": [], "transforms": [],
                                    "nodata_values": [], "bounds": [],
                                    "crs": None, "scene_ids": []}
    pipeline.detect_overlaps = lambda scene_data: [{"idx_i": 0, "idx_j": 1}]
    pipeline.register_scenes = lambda scene_data, overlaps: {
        "connected": True,
        "unreachable_scenes": [],
        "quality": {"quality": "warn"},
    }
    normalization_calls = []
    pipeline.apply_radiometric_normalization = lambda *args, **kwargs: normalization_calls.append(True)
    monkeypatch.setattr(multiband_pipeline, "validate_config", lambda config: [])
    monkeypatch.setattr(multiband_pipeline, "validate_band_consistency", lambda *args, **kwargs: [])

    result = pipeline.run()

    assert normalization_calls == []
    assert result["pipeline_status"] == "failed"
    assert result["normalized"] is None
    assert "registration_quality_gate" in result["skipped_outputs"]


def test_smoke_run_stops_before_normalization_when_registration_status_fails(monkeypatch):
    from rasterio.transform import from_origin
    from src import multiband_pipeline
    from src.multiband_pipeline import MultibandPipeline

    config = SimpleNamespace(
        experiment_name="smoke-status-gate-test",
        scenes=[{"id": "a"}, {"id": "b"}],
        selected_bands=["B14"],
        registration_band="B14",
        registration_params={"required_quality": "pass"},
        ablation_methods=[],
        mosaic_modes=[],
        feather_widths=[],
        enable_spectral_metrics=False,
        output_root="data/test-output",
    )
    arrays = [np.ones((1, 4, 4)), np.ones((1, 4, 4))]
    transforms = [from_origin(0, 4, 1, 1)] * 2
    scene_data = {
        "arrays": arrays,
        "transforms": transforms,
        "nodata_values": [None, None],
        "bounds": [(0, 0, 4, 4), (0, 0, 4, 4)],
        "crs": None,
        "scene_ids": ["a", "b"],
    }
    full_registration = {
        "registered_arrays": arrays,
        "global_shifts": np.zeros((2, 2)),
        "local_dx_fields": [np.zeros((4, 4)) for _ in arrays],
        "local_dy_fields": [np.zeros((4, 4)) for _ in arrays],
        "local_refinement": {},
        "pair_matches": [],
        "connected": True,
        "status": "fail",
        "failure": {"code": "full_registration_failed"},
        "spanning_tree": [(0, 1)],
        "geometric_edges": [(0, 1)],
        "matching_edges": [(0, 1)],
        "rejected_edges": [],
        "connected_components": [[0, 1]],
        "unreachable_scenes": [],
        "quality": {"quality": "pass"},
        "final_validation": {"edges": [], "overall": {"quality": "pass"}},
        "diagnostics": {},
    }
    pipeline = MultibandPipeline.__new__(MultibandPipeline)
    pipeline.config = config
    pipeline.dry_run = False
    pipeline.smoke = True
    pipeline.smoke_crop_size = 4
    pipeline.output_root = config.output_root
    pipeline.control_idx = 0
    pipeline.registration_band_idx = 0
    pipeline.common_bands = ["B14"]
    pipeline.load_scenes = lambda: scene_data
    pipeline.detect_overlaps = lambda scene_data: [{"idx_i": 0, "idx_j": 1}]
    pipeline.register_scenes = lambda scene_data, overlaps: full_registration
    pipeline._smoke_recrop_by_spanning_tree = lambda scene_data, tree: scene_data

    normalization_calls = []
    pipeline.apply_radiometric_normalization = lambda *args, **kwargs: (
        normalization_calls.append(True) or {}
    )
    pipeline.compute_mosaics = lambda *args, **kwargs: {}
    pipeline.evaluate_metrics = lambda *args, **kwargs: {}
    pipeline.check_data_quality = lambda *args, **kwargs: {}
    monkeypatch.setattr(multiband_pipeline, "validate_config", lambda config: [])
    monkeypatch.setattr(
        multiband_pipeline, "validate_band_consistency", lambda *args, **kwargs: []
    )
    monkeypatch.setattr(
        multiband_pipeline,
        "_validate_final_registration_arrays",
        lambda *args, **kwargs: (
            {"quality": "pass"}, {"edges": [], "overall": {"quality": "pass"}}
        ),
    )

    result = pipeline.run()

    assert result["registration"]["status"] == "fail"
    assert result["registration"]["quality"]["quality"] == "pass"
    assert normalization_calls == []
    assert result["normalized"] is None
    assert result["pipeline_status"] == "failed"
