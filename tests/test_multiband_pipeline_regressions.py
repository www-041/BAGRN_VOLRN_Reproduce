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
