"""
Stage 2 problem_discovery tests (10+ cases).

Uses synthetic data only — no real GeoTIFF files needed.
"""

import os
import json
import tempfile
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

def _make_synthetic_data(
    n_scenes: int = 4,
    n_bands: int = 3,
    rows: int = 256,
    cols: int = 256,
    seed: int = 42,
) -> dict:
    """Create synthetic registered arrays, overlaps, normalized results."""
    rng = np.random.RandomState(seed)

    scene_ids = [f"scene_{i}" for i in range(n_scenes)]
    band_names = [f"B{i+1:02d}" for i in range(n_bands)]

    # Registered arrays: each scene is different but overlapping in stats
    registered_arrays = []
    for s in range(n_scenes):
        arr = rng.uniform(100, 500, (n_bands, rows, cols)).astype(np.float64)
        arr += s * 50  # scene-specific offset
        registered_arrays.append(arr)

    # Transforms as Affine-compatible objects (image_blocking needs .a, .e, ~, *)
    class _FakeAffine:
        def __init__(self, rows):
            self.a = 1.0
            self.e = -1.0
            self.c = 0.0
            self.f = float(rows)

        def __invert__(self):
            inv = _FakeAffine(int(self.f))
            inv.a = 1.0 / self.a
            inv.e = 1.0 / self.e
            inv.c = -self.c / self.a
            inv.f = -self.f / self.e
            return inv

        def __mul__(self, other):
            if isinstance(other, (tuple, list)) and len(other) == 2:
                x, y = other
                new_x = self.a * x + self.c
                new_y = self.e * y + self.f
                return (new_x, new_y)
            return NotImplemented

    transforms = [_FakeAffine(rows) for _ in range(n_scenes)]
    bounds = [(0, 0, cols, rows) for _ in range(n_scenes)]
    nodata_values = [None] * n_scenes

    # Overlaps: each adjacent pair
    overlaps = []
    for i in range(n_scenes - 1):
        overlap_rows = min(64, rows // 4)
        overlap_cols = min(64, cols // 4)
        overlaps.append({
            "idx_i": i,
            "idx_j": i + 1,
            "window_i": (0, overlap_rows, 0, overlap_cols),
            "window_j": (0, overlap_rows, 0, overlap_cols),
        })

    # Normalized results: original + 3 methods
    normalized = {
        "original": [a.copy() for a in registered_arrays],
        "bagrn": [a.copy() + rng.normal(0, 5, a.shape) for a in registered_arrays],
        "volrn_only": [a.copy() + rng.normal(0, 10, a.shape) for a in registered_arrays],
        "bagrn_volrn": [a.copy() + rng.normal(0, 3, a.shape) for a in registered_arrays],
    }

    return {
        "scene_ids": scene_ids,
        "band_names": band_names,
        "transforms": transforms,
        "bounds": bounds,
        "nodata_values": nodata_values,
        "overlaps": overlaps,
        "normalized": normalized,
        "registered_arrays": registered_arrays,
    }


# ---------------------------------------------------------------------------
# Test 1: band_attribution runs and produces expected keys
# ---------------------------------------------------------------------------

def test_band_attribution_runs():
    from src.problem_discovery import run_band_attribution

    data = _make_synthetic_data()
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_band_attribution(
            data, tmpdir,
            sensor_id="VNIR",
            band_wavelengths={
                "B01": {"center_nm": 421.0},
                "B02": {"center_nm": 452.5},
                "B03": {"center_nm": 490.0},
            },
            metadata_status="test",
        )
        assert "experiments" in result
        assert result.get("sensor_id") == "VNIR"
        for method in data["normalized"]:
            assert method in result["experiments"]
            exp = result["experiments"][method]
            assert "per_band_ave" in exp
            assert len(exp["per_band_ave"]) == len(data["band_names"])
            assert "top3_worst_bands" in exp
            assert "top3_best_bands" in exp
            assert exp.get("sensor_id") == "VNIR"
        # Check CSV was created and has new fields
        csv_files = [f for f in os.listdir(tmpdir) if f.endswith(".csv")]
        assert len(csv_files) > 0
        # Read first CSV and verify headers
        csv_path = os.path.join(tmpdir, csv_files[0])
        with open(csv_path, "r") as f:
            header = f.readline().strip()
        assert "sensor_id" in header
        assert "qualified_band_name" in header
        assert "wavelength_center_nm" in header
        assert "metadata_status" in header


# ---------------------------------------------------------------------------
# Test 2: scene_attribution runs and identifies worst scene
# ---------------------------------------------------------------------------

def test_scene_attribution_runs():
    from src.problem_discovery import run_scene_attribution

    data = _make_synthetic_data()
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_scene_attribution(data, tmpdir)
        assert "experiments" in result
        for method in data["normalized"]:
            assert method in result["experiments"]
            exp = result["experiments"][method]
            assert "worst_scene_id" in exp
            assert "per_scene_ave" in exp
            assert len(exp["per_scene_ave"]) == len(data["scene_ids"])


# ---------------------------------------------------------------------------
# Test 3: gain_offset_ablation runs
# ---------------------------------------------------------------------------

def test_gain_offset_ablation_runs():
    from src.problem_discovery import run_gain_offset_ablation

    data = _make_synthetic_data()
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_gain_offset_ablation(data, tmpdir)
        assert "dominant_effect" in result
        assert "radiometric_global" in result
        assert "spectral_global" in result
        assert result["dominant_effect"] in (
            "gain_dominant", "offset_dominant", "joint_effect", "interaction_or_nonlinear"
        )


# ---------------------------------------------------------------------------
# Test 4: spatial_attribution runs and interior < boundary (or equal)
# ---------------------------------------------------------------------------

def test_spatial_attribution_runs():
    from src.problem_discovery import run_spatial_attribution

    data = _make_synthetic_data()
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_spatial_attribution(data, tmpdir, boundary_width=32)
        assert "experiments" in result
        for method in data["normalized"]:
            assert method in result["experiments"]
            exp = result["experiments"][method]
            assert "interior_ave" in exp
            assert "boundary_ave" in exp
            assert "degradation_ratio" in exp


# ---------------------------------------------------------------------------
# Test 5: multiwindow runs and produces per-window Ave
# ---------------------------------------------------------------------------

def test_multiwindow_runs():
    from src.problem_discovery import run_multiwindow

    data = _make_synthetic_data()
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_multiwindow(data, tmpdir, window_size=128, max_windows=3)
        assert "experiments" in result
        assert "window_positions" in result
        assert len(result["window_positions"]) <= 3
        for method in data["normalized"]:
            assert method in result["experiments"]
            exp = result["experiments"][method]
            assert "per_window_ave" in exp
            assert "mean_ave" in exp


# ---------------------------------------------------------------------------
# Test 6: nan_trace runs with zero nan (synthetic has no nan)
# ---------------------------------------------------------------------------

def test_nan_trace_runs():
    from src.problem_discovery import run_nan_trace

    data = _make_synthetic_data()
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_nan_trace(data, tmpdir)
        assert "input_nan_counts" in result
        assert "output_nan_counts" in result
        assert "band_converged" in result
        assert "block_details" in result
        # Synthetic data has no nan → input_nan_counts should be 0
        assert all(c == 0 for c in result["input_nan_counts"])


# ---------------------------------------------------------------------------
# Test 7: run_problem_discovery master runner
# ---------------------------------------------------------------------------

def test_run_problem_discovery_master():
    from src.problem_discovery import run_problem_discovery

    data = _make_synthetic_data()
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save synthetic data as if Stage 1 output
        baseline_dir = os.path.join(tmpdir, "stage1")
        registered_dir = os.path.join(baseline_dir, "registered")
        normalized_dir = os.path.join(baseline_dir, "normalized")
        os.makedirs(registered_dir)
        os.makedirs(normalized_dir)

        for i, arr in enumerate(data["registered_arrays"]):
            np.save(os.path.join(registered_dir, f"scene_{i}.npy"), arr)

        meta = {
            "transforms": [{"a": t.a, "e": t.e, "c": t.c, "f": t.f} for t in data["transforms"]],
            "nodata_values": data["nodata_values"],
            "bounds": data["bounds"],
            "band_names": data["band_names"],
        }
        with open(os.path.join(registered_dir, "metadata.json"), "w") as f:
            json.dump(meta, f)

        with open(os.path.join(baseline_dir, "overlaps.json"), "w") as f:
            json.dump(data["overlaps"], f)

        for method, arrays in data["normalized"].items():
            method_dir = os.path.join(normalized_dir, method)
            os.makedirs(method_dir, exist_ok=True)
            for i, arr in enumerate(arrays):
                np.save(os.path.join(method_dir, f"scene_{i}.npy"), arr)

        # Mock config
        class MockConfig:
            problem_discovery = {
                "baseline_output": baseline_dir,
                "crop_size": 256,
                "max_windows": 3,
                "experiments": ["band_attribution"],
                "nan_trace": {"enabled": True},
            }
            volrn_params = {"block_size": 100, "lambda": 0.5, "rho": 1.0, "max_iter": 50, "tol": 1e-4}
            output_root = tmpdir
            experiment_name = "test_stage2"
            scenes = []
            sensor = {}

        config = MockConfig()
        output_dir = os.path.join(tmpdir, "test_stage2")
        os.makedirs(output_dir, exist_ok=True)

        result = run_problem_discovery(config, output_dir, experiments=["band_attribution"])
        assert "band_attribution" in result
        assert "experiments_run" in result
        assert result["experiments_run"] == ["band_attribution"]


# ---------------------------------------------------------------------------
# Test 8: load_stage1_data handles missing directory
# ---------------------------------------------------------------------------

def test_load_stage1_data_missing_dir():
    from src.problem_discovery import load_stage1_data

    with pytest.raises(FileNotFoundError):
        load_stage1_data("/nonexistent/path")


# ---------------------------------------------------------------------------
# Test 9: band_attribution with empty overlaps
# ---------------------------------------------------------------------------

def test_band_attribution_empty_overlaps():
    from src.problem_discovery import run_band_attribution

    data = _make_synthetic_data()
    data["overlaps"] = []
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_band_attribution(data, tmpdir)
        assert "experiments" in result
        for method in data["normalized"]:
            assert method in result["experiments"]
            # All bands should be 0 with no overlaps
            assert all(v == 0.0 for v in result["experiments"][method]["per_band_ave"])


# ---------------------------------------------------------------------------
# Test 10: scene_attribution with single scene
# ---------------------------------------------------------------------------

def test_scene_attribution_single_scene():
    from src.problem_discovery import run_scene_attribution

    data = _make_synthetic_data(n_scenes=1)
    data["overlaps"] = []
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_scene_attribution(data, tmpdir)
        assert "experiments" in result
        for method in data["normalized"]:
            assert method in result["experiments"]
            exp = result["experiments"][method]
            assert len(exp["per_scene_ave"]) == 1


# ---------------------------------------------------------------------------
# Test 11: spatial_attribution with zero boundary_width
# ---------------------------------------------------------------------------

def test_spatial_attribution_small_boundary():
    from src.problem_discovery import run_spatial_attribution

    data = _make_synthetic_data()
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_spatial_attribution(data, tmpdir, boundary_width=1)
        assert "experiments" in result
        for method in data["normalized"]:
            exp = result["experiments"][method]
            assert "interior_ave" in exp
            assert "boundary_ave" in exp


# ---------------------------------------------------------------------------
# Test 12: compute_pair_metrics helper
# ---------------------------------------------------------------------------

def test_compute_pair_metrics():
    from src.problem_discovery import _compute_pair_metrics

    rng = np.random.RandomState(42)
    arr_i = rng.uniform(100, 500, (3, 64, 64))
    arr_j = arr_i + 10  # offset

    result = _compute_pair_metrics(
        arr_i, arr_j,
        window_i=(0, 64, 0, 64),
        window_j=(0, 64, 0, 64),
        nodata_i=None,
        nodata_j=None,
    )
    assert "adm" in result
    assert "adsd" in result
    assert "ave" in result
    assert result["adm"] > 0  # There should be some difference


# ---------------------------------------------------------------------------
# Test 13: config validation for problem_discovery
# ---------------------------------------------------------------------------

def test_config_validation_problem_discovery():
    from src.experiment_config import ExperimentConfig, validate_config

    config = ExperimentConfig(
        experiment_name="test",
        scenes=[],
        output_root="/tmp",
        problem_discovery={
            "baseline_output": "/nonexistent",
            "crop_size": 1024,
            "max_windows": 6,
            "spatial_strata": {
                "boundary_width_pixels": 100,
                "texture_quantiles": [0.25, 0.75],
                "brightness_quantiles": [0.25, 0.75],
                "minimum_valid_pixels": 1000,
            },
            "multiwindow": {
                "window_size": 1024,
                "minimum_valid_ratio": 0.30,
            },
        },
    )
    errors = validate_config(config)
    # Should flag missing baseline_output directory
    assert any("baseline_output" in e for e in errors)
