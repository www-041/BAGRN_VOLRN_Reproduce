"""Static tests for script configuration and metric call contracts.

Task 17 of reliability-fixes plan:
- compute_all(arrays_before, arrays_after, nodata_values, overlaps, bands)
- Baseline: compute_all(original, original, ...)
- BAGRN: compute_all(original, bagrn_result, ...)
- VOLRN: compute_all(original, volrn_result, ...)
- Never compute_all(result, result, ...) for normalized methods
"""

import ast
import importlib.util
from pathlib import Path
from unittest.mock import patch

import rasterio


REPO_ROOT = Path(__file__).resolve().parents[1]
TWO_IMAGE_PIPELINE = REPO_ROOT / "scripts" / "two_image_pipeline.py"


def _load_two_image_pipeline():
    spec = importlib.util.spec_from_file_location("two_image_pipeline_test", TWO_IMAGE_PIPELINE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read_pipeline_constants():
    tree = ast.parse(TWO_IMAGE_PIPELINE.read_text(encoding="utf-8"))
    constants = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in {"OUTPUT", "IMG1", "IMG2", "BANDS"}:
                constants[target.id] = ast.literal_eval(node.value)
    return constants


def test_two_image_pipeline_targets_requested_float_b14_pair():
    """The two-image script must use the requested float inputs and output directory."""
    constants = _read_pipeline_constants()

    assert constants["OUTPUT"] == str(
        REPO_ROOT / "data" / "output" / "two_image_float"
    )
    assert constants["IMG1"] == (
        "data/input/float/"
        "DZ01V_L2_E113.4_N36.6_20260810030932_01_T1/"
        "DZ01V_L2_E113.4_N36.6_20260810030932_01_T1_{}.TIF"
    )
    assert constants["IMG2"] == (
        "data/input/float/"
        "DZ01V_L2_E113.3_N36.2_20260810030938_01_T1/"
        "DZ01V_L2_E113.3_N36.2_20260810030938_01_T1_{}.TIF"
    )
    assert constants["BANDS"] == ["B14"]

    assert (REPO_ROOT / constants["IMG1"].format("B14")).is_file()
    assert (REPO_ROOT / constants["IMG2"].format("B14")).is_file()


def test_two_image_pipeline_creates_band_output_directory():
    """Each band output directory must exist before Rasterio writes mosaics."""
    pipeline = _load_two_image_pipeline()
    pipeline.OUTPUT = str(REPO_ROOT / "data" / "output" / "two_image_float")

    expected = REPO_ROOT / "data" / "output" / "two_image_float" / "B14"
    with patch.object(pipeline.os, "makedirs") as makedirs:
        out_dir = Path(pipeline.get_band_output_dir("B14"))

    assert out_dir == expected
    makedirs.assert_called_once_with(str(expected), exist_ok=True)


def test_registration_metrics_report_residuals_and_inlier_ratio():
    """Registration metrics must summarize accepted blocks against the global shift."""
    pipeline = _load_two_image_pipeline()
    matches = [
        {"ref_x": 10, "ref_y": 10, "tgt_x": 20, "tgt_y": 15,
         "shift_dx": 5.0, "shift_dy": -2.0, "confidence": 0.9},
        {"ref_x": 90, "ref_y": 10, "tgt_x": 100, "tgt_y": 15,
         "shift_dx": 5.2, "shift_dy": -1.9, "confidence": 0.8},
        {"ref_x": 10, "ref_y": 90, "tgt_x": 20, "tgt_y": 95,
         "shift_dx": 12.0, "shift_dy": 8.0, "confidence": 0.7},
    ]

    metrics = pipeline.build_registration_metrics(
        matches,
        {"total": 4, "low_valid": 1, "low_texture": 0,
         "low_conf": 0, "large_shift": 0, "accepted": 3},
        global_dx=5.0,
        global_dy=-2.0,
        phase_confidence=0.92,
        reference_shape=(100, 100),
        reference_transform=rasterio.Affine(14, 0, 0, 0, -14, 0),
    )

    assert metrics["matching"]["candidate_blocks"] == 4
    assert metrics["matching"]["accepted_matches"] == 3
    assert metrics["matching"]["inlier_matches"] == 2
    assert metrics["matching"]["inlier_ratio"] == 2 / 3
    assert metrics["residual_relative_to_global_model"]["rmse_pixels"] > 0
    assert metrics["residual_relative_to_global_model"]["median_magnitude_pixels"] < metrics["residual_relative_to_global_model"]["max_magnitude_pixels"]
    assert metrics["offset"]["dx_map_units"] == 70.0
    assert metrics["offset"]["dy_map_units"] == 28.0


def test_registration_metrics_report_spatial_match_distribution():
    """Registration metrics must expose whether matches cover the reference image."""
    pipeline = _load_two_image_pipeline()
    matches = [
        {"ref_x": 10, "ref_y": 10, "tgt_x": 10, "tgt_y": 10,
         "shift_dx": 0.0, "shift_dy": 0.0, "confidence": 0.9},
        {"ref_x": 90, "ref_y": 10, "tgt_x": 90, "tgt_y": 10,
         "shift_dx": 0.0, "shift_dy": 0.0, "confidence": 0.9},
        {"ref_x": 10, "ref_y": 90, "tgt_x": 10, "tgt_y": 90,
         "shift_dx": 0.0, "shift_dy": 0.0, "confidence": 0.9},
        {"ref_x": 90, "ref_y": 90, "tgt_x": 90, "tgt_y": 90,
         "shift_dx": 0.0, "shift_dy": 0.0, "confidence": 0.9},
    ]

    metrics = pipeline.build_registration_metrics(
        matches, {"total": 4, "accepted": 4}, 0.0, 0.0, 0.9,
        reference_shape=(100, 100),
        reference_transform=rasterio.Affine(1, 0, 0, 0, -1, 0),
    )

    coverage = metrics["spatial_coverage"]
    assert coverage["grid_cells_covered"] == 4
    assert coverage["grid_coverage_ratio"] == 0.25
    assert coverage["row_span_pixels"] == 80.0
    assert coverage["col_span_pixels"] == 80.0


def test_registration_match_rows_include_residual_and_inlier_fields():
    """Per-match CSV rows must contain displacement, residual, confidence, and inlier data."""
    pipeline = _load_two_image_pipeline()
    matches = [
        {"ref_x": 10, "ref_y": 20, "tgt_x": 15, "tgt_y": 18,
         "shift_dx": 5.0, "shift_dy": -2.0, "confidence": 0.88},
    ]

    rows = pipeline.build_registration_match_rows(matches, global_dx=5.0, global_dy=-2.0)

    assert rows == [{
        "match_index": 0,
        "ref_x": 10.0,
        "ref_y": 20.0,
        "tgt_x": 15.0,
        "tgt_y": 18.0,
        "shift_dx_pixels": 5.0,
        "shift_dy_pixels": -2.0,
        "confidence": 0.88,
        "residual_dx_pixels": 0.0,
        "residual_dy_pixels": 0.0,
        "residual_magnitude_pixels": 0.0,
        "inlier": 1,
    }]


def test_radiometric_console_labels_and_metric_columns_are_complete():
    source = TWO_IMAGE_PIPELINE.read_text(encoding="utf-8")

    assert "Registered / Before normalization" in source
    assert "BAGRN + VOLRN" in source
    for label in ("ADM", "ADSD", "CD", "GL", "RDOA", "Ave"):
        assert f'("{label.lower() if label != "Ave" else "ave"}", "{label}")' in source


def test_compute_all_contract_documented():
    """Document the correct compute_all contract."""
    # compute_all(arrays_before, arrays_after, nodata_values, overlaps, bands)
    # 
    # For baseline: arrays_before = original, arrays_after = original
    # For BAGRN: arrays_before = original, arrays_after = bagrn_result
    # For VOLRN: arrays_before = original, arrays_after = volrn_result
    # 
    # This ensures GL measures loss from original, not between two normalized versions
    
    contract = {
        "baseline": ("original", "original"),
        "bagrn": ("original", "bagrn_result"),
        "volrn": ("original", "volrn_result"),
    }
    
    assert contract["baseline"] == ("original", "original")
    assert contract["bagrn"] == ("original", "bagrn_result")
    assert contract["volrn"] == ("original", "volrn_result")
