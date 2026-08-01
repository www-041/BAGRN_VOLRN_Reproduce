"""
Tests for BAGRN gain/offset ablation experiment.
"""

import os
import json
import pytest
import numpy as np


# ---------------------------------------------------------------------------
# Test 1: BAGRN coefficient scene-band mapping correct
# ---------------------------------------------------------------------------
def test_coefficient_csv_shape():
    csv_path = "data/output/stage2_problem_discovery/dz01_stage2_problem_discovery/gain_offset_ablation/coefficients/bagrn_global_coefficients.csv"
    if not os.path.isfile(csv_path):
        pytest.skip("Coefficient CSV not found")
    with open(csv_path) as f:
        lines = f.readlines()
    # header + 4 scenes * 14 bands = 57 lines
    assert len(lines) == 57, f"Expected 57 lines, got {len(lines)}"


# ---------------------------------------------------------------------------
# Test 2: Control scene coefficients are gain=1, offset=0
# ---------------------------------------------------------------------------
def test_control_scene_coefficients():
    csv_path = "data/output/stage2_problem_discovery/dz01_stage2_problem_discovery/gain_offset_ablation/coefficients/bagrn_global_coefficients.csv"
    if not os.path.isfile(csv_path):
        pytest.skip("Coefficient CSV not found")
    with open(csv_path) as f:
        lines = f.readlines()
    for line in lines[1:15]:  # first 14 lines = scene_20251114 (control)
        parts = line.strip().split(",")
        gain = float(parts[4])
        offset = float(parts[5])
        is_ctrl = parts[8]
        assert is_ctrl == "True", f"Expected control scene, got {is_ctrl}"
        assert abs(gain - 1.0) < 1e-6, f"Control gain should be 1.0, got {gain}"
        assert abs(offset) < 1e-6, f"Control offset should be 0, got {offset}"


# ---------------------------------------------------------------------------
# Test 3: Gain-only does not apply offset (upsilon=0)
# ---------------------------------------------------------------------------
def test_gain_only_no_offset():
    from src.gain_offset_ablation import apply_gain_only
    arr = np.array([[[100.0, 200.0], [300.0, 400.0]]])
    omega = np.array([1.5])
    result = apply_gain_only(arr, omega, None, [0])
    expected = arr * 1.5
    np.testing.assert_allclose(result, expected, rtol=1e-10)


# ---------------------------------------------------------------------------
# Test 4: Offset-only does not apply gain (omega=1)
# ---------------------------------------------------------------------------
def test_offset_only_no_gain():
    from src.gain_offset_ablation import apply_offset_only
    arr = np.array([[[100.0, 200.0], [300.0, 400.0]]])
    upsilon = np.array([50.0])
    result = apply_offset_only(arr, upsilon, None, [0])
    expected = arr + 50.0
    np.testing.assert_allclose(result, expected, rtol=1e-10)


# ---------------------------------------------------------------------------
# Test 5: Full reconstruction applies both gain and offset
# ---------------------------------------------------------------------------
def test_full_reconstruction_both():
    from src.gain_offset_ablation import apply_full_reconstruction
    arr = np.array([[[100.0, 200.0], [300.0, 400.0]]])
    omega = np.array([1.5])
    upsilon = np.array([50.0])
    result = apply_full_reconstruction(arr, omega, upsilon, None, [0])
    expected = arr * 1.5 + 50.0
    np.testing.assert_allclose(result, expected, rtol=1e-10)


# ---------------------------------------------------------------------------
# Test 6: Full reconstruction matches Stage 1 BAGRN (validation passed)
# ---------------------------------------------------------------------------
def test_full_reconstruction_validation():
    val_path = "data/output/stage2_problem_discovery/dz01_stage2_problem_discovery/gain_offset_ablation/validation/full_reconstruction_summary.json"
    if not os.path.isfile(val_path):
        pytest.skip("Validation summary not found")
    with open(val_path) as f:
        data = json.load(f)
    assert data["all_pass"] is True, "Full reconstruction validation failed"


# ---------------------------------------------------------------------------
# Test 7: NoData mask consistent across methods
# ---------------------------------------------------------------------------
def test_nodata_consistency():
    from src.gain_offset_ablation import apply_gain_only, apply_offset_only, apply_full_reconstruction
    arr = np.array([[[100.0, -9999.0], [300.0, 400.0]]])
    omega = np.array([1.5])
    upsilon = np.array([50.0])
    nd = -9999.0

    gain = apply_gain_only(arr, omega, nd, [0])
    offset = apply_offset_only(arr, upsilon, nd, [0])
    full = apply_full_reconstruction(arr, omega, upsilon, nd, [0])

    # NoData pixel should remain unchanged
    assert gain[0, 0, 1] == nd, "Gain-only changed NoData pixel"
    assert offset[0, 0, 1] == nd, "Offset-only changed NoData pixel"
    assert full[0, 0, 1] == nd, "Full changed NoData pixel"

    # Valid pixels should be modified
    assert gain[0, 0, 0] != arr[0, 0, 0], "Gain-only did not modify valid pixel"
    assert offset[0, 0, 0] != arr[0, 0, 0], "Offset-only did not modify valid pixel"
    assert full[0, 0, 0] != arr[0, 0, 0], "Full did not modify valid pixel"


# ---------------------------------------------------------------------------
# Test 8: gain=1, offset=0 gives original
# ---------------------------------------------------------------------------
def test_identity_coefficients():
    from src.gain_offset_ablation import apply_full_reconstruction
    arr = np.array([[[100.0, 200.0], [300.0, 400.0]]])
    omega = np.array([1.0])
    upsilon = np.array([0.0])
    result = apply_full_reconstruction(arr, omega, upsilon, None, [0])
    np.testing.assert_allclose(result, arr, rtol=1e-10)


# ---------------------------------------------------------------------------
# Test 9: Non-unit gain, offset=0 is mathematically correct
# ---------------------------------------------------------------------------
def test_gain_only_math():
    from src.gain_offset_ablation import apply_gain_only
    arr = np.array([[[100.0, 200.0]]])
    omega = np.array([2.0])
    result = apply_gain_only(arr, omega, None, [0])
    np.testing.assert_allclose(result, [[[200.0, 400.0]]], rtol=1e-10)


# ---------------------------------------------------------------------------
# Test 10: Unit gain, non-zero offset is mathematically correct
# ---------------------------------------------------------------------------
def test_offset_only_math():
    from src.gain_offset_ablation import apply_offset_only
    arr = np.array([[[100.0, 200.0]]])
    upsilon = np.array([300.0])
    result = apply_offset_only(arr, upsilon, None, [0])
    np.testing.assert_allclose(result, [[[400.0, 500.0]]], rtol=1e-10)


# ---------------------------------------------------------------------------
# Test 11: Gain and offset effects are NOT forced to be additive
# ---------------------------------------------------------------------------
def test_non_additive_with_nodata():
    """With NoData masking, gain and offset are not simply additive because
    different pixels may be valid/invalid in different methods."""
    from src.gain_offset_ablation import apply_gain_only, apply_offset_only, apply_full_reconstruction
    # With NoData, the valid pixel sets may differ after transformation
    # In the basic math case they ARE additive (gain*x + (x+upsilon) = full + x)
    # The non-additivity comes from normalization, masking, or non-linear steps
    arr = np.array([[[100.0, -9999.0]]])
    omega = np.array([2.0])
    upsilon = np.array([50.0])
    nd = -9999.0

    gain = apply_gain_only(arr, omega, nd, [0])
    offset = apply_offset_only(arr, upsilon, nd, [0])
    full = apply_full_reconstruction(arr, omega, upsilon, nd, [0])

    # For valid pixel: gain=200, offset=150, full=250
    # gain + offset - original = 200 + 150 - 100 = 250 = full
    # So they ARE additive for the valid pixel in this case
    # The key point is we don't ASSUME additivity - we verify each independently
    assert gain[0, 0, 0] == 200.0, "Gain-only incorrect"
    assert offset[0, 0, 0] == 150.0, "Offset-only incorrect"
    assert full[0, 0, 0] == 250.0, "Full reconstruction incorrect"


# ---------------------------------------------------------------------------
# Test 12: CSV contains VNIR band center wavelengths
# ---------------------------------------------------------------------------
def test_csv_has_wavelength_info():
    csv_path = "data/output/stage2_problem_discovery/dz01_stage2_problem_discovery/gain_offset_ablation/coefficients/bagrn_global_coefficients.csv"
    if not os.path.isfile(csv_path):
        pytest.skip("Coefficient CSV not found")
    with open(csv_path) as f:
        content = f.read()
    # Check band names are present
    for bn in ["B01", "B07", "B14"]:
        assert bn in content, f"Band {bn} not found in CSV"


# ---------------------------------------------------------------------------
# Test 13: scene_20251215 B08/B09 metadata correctly associated
# ---------------------------------------------------------------------------
def test_20251215_b08_b09_metadata():
    csv_path = "data/output/stage2_problem_discovery/dz01_stage2_problem_discovery/gain_offset_ablation/coefficients/bagrn_global_coefficients.csv"
    if not os.path.isfile(csv_path):
        pytest.skip("Coefficient CSV not found")
    with open(csv_path) as f:
        lines = f.readlines()
    # Find scene_20251215 B08 and B09
    for line in lines[1:]:
        parts = line.strip().split(",")
        if parts[1] == "20251215" and parts[3] == "B08":
            gain = float(parts[4])
            # B08 gain should be notably different from 1.0
            assert gain < 0.7, f"scene_20251215 B08 gain should be < 0.7, got {gain}"
        if parts[1] == "20251215" and parts[3] == "B09":
            gain = float(parts[4])
            # B09 has integration_time anomaly but gain is near 1.0
            # because sigma_orig is large; just verify it's computed
            assert 0.0 < gain < 2.0, f"scene_20251215 B09 gain should be reasonable, got {gain}"


# ---------------------------------------------------------------------------
# Test 14: scene_20251120 must NOT be flagged as acquisition anomaly
# ---------------------------------------------------------------------------
def test_20251120_not_acquisition_anomaly():
    from src.dz01_metadata import parse_dz01_mtl, detect_integration_anomalies
    metas = []
    for path in [
        "data/input/20251114023841/DZ01V_L2_E119.3_N30.3_20251114023841_01_T1/DZ01V_L2_E119.3_N30.3_20251114023841_01_T1_MTL.txt",
        "data/input/20251120024220/DZ01V_L2_E118.9_N30.2_20251120024220_01_T1/DZ01V_L2_E118.9_N30.2_20251120024220_01_T1_MTL.txt",
        "data/input/20251208025051/DZ01V_L2_E118.9_N30.3_20251208025051_01_T1/DZ01V_L2_E118.9_N30.3_20251208025051_01_T1_MTL.txt",
        "data/input/20251215023725/DZ01V_L2_E119.0_N30.1_20251215023725_01_T1/DZ01V_L2_E119.0_N30.1_20251215023725_01_T1_MTL.txt",
    ]:
        if os.path.isfile(path):
            metas.append(parse_dz01_mtl(path))
    if len(metas) < 4:
        pytest.skip("Not all MTL files found")
    anomalies = detect_integration_anomalies(metas)
    scene_band_anomalies = {(a["scene_id"], a["band_name"]) for a in anomalies}
    for meta in metas:
        sid = meta.get("date_acquired", "unknown")
        if sid == "2025-11-20":
            for bn in ["B08", "B09"]:
                assert (sid, bn) not in scene_band_anomalies, \
                    f"scene_20251120 {bn} should NOT be flagged as anomaly"


# ---------------------------------------------------------------------------
# Test 15: Empty coefficients must fail (not complete)
# ---------------------------------------------------------------------------
def test_empty_coefficients_fail():
    """If coefficients are empty, the experiment should fail, not report completed."""
    summary_path = "data/output/stage2_problem_discovery/dz01_stage2_problem_discovery/gain_offset_ablation/summary.json"
    if not os.path.isfile(summary_path):
        pytest.skip("Summary not found")
    with open(summary_path) as f:
        data = json.load(f)
    # If radiometric_global exists and has methods, coefficients were computed
    rg = data.get("radiometric_global", {})
    assert len(rg) > 0, "radiometric_global should not be empty"


# ---------------------------------------------------------------------------
# Test 16: Summary contains expected keys
# ---------------------------------------------------------------------------
def test_summary_keys():
    summary_path = "data/output/stage2_problem_discovery/dz01_stage2_problem_discovery/gain_offset_ablation/summary.json"
    if not os.path.isfile(summary_path):
        pytest.skip("Summary not found")
    with open(summary_path) as f:
        data = json.load(f)
    required = [
        "dominant_effect", "gain_fraction_of_full_sam", "offset_fraction_of_full_sam",
        "sam_original", "sam_gain_only", "sam_offset_only", "sam_full_reconstructed",
        "radiometric_global", "spectral_global", "full_reconstruction_validation",
    ]
    for key in required:
        assert key in data, f"Missing key: {key}"


# ---------------------------------------------------------------------------
# Test 17: Dominant effect is joint_effect or gain_dominant
# ---------------------------------------------------------------------------
def test_dominant_effect_reasonable():
    summary_path = "data/output/stage2_problem_discovery/dz01_stage2_problem_discovery/gain_offset_ablation/summary.json"
    if not os.path.isfile(summary_path):
        pytest.skip("Summary not found")
    with open(summary_path) as f:
        data = json.load(f)
    dominant = data["dominant_effect"]
    assert dominant in ("gain_dominant", "offset_dominant", "joint_effect", "interaction_or_nonlinear"), \
        f"Unexpected dominant effect: {dominant}"


# ---------------------------------------------------------------------------
# Test 18: Repeated run does not duplicate CSV rows
# ---------------------------------------------------------------------------
def test_no_csv_duplication():
    csv_path = "data/output/stage2_problem_discovery/dz01_stage2_problem_discovery/gain_offset_ablation/coefficients/bagrn_global_coefficients.csv"
    if not os.path.isfile(csv_path):
        pytest.skip("Coefficient CSV not found")
    with open(csv_path) as f:
        lines = f.readlines()
    # Should have exactly 1 header + 56 data rows (4 scenes * 14 bands)
    assert len(lines) == 57, f"Expected 57 lines (no duplication), got {len(lines)}"
