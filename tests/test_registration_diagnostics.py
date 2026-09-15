import numpy as np
from rasterio.transform import Affine
from scipy.ndimage import shift as ndimage_shift

from src.coregistration import collect_block_matches, phase_correlation
from src.registration_diagnostics import (
    build_reference_common_grid_overlap,
    collect_common_grid_candidate_diagnostics,
    collect_raw_grid_candidate_diagnostics,
    compute_grid_relationship,
    masked_ncc,
    summarize_candidate_rows,
)


def test_masked_ncc_identical_arrays_is_one():
    yy, xx = np.mgrid[0:20, 0:20]
    ref = np.sin(xx / 3.0) + np.cos(yy / 4.0)
    tgt = ref.copy()
    valid = np.ones(ref.shape, dtype=bool)

    value = masked_ncc(ref, tgt, valid)

    assert value is not None
    assert np.isclose(value, 1.0)


def test_masked_ncc_too_few_valid_pixels_returns_none():
    ref = np.arange(400, dtype=float).reshape(20, 20)
    tgt = ref.copy()
    valid = np.zeros((20, 20), dtype=bool)
    valid[:5, :5] = True

    assert masked_ncc(ref, tgt, valid) is None


def test_grid_relationship_detects_fractional_pixel_phase():
    ref = Affine(14.0, 0.0, 1000.0, 0.0, -14.0, 2000.0)
    tgt = Affine(
        14.0,
        0.0,
        1000.0 + 14.0 * 10.25,
        0.0,
        -14.0,
        2000.0 - 14.0 * 3.5,
    )

    result = compute_grid_relationship(ref, tgt)

    assert np.isclose(result["target_origin_in_ref_pixels"]["col"], 10.25)
    assert np.isclose(result["target_origin_in_ref_pixels"]["row"], 3.5)
    assert np.isclose(abs(result["fractional_phase_pixels"]["col"]), 0.25)
    assert np.isclose(abs(result["fractional_phase_pixels"]["row"]), 0.5)


def test_candidate_summary_keeps_reject_counts_and_measured_distributions():
    rows = [
        {
            "shift_dx_pixels": 1.0,
            "shift_dy_pixels": 0.0,
            "shift_magnitude_pixels": 1.0,
            "confidence": 0.4,
            "reject_reason": "low_conf",
        },
        {
            "shift_dx_pixels": 0.0,
            "shift_dy_pixels": 2.0,
            "shift_magnitude_pixels": 2.0,
            "confidence": 0.8,
            "reject_reason": "accepted",
        },
        {
            "shift_dx_pixels": None,
            "shift_dy_pixels": None,
            "shift_magnitude_pixels": None,
            "confidence": None,
            "reject_reason": "low_valid",
        },
    ]

    result = summarize_candidate_rows(rows)

    assert result["total_rows"] == 3
    assert result["measured_rows"] == 2
    assert result["reject_reason_counts"] == {
        "low_conf": 1,
        "accepted": 1,
        "low_valid": 1,
    }
    assert np.isclose(result["confidence"]["median"], 0.6)
    assert np.isclose(result["shift_magnitude_pixels"]["max"], 2.0)


def test_raw_grid_candidates_preserve_low_confidence_rows(monkeypatch):
    yy, xx = np.mgrid[0:1024, 0:1024]
    arr = np.sin(xx / 15.0) + np.cos(yy / 19.0)
    tr = Affine(14.0, 0.0, 1000.0, 0.0, -14.0, 5000.0)

    monkeypatch.setattr(
        "src.registration_diagnostics.phase_correlation",
        lambda *args, **kwargs: (0.2, -0.1, 0.49),
    )

    result = collect_raw_grid_candidate_diagnostics(
        arr, tr, arr, tr, block_size=512, confidence_threshold=0.5)

    assert result["screening"]["accepted"] == 0
    assert result["screening"]["low_conf"] > 0
    assert any(
        row["reject_reason"] == "low_conf"
        for row in result["candidates"]
    )
    assert all(
        row["confidence"] == 0.49
        for row in result["candidates"]
        if row["reject_reason"] == "low_conf"
    )


def test_raw_diagnostic_accepted_rows_match_baseline_matcher(monkeypatch):
    yy, xx = np.mgrid[0:1024, 0:1024]
    arr = np.sin(xx / 15.0) + np.cos(yy / 19.0)
    tr = Affine(14.0, 0.0, 1000.0, 0.0, -14.0, 5000.0)

    def fake_phase(*args, **kwargs):
        return 0.2, -0.1, 0.8

    monkeypatch.setattr("src.registration_diagnostics.phase_correlation", fake_phase)
    monkeypatch.setattr("src.coregistration.phase_correlation", fake_phase)

    diagnostic = collect_raw_grid_candidate_diagnostics(
        arr, tr, arr, tr, block_size=512, confidence_threshold=0.5)
    matches, _ = collect_block_matches(
        arr, tr, arr, tr, block_size=512, confidence_threshold=0.5)
    accepted = [
        row for row in diagnostic["candidates"]
        if row["reject_reason"] == "accepted"
    ]

    assert len(accepted) == len(matches)
    for row, match in zip(accepted, matches):
        assert np.isclose(row["shift_dx_pixels"], match["shift_dx"])
        assert np.isclose(row["shift_dy_pixels"], match["shift_dy"])
        assert np.isclose(row["confidence"], match["confidence"])


def test_common_grid_identical_transforms_use_same_values_without_reprojection():
    yy, xx = np.mgrid[0:128, 0:128]
    arr_ref = np.sin(xx / 7.0) + np.cos(yy / 11.0)
    arr_tgt = arr_ref.copy()
    tr = Affine(14.0, 0.0, 1000.0, 0.0, -14.0, 5000.0)

    result = build_reference_common_grid_overlap(
        arr_ref, tr, arr_tgt, tr, "EPSG:4326", "EPSG:4326")

    assert result["available"] is True
    assert np.allclose(
        result["ref_overlap"], result["tgt_on_ref_grid"], equal_nan=True)
    assert result["reprojected_target"] is False


def test_common_grid_reprojection_removes_fractional_source_phase():
    def world_signal(x, y):
        return (
            np.sin(x / 70.0)
            + 0.7 * np.cos(y / 55.0)
            + 0.2 * np.sin((x + y) / 31.0)
        )

    ref_transform = Affine(14.0, 0.0, 1000.0, 0.0, -14.0, 5000.0)
    tgt_transform = Affine(
        14.0, 0.0, 1000.0 + 14.0 * 0.35,
        0.0, -14.0, 5000.0 - 14.0 * 0.40,
    )
    yy, xx = np.mgrid[0:128, 0:128]
    ref_x, ref_y = ref_transform * (xx + 0.5, yy + 0.5)
    tgt_x, tgt_y = tgt_transform * (xx + 0.5, yy + 0.5)
    arr_ref = world_signal(ref_x, ref_y)
    arr_tgt = world_signal(tgt_x, tgt_y)

    result = build_reference_common_grid_overlap(
        arr_ref,
        ref_transform,
        arr_tgt,
        tgt_transform,
        "EPSG:4326",
        "EPSG:4326",
    )

    value = masked_ncc(
        result["ref_overlap"],
        result["tgt_on_ref_grid"],
        result["common_valid"],
    )
    assert result["available"] is True
    assert result["reprojected_target"] is True
    assert value is not None
    assert value > 0.95


def test_common_grid_identity_has_high_zero_shift_ncc():
    yy, xx = np.mgrid[0:512, 0:512]
    arr = np.sin(xx / 15.0) + np.cos(yy / 19.0)
    result = collect_common_grid_candidate_diagnostics(
        arr, arr.copy(), np.ones(arr.shape, dtype=bool), block_size=512)

    measured = [
        row for row in result["candidates"]
        if row["confidence"] is not None
    ]

    assert result["available"] is True
    assert measured
    row = measured[0]
    assert row["zero_shift_ncc"] > 0.99
    assert abs(row["shift_dx_pixels"]) < 0.1
    assert abs(row["shift_dy_pixels"]) < 0.1
    assert abs(row["ncc_gain"]) < 0.05


def test_phase_correlation_return_is_a_correction_for_target():
    yy, xx = np.mgrid[0:256, 0:256]
    ref = np.sin(xx / 11.0) + np.cos(yy / 17.0)
    tgt = ndimage_shift(ref, [2.0, -3.0], order=1, mode="nearest")
    valid = np.ones(ref.shape, dtype=bool)
    shift_y, shift_x, _ = phase_correlation(
        ref, tgt, valid_ref=valid, valid_tgt=valid)

    before = np.mean((ref[8:-8, 8:-8] - tgt[8:-8, 8:-8]) ** 2)
    corrected = ndimage_shift(
        tgt, [shift_y, shift_x], order=1, mode="nearest")
    after = np.mean((ref[8:-8, 8:-8] - corrected[8:-8, 8:-8]) ** 2)

    assert after < before


def test_common_grid_recovers_known_small_shift_and_improves_ncc():
    yy, xx = np.mgrid[0:512, 0:512]
    ref = np.sin(xx / 15.0) + np.cos(yy / 19.0)
    tgt = ndimage_shift(ref, [2.0, -3.0], order=1, mode="nearest")
    common_valid = np.ones(ref.shape, dtype=bool)
    result = collect_common_grid_candidate_diagnostics(
        ref, tgt, common_valid, block_size=256)

    measured = [
        row for row in result["candidates"]
        if row["confidence"] is not None
    ]

    assert measured
    assert any(abs(row["shift_dy_pixels"] + 2.0) < 0.6 for row in measured)
    assert any(abs(row["shift_dx_pixels"] - 3.0) < 0.6 for row in measured)
    accepted = [
        row for row in measured if row["reject_reason"] == "accepted"
    ]
    assert accepted
    assert all(row["best_shift_ncc"] > row["zero_shift_ncc"] for row in accepted)
