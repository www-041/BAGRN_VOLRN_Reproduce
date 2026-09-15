import numpy as np
from rasterio.transform import Affine

from src.registration_diagnostics import (
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
