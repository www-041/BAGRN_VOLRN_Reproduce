"""Synthetic behavior tests for Task13A's symmetric seam-local correction."""

import numpy as np

from src.seam_local.local_moment import (
    apply_seam_local_correction,
    estimate_seam_segment_moments,
)
from src.seam_local.seam import SeamResult


def _vertical_seam(height: int, column: int) -> SeamResult:
    return SeamResult(
        orientation="vertical",
        row_col_path=np.column_stack((np.arange(height), np.full(height, column))),
        total_cost=0.0,
        mean_cost=0.0,
        p95_cost=0.0,
        status="OK",
    )


def _affine_pair() -> tuple[np.ndarray, np.ndarray, np.ndarray, SeamResult]:
    row, col = np.indices((64, 64))
    a = 30.0 + 0.8 * row + 0.4 * col + 2.0 * np.sin(row / 3)
    b = 1.5 * a + 10.0
    return a, b, np.ones(a.shape, dtype=bool), _vertical_seam(64, 32)


def test_symmetric_local_correction_reduces_centerline_difference():
    """Catches an affine correction that moves only one source or misses the seam."""
    a, b, valid, seam = _affine_pair()
    result = apply_seam_local_correction(
        a, b, valid, valid, seam,
        half_width=8, segment_length=32, min_valid_pairs=100,
    )
    assert result.status == "OK"
    assert np.mean(np.abs(result.corrected_a[:, 32] - result.corrected_b[:, 32])) < 1e-10
    assert np.mean(np.abs(result.corrected_a[:, 32] - result.corrected_b[:, 32])) < np.mean(np.abs(a[:, 32] - b[:, 32]))
    assert np.any(np.abs(result.corrected_a[:, 32] - a[:, 32]) > 1)
    assert np.any(np.abs(result.corrected_b[:, 32] - b[:, 32]) > 1)


def test_taper_reaches_identity_at_boundary_and_outside():
    """Catches an abrupt corridor edge or correction leaked outside its width."""
    a, b, valid, seam = _affine_pair()
    result = apply_seam_local_correction(
        a, b, valid, valid, seam,
        half_width=8, segment_length=32, min_valid_pairs=100,
    )
    np.testing.assert_array_equal(result.corrected_a[:, :24], a[:, :24])
    np.testing.assert_array_equal(result.corrected_b[:, 40:], b[:, 40:])
    np.testing.assert_allclose(result.corrected_a[:, 24], a[:, 24], atol=1e-12)
    np.testing.assert_allclose(result.corrected_b[:, 40], b[:, 40], atol=1e-12)
    assert np.mean(np.abs(result.corrected_a[:, 31] - a[:, 31])) > np.mean(np.abs(result.corrected_a[:, 25] - a[:, 25]))


def test_swapping_sources_swaps_corrected_outputs():
    """Catches asymmetric target moments or scene-index-dependent behavior."""
    a, b, valid, seam = _affine_pair()
    ab = apply_seam_local_correction(a, b, valid, valid, seam, half_width=8, segment_length=32, min_valid_pairs=100)
    ba = apply_seam_local_correction(b, a, valid, valid, seam, half_width=8, segment_length=32, min_valid_pairs=100)
    np.testing.assert_allclose(ab.corrected_a, ba.corrected_b)
    np.testing.assert_allclose(ab.corrected_b, ba.corrected_a)


def test_degenerate_segment_borrows_nearest_valid_coefficients():
    """Catches division by zero or identity fallback for a local flat segment."""
    a, b, valid, seam = _affine_pair()
    a[32:] = 20.0
    b[32:] = 40.0
    estimate = estimate_seam_segment_moments(
        a, b, valid, valid, seam,
        half_width=8, segment_length=32, min_valid_pairs=100,
    )
    assert estimate.status == "OK"
    assert len(estimate.segments) >= 2
    assert any(segment.fallback_from is not None for segment in estimate.segments)
    assert all(np.isfinite(segment.a_a) and np.isfinite(segment.a_b) for segment in estimate.segments)


def test_all_degenerate_segments_are_explicitly_unavailable():
    """Catches fabricated correction coefficients when no segment is solvable."""
    a = np.full((64, 64), 20.0)
    b = np.full((64, 64), 40.0)
    valid = np.ones_like(a, dtype=bool)
    result = apply_seam_local_correction(a, b, valid, valid, _vertical_seam(64, 32), half_width=8, segment_length=32, min_valid_pairs=100)
    assert result.status == "LOCAL_MOMENT_UNAVAILABLE"
    np.testing.assert_array_equal(result.corrected_a, a)
    np.testing.assert_array_equal(result.corrected_b, b)


def test_nonfinite_unpaired_pixels_cannot_poison_coefficients():
    """Catches NaN propagation from samples excluded by the paired validity mask."""
    a, b, valid, seam = _affine_pair()
    a[0, 32] = np.nan
    b[1, 32] = np.inf
    result = apply_seam_local_correction(a, b, valid, valid, seam, half_width=8, segment_length=32, min_valid_pairs=100)
    assert result.status == "OK"
    joint_finite = np.isfinite(a) & np.isfinite(b)
    assert np.isfinite(result.corrected_a[joint_finite]).all()
    assert np.isfinite(result.corrected_b[joint_finite]).all()


def test_unstable_gain_is_reported_without_clipping():
    """Catches silent coefficient clipping when source contrast is far apart."""
    a, _, valid, seam = _affine_pair()
    b = 10.0 * a + 4.0
    result = apply_seam_local_correction(a, b, valid, valid, seam, half_width=8, segment_length=32, min_valid_pairs=100)
    assert result.status == "UNSTABLE_LOCAL_GAIN"
    assert result.gain_max > 2.0
    # Symmetric target sigma makes the high-contrast source gain 0.55 here.
    assert np.isclose(result.gain_min, 0.55)


def test_offset_extrema_are_reported_for_both_sources():
    """Catches dropped offset diagnostics while keeping symmetric coefficients."""
    a, b, valid, seam = _affine_pair()
    estimate = estimate_seam_segment_moments(a, b, valid, valid, seam, half_width=8, segment_length=32, min_valid_pairs=100)
    result = apply_seam_local_correction(a, b, valid, valid, seam, half_width=8, segment_length=32, min_valid_pairs=100)
    for report in (estimate, result):
        assert np.isclose(report.b_min, -10 / 3)
        assert np.isclose(report.b_max, 5.0)


def test_finite_input_overflow_returns_original_pair_with_explicit_status():
    """Catches Inf output when a finite outlier overflows its local affine map."""
    a, b, valid, seam = _affine_pair()
    a[0, 32] = np.finfo(np.float64).max
    b = 10.0 * np.where(a == np.finfo(np.float64).max, 100.0, a) + 4.0
    result = apply_seam_local_correction(a, b, valid, valid, seam, half_width=8, segment_length=32, min_valid_pairs=100)
    assert result.status == "NUMERICAL_INVALID"
    np.testing.assert_array_equal(result.corrected_a, a)
    np.testing.assert_array_equal(result.corrected_b, b)
    assert np.isfinite(result.corrected_a).all()
    assert np.isfinite(result.corrected_b).all()
