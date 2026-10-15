import numpy as np
import pytest

from src.benchmark_v1_lsstaa.core import (
    _matlab_int16,
    _matlab_hist_counts,
    absolute_day_offsets,
    apply_coefficients,
    choose_control,
    internal_bias_compensation,
    matlab_quantile,
    pif_from_series,
    select_pif_subset,
    solve_joint,
)


def test_matlab_int16_rounds_ties_away_and_saturates():
    values = np.array([-40000, -2.5, -1.5, -.5, .5, 1.5, 2.5, 40000])
    assert _matlab_int16(values).tolist() == [-32768, -3, -2, -1, 1, 2, 3, 32767]


def test_bias_compensation_retains_official_single_storage():
    base = np.arange(1600, dtype=np.float32).reshape(40, 40) + np.float32(1000.125)
    series = np.stack([base, base + np.float32(.03125)], axis=2)
    corrected, _ = internal_bias_compensation(series, [0, 1])
    assert corrected.dtype == np.float32


def test_matlab_hist_assigns_exact_midpoint_to_left_bin():
    # MathWorks hist migration guide: edge equality belongs to left bin.
    counts = _matlab_hist_counts(np.array([0., .5, 1., 1.5, 2.]),
                                 np.array([0., 1., 2.]))
    assert counts.tolist() == [2, 2, 1]


def test_absolute_dates_do_not_wrap_at_new_year():
    assert absolute_day_offsets(["2025-12-31", "2026-01-02"]) == [0, 2]


def test_internal_bias_compensation_preserves_invalid_and_reduces_offset():
    base = (1000 + np.arange(1600, dtype=float).reshape(40, 40) % 71)
    a = base.copy()
    b = base + 20
    a[0, 0] = 0
    b[0, 0] = 0
    corrected, medians = internal_bias_compensation(np.stack([a, b], axis=2), [0, 2])
    assert medians == pytest.approx([-20, 20])
    assert corrected[0, 0, :].tolist() == [0, 0]
    assert np.median(corrected[1:, :, 0] - corrected[1:, :, 1]) == pytest.approx(0)


def test_control_uses_thresholded_weighted_r2_not_largest_raw_r2():
    rsq = np.array([[0, .95, .85], [.95, 0, .90], [.85, .90, 0]])
    gain = np.ones((3, 3))
    offset = np.zeros((3, 3))
    overlap = np.array([[0, .1, .1], [.1, 0, .7], [.1, .7, 0]])
    gain[1, 2] = gain[2, 1] = 1.5
    assert choose_control(rsq, gain, offset, overlap) == 0


def test_subset_samples_axis_grid_and_remains_finite():
    x = np.linspace(20, 120, 1000)
    y = 1.2 * x + 5
    xs, ys, r2 = select_pif_subset(x, y, bins=100)
    assert len(xs) > 10
    assert len(xs) <= 204
    assert np.all(np.isfinite(xs)) and np.all(np.isfinite(ys))
    assert r2 == pytest.approx(1)


def test_subset_quantile_range_precedes_zero_removal_as_in_official_source(monkeypatch):
    import src.benchmark_v1_lsstaa.core as core
    seen = []
    original = core.matlab_quantile
    def spy(values, q):
        seen.append(np.asarray(values).copy())
        return original(values, q)
    monkeypatch.setattr(core, "matlab_quantile", spy)
    x = np.r_[np.zeros(20), np.linspace(10, 100, 400)]
    y = np.r_[np.zeros(20), np.linspace(20, 200, 400)]
    select_pif_subset(x, y, bins=10)
    assert any(np.any(values == 0) for values in seen)


def test_joint_solution_recovers_known_gains_offsets():
    x = np.linspace(100, 400, 200)
    common = np.tile(x, (2, 1))
    observed = np.vstack([common[0], (common[1] - 5) / 1.1, (common[1] + 9) / .9])
    edges = [(0, 1, observed[0], observed[1]), (1, 2, observed[1], observed[2]), (0, 2, observed[0], observed[2])]
    result = solve_joint(3, 0, edges)
    assert result.gains == pytest.approx([1, 1.1, .9], abs=1e-8)
    assert result.offsets == pytest.approx([0, 5, -9], abs=1e-7)
    assert result.rank == 4


def test_joint_solution_rejects_component_without_control():
    x = np.linspace(10, 100, 100)
    with pytest.raises(ValueError, match="DISCONNECTED"):
        solve_joint(3, 0, [(1, 2, x, x)])


def test_application_is_float_and_preserves_invalid_zero():
    src = np.array([[0, 60000], [500, 1]], dtype=np.uint16)
    out = apply_coefficients(src, src != 0, 1.2, 3)
    assert out.dtype == np.float32
    assert np.isnan(out[0, 0])
    assert out[0, 1] == pytest.approx(72003)


def test_pif_fails_closed_on_empty_temporal_support():
    with pytest.raises(ValueError, match="EMPTY_OBSERVATIONS"):
        pif_from_series(np.zeros((40, 40, 3)), [0, 0, 6])


def test_matlab_midpoint_quantile_not_numpy_default_inclusive():
    assert matlab_quantile(np.array([1, 2, 3, 4, 5]), .25) == pytest.approx(1.75)
