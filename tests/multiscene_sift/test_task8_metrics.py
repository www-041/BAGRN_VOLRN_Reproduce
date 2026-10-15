"""Task10C metric evaluator regressions independent of paper-formula claims."""

import numpy as np

from src.metrics import _gradient_support_mask, compute_gl
from src.multiscene_sift.radiometric_runner import _pair_metrics


def test_radiometric_histogram_diagnostics_read_after_arrays():
    before = [np.zeros((1, 4, 4)), np.zeros((1, 4, 4))]
    after = [np.zeros((1, 4, 4)), np.ones((1, 4, 4))]
    valid = [np.ones((4, 4), dtype=bool), np.ones((4, 4), dtype=bool)]

    result = _pair_metrics(before, after, valid)
    histogram = result["pairs"][0]["histogram"]

    assert histogram["mean_abs_difference"] > 0.0
    assert histogram["histogram_tv_distance"] > 0.0


def test_gl_ignores_gradient_stencil_touching_nodata_boundary():
    before = np.full((1, 8, 8), 10.0)
    after = before.copy()
    after[0, :, 0] = -9999.0

    gl = compute_gl([before], [after], [-9999.0], bands=[0])

    assert gl == 0.0


def test_gradient_support_erodes_pixels_adjacent_to_invalid_boundary():
    valid = np.ones((8, 8), dtype=bool)
    valid[:, 0] = False

    support = _gradient_support_mask(valid)

    assert not support[:, 1].any()
    assert support[2:6, 3:6].all()
