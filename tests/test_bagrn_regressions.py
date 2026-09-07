"""Regression tests for BAGRN valid-pixel handling and input validation.

Task 3 of reliability-fixes plan:
- Explicit NoData masks must also exclude NaN/Inf.
- Whole-image moments must use the same valid-pixel semantics.
- Validate inputs: control_idx, list lengths, 3-D input, common band count.
- Single-image input returns identity.
"""

import numpy as np
import pytest

from src.bagrn import bagrn_normalize


def test_bagrn_explicit_nodata_also_excludes_nan_inf():
    a = np.array([[[10.0, 20.0], [-9999.0, np.nan]]])
    b = np.array([[[12.0, 22.0], [-9999.0, 30.0]]])

    ov = [{
        "idx_i": 0,
        "idx_j": 1,
        "window_i": (0, 2, 0, 2),
        "window_j": (0, 2, 0, 2),
    }]

    result, theta_mu, theta_sigma = bagrn_normalize(
        [a, b], [-9999.0, -9999.0], ov, control_idx=0
    )

    assert np.isfinite(theta_mu).all()
    assert np.isfinite(theta_sigma).all()
    assert result[0][0, 1, 0] == -9999.0
    assert np.isnan(result[0][0, 1, 1])


def test_bagrn_rejects_bad_control_index():
    arr = np.ones((1, 4, 4))
    with pytest.raises(ValueError):
        bagrn_normalize([arr, arr], [None, None], [], control_idx=2)


def test_bagrn_single_image_is_identity():
    arr = np.arange(16, dtype=float).reshape(1, 4, 4)
    out, theta_mu, theta_sigma = bagrn_normalize([arr], [None], [], 0)
    assert np.allclose(out[0], arr)
    assert theta_mu.shape == (1, 1)
    assert theta_sigma.shape == (1, 1)
