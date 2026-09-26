"""Hard-fail contracts for invalid shared affine geometry inputs."""

from __future__ import annotations

import numpy as np

from src.registration_benchmark.geometry import (
    STATUS_INVALID_GEOMETRY,
    fit_affine_ransac,
)
from src.registration_benchmark.models import MatchSet


def _matches(ref_xy: np.ndarray, tgt_xy: np.ndarray) -> MatchSet:
    return MatchSet(
        method="synthetic",
        ref_xy=ref_xy,
        tgt_xy=tgt_xy,
        confidence=np.ones(len(ref_xy), dtype=np.float64),
        runtime_sec=0.0,
    )


def test_collinear_affine_points_return_structured_failure():
    x = np.arange(12, dtype=np.float64)
    points = np.column_stack((x, 2.0 * x + 1.0))

    result = fit_affine_ransac(_matches(points, points))

    assert result.status == STATUS_INVALID_GEOMETRY
    assert result.model is None
    assert result.inlier_mask.shape == (12,)


def test_nonfinite_affine_points_return_structured_failure():
    ref = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], dtype=np.float64)
    tgt = ref.copy()
    tgt[1, 0] = np.nan

    result = fit_affine_ransac(_matches(ref, tgt))

    assert result.status == STATUS_INVALID_GEOMETRY
    assert result.model is None
    assert result.n_raw == 3
