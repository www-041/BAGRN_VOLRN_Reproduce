"""Tests for the shared memory-safe affine estimator."""

from __future__ import annotations

import numpy as np
import pytest

from src.registration_benchmark.geometry import estimate_affine_lstsq


def _apply(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack([points, np.ones(len(points))])
    return (matrix @ homogeneous.T).T[:, :2]


def test_exact_affine_is_recovered_from_non_collinear_points():
    matrix = np.array(
        [
            [1.002, -0.015, 12.5],
            [0.010, 0.998, -7.25],
            [0.0, 0.0, 1.0],
        ]
    )
    src = np.column_stack(
        [np.linspace(-200.0, 200.0, 100), np.sin(np.linspace(-3.0, 3.0, 100)) * 80]
    )
    dst = _apply(matrix, src)

    estimated = estimate_affine_lstsq(src, dst)

    np.testing.assert_allclose(estimated, matrix, rtol=0, atol=1e-9)


def test_noisy_affine_estimate_is_numerically_stable():
    rng = np.random.default_rng(123)
    matrix = np.array(
        [
            [1.002, -0.015, 12.5],
            [0.010, 0.998, -7.25],
            [0.0, 0.0, 1.0],
        ]
    )
    src = rng.uniform(-500.0, 500.0, size=(500, 2))
    dst = _apply(matrix, src) + rng.normal(0.0, 0.05, size=(500, 2))

    estimated = estimate_affine_lstsq(src, dst)

    np.testing.assert_allclose(estimated, matrix, rtol=0, atol=2e-3)


@pytest.mark.parametrize(
    "src,dst,match",
    [
        (np.zeros((3, 2)), np.zeros((2, 2)), "same shape"),
        (np.zeros((3, 3)), np.zeros((3, 2)), "shape (N, 2)"),
        (np.zeros((2, 2)), np.zeros((2, 2)), "at least 3"),
        (np.array([[0.0, 0.0], [1.0, 0.0], [np.nan, 1.0]]), np.zeros((3, 2)), "finite"),
        (np.zeros((3, 2)), np.array([[0.0, 0.0], [1.0, 0.0], [np.inf, 1.0]]), "finite"),
        (np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]), np.zeros((3, 2)), "rank"),
    ],
)
def test_invalid_affine_inputs_fail_with_controlled_value_error(src, dst, match):
    with pytest.raises(ValueError, match=match.replace("(", "\\(").replace(")", "\\)")):
        estimate_affine_lstsq(src, dst)
