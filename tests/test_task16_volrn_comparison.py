from __future__ import annotations

import numpy as np
import pytest

from src.task16_volrn_comparison import (
    TASK16_PARAMS,
    classify_volrn_status,
    compare_metric_values,
    validate_frozen_grid,
)


def test_task16_parameters_are_frozen():
    assert TASK16_PARAMS == {
        "block_size_pixels": 400,
        "lambda": 0.5,
        "rho": 1.0,
        "max_iter": 200,
        "tol": 1e-4,
    }


def test_finite_iter200_nonconverged_is_distinguished_from_invalid():
    finite = {"finite_state": True, "cg_failed": False, "iterations": 200, "strict_admm_converged": False}
    assert classify_volrn_status(finite) == "COMPLETED_FINITE_NONCONVERGED"
    invalid = {**finite, "finite_state": False}
    assert classify_volrn_status(invalid) == "NUMERICAL_INVALID_ITER200"
    converged = {**finite, "strict_admm_converged": True}
    assert classify_volrn_status(converged) == "PASS_STRICT_CONVERGED"


def test_nonfinite_or_cg_failure_is_invalid_even_if_iteration_count_is_200():
    assert classify_volrn_status({"finite_state": True, "cg_failed": True, "iterations": 200}) == "NUMERICAL_INVALID_ITER200"
    assert classify_volrn_status({"finite_state": True, "cg_failed": False, "iterations": 199}) == "ITERATION_COUNT_MISMATCH"


def test_frozen_grid_rejects_any_identity_change():
    grid = {
        "crs": "EPSG:32650",
        "width": 4,
        "height": 3,
        "resolution": 14.0,
        "transform": [14.0, 0.0, 100.0, 0.0, -14.0, 200.0],
    }
    assert validate_frozen_grid(grid, grid) is True
    changed = {**grid, "width": 5}
    with pytest.raises(ValueError, match="HARD_STOP_PROVENANCE_MISMATCH"):
        validate_frozen_grid(grid, changed)


def test_compare_metric_values_reports_absolute_and_relative_change():
    row = compare_metric_values("MAMD", 10.0, 8.0)
    assert row["absolute_change"] == -2.0
    assert row["relative_change_percent"] == -20.0
    missing = compare_metric_values("MAMD", None, 8.0)
    assert missing["absolute_change"] is None
    assert missing["relative_change_percent"] is None
