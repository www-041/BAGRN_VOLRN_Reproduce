"""Task10C VOLRN solver invariants.

These tests intentionally exercise the solver directly so a non-converged
solution cannot be hidden by the public image wrapper.
"""

import numpy as np
from scipy import sparse

import src.volrn as volrn


def test_max_iter_returns_last_finite_iterate_and_diagnostics():
    B = sparse.csr_matrix(np.diag([2.0, 2.0]))
    A = sparse.identity(2, format="csr")
    b = np.array([2.0, 0.0])

    x, converged, iterations, diagnostics = volrn._admm_solver(
        B,
        A,
        b,
        lambda_param=0.5,
        rho=1.0,
        max_iter=1,
        tol=1e-12,
        return_diagnostics=True,
    )

    assert not converged
    assert iterations == 1
    assert np.isfinite(x).all()
    assert not np.allclose(x, np.array([1.0, 0.0]))
    for key in (
        "iterations",
        "primal_residual",
        "dual_residual",
        "objective",
        "x_relative_change",
        "cg_status_history",
    ):
        assert key in diagnostics


def test_x_change_alone_cannot_declare_admm_convergence(monkeypatch):
    B = sparse.csr_matrix((1, 2))
    A = sparse.csr_matrix([[1.0, 0.0]])
    b = np.array([10.0])

    def frozen_cg(matrix, rhs, x0, **kwargs):
        return np.asarray(x0, dtype=float).copy(), 0

    monkeypatch.setattr(volrn, "cg", frozen_cg)
    _, converged, _, diagnostics = volrn._admm_solver(
        B,
        A,
        b,
        lambda_param=0.5,
        rho=1.0,
        max_iter=1,
        tol=1e-12,
        return_diagnostics=True,
    )

    assert not converged
    assert diagnostics["x_relative_change"] == 0.0
    assert diagnostics["primal_residual"] > diagnostics["primal_tolerance"]


def test_cg_failure_is_propagated_and_not_science_pass(monkeypatch):
    B = sparse.csr_matrix([[1.0, 0.0]])
    A = sparse.csr_matrix([[1.0, 0.0]])
    b = np.array([2.0])

    def failed_cg(matrix, rhs, x0, **kwargs):
        return np.asarray(x0, dtype=float).copy(), 7

    monkeypatch.setattr(volrn, "cg", failed_cg)
    x, converged, _, diagnostics = volrn._admm_solver(
        B,
        A,
        b,
        lambda_param=0.5,
        rho=1.0,
        max_iter=2,
        tol=1e-4,
        return_diagnostics=True,
    )

    assert np.isfinite(x).all()
    assert not converged
    assert diagnostics["cg_failed"] is True
    assert diagnostics["science_pass"] is False
    assert diagnostics["cg_status_history"] == [7, 7]


def test_volrn_solver_uses_original_units_without_percentile_rescaling(monkeypatch):
    from tests.test_volrn import _make_test_pair

    arrays, transforms, bounds, nodata = _make_test_pair()
    captured_mu = []
    original_build = volrn._build_volrn_system

    def capture_build(blocks, pairs, band_idx):
        captured_mu.append(float(blocks[0].mu[band_idx]))
        return original_build(blocks, pairs, band_idx)

    def immediate_solver(*args, **kwargs):
        B = args[0]
        x = np.zeros(B.shape[1], dtype=float)
        x[0::2] = 1.0
        diagnostics = {
            "iterations": 0,
            "primal_residual": 0.0,
            "dual_residual": 0.0,
            "objective": 0.0,
            "x_relative_change": 0.0,
            "cg_status_history": [],
            "cg_failed": False,
            "science_pass": True,
        }
        if kwargs.get("return_diagnostics"):
            return x, True, 0, diagnostics
        return x, True, 0

    monkeypatch.setattr(volrn, "_build_volrn_system", capture_build)
    monkeypatch.setattr(volrn, "_admm_solver", immediate_solver)
    volrn.volrn_normalize(
        arrays,
        transforms,
        bounds,
        nodata,
        block_size_pixels=4,
        max_iter=1,
    )

    assert captured_mu
    assert captured_mu[0] > 10.0
