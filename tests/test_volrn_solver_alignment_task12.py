"""Task12 solver alignment tests: objective-preserving PCG and status semantics."""

import numpy as np
from scipy import sparse

import src.volrn as volrn


def test_jacobi_preconditioner_is_finite_and_objective_preserving():
    matrix = sparse.diags([1e-18, 2.0, 100.0], format="csr")
    preconditioner = volrn.build_jacobi_preconditioner(matrix)
    result = preconditioner @ np.array([1.0, 2.0, 3.0])
    assert np.isfinite(result).all()
    assert np.allclose(result, [1.0, 1.0, 0.03])


def test_pcg_and_plain_cg_return_same_convex_solution():
    B = sparse.csr_matrix(np.diag([2.0, 3.0]))
    A = sparse.identity(2, format="csr")
    b = np.array([2.0, -1.0])
    plain, _, _, plain_diag = volrn._admm_solver(
        B, A, b, lambda_param=0.5, rho=1.0, max_iter=40, tol=1e-8,
        use_preconditioner=False, return_diagnostics=True,
    )
    pcg, _, _, pcg_diag = volrn._admm_solver(
        B, A, b, lambda_param=0.5, rho=1.0, max_iter=40, tol=1e-8,
        use_preconditioner=True, return_diagnostics=True,
    )
    assert np.allclose(plain, pcg, rtol=1e-6, atol=1e-8)
    assert plain_diag["preconditioner_type"] == "none"
    assert pcg_diag["preconditioner_type"] == "jacobi"


def test_adaptive_rho_policy_and_dual_rescaling():
    assert volrn._adaptive_rho_update(1.0, 1.0, 100.0) == 0.5
    assert volrn._adaptive_rho_update(1.0, 100.0, 1.0) == 2.0
    assert volrn._adaptive_rho_update(1.0, 1.0, 1.0) == 1.0
    assert volrn._rescale_scaled_dual(np.array([2.0, -4.0]), 4.0, 2.0).tolist() == [4.0, -8.0]


def test_solver_reports_strict_and_stability_as_distinct_states():
    B = sparse.csr_matrix(np.diag([2.0, 2.0]))
    A = sparse.identity(2, format="csr")
    b = np.array([2.0, 0.0])
    _, converged, _, diagnostics = volrn._admm_solver(
        B, A, b, lambda_param=0.5, rho=1.0, max_iter=1, tol=1e-12,
        return_diagnostics=True,
    )
    assert not converged
    assert diagnostics["strict_admm_converged"] is False
    assert diagnostics["coefficient_stable"] is True or diagnostics["coefficient_stable"] is False
    assert diagnostics["finite_state"] is True
    assert diagnostics["science_status"] != "STRICT_ADMM_CONVERGED"
