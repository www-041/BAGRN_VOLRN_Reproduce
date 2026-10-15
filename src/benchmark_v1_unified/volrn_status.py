"""Status adapter for VOLRN scientific state versus execution failures."""
from __future__ import annotations


def classify_volrn_execution(diagnostics, *, exception=None):
    """Map existing solver diagnostics into explicit non-success states.

    This adapter does not alter coefficients or solver behavior. It prevents
    empty support, finite nonconvergence, and failures from being represented
    as strict scientific convergence.
    """
    if exception is not None:
        return "RESOURCE_STOPPED" if isinstance(exception, (MemoryError, TimeoutError)) else "FAILED"
    diagnostics = diagnostics or {}
    if int(diagnostics.get("n_pairs", -1)) == 0 or int(diagnostics.get("n_blocks", -1)) == 0:
        return "NO_VALID_BLOCK_PAIRS"
    bands = diagnostics.get("band_solver_diagnostics") or []
    if not bool(diagnostics.get("finite_state", True)):
        return "NUMERICAL_INVALID"
    if any(not bool(band.get("finite_state", True)) or bool(band.get("cg_failed", False)) for band in bands):
        return "NUMERICAL_INVALID"
    if bool(diagnostics.get("strict_admm_converged", diagnostics.get("all_converged", False))):
        return "STRICT_ADMM_CONVERGED"
    if diagnostics.get("n_pairs", -1) < 0 and not bands:
        return "FAILED"
    return "COMPLETED_FINITE_NONCONVERGED"


def standardize_task16_status(status):
    value = str(status or "").upper()
    if value in {"PASS_STRICT_CONVERGED", "STRICT_ADMM_CONVERGED"}:
        return "STRICT_ADMM_CONVERGED"
    if value == "COMPLETED_FINITE_NONCONVERGED":
        return value
    if value.startswith("NUMERICAL_INVALID") or value == "ITERATION_COUNT_MISMATCH":
        return "NUMERICAL_INVALID"
    if value == "RESOURCE_STOPPED":
        return value
    if value in {"NO_VALID_BLOCK_PAIRS", "NO_VALID_OVERLAP", "NO_VALID_PAIRS"}:
        return "NO_VALID_BLOCK_PAIRS"
    return "FAILED"


def task16_exit_code(status):
    return 0 if standardize_task16_status(status) == "STRICT_ADMM_CONVERGED" else 2
