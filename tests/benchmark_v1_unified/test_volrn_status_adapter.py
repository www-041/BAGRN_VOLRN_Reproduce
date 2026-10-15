from src.benchmark_v1_unified.volrn_status import classify_volrn_execution, task16_exit_code


def test_volrn_statuses_separate_convergence_empty_support_numeric_failure_and_exception():
    assert classify_volrn_execution({"n_pairs": 0, "n_blocks": 0}) == "NO_VALID_BLOCK_PAIRS"
    assert classify_volrn_execution({"n_pairs": 2, "finite_state": True,
                                     "strict_admm_converged": True}) == "STRICT_ADMM_CONVERGED"
    assert classify_volrn_execution({"n_pairs": 2, "finite_state": True,
                                     "strict_admm_converged": False}) == "COMPLETED_FINITE_NONCONVERGED"
    assert classify_volrn_execution({"n_pairs": 2, "finite_state": False}) == "NUMERICAL_INVALID"


def test_volrn_resource_stop_and_generic_failure_are_distinct():
    assert classify_volrn_execution({}, exception=MemoryError("allocation limit")) == "RESOURCE_STOPPED"
    assert classify_volrn_execution({}, exception=RuntimeError("unexpected")) == "FAILED"


def test_task16_exit_code_does_not_call_finite_nonconvergence_success():
    assert task16_exit_code("STRICT_ADMM_CONVERGED") == 0
    for status in ("COMPLETED_FINITE_NONCONVERGED", "NO_VALID_BLOCK_PAIRS", "NUMERICAL_INVALID",
                   "RESOURCE_STOPPED", "FAILED"):
        assert task16_exit_code(status) != 0
