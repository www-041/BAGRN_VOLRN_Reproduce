import pytest

from scripts.benchmark.run_spectralmatch_phase2 import can_start_worker, validate_pair_selection


def test_runner_accepts_only_the_two_predeclared_pairs():
    assert validate_pair_selection(["00_01", "00_10"]) == ["00_01", "00_10"]
    with pytest.raises(ValueError, match="unsupported pair"):
        validate_pair_selection(["00_02"])
    with pytest.raises(ValueError, match="duplicate"):
        validate_pair_selection(["00_01", "00_01"])


def test_runner_refuses_worker_when_environment_or_monitor_is_unavailable():
    assert can_start_worker({"status": "COMPATIBLE"}, monitor_available=True) is True
    assert can_start_worker({"status": "INCOMPATIBLE"}, monitor_available=True) is False
    assert can_start_worker({"status": "COMPATIBLE"}, monitor_available=False) is False
