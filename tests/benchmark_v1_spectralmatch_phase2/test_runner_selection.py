import pytest

from scripts.benchmark.run_spectralmatch_phase2 import validate_pair_selection


def test_runner_accepts_only_the_two_predeclared_pairs():
    assert validate_pair_selection(["00_01", "00_10"]) == ["00_01", "00_10"]
    with pytest.raises(ValueError, match="unsupported pair"):
        validate_pair_selection(["00_02"])
    with pytest.raises(ValueError, match="duplicate"):
        validate_pair_selection(["00_01", "00_01"])
