import numpy as np

from src.benchmark_v1_spectralmatch_phase2.coverage_seams import (
    assign_complete_coverage,
    validate_assignment,
)


def test_assignment_preserves_official_valid_labels_and_fills_only_union():
    valid_i = np.array([[1, 1, 0, 0], [0, 1, 0, 0]], dtype=bool)
    valid_j = np.array([[0, 1, 1, 0], [0, 1, 1, 0]], dtype=bool)
    official = np.array([[0, -1, 1, -1], [-1, 1, 1, -1]], dtype=np.int8)

    labels, audit = assign_complete_coverage(valid_i, valid_j, official)

    assert labels.tolist() == [[0, 0, 1, -1], [-1, 1, 1, -1]]
    assert audit["official_assignments"] == 4
    assert audit["fallback_assignments"] == 1
    assert audit["source_invalid_pixels"] == 0
    assert validate_assignment(labels, valid_i, valid_j, valid_i | valid_j)["status"] == "PASS"


def test_invalid_official_source_is_replaced_by_valid_source():
    valid_i = np.array([[True, False]])
    valid_j = np.array([[False, True]])
    official = np.array([[1, 0]], dtype=np.int8)

    labels, audit = assign_complete_coverage(valid_i, valid_j, official)

    assert labels.tolist() == [[0, 1]]
    assert audit["official_source_invalid_pixels"] == 2
    assert audit["fallback_assignments"] == 2
    assert validate_assignment(labels, valid_i, valid_j, valid_i | valid_j)["status"] == "PASS"


def test_no_source_cell_stays_unassigned_and_cannot_be_claimed_as_support():
    valid_i = np.array([[True, False]])
    valid_j = np.array([[False, False]])
    official = np.array([[-1, -1]], dtype=np.int8)
    labels, audit = assign_complete_coverage(valid_i, valid_j, official)

    assert labels.tolist() == [[0, -1]]
    assert audit["no_source_pixels"] == 0
    invalid_expectation = validate_assignment(labels, valid_i, valid_j, np.array([[True, True]]))
    assert invalid_expectation["status"] == "FAIL"
    assert invalid_expectation["missing_pixels"] == 1


def test_unexpected_shape_mismatch_is_rejected():
    with np.testing.assert_raises_regex(ValueError, "same 2D shape"):
        assign_complete_coverage(np.zeros((2, 2), bool), np.zeros((1, 2), bool), np.zeros((2, 2)))
