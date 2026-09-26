"""Task5 tests for statistically valid BAGRN overlap observations."""

from __future__ import annotations

import numpy as np
import pytest

from src.bagrn import bagrn_normalize


def _overlap(i, j, width=4):
    return {
        "idx_i": i,
        "idx_j": j,
        "window_i": (0, 4, 0, width),
        "window_j": (0, 4, 0, width),
    }


def test_empty_valid_overlap_is_not_solved_as_zero_moments():
    arrays = [
        np.full((1, 4, 4), 10.0, dtype=np.float32),
        np.zeros((1, 4, 4), dtype=np.float32),
    ]

    with pytest.raises(ValueError, match="BAGRN_NO_VALID_OVERLAP"):
        bagrn_normalize(
            arrays,
            [None, 0.0],
            [_overlap(0, 1)],
            control_idx=0,
        )


def test_invalid_overlap_filtering_cannot_hide_a_disconnected_network():
    arrays = [
        np.full((1, 4, 4), 10.0, dtype=np.float32),
        np.full((1, 4, 4), 20.0, dtype=np.float32),
        np.zeros((1, 4, 4), dtype=np.float32),
    ]
    overlaps = [_overlap(0, 1), _overlap(1, 2)]

    with pytest.raises(ValueError, match="BAGRN_RADIO_NETWORK_DISCONNECTED"):
        bagrn_normalize(arrays, [None, None, 0.0], overlaps, control_idx=0)


def test_audit_records_geometric_and_clear_valid_pixel_counts():
    arrays = [
        np.full((1, 4, 4), 10.0, dtype=np.float32),
        np.full((1, 4, 4), 20.0, dtype=np.float32),
        np.full((1, 4, 4), 30.0, dtype=np.float32),
    ]
    overlaps = [_overlap(0, 1, width=2), _overlap(1, 2, width=4)]

    bagrn_normalize(arrays, [None, None, None], overlaps, control_idx=0)

    assert overlaps[0]["geometric_pixel_count"] == 8
    assert overlaps[1]["geometric_pixel_count"] == 16
    assert overlaps[0]["clear_valid_pixel_count_by_band"]["0"] == [8, 8]
    assert overlaps[1]["clear_valid_pixel_count_by_band"]["0"] == [16, 16]
