import numpy as np


def test_validation_size_selection_uses_geometry_not_residual_quality():
    from src.coregistration import select_validation_block_size

    common = np.ones((96, 160), dtype=bool)
    holdout = np.zeros_like(common)
    holdout[0:64, 0:128] = True

    result = select_validation_block_size(
        common, holdout, [64, 32], required_candidate_count=2,
        step=32, offset_row=0, offset_col=0,
    )

    assert result["selected_block_size"] == 64
    assert result["candidate_counts"][64] >= 2
    assert result["candidate_counts"][32] > result["candidate_counts"][64]


def test_narrow_overlap_can_fall_back_from_384_to_256_or_192():
    from src.coregistration import select_validation_block_size

    common = np.ones((512, 512), dtype=bool)
    holdout = np.zeros_like(common)
    holdout[64:320, 64:320] = True

    result = select_validation_block_size(
        common, holdout, [384, 256, 192], required_candidate_count=2,
        step=64, offset_row=0, offset_col=0,
    )

    assert result["selected_block_size"] == 192
    assert result["candidate_counts"][384] == 0
    assert result["candidate_counts"][256] == 1
    assert result["candidate_counts"][192] >= 2


def test_validation_reports_insufficient_geometry_when_no_candidate_size_works():
    from src.coregistration import select_validation_block_size

    common = np.ones((128, 128), dtype=bool)
    holdout = np.zeros_like(common)
    holdout[32:96, 32:96] = True

    result = select_validation_block_size(
        common, holdout, [384, 256, 192], required_candidate_count=10,
        step=32, offset_row=0, offset_col=0,
    )

    assert result["selected_block_size"] is None
    assert "insufficient" in result["failure_reason"]
