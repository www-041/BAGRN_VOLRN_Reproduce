import numpy as np


def test_validation_windows_are_reserved_before_training():
    from src.coregistration import reserve_validation_windows

    common = np.ones((160, 320), dtype=bool)
    result = reserve_validation_windows(
        common, [64, 32], final_min_blocks=5, step=32,
        offset_row=0, offset_col=0, reservation_margin=0,
    )

    assert result["available"] is True
    assert result["reserved_count"] >= 5
    assert result["holdout_mask"].any()
    assert np.all(~(result["train_sampling_mask"] & result["holdout_region_mask"]))


def test_narrow_diagonal_overlap_can_reserve_five_windows_when_geometry_allows():
    from src.coregistration import reserve_validation_windows

    yy, xx = np.mgrid[:256, :768]
    common = np.abs(yy - (0.25 * xx + 24.0)) <= 16.0
    result = reserve_validation_windows(
        common, [64, 48, 32], final_min_blocks=5, step=16,
        offset_row=0, offset_col=0, min_common_valid_ratio=0.30,
        reservation_margin=0,
    )

    assert result["available"] is True
    assert result["reserved_count"] >= 5
    assert result["selected_block_size"] in {32, 48, 64}


def test_training_blocks_do_not_intersect_reserved_validation_windows():
    from src.coregistration import reserve_validation_windows

    common = np.ones((128, 256), dtype=bool)
    result = reserve_validation_windows(
        common, [32], final_min_blocks=4, step=16,
        offset_row=0, offset_col=0, reservation_margin=0,
        buffer_pixels=4,
    )

    assert result["available"] is True
    assert not np.any(result["train_mask"] & result["holdout_region_mask"])
    assert not np.any(result["train_sampling_mask"] & result["holdout_mask"])


def test_validation_block_size_selection_uses_geometry_only():
    from src.coregistration import reserve_validation_windows

    common = np.ones((160, 320), dtype=bool)
    result = reserve_validation_windows(
        common, [64, 32], final_min_blocks=5, step=32,
        offset_row=0, offset_col=0, reservation_margin=0,
    )

    assert result["selected_block_size"] == 64
    assert result["candidate_counts"][64] >= 5
    assert result["candidate_counts"][32] > result["candidate_counts"][64]


def test_holdout_diagnostics_account_for_all_candidate_cells():
    from src.coregistration import reserve_validation_windows

    common = np.ones((128, 256), dtype=bool)
    result = reserve_validation_windows(
        common, [32], final_min_blocks=4, step=16,
        offset_row=0, offset_col=0, reservation_margin=0,
    )

    assert result["candidates_before_spatial_filter"] == len(result["candidate_cells"])
    assert result["reserved_count"] == len(result["reserved_windows"])
    assert len(result["candidate_cells"]) == (
        result["reserved_count"] + len(result["unused_cells"])
    )


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
