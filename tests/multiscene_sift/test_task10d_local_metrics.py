"""RED tests for the independent fixed-grid local metrics."""

import numpy as np
import pytest

from src.multiscene_sift.radiometric_metrics import (
    build_fixed_grid_tiles,
    compute_local_pair_metrics,
    compute_pair_mamd,
)


def test_local_tiles_are_anchored_to_canonical_origin():
    tiles = build_fixed_grid_tiles((600, 600))

    assert tiles[0]["tile_id"] == "r000_c000"
    assert tiles[0]["row_start"] == 0
    assert tiles[0]["col_start"] == 0
    assert tiles[-1]["tile_id"] == "r002_c002"
    assert tiles[-1]["row_start"] == 512
    assert tiles[-1]["col_start"] == 512


def test_tile_ids_depend_only_on_geometry_not_method_values():
    first = build_fixed_grid_tiles((512, 768))
    second = build_fixed_grid_tiles((512, 768))

    assert [tile["tile_id"] for tile in first] == [tile["tile_id"] for tile in second]
    assert [tile["support"] for tile in first] == [tile["support"] for tile in second]


def test_tiles_below_shared_valid_threshold_are_excluded():
    left = np.zeros((256, 512), dtype=np.float64)
    right = np.zeros_like(left)
    valid = np.ones_like(left, dtype=bool)
    valid[:, :256] = False

    summary = compute_local_pair_metrics(left, right, valid)

    assert summary["valid_tile_count"] == 1
    assert [tile["tile_id"] for tile in summary["tiles"]] == ["r000_c001"]


def test_local_rdd_detects_opposing_tile_shifts_hidden_by_global_mean():
    left = np.zeros((256, 512), dtype=np.float64)
    right = np.concatenate(
        [np.full((256, 256), 10.0), np.full((256, 256), -10.0)], axis=1
    )
    valid = np.ones_like(left, dtype=bool)

    assert compute_pair_mamd(left, right, valid) == pytest.approx(0.0)
    summary = compute_local_pair_metrics(left, right, valid)

    assert summary["mamd"]["median"] == pytest.approx(10.0)
    assert summary["rdd"]["median"] == pytest.approx(10.0)
    assert summary["rdd"]["mean"] == pytest.approx(10.0)
    assert summary["rdd"]["p95"] == pytest.approx(10.0)
    assert summary["rdd"]["worst_tile"]["tile_id"] == "r000_c000"
