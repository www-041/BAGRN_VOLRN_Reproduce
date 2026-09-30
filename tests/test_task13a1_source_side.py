"""Synthetic geometry-only tests for Task13A.1 source-side resolution."""

import numpy as np
from shapely.geometry import box

from src.seam_local.seam import SeamResult
from src.seam_local.source_side import resolve_source_sides


def _seam(shape: tuple[int, int], orientation: str, coordinate: int) -> SeamResult:
    length = shape[0] if orientation == "vertical" else shape[1]
    lines = np.arange(length, dtype=np.int32)
    if orientation == "vertical":
        path = np.column_stack((lines, np.full(length, coordinate, dtype=np.int32)))
    else:
        path = np.column_stack((np.full(length, coordinate, dtype=np.int32), lines))
    return SeamResult(orientation, path, 0.0, 0.0, 0.0, "OK")


def _exclusive_masks(shape: tuple[int, int], orientation: str, reversed_order: bool = False):
    rows, cols = np.indices(shape)
    if orientation == "vertical":
        first = cols < 10
        second = cols > 10
    else:
        first = rows < 10
        second = rows > 10
    if not reversed_order:
        valid_a, valid_b = first | (rows == 0), second | (rows == 0)
    else:
        valid_a, valid_b = second | (rows == 0), first | (rows == 0)
    # Keep a shared-valid corridor around the seam so side partitioning is
    # independent of scene order and does not use a flood fill.
    seam_line = (cols == 10) if orientation == "vertical" else (rows == 10)
    valid_a |= seam_line
    valid_b |= seam_line
    return valid_a, valid_b


def test_vertical_exclusive_contact_resolves_left_right_without_scene_id_assumption():
    shape = (20, 20)
    seam = _seam(shape, "vertical", 10)
    valid_a, valid_b = _exclusive_masks(shape, "vertical")
    result = resolve_source_sides(
        seam,
        np.ones(shape, dtype=bool),
        valid_a,
        valid_b,
        box(0, 0, 12, 20),
        box(8, 0, 20, 20),
    )
    assert result.status == "EXCLUSIVE_CONTACT_RESOLVABLE"
    assert (result.side_1_source, result.side_2_source) == ("A", "B")
    assert result.method == "EXCLUSIVE_BOUNDARY_CONTACT"
    assert result.diagnostics["exclusive_contact_a_side1"] > result.diagnostics["exclusive_contact_a_side2"]


def test_vertical_reversed_spatial_order_assigns_b_to_low_side():
    shape = (20, 20)
    valid_a, valid_b = _exclusive_masks(shape, "vertical", reversed_order=True)
    result = resolve_source_sides(
        _seam(shape, "vertical", 10),
        np.ones(shape, dtype=bool),
        valid_a,
        valid_b,
        box(8, 0, 20, 20),
        box(0, 0, 12, 20),
    )
    assert result.status == "EXCLUSIVE_CONTACT_RESOLVABLE"
    assert (result.side_1_source, result.side_2_source) == ("B", "A")


def test_horizontal_top_bottom_assignments_follow_geometry():
    shape = (20, 20)
    valid_a, valid_b = _exclusive_masks(shape, "horizontal")
    result = resolve_source_sides(
        _seam(shape, "horizontal", 10),
        np.ones(shape, dtype=bool),
        valid_a,
        valid_b,
        box(0, 0, 20, 12),
        box(0, 8, 20, 20),
    )
    assert result.status == "EXCLUSIVE_CONTACT_RESOLVABLE"
    assert (result.side_1_source, result.side_2_source) == ("A", "B")


def test_contact_score_uses_all_exclusive_boundary_evidence():
    shape = (20, 20)
    rows, cols = np.indices(shape)
    boundary = (rows == 0) | (rows == 19) | (cols == 0) | (cols == 19)
    valid_a = np.where(boundary, cols <= 10, cols >= 10)
    valid_b = np.where(boundary, cols >= 10, cols <= 10)
    result = resolve_source_sides(
        _seam(shape, "vertical", 10),
        np.ones(shape, dtype=bool),
        valid_a,
        valid_b,
        box(0, 0, 12, 20),
        box(8, 0, 20, 20),
    )
    assert result.status == "EXCLUSIVE_CONTACT_RESOLVABLE"
    assert (result.side_1_source, result.side_2_source) == ("B", "A")


def test_stray_outer_contact_does_not_override_stronger_shared_boundary_contact():
    shape = (20, 20)
    rows, cols = np.indices(shape)
    overlap = (cols >= 8) & (cols <= 12)
    valid_a = overlap.copy()
    valid_b = overlap.copy()
    valid_a[1:19, 7] = True
    valid_b[1:19, 12] = True
    valid_b[0, 7] = True
    result = resolve_source_sides(
        _seam(shape, "vertical", 10), overlap, valid_a, valid_b,
        box(0, 0, 13, 20), box(7, 0, 20, 20),
    )
    assert result.status == "EXCLUSIVE_CONTACT_RESOLVABLE"
    assert (result.side_1_source, result.side_2_source) == ("A", "B")


def test_centroid_projection_is_used_only_when_contact_is_not_decisive():
    shape = (20, 20)
    both = np.ones(shape, dtype=bool)
    result = resolve_source_sides(
        _seam(shape, "vertical", 10),
        both,
        both,
        both,
        box(0, 0, 8, 20),
        box(12, 0, 20, 20),
    )
    assert result.status == "CENTROID_RESOLVABLE"
    assert (result.side_1_source, result.side_2_source) == ("A", "B")
    assert result.method == "FOOTPRINT_CENTROID_PROJECTION"
    assert result.diagnostics["projection_separation"] > 0


def test_horizontal_centroid_projection_maps_larger_world_y_to_top():
    shape = (20, 20)
    both = np.ones(shape, dtype=bool)
    result = resolve_source_sides(
        _seam(shape, "horizontal", 10),
        both, both, both,
        box(0, 12, 20, 20),
        box(0, 0, 20, 8),
    )
    assert result.status == "CENTROID_RESOLVABLE"
    assert (result.side_1_source, result.side_2_source) == ("A", "B")


def test_nested_footprints_remain_unresolved():
    shape = (20, 20)
    both = np.ones(shape, dtype=bool)
    result = resolve_source_sides(
        _seam(shape, "vertical", 10),
        both,
        both,
        both,
        box(0, 0, 20, 20),
        box(4, 4, 16, 16),
    )
    assert result.status == "REQUIRES_MULTISCENE_LABELING"
    assert result.side_1_source is None and result.side_2_source is None


def test_identical_centroids_do_not_receive_arbitrary_assignment():
    shape = (20, 20)
    both = np.ones(shape, dtype=bool)
    result = resolve_source_sides(
        _seam(shape, "horizontal", 10),
        both,
        both,
        both,
        box(0, 0, 20, 20),
        box(0, 0, 20, 20),
    )
    assert result.status == "REQUIRES_MULTISCENE_LABELING"
    assert result.method == "UNRESOLVED_GEOMETRY"


def test_swapping_a_and_b_swaps_ownership_but_not_spatial_partition():
    shape = (20, 20)
    seam = _seam(shape, "vertical", 10)
    valid_a, valid_b = _exclusive_masks(shape, "vertical")
    footprint_a, footprint_b = box(0, 0, 12, 20), box(8, 0, 20, 20)
    first = resolve_source_sides(seam, np.ones(shape, bool), valid_a, valid_b, footprint_a, footprint_b)
    swapped = resolve_source_sides(seam, np.ones(shape, bool), valid_b, valid_a, footprint_b, footprint_a)
    assert np.array_equal(seam.row_col_path, _seam(shape, "vertical", 10).row_col_path)
    assert (first.side_1_source, first.side_2_source) == ("A", "B")
    assert (swapped.side_1_source, swapped.side_2_source) == ("B", "A")
    assert first.diagnostics["side_1_pixel_count"] == swapped.diagnostics["side_1_pixel_count"]


def test_seam_path_is_bitwise_unchanged_and_diagnostics_are_geometry_only():
    shape = (20, 20)
    seam = _seam(shape, "vertical", 10)
    before = seam.row_col_path.copy()
    both = np.ones(shape, dtype=bool)
    result = resolve_source_sides(seam, both, both, both, box(0, 0, 8, 20), box(12, 0, 20, 20))
    assert np.array_equal(seam.row_col_path, before)
    assert not any(token in str(result.diagnostics).lower() for token in ("mae", "rmse", "rdd"))


def test_shared_overlap_mask_still_uses_exclusive_support_outside_overlap():
    shape = (20, 20)
    rows, cols = np.indices(shape)
    valid_a = cols <= 11
    valid_b = cols >= 8
    overlap = valid_a & valid_b
    result = resolve_source_sides(
        _seam(shape, "vertical", 10), overlap, valid_a, valid_b,
        box(0, 0, 12, 20), box(8, 0, 20, 20),
    )
    assert result.status == "EXCLUSIVE_CONTACT_RESOLVABLE"
    assert (result.side_1_source, result.side_2_source) == ("A", "B")
    assert result.diagnostics["assignment_score_ab"] > result.diagnostics["assignment_score_ba"]


def test_off_center_nested_footprint_never_uses_centroid_assignment():
    shape = (20, 20)
    valid_a = np.ones(shape, dtype=bool)
    valid_b = np.zeros(shape, dtype=bool)
    valid_b[:, 2:14] = True
    result = resolve_source_sides(
        _seam(shape, "vertical", 8), valid_a & valid_b, valid_a, valid_b,
        box(0, 0, 20, 20), box(2, 0, 14, 20),
    )
    assert result.status == "REQUIRES_MULTISCENE_LABELING"
    assert result.method == "CONTAINMENT_OR_NESTED_FOOTPRINT"


def test_nearly_contained_footprint_is_not_resolved_by_centroid():
    shape = (20, 20)
    both = np.ones(shape, dtype=bool)
    result = resolve_source_sides(
        _seam(shape, "vertical", 10), both, both, both,
        box(0, 0, 20, 20), box(1, 0, 20.000001, 20),
    )
    assert result.status == "REQUIRES_MULTISCENE_LABELING"


def test_nearly_equal_centroid_projection_remains_unresolved():
    shape = (20, 20)
    both = np.ones(shape, dtype=bool)
    result = resolve_source_sides(
        _seam(shape, "vertical", 10), both, both, both,
        box(0, 0, 20, 20), box(1e-8, -1, 20 + 1e-8, 21),
    )
    assert result.status == "REQUIRES_MULTISCENE_LABELING"
