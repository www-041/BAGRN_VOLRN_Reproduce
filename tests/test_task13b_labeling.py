"""Focused synthetic tests for Task13B multiscene source labeling."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt

from src.seam_local.multiscene_label import (
    LABEL_UNRESOLVED,
    PairwisePreferenceField,
    aggregate_multiscene_labels,
    build_pairwise_preference_field,
)


def _vertical_path(height: int, center: int) -> np.ndarray:
    return np.column_stack((np.arange(height), np.full(height, center)))


def _horizontal_path(width: int, center: int) -> np.ndarray:
    return np.column_stack((np.full(width, center), np.arange(width)))


def _constant_field(a: int, b: int, vote: float, shape: tuple[int, int]) -> PairwisePreferenceField:
    return PairwisePreferenceField(
        a, b, np.full(shape, vote, dtype=np.float32), np.ones(shape, dtype=bool),
        np.ones(shape, dtype=np.float32), "synthetic",
    )


def test_vertical_and_horizontal_fields_use_64px_distance_and_ownership() -> None:
    valid = np.ones((3, 5), dtype=bool)
    field = build_pairwise_preference_field(valid, valid, _vertical_path(3, 2), "A_LOW_B_HIGH")
    assert np.isclose(field.vote[1, 0], 2.0 / 64.0)
    assert np.isclose(field.vote[1, 1], 1.0 / 64.0)
    assert np.isclose(field.vote[1, 4], -2.0 / 64.0)
    assert np.isclose(field.confidence[1, 4], 2.0 / 64.0)

    reverse = build_pairwise_preference_field(valid, valid, _vertical_path(3, 2), "B_LOW_A_HIGH")
    assert np.allclose(reverse.vote, -field.vote)

    horizontal_valid = np.ones((5, 3), dtype=bool)
    horizontal = build_pairwise_preference_field(
        horizontal_valid, horizontal_valid, _horizontal_path(3, 2), "A_LOW_B_HIGH"
    )
    assert np.isclose(horizontal.vote[0, 1], 2.0 / 64.0)
    assert np.isclose(horizontal.vote[4, 1], -2.0 / 64.0)


def test_invalid_pixels_and_unresolved_pair_contribute_no_vote() -> None:
    valid_a = np.ones((2, 4), dtype=bool)
    valid_b = valid_a.copy()
    valid_b[:, 0] = False
    unresolved = build_pairwise_preference_field(valid_a, valid_b, _vertical_path(2, 2), None)
    assert not unresolved.available.any()
    assert not unresolved.vote.any()
    result = aggregate_multiscene_labels(np.stack((valid_a, valid_b)), {})
    assert result.labels[0, 0] == 0
    # Geometry-only interiority may resolve a score tie even when no pair
    # preference is available; the invalid pixel itself remains scene 0.
    assert result.labels[0, 1] == 0
    assert result.diagnostics["invalid_pixels"] == 0


def test_two_scene_resolved_labels_are_consistent_except_seam_center() -> None:
    valid = np.ones((3, 5), dtype=bool)
    field = build_pairwise_preference_field(valid, valid, _vertical_path(3, 2), "A_LOW_B_HIGH")
    result = aggregate_multiscene_labels(np.stack((valid, valid)), {(0, 1): field})
    assert np.all(result.labels[:, 0:2] == 0)
    assert np.all(result.labels[:, 3:5] == 1)
    assert np.all(result.labels[:, 2] == LABEL_UNRESOLVED)
    assert result.diagnostics["two_scene_disagreement_pixels"] == 0


def test_normalized_scores_do_not_penalize_scene_with_missing_edge() -> None:
    masks = np.ones((3, 1, 1), dtype=bool)
    # Scene 0 wins its only edge. Scene 1 has two edges, but its score is
    # normalized by two available comparisons; no degree/index tie-break is used.
    fields = {
        (0, 1): _constant_field(0, 1, 1.0, (1, 1)),
        (1, 2): _constant_field(1, 2, 1.0, (1, 1)),
    }
    result = aggregate_multiscene_labels(masks, fields)
    assert np.allclose(result.scores[:, 0, 0], [1.0, 0.0, -1.0])
    assert result.labels[0, 0] == 0


def test_three_scene_consistent_cycle_and_missing_edge_diagnostics() -> None:
    masks = np.ones((3, 2, 2), dtype=bool)
    consistent = {
        (0, 1): _constant_field(0, 1, 1.0, (2, 2)),
        (0, 2): _constant_field(0, 2, 1.0, (2, 2)),
        (1, 2): _constant_field(1, 2, 1.0, (2, 2)),
    }
    result = aggregate_multiscene_labels(masks, consistent)
    assert np.all(result.labels == 0)
    assert result.diagnostics["cycle_pixels"] == 0
    assert result.diagnostics["multiscene_pixels"] == 4

    cycle = dict(consistent)
    cycle[(0, 2)] = _constant_field(0, 2, -1.0, (2, 2))
    cycled = aggregate_multiscene_labels(masks, cycle)
    assert cycled.diagnostics["cycle_pixels"] == 4

    missing = aggregate_multiscene_labels(masks, {(0, 1): consistent[(0, 1)]})
    assert missing.diagnostics["resolved_pair_count"] == 1
    assert missing.diagnostics["cycle_pixels"] == 0


def test_cycle_diagnostics_require_triple_scene_support() -> None:
    masks = np.ones((3, 2, 2), dtype=bool)
    masks[2, 0, 0] = False
    cycle_fields = {
        (0, 1): _constant_field(0, 1, 1.0, (2, 2)),
        (0, 2): _constant_field(0, 2, -1.0, (2, 2)),
        (1, 2): _constant_field(1, 2, 1.0, (2, 2)),
    }
    result = aggregate_multiscene_labels(masks, cycle_fields)
    assert result.diagnostics["cycle_pixels"] == 3


def test_equal_score_equal_interiority_remains_unresolved_without_index_fallback() -> None:
    masks = np.ones((2, 2, 2), dtype=bool)
    zero = _constant_field(0, 1, 0.0, (2, 2))
    result = aggregate_multiscene_labels(masks, {(0, 1): zero}, interiority=np.ones_like(masks, dtype=float))
    assert np.all(result.labels == LABEL_UNRESOLVED)
    assert result.diagnostics["top_score_tie_pixels"] == 4
    assert result.diagnostics["interiority_fallback_pixels"] == 0
    assert result.diagnostics["unresolved_pixels"] == 4


def test_geometry_only_interiority_breaks_tie() -> None:
    masks = np.ones((2, 1, 1), dtype=bool)
    zero = _constant_field(0, 1, 0.0, (1, 1))
    # Explicit geometry values are normalized internally; scene 1 is deeper.
    # Values below one retain their relative depth because the denominator is
    # max(P95, 1); a one-pixel scene with values 0.5 and 2 therefore remains
    # an unambiguous geometry tie-break.
    interiority = np.array([[[0.5]], [[2.0]]])
    result = aggregate_multiscene_labels(masks, {(0, 1): zero}, interiority=interiority)
    assert result.labels[0, 0] == 1
    assert result.diagnostics["interiority_fallback_pixels"] == 1


def test_nested_footprints_use_only_p95_normalized_interiority() -> None:
    """An unresolved pair is selected from footprint depth, never intensity."""

    masks = np.zeros((2, 20, 20), dtype=bool)
    masks[0, 2:18, 2:18] = True
    masks[1, 3:8, 3:8] = True
    result = aggregate_multiscene_labels(masks, {})

    # Recompute the protocol quantity independently and require every
    # non-tied overlap pixel to follow it.  No pairwise or radiometric field
    # is supplied, so this is a geometry-only assertion.
    normalized = []
    for mask in masks:
        distance = distance_transform_edt(mask)
        p95 = np.percentile(distance[mask], 95)
        normalized.append(np.minimum(distance / max(p95, 1.0), 1.0))
    normalized = np.asarray(normalized)
    overlap = masks.all(axis=0)
    strict = overlap & (np.abs(normalized[0] - normalized[1]) > 1e-6)
    expected = np.where(normalized[1] > normalized[0], 1, 0)
    assert np.array_equal(result.labels[strict], expected[strict])
    assert result.diagnostics["resolved_pair_count"] == 0
    assert result.diagnostics["interiority_fallback_pixels"] == int(np.count_nonzero(overlap))


def test_identical_interiority_remains_unresolved() -> None:
    masks = np.ones((2, 5, 5), dtype=bool)
    # Identical explicit geometry values make the normalized tie exact.
    interiority = np.full_like(masks, 3.0, dtype=np.float32)
    result = aggregate_multiscene_labels(masks, {}, interiority=interiority)
    assert np.all(result.labels == LABEL_UNRESOLVED)
    assert result.diagnostics["unresolved_pixels"] == 25
    assert result.diagnostics["interiority_fallback_pixels"] == 0


def test_swapping_scene_ids_preserves_spatial_geometry_ownership() -> None:
    masks = np.zeros((2, 20, 20), dtype=bool)
    masks[0, 2:18, 2:18] = True
    masks[1, 3:8, 3:8] = True
    original = aggregate_multiscene_labels(masks, {})
    swapped = aggregate_multiscene_labels(masks[::-1], {})
    # Convert swapped scene IDs back to the original IDs before comparing.
    remapped = np.where(swapped.labels == 0, 1, np.where(swapped.labels == 1, 0, LABEL_UNRESOLVED))
    assert np.array_equal(original.labels, remapped)

