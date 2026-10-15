"""Unresolved-only geometry tie resolution for Task13B.1.

This module deliberately has no radiometric or scene-order dependency.  It
receives the already-computed Task13B scores and geometry distances and only
touches pixels whose old label is ``LABEL_UNRESOLVED``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .multiscene_label import LABEL_UNRESOLVED


@dataclass(frozen=True)
class TieResolutionResult:
    labels: np.ndarray
    methods: np.ndarray
    changed_old_resolved_pixels: int
    resolved_by_unclipped: int
    resolved_by_raw_edt: int
    still_unresolved: int
    all_clipped_equal_count: int
    non_saturated_unique_winner_count: int
    non_saturated_still_tied_count: int
    raw_edt_unique_winner_after_normalized_tie_count: int
    still_exactly_tied_after_all_geometry_count: int


def _unique_winner(values: np.ndarray, candidates: np.ndarray, epsilon: float) -> tuple[np.ndarray, np.ndarray]:
    """Return winner and strict-margin mask for (scene,row,col) values."""

    masked = np.where(candidates, values, -np.inf)
    order = np.argsort(masked, axis=0)
    best = order[-1]
    second = order[-2]
    best_value = np.take_along_axis(masked, best[None], axis=0)[0]
    second_value = np.take_along_axis(masked, second[None], axis=0)[0]
    unique = candidates.sum(axis=0) > 1
    with np.errstate(invalid="ignore"):
        margin = best_value - second_value
    has_candidate = candidates.any(axis=0)
    return best.astype(np.int16), has_candidate & np.isfinite(best_value) & (margin > epsilon)


def resolve_unresolved_geometry(
    old_labels: np.ndarray,
    candidate_masks: np.ndarray,
    pairwise_scores: np.ndarray,
    raw_edt: np.ndarray,
    p95_edt: np.ndarray,
    *,
    score_tie_epsilon: float = 1e-6,
    geometry_tie_epsilon: float = 1e-6,
) -> TieResolutionResult:
    """Apply Task13B.1 levels 3 and 4 only to old unresolved pixels.

    ``pairwise_scores`` is retained for the audit and is not used to change a
    resolved pixel.  Level 3 compares ``raw_edt / max(P95, eps)`` without an
    upper clip; level 4 compares raw EDT.  A strict margin is required.
    """

    labels = np.asarray(old_labels).copy()
    masks = np.asarray(candidate_masks, dtype=bool)
    scores = np.asarray(pairwise_scores, dtype=np.float32)
    raw = np.asarray(raw_edt, dtype=np.float32)
    p95 = np.asarray(p95_edt, dtype=np.float32)
    if labels.ndim != 2 or masks.ndim != 3 or masks.shape[1:] != labels.shape:
        raise ValueError("label and candidate-mask shapes disagree")
    if scores.shape != masks.shape or raw.shape != masks.shape:
        raise ValueError("geometry/scores must have scene-first mask shape")
    if p95.shape != (masks.shape[0],):
        raise ValueError("p95_edt must have one value per scene")
    unresolved = labels == LABEL_UNRESOLVED
    candidate = masks & unresolved[None]
    if not np.any(unresolved):
        return TieResolutionResult(labels, np.zeros(labels.shape, np.uint8), 0, 0, 0, 0, 0, 0, 0, 0, 0)

    safe_p95 = np.maximum(p95, 1e-6)
    unclipped = raw / safe_p95[:, None, None]
    clipped = np.minimum(unclipped, 1.0)
    methods = np.zeros(labels.shape, dtype=np.uint8)

    # Level 3 is evaluated only among the current candidates.  The existing
    # Task13B score is an audit input; only old unresolved pixels are touched.
    clipped_masked = np.where(candidate, clipped, -np.inf)
    clipped_best = np.max(clipped_masked, axis=0)
    clipped_tie = candidate & (np.abs(clipped - clipped_best[None]) <= score_tie_epsilon)
    all_clipped_equal = unresolved & (candidate.sum(axis=0) > 1) & (np.count_nonzero(clipped_tie, axis=0) == candidate.sum(axis=0))
    winner3, unique3 = _unique_winner(unclipped, candidate, geometry_tie_epsilon)
    assign3 = unresolved & unique3
    labels[assign3] = winner3[assign3]
    methods[assign3] = 3  # UNCLIPPED_NORMALIZED_INTERIORITY

    # Level 4 sees only pixels still unresolved after level 3.
    remain = unresolved & ~assign3
    candidate4 = masks & remain[None]
    winner4, unique4 = _unique_winner(raw, candidate4, geometry_tie_epsilon)
    assign4 = remain & unique4
    labels[assign4] = winner4[assign4]
    methods[assign4] = 4  # RAW_EDT_DEEP_INTERIOR
    still = remain & ~assign4
    methods[still] = 0  # UNRESOLVED_GEOMETRY

    old_resolved = ~unresolved
    changed = int(np.count_nonzero(labels[old_resolved] != old_labels[old_resolved]))
    if changed:
        raise AssertionError("Task13B.1 changed an old resolved label")
    return TieResolutionResult(
        labels=labels,
        methods=methods,
        changed_old_resolved_pixels=changed,
        resolved_by_unclipped=int(assign3.sum()),
        resolved_by_raw_edt=int(assign4.sum()),
        still_unresolved=int(still.sum()),
        all_clipped_equal_count=int(all_clipped_equal.sum()),
        non_saturated_unique_winner_count=int(assign3.sum()),
        non_saturated_still_tied_count=int(remain.sum()),
        raw_edt_unique_winner_after_normalized_tie_count=int(assign4.sum()),
        still_exactly_tied_after_all_geometry_count=int(still.sum()),
    )


__all__ = ["TieResolutionResult", "resolve_unresolved_geometry"]
