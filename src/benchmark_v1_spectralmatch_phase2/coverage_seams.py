"""Coverage-preserving, source-valid assignment separate from upstream seams."""

from __future__ import annotations

from typing import Iterable

import numpy as np


def _validate_inputs(valid_i, valid_j, official_labels):
    vi = np.asarray(valid_i, dtype=bool)
    vj = np.asarray(valid_j, dtype=bool)
    labels = np.asarray(official_labels)
    if vi.ndim != 2 or vi.shape != vj.shape or vi.shape != labels.shape:
        raise ValueError("valid_i, valid_j and official_labels must have the same 2D shape")
    return vi, vj, labels


def assign_complete_coverage(
    valid_i: np.ndarray,
    valid_j: np.ndarray,
    official_labels: np.ndarray,
    priority: Iterable[int] = (0, 1),
) -> tuple[np.ndarray, dict]:
    """Keep valid official source labels and fill gaps from actual source masks.

    Labels are 0 for scene_i, 1 for scene_j and -1 outside input support.
    Priority is fixed by the caller and is never inferred from radiometric score.
    """
    vi, vj, official = _validate_inputs(valid_i, valid_j, official_labels)
    order = tuple(int(value) for value in priority)
    if sorted(order) != [0, 1]:
        raise ValueError("priority must contain scene labels 0 and 1 exactly once")
    union = vi | vj
    labels = np.full(vi.shape, -1, dtype=np.int8)
    official_i = (official == 0) & vi
    official_j = (official == 1) & vj
    keep = official_i | official_j
    labels[official_i] = 0
    labels[official_j] = 1
    invalid_official = ((official == 0) & ~vi) | ((official == 1) & ~vj) | (~np.isin(official, [-1, 0, 1]))
    fallback = union & ~keep
    for source in order:
        available = vi if source == 0 else vj
        select = fallback & available
        labels[select] = source
        fallback[select] = False
    audit = {
        "official_assignments": int(keep.sum()),
        "official_source_invalid_pixels": int((invalid_official & union).sum()),
        "fallback_assignments": int((union & ~keep & (labels >= 0)).sum()),
        "source_invalid_pixels": int((((labels == 0) & ~vi) | ((labels == 1) & ~vj)).sum()),
        "no_source_pixels": int((union & (labels < 0)).sum()),
        "outside_union_assignments": int(((labels >= 0) & ~union).sum()),
        "input_union_pixels": int(union.sum()),
    }
    return labels, audit


def validate_assignment(labels, valid_i, valid_j, expected_union) -> dict:
    vi, vj, assigned = _validate_inputs(valid_i, valid_j, labels)
    expected = np.asarray(expected_union, dtype=bool)
    if expected.shape != vi.shape:
        raise ValueError("expected_union must have the same shape as source masks")
    selected_i = assigned == 0
    selected_j = assigned == 1
    selected = selected_i | selected_j
    source_invalid = (selected_i & ~vi) | (selected_j & ~vj)
    missing = expected & ~selected
    extras = selected & ~expected
    no_source = expected & ~(vi | vj)
    invalid_labels = ~np.isin(assigned, [-1, 0, 1])
    passed = not (missing.any() or extras.any() or source_invalid.any() or no_source.any() or invalid_labels.any())
    return {
        "status": "PASS" if passed else "FAIL",
        "expected_pixels": int(expected.sum()),
        "assigned_pixels": int(selected.sum()),
        "missing_pixels": int(missing.sum()),
        "extra_pixels": int(extras.sum()),
        "source_invalid_pixels": int(source_invalid.sum()),
        "no_source_pixels": int(no_source.sum()),
        "invalid_label_pixels": int(invalid_labels.sum()),
    }
