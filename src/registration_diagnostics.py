"""Geometry-only diagnostics for two-image registration evidence.

This module deliberately does not feed measurements back into the formal
Global + RBF registration pipeline.
"""

from collections import Counter

import numpy as np


def masked_ncc(ref, tgt, valid_mask):
    """Return absolute mean-centered NCC on valid pixels, or ``None``."""
    ref = np.asarray(ref, dtype=np.float64)
    tgt = np.asarray(tgt, dtype=np.float64)
    valid = (
        np.asarray(valid_mask, dtype=bool)
        & np.isfinite(ref)
        & np.isfinite(tgt)
    )
    if int(valid.sum()) < 100:
        return None

    r = ref[valid].astype(np.float64, copy=True)
    t = tgt[valid].astype(np.float64, copy=True)
    r -= r.mean()
    t -= t.mean()
    denom = np.linalg.norm(r) * np.linalg.norm(t)
    if denom <= 1e-12:
        return None
    return float(abs(np.dot(r, t) / denom))


def compute_grid_relationship(ref_transform, tgt_transform):
    """Describe the target pixel-grid origin/phase in reference pixels."""
    target_center_world = tgt_transform * (0.5, 0.5)
    target_center_ref_edge = (~ref_transform) * target_center_world
    target_col = float(target_center_ref_edge[0] - 0.5)
    target_row = float(target_center_ref_edge[1] - 0.5)
    nearest_col = round(target_col)
    nearest_row = round(target_row)

    return {
        'ref_pixel_size': {
            'x': float(np.hypot(ref_transform.a, ref_transform.d)),
            'y': float(np.hypot(ref_transform.b, ref_transform.e)),
        },
        'tgt_pixel_size': {
            'x': float(np.hypot(tgt_transform.a, tgt_transform.d)),
            'y': float(np.hypot(tgt_transform.b, tgt_transform.e)),
        },
        'target_origin_in_ref_pixels': {
            'col': target_col,
            'row': target_row,
        },
        'nearest_integer_offset_pixels': {
            'col': int(nearest_col),
            'row': int(nearest_row),
        },
        'fractional_phase_pixels': {
            'col': float(target_col - nearest_col),
            'row': float(target_row - nearest_row),
        },
    }


def _summary_values(values, names):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    result = {name: None for name in names}
    if values.size == 0:
        return result
    percentiles = {
        'p10': 10,
        'median': 50,
        'p90': 90,
    }
    for name, percentile in percentiles.items():
        if name in result:
            result[name] = float(np.percentile(values, percentile))
    if 'mean' in result:
        result['mean'] = float(np.mean(values))
    if 'min' in result:
        result['min'] = float(np.min(values))
    if 'max' in result:
        result['max'] = float(np.max(values))
    return result


def summarize_candidate_rows(rows):
    """Summarize candidate distributions while retaining rejection counts."""
    rows = list(rows or [])
    measured_dx = []
    measured_dy = []
    measured_magnitude = []
    confidence = []
    reject_reasons = Counter()

    for row in rows:
        reason = row.get('reject_reason')
        if reason is not None:
            reject_reasons[str(reason)] += 1

        dx = row.get('shift_dx_pixels')
        dy = row.get('shift_dy_pixels')
        magnitude = row.get('shift_magnitude_pixels')
        try:
            dx = float(dx)
            dy = float(dy)
        except (TypeError, ValueError):
            dx = dy = None
        if dx is not None and dy is not None and np.isfinite(dx) and np.isfinite(dy):
            measured_dx.append(dx)
            measured_dy.append(dy)
            try:
                magnitude = float(magnitude)
            except (TypeError, ValueError):
                magnitude = float(np.hypot(dx, dy))
            if np.isfinite(magnitude):
                measured_magnitude.append(magnitude)

        try:
            confidence_value = float(row.get('confidence'))
        except (TypeError, ValueError):
            confidence_value = None
        if confidence_value is not None and np.isfinite(confidence_value):
            confidence.append(confidence_value)

    return {
        'total_rows': int(len(rows)),
        'measured_rows': int(len(measured_dx)),
        'reject_reason_counts': dict(reject_reasons),
        'confidence': _summary_values(
            confidence, ['min', 'median', 'mean', 'p90', 'max']),
        'shift_dx_pixels': _summary_values(
            measured_dx, ['median', 'p10', 'p90']),
        'shift_dy_pixels': _summary_values(
            measured_dy, ['median', 'p10', 'p90']),
        'shift_magnitude_pixels': _summary_values(
            measured_magnitude, ['median', 'p90', 'max']),
    }
