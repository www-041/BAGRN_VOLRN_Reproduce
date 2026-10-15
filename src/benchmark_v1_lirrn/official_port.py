"""Deterministic NumPy translation of the pinned single-band LIRRN steps."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .frozen_crop import RasterCrop


@dataclass(frozen=True)
class RegressionResult:
    scale: float
    offset: float
    fit_rmse: float
    adjusted_r2: float | None
    rank: int
    condition_number: float
    sample_count: int


def classify_three_levels(values: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, dict]:
    values = np.asarray(values, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if values.shape != valid.shape or values.ndim != 2:
        raise ValueError("VALUES_MASK_SHAPE_OR_DIMENSION_INVALID")
    selected = values[valid & (values != 0)]
    if selected.size < 3 or not np.isfinite(selected).all():
        raise ValueError("OTSU_INSUFFICIENT_OR_NONFINITE_SAMPLES")
    if np.any(selected < 0):
        raise ValueError("OTSU_NEGATIVE_VALUES_BLOCKED")
    low, high = float(selected.min()), float(selected.max())
    if low == high:
        raise ValueError("OTSU_THRESHOLDS_NOT_SEPARABLE")

    counts, edges = np.histogram(selected, bins=256, range=(low, high))
    centers = (edges[:-1] + edges[1:]) * 0.5
    weights = counts.astype(np.float64)
    cumulative_weight = np.cumsum(weights)
    cumulative_moment = np.cumsum(weights * centers)
    total_weight = cumulative_weight[-1]
    total_mean = cumulative_moment[-1] / total_weight
    best_score = -np.inf
    best_pair: tuple[int, int] | None = None
    # Ascending scan plus strict > gives deterministic lowest-index tie resolution.
    for first in range(0, 254):
        w1 = cumulative_weight[first]
        if w1 <= 0:
            continue
        for second in range(first + 1, 255):
            w2 = cumulative_weight[second] - w1
            w3 = total_weight - cumulative_weight[second]
            if w2 <= 0 or w3 <= 0:
                continue
            m1 = cumulative_moment[first] / w1
            m2 = (cumulative_moment[second] - cumulative_moment[first]) / w2
            m3 = (cumulative_moment[-1] - cumulative_moment[second]) / w3
            score = w1 * (m1-total_mean)**2 + w2 * (m2-total_mean)**2 + w3 * (m3-total_mean)**2
            if score > best_score:
                best_score, best_pair = score, (first, second)
    if best_pair is None:
        raise ValueError("OTSU_THRESHOLDS_NOT_SEPARABLE")
    t1, t2 = float(edges[best_pair[0] + 1]), float(edges[best_pair[1] + 1])
    if not t1 < t2:
        raise ValueError("OTSU_THRESHOLDS_NOT_SEPARABLE")
    labels = np.zeros(values.shape, dtype=np.uint8)
    labels[valid & (values != 0) & (values <= t1)] = 1
    labels[valid & (values != 0) & (values > t1) & (values <= t2)] = 2
    labels[valid & (values != 0) & (values > t2)] = 3
    class_counts = [int(np.count_nonzero(labels == c)) for c in (1, 2, 3)]
    if min(class_counts) == 0:
        raise ValueError("OTSU_EMPTY_CLASS")
    return labels, {"thresholds": [t1, t2], "class_counts": class_counts,
                    "valid_nonzero_count": int(selected.size), "histogram_bins": 256,
                    "range": [low, high], "tie_break": "FIRST_LOWEST_BIN_PAIR",
                    "backend": "NUMPY_256BIN_PORT_V1"}


def draw_shared_ids(p_n: int, seed: int) -> np.ndarray:
    if p_n <= 0:
        raise ValueError("PN_MUST_BE_POSITIVE")
    count = int(math.floor(0.1 * p_n + 0.5))
    if count <= 0:
        raise ValueError("PN_PRODUCES_ZERO_SHARED_IDS")
    return np.random.default_rng(seed).integers(0, p_n, size=count, dtype=np.int64)


def _class_candidates(crop: RasterCrop, labels: np.ndarray, class_id: int) -> np.ndarray:
    labels = np.asarray(labels)
    if labels.shape != crop.values.shape:
        raise ValueError("LABEL_SHAPE_MISMATCH")
    valid = crop.valid & (labels == class_id) & (crop.values != 0)
    values = flatten_lirrn_values(crop.values, valid)
    if not np.isfinite(values).all():
        raise ValueError("NONFINITE_SAMPLE_VALUE")
    return values


def flatten_lirrn_values(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Extract selected raster values in MATLAB's column-major linear order."""
    values = np.asarray(values)
    valid = np.asarray(valid, dtype=bool)
    if values.shape != valid.shape:
        raise ValueError("VALUES_MASK_SHAPE_MISMATCH")
    flat_values = values.ravel(order="F")
    flat_valid = valid.ravel(order="F")
    return flat_values[flat_valid]


def select_lirrn_samples(ref: RasterCrop, sub: RasterCrop, ref_labels: np.ndarray,
                         sub_labels: np.ndarray, shared_ids: np.ndarray, p_n: int
                         ) -> tuple[np.ndarray, np.ndarray, dict]:
    ids = np.asarray(shared_ids, dtype=np.int64).reshape(-1)
    if p_n <= 0 or ids.size == 0 or np.any(ids < 0) or np.any(ids >= p_n):
        raise ValueError("SHARED_ID_VECTOR_INVALID")
    if ref.values.shape != sub.values.shape:
        raise ValueError("INPUT_CROP_SHAPE_MISMATCH")
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    class_audit = []
    for class_id in (1, 2, 3):
        a = _class_candidates(ref, ref_labels, class_id)
        b = _class_candidates(sub, sub_labels, class_id)
        if a.size < p_n or b.size < p_n:
            raise ValueError(f"INSUFFICIENT_STRATUM_SAMPLES:class={class_id}:ref={a.size}:sub={b.size}:p_n={p_n}")
        ref_order = np.argsort(np.abs(a - np.max(a)), kind="stable")[:p_n]
        sub_order = np.argsort(np.abs(b - np.max(b)), kind="stable")[:p_n]
        ref_top, sub_top = a[ref_order], b[sub_order]
        # The official local ComputeSample performs this same call three times.
        for _ in range(3):
            ys.append(ref_top[ids])
            xs.append(sub_top[ids])
        class_audit.append({"class": class_id, "ref_candidates": int(a.size),
                            "sub_candidates": int(b.size), "ref_top_min": float(ref_top.min()),
                            "sub_top_min": float(sub_top.min()), "duplicate_max_calls": 3})
    return np.concatenate(xs), np.concatenate(ys), {
        "three_identical_max_sampling_calls": True, "flatten_order": "F",
        "stable_sort": "numpy.argsort(kind=stable)", "shared_id_count": int(ids.size),
        "samples_per_class": int(3 * ids.size), "sample_count": int(9 * ids.size),
        "classes": class_audit,
    }


def fit_lirrn_linear(x_sub: np.ndarray, y_ref: np.ndarray) -> RegressionResult:
    x = np.asarray(x_sub, dtype=np.float64).reshape(-1)
    y = np.asarray(y_ref, dtype=np.float64).reshape(-1)
    if x.shape != y.shape or x.size < 3 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("REGRESSION_INVALID_SAMPLES")
    design = np.column_stack((x, np.ones_like(x)))
    coef, _, rank, _ = np.linalg.lstsq(design, y, rcond=None)
    condition = float(np.linalg.cond(design))
    if rank != 2 or not np.isfinite(coef).all() or not np.isfinite(condition):
        raise ValueError("REGRESSION_DEGENERATE")
    pred = design @ coef
    residual = y - pred
    sse = float(residual @ residual)
    fit_rmse = float(np.sqrt(sse / (x.size - 2)))
    total = float(np.sum((y - y.mean())**2))
    adjusted = None if total == 0 else float(1.0 - (sse/(x.size-2))/(total/(x.size-1)))
    if not np.isfinite(fit_rmse) or (adjusted is not None and not np.isfinite(adjusted)):
        raise ValueError("REGRESSION_NONFINITE_DIAGNOSTIC")
    return RegressionResult(float(coef[0]), float(coef[1]), fit_rmse, adjusted,
                            int(rank), condition, int(x.size))


def apply_lirrn_linear(sub: RasterCrop, model: RegressionResult) -> np.ndarray:
    if not np.isfinite(model.scale) or not np.isfinite(model.offset):
        raise ValueError("NONFINITE_REGRESSION_COEFFICIENT")
    out = sub.values.astype(np.float32, copy=True)
    corrected = model.scale * sub.values[sub.valid] + model.offset
    if not np.isfinite(corrected).all():
        raise ValueError("NONFINITE_CORRECTED_VALID_PIXELS")
    out[sub.valid] = corrected.astype(np.float32)
    if not np.isfinite(out[sub.valid]).all():
        raise ValueError("NONFINITE_OUTPUT_VALID_PIXELS")
    return out
