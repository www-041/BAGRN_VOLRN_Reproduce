from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from scipy.ndimage import binary_erosion, sobel, uniform_filter
from scipy.stats import wasserstein_distance


def _shared_values(reference, corrected, mask) -> tuple[np.ndarray, np.ndarray]:
    a, b, valid = np.asarray(reference, dtype=np.float64), np.asarray(corrected, dtype=np.float64), np.asarray(mask, dtype=bool)
    if a.shape != b.shape or a.shape != valid.shape:
        raise ValueError("reference, corrected, and mask must have identical shapes")
    support = valid & np.isfinite(a) & np.isfinite(b)
    return a[support], b[support]


def pair_metrics(reference, corrected, mask) -> dict[str, Any]:
    """Per-ROI paired metrics on shared finite support; RDD is exact Wasserstein-1."""
    a, b = _shared_values(reference, corrected, mask)
    count = int(a.size)
    if not count:
        return {"status": "NO_VALID_SUPPORT", "shared_pixels": 0, "MAMD": None, "MSDD": None, "RDD": None, "MAE": None, "RMSE": None, "P95": None, "MAX_ABS_ERROR": None}
    error = np.abs(a - b)
    return {
        "status": "EVALUABLE", "shared_pixels": count,
        "MAMD": float(abs(np.mean(a) - np.mean(b))),
        "MSDD": float(abs(np.std(a) - np.std(b))),
        "RDD": float(wasserstein_distance(a, b)),
        "MAE": float(np.mean(error)), "RMSE": float(np.sqrt(np.mean(error * error))),
        "P95": float(np.percentile(error, 95)), "MAX_ABS_ERROR": float(np.max(error)),
    }


def change_preservation_metrics(
    target_change, output_change, change_mask, *, truth_status: str = "UNVERIFIED", output_change_mask=None,
) -> dict[str, Any]:
    """Score change only when the supplied change truth has explicit human approval."""
    if truth_status != "APPROVED" or change_mask is None:
        return {"status": "CONDITIONAL_NOT_EVALUABLE", "reason": "independent approved change truth is required", "change_magnitude_MAE": None, "change_magnitude_RMSE": None, "change_F1": None, "change_IoU": None}
    truth, output = np.asarray(target_change, dtype=np.float64), np.asarray(output_change, dtype=np.float64)
    truth_mask = np.asarray(change_mask, dtype=bool)
    if truth.shape != output.shape or truth.shape != truth_mask.shape:
        raise ValueError("change truth, output, and change mask shapes must match")
    support = np.isfinite(truth) & np.isfinite(output)
    magnitude_support = support & truth_mask
    magnitude_error = np.abs(truth[magnitude_support] - output[magnitude_support])
    if magnitude_error.size:
        magnitude_mae = float(np.mean(magnitude_error))
        magnitude_rmse = float(np.sqrt(np.mean(magnitude_error ** 2)))
    else:
        magnitude_mae = magnitude_rmse = None
    f1 = iou = None
    if output_change_mask is not None:
        predicted = np.asarray(output_change_mask, dtype=bool)
        if predicted.shape != truth.shape:
            raise ValueError("output change mask shape must match change truth")
        tp = int(np.count_nonzero(support & truth_mask & predicted))
        fp = int(np.count_nonzero(support & ~truth_mask & predicted))
        fn = int(np.count_nonzero(support & truth_mask & ~predicted))
        f1 = float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else 1.0
        iou = float(tp / (tp + fp + fn)) if tp + fp + fn else 1.0
    return {"status": "EVALUABLE" if output_change_mask is not None else "MAGNITUDE_ONLY", "truth_status": truth_status,
            "change_pixels": int(magnitude_support.sum()), "change_magnitude_MAE": magnitude_mae,
            "change_magnitude_RMSE": magnitude_rmse, "change_F1": f1, "change_IoU": iou}


def _scales_for_policy(policy: Mapping[str, Any], band_count: int) -> tuple[str, np.ndarray]:
    name = str(policy.get("name", ""))
    values = policy.get("scales")
    if isinstance(values, Mapping):
        values = [values.get(str(index), values.get(f"B{index + 1}")) for index in range(band_count)]
    scales = np.asarray(values, dtype=np.float64)
    if not name or scales.shape != (band_count,) or not np.all(np.isfinite(scales)) or np.any(scales <= 0):
        raise ValueError("scale_policy must name a frozen positive finite scale for every band")
    return name, scales


def multiband_metrics(reference_cube, corrected_cube, mask, scale_policy: Mapping[str, Any], *, spectral_coregistered: bool = False, spectral_scale_confirmed: bool = False) -> dict[str, Any]:
    reference, corrected = np.asarray(reference_cube, dtype=np.float64), np.asarray(corrected_cube, dtype=np.float64)
    if reference.shape != corrected.shape or reference.ndim != 3:
        raise ValueError("multiband inputs must be identical bands-first (bands, rows, cols) arrays")
    band_count = reference.shape[0]
    name, scales = _scales_for_policy(scale_policy, band_count)
    masks = np.asarray(mask, dtype=bool)
    if masks.shape == reference.shape[1:]:
        masks = np.broadcast_to(masks, reference.shape)
    if masks.shape != reference.shape:
        raise ValueError("multiband mask must be spatial or match all bands")
    rows = []
    for index in range(band_count):
        metrics = pair_metrics(reference[index], corrected[index], masks[index])
        rows.append({"band_index": index + 1, **metrics,
                     "normalized_MAE": metrics["MAE"] / scales[index] if metrics["MAE"] is not None else None,
                     "scale": float(scales[index]), "scale_policy": name})
    normalized = [row["normalized_MAE"] for row in rows if row["normalized_MAE"] is not None]
    sam = {"status": "CONDITIONAL_NOT_EVALUABLE", "mean_angle_rad": None, "joint_pixels": 0}
    if spectral_coregistered and spectral_scale_confirmed:
        support = np.all(masks & np.isfinite(reference) & np.isfinite(corrected), axis=0)
        a = reference[:, support].T; b = corrected[:, support].T
        norms = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
        keep = norms > 0
        if np.any(keep):
            cosine = np.sum(a[keep] * b[keep], axis=1) / norms[keep]
            angle = np.arccos(np.clip(cosine, -1.0, 1.0))
            sam = {"status": "EVALUABLE", "mean_angle_rad": float(np.mean(angle)), "mean_angle_deg": float(np.rad2deg(np.mean(angle))), "joint_pixels": int(keep.sum())}
    return {"status": "EVALUABLE" if all(row["status"] == "EVALUABLE" for row in rows) else "PARTIAL_SUPPORT",
            "per_band": rows, "scale_policy": name,
            "normalized_MAE_macro_mean": float(np.mean(normalized)) if normalized else None,
            "raw_MAE_pooled": None,
            "SAM": sam,
            "SAC": {"status": "CONDITIONAL_NOT_EVALUABLE", "reason": "SAC definition not frozen in R1"}}


def structure_metrics(reference, corrected, mask, *, co_registered: bool, scale_range: float | None, window_size: int = 7) -> dict[str, Any]:
    if not co_registered or scale_range is None or not np.isfinite(scale_range) or scale_range <= 0:
        return {"status": "CONDITIONAL_NOT_EVALUABLE", "reason": "confirmed common grid and positive data scale range are required", "gradient_NCC": None, "SSIM": None, "support_pixels": 0}
    a, b, valid = np.asarray(reference, dtype=np.float64), np.asarray(corrected, dtype=np.float64), np.asarray(mask, dtype=bool)
    if a.shape != b.shape or a.shape != valid.shape or a.ndim != 2:
        raise ValueError("structure metric inputs must be identical 2D arrays")
    support = valid & np.isfinite(a) & np.isfinite(b)
    if not np.any(support):
        return {"status": "NO_VALID_SUPPORT", "gradient_NCC": None, "SSIM": None, "support_pixels": 0}
    safe_a, safe_b = np.where(support, a, 0.0), np.where(support, b, 0.0)
    safe_support = support.astype(np.float64)
    eroded = binary_erosion(support, structure=np.ones((3, 3), bool))
    gx_a, gy_a = sobel(safe_a, axis=1), sobel(safe_a, axis=0)
    gx_b, gy_b = sobel(safe_b, axis=1), sobel(safe_b, axis=0)
    mag_a, mag_b = np.hypot(gx_a, gy_a)[eroded], np.hypot(gx_b, gy_b)[eroded]
    if not mag_a.size:
        return {"status": "NO_VALID_SUPPORT", "gradient_NCC": None, "SSIM": None, "support_pixels": 0}
    centered_a, centered_b = mag_a - mag_a.mean(), mag_b - mag_b.mean()
    denom = float(np.sqrt(np.sum(centered_a ** 2) * np.sum(centered_b ** 2)))
    gradient_ncc = (1.0 if np.array_equal(mag_a, mag_b) else None) if denom == 0 else float(np.sum(centered_a * centered_b) / denom)
    size = int(window_size)
    if size < 3 or size % 2 == 0:
        raise ValueError("SSIM window_size must be odd and at least 3")
    count = uniform_filter(safe_support, size=size, mode="constant", cval=0.0)
    full_windows = count >= 1.0 - 1e-12
    if not full_windows.any():
        return {"status": "PARTIAL_SUPPORT", "gradient_NCC": gradient_ncc, "SSIM": None, "support_pixels": int(support.sum()), "ssim_windows": 0}
    mu_a = uniform_filter(safe_a, size=size, mode="constant", cval=0.0)
    mu_b = uniform_filter(safe_b, size=size, mode="constant", cval=0.0)
    var_a = np.maximum(0.0, uniform_filter(safe_a ** 2, size=size, mode="constant", cval=0.0) - mu_a ** 2)
    var_b = np.maximum(0.0, uniform_filter(safe_b ** 2, size=size, mode="constant", cval=0.0) - mu_b ** 2)
    cov = uniform_filter(safe_a * safe_b, size=size, mode="constant", cval=0.0) - mu_a * mu_b
    c1, c2 = (0.01 * scale_range) ** 2, (0.03 * scale_range) ** 2
    ssim_map = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (var_a + var_b + c2))
    return {"status": "EVALUABLE", "gradient_NCC": gradient_ncc, "SSIM": float(np.mean(ssim_map[full_windows])),
            "support_pixels": int(support.sum()), "ssim_windows": int(full_windows.sum()), "scale_range": float(scale_range)}


def aggregate_by_roi_pair_band_time(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize evaluable ROI rows with ROI-macro then pair-macro aggregation."""
    grouped: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    pair_values: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    nonevaluable: set[tuple[str, str, str]] = set()
    for row in rows:
        pair, band, time_bin = str(row.get("pair_id", "")), str(row.get("band", "")), str(row.get("time_bin", ""))
        pair_key = (pair, band, time_bin)
        value = row.get("MAE")
        if row.get("status") != "EVALUABLE" or value is None or not np.isfinite(float(value)):
            nonevaluable.add(pair_key)
            continue
        key = (pair, str(row.get("roi_id", "")), band, time_bin)
        grouped[key].append(float(value))
    for (pair, _roi, band, time_bin), values in grouped.items():
        pair_values[(pair, band, time_bin)].append(float(np.mean(values)))
    pair_means = [float(np.mean(values)) for values in pair_values.values()]
    return {"groups": {"|".join(key): float(np.mean(value)) for key, value in sorted(grouped.items())},
            "pair_means": {"|".join(key): float(np.mean(value)) for key, value in sorted(pair_values.items())},
            "pair_macro_mean_MAE": float(np.mean(pair_means)) if pair_means else None,
            "pair_MAE_distribution": sorted(pair_means), "evaluable_pairs": len(pair_values),
            "nonevaluable_pairs": len(nonevaluable), "raw_pixel_weighted_MAE": None}
