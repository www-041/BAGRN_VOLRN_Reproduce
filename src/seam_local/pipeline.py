"""Pure, deterministic Task13A processing of one frozen BAGRN scene pair.

This module accepts arrays and masks only. File loading, artifact writing, and
batch hard-stop accounting belong to the Task13A runner.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt, label, sobel

from src.multiscene_sift.radiometric_metrics import compute_cgl, compute_seam_zone_metrics
from src.registration_benchmark.metrics import gradient_ncc

from .blend import blend_across_seam
from .cost import compute_seam_cost
from .local_moment import LocalMomentSegment, apply_seam_local_correction
from .seam import SeamResult, find_monotonic_seam


@dataclass(frozen=True)
class PairResult:
    status: str
    v0_status: str
    v1_status: str
    v2_status: str
    initial_seam: SeamResult | None
    refined_seam: SeamResult | None
    v0_mosaic: np.ndarray | None
    v1_mosaic: np.ndarray | None
    v2_mosaic: np.ndarray | None
    corrected_a: np.ndarray | None
    corrected_b: np.ndarray | None
    valid_union: np.ndarray | None
    crop_origin: tuple[int, int] | None
    metrics: dict[str, Any]
    diagnostics: dict[str, Any]
    local_segments: tuple[LocalMomentSegment, ...]


def _failed(
    status: str, *, v0_status: str | None = None, v1_status: str | None = None,
    v2_status: str | None = None, **updates: Any,
) -> PairResult:
    fields: dict[str, Any] = dict(
        status=status,
        v0_status=v0_status or status,
        v1_status="PASS" if v1_status == "OK" else (v1_status or status),
        v2_status="PASS" if v2_status == "OK" else (v2_status or status),
        initial_seam=None, refined_seam=None, v0_mosaic=None, v1_mosaic=None,
        v2_mosaic=None, corrected_a=None, corrected_b=None,
        valid_union=None, crop_origin=None, metrics={}, diagnostics={},
        local_segments=(),
    )
    fields.update(updates)
    return PairResult(**fields)


def _corridor(seam: SeamResult, shape: tuple[int, int], half_width: int) -> np.ndarray:
    """Boolean per-pixel corridor around an ordered spanning seam."""
    if seam.orientation == "vertical":
        centers = seam.row_col_path[:, 1][:, None]
        return np.abs(np.arange(shape[1])[None, :] - centers) <= half_width
    centers = seam.row_col_path[:, 0][None, :]
    return np.abs(np.arange(shape[0])[:, None] - centers) <= half_width


def _move_seam(seam: SeamResult, row_offset: int, col_offset: int) -> SeamResult:
    path = seam.row_col_path.copy()
    path[:, 0] += row_offset
    path[:, 1] += col_offset
    return replace(seam, row_col_path=path)


def _crop_for_pair(
    a: np.ndarray, b: np.ndarray, va: np.ndarray, vb: np.ndarray,
) -> tuple[tuple[slice, slice], tuple[slice, slice], tuple[int, int]] | None:
    """Crop long axis to overlap and retain transverse exclusive support.

    The overlap window drives seam orientation; the wider output crop retains
    exclusive source pixels needed to determine the A/B side geometrically.
    """
    joint = va & vb
    rows, cols = np.where(joint)
    if rows.size == 0:
        return None
    r0, r1, c0, c1 = int(rows.min()), int(rows.max()) + 1, int(cols.min()), int(cols.max()) + 1
    if r1 - r0 >= c1 - c0:
        support = va[r0:r1] | vb[r0:r1]
        transverse = np.flatnonzero(np.any(support, axis=0))
        outer = (slice(r0, r1), slice(int(transverse[0]), int(transverse[-1]) + 1))
    else:
        support = va[:, c0:c1] | vb[:, c0:c1]
        transverse = np.flatnonzero(np.any(support, axis=1))
        outer = (slice(int(transverse[0]), int(transverse[-1]) + 1), slice(c0, c1))
    overlap = (slice(r0, r1), slice(c0, c1))
    return outer, overlap, (r0, c0)


def _weighted_feather_pair(
    a: np.ndarray, b: np.ndarray, va: np.ndarray, vb: np.ndarray,
    outer: tuple[slice, slice],
) -> np.ndarray:
    """Reproduce the existing weighted mode's full-grid EDT weights for two scenes."""
    # Compute on the canonical masks before cropping: EDT values otherwise
    # change at crop boundaries and V0 ceases to match existing mosaicking.
    weight_a = distance_transform_edt(va)[outer]
    weight_b = distance_transform_edt(vb)[outer]
    va_crop, vb_crop = va[outer], vb[outer]
    weight_a = np.where(va_crop, weight_a + 1e-6, 0.0)
    weight_b = np.where(vb_crop, weight_b + 1e-6, 0.0)
    numerator = np.where(va_crop, a[outer], 0.0) * weight_a
    numerator += np.where(vb_crop, b[outer], 0.0) * weight_b
    result = np.full(va_crop.shape, np.nan, dtype=np.float64)
    denominator = weight_a + weight_b
    np.divide(numerator, denominator, out=result, where=denominator > 1e-12)
    return result


def _radiometric_metrics(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> dict[str, float | int | None]:
    result = compute_seam_zone_metrics(a, b, mask)
    if result["status"] != "PASS":
        return {"valid_pixels": 0, "mae": None, "rmse": None, "rdd": None,
                "mean_difference": None, "std_difference": None}
    delta = a[mask] - b[mask]
    return {
        "valid_pixels": result["valid_pixels"],
        "mae": result["seam_mae"], "rmse": result["seam_rmse"],
        "rdd": result["seam_rdd"],
        "mean_difference": float(np.mean(delta)),
        "std_difference": float(np.std(delta)),
    }


def _actual_seam_metrics(
    a: np.ndarray, b: np.ndarray, valid: np.ndarray, seam: SeamResult,
) -> dict[str, float | int | None]:
    metrics = _radiometric_metrics(a, b, valid & _corridor(seam, a.shape, 64))
    metrics["mean_absolute_cost"] = seam.mean_cost
    metrics["p95_cost"] = seam.p95_cost
    return metrics


def _structure_metrics(
    source: np.ndarray, corrected: np.ndarray, valid: np.ndarray,
) -> dict[str, Any]:
    # CGL remains the existing Task10D definition. The cosine statistic uses
    # the same eroded, nonzero-gradient support and independent Sobel vectors.
    cgl = compute_cgl(source, corrected, valid)
    finite_stencil = binary_erosion(
        valid & np.isfinite(source) & np.isfinite(corrected),
        structure=np.ones((3, 3), dtype=bool),
    )
    ncc = float(gradient_ncc(source, corrected, finite_stencil.copy()))
    safe_source = np.where(np.isfinite(source), source, 0.0)
    safe_corrected = np.where(np.isfinite(corrected), corrected, 0.0)
    ax, ay = sobel(safe_source, axis=1, mode="nearest"), sobel(safe_source, axis=0, mode="nearest")
    bx, by = sobel(safe_corrected, axis=1, mode="nearest"), sobel(safe_corrected, axis=0, mode="nearest")
    norm_a, norm_b = np.hypot(ax, ay), np.hypot(bx, by)
    eligible = binary_erosion(valid & np.isfinite(source) & np.isfinite(corrected),
                              structure=np.ones((3, 3), dtype=bool))
    eligible &= (norm_a > 1e-12) & (norm_b > 1e-12)
    cosine = None
    if np.any(eligible):
        cosine = float(np.mean((ax[eligible] * bx[eligible] + ay[eligible] * by[eligible])
                               / (norm_a[eligible] * norm_b[eligible])))
    return {
        "cgl": cgl,
        "gradient_magnitude_ncc": ncc if np.isfinite(ncc) else None,
        "gradient_orientation_cosine": cosine,
    }


def process_pair(
    image_a: np.ndarray, image_b: np.ndarray,
    valid_a: np.ndarray, valid_b: np.ndarray,
) -> PairResult:
    """Run V1 and V2 on one pair; return diagnostics even for pair failures."""
    a, b = np.asarray(image_a, dtype=np.float64), np.asarray(image_b, dtype=np.float64)
    va, vb = np.asarray(valid_a, dtype=bool), np.asarray(valid_b, dtype=bool)
    if a.ndim != 2 or 0 in a.shape or any(x.shape != a.shape for x in (b, va, vb)):
        raise ValueError("images and masks must be same-shape nonempty 2D arrays")
    if not np.isfinite(a[va]).all() or not np.isfinite(b[vb]).all():
        return _failed("NUMERICAL_INVALID")
    windows = _crop_for_pair(a, b, va, vb)
    if windows is None:
        return _failed("UNSUPPORTED_TOPOLOGY")
    outer, overlap, (overlap_r0, overlap_c0) = windows
    v0_mosaic = _weighted_feather_pair(a, b, va, vb, outer)
    if not np.isfinite(v0_mosaic[(va | vb)[outer]]).all():
        return _failed("NUMERICAL_INVALID", v0_status="NUMERICAL_INVALID")
    origin = (outer[0].start, outer[1].start)
    local_overlap = (
        slice(overlap[0].start - origin[0], overlap[0].stop - origin[0]),
        slice(overlap[1].start - origin[1], overlap[1].stop - origin[1]),
    )
    a, b, va, vb = a[outer], b[outer], va[outer], vb[outer]
    joint = va & vb
    union = va | vb
    overlap_joint = joint[local_overlap]
    component_count = label(overlap_joint, structure=np.ones((3, 3), dtype=np.uint8))[1]
    if component_count != 1:
        return _failed("UNSUPPORTED_TOPOLOGY", v0_status="PASS",
                       v0_mosaic=v0_mosaic, valid_union=union,
                       crop_origin=origin,
                       diagnostics={"joint_overlap_components": int(component_count)})
    initial_cost = compute_seam_cost(a[local_overlap], b[local_overlap], overlap_joint)
    initial_overlap = find_monotonic_seam(initial_cost, overlap_joint)
    diagnostics: dict[str, Any] = {
        "joint_overlap_components": int(component_count),
        "initial_search_mode": initial_overlap.search_mode,
        "crop_origin": origin,
        "overlap_origin": (overlap_r0, overlap_c0),
    }
    if initial_overlap.status != "OK":
        return _failed("UNSUPPORTED_TOPOLOGY", v0_status="PASS", v0_mosaic=v0_mosaic,
                       valid_union=union,
                       crop_origin=origin, diagnostics=diagnostics)
    initial = _move_seam(initial_overlap, local_overlap[0].start, local_overlap[1].start)
    fixed_mask = joint & _corridor(initial, a.shape, 128)
    metrics: dict[str, Any] = {
        "v0": {"blend_method": "existing_distance_weighted_feather",
               "valid_pixels": int(np.count_nonzero(union)),
               "finite_pixels": int(np.count_nonzero(np.isfinite(v0_mosaic) & union))},
        "fixed_corridor": {"bagrn": _radiometric_metrics(a, b, fixed_mask)},
        "actual_seam": {"v1": _actual_seam_metrics(a, b, joint, initial)},
    }
    v1 = blend_across_seam(a, b, initial, va, vb)
    diagnostics["v1_blend_pixels"] = v1.blend_pixels
    diagnostics["v1_source_side"] = v1.source_side
    base = dict(initial_seam=initial, v0_status="PASS", v0_mosaic=v0_mosaic,
                v1_mosaic=v1.image,
                valid_union=union, crop_origin=origin,
                metrics=metrics, diagnostics=diagnostics)
    local = apply_seam_local_correction(a, b, va, vb, initial)
    diagnostics.update(gain_min=local.gain_min, gain_max=local.gain_max,
                       offset_min=local.b_min, offset_max=local.b_max,
                       fallback_segments=sum(segment.fallback_from is not None
                                             for segment in local.segments))
    base.update(corrected_a=local.corrected_a, corrected_b=local.corrected_b,
                local_segments=local.segments)
    if local.status != "OK":
        return _failed(local.status, v1_status=v1.status, **base)
    metrics["fixed_corridor"]["local"] = _radiometric_metrics(
        local.corrected_a, local.corrected_b, fixed_mask)
    metrics["structure"] = {
        "a": _structure_metrics(a, local.corrected_a, fixed_mask & va),
        "b": _structure_metrics(b, local.corrected_b, fixed_mask & vb),
    }
    refined_cost = compute_seam_cost(
        local.corrected_a[local_overlap], local.corrected_b[local_overlap], overlap_joint)
    restricted = overlap_joint & _corridor(initial_overlap, overlap_joint.shape, 64)
    refined_overlap = find_monotonic_seam(refined_cost, restricted, coarse_factor=1)
    diagnostics["refined_search_mode"] = refined_overlap.search_mode
    if refined_overlap.status != "OK":
        return _failed("UNSUPPORTED_TOPOLOGY", v1_status=v1.status, **base)
    refined = _move_seam(refined_overlap, local_overlap[0].start, local_overlap[1].start)
    metrics["actual_seam"]["v2"] = _actual_seam_metrics(
        local.corrected_a, local.corrected_b, joint, refined)
    v2 = blend_across_seam(local.corrected_a, local.corrected_b, refined, va, vb)
    diagnostics["v2_blend_pixels"] = v2.blend_pixels
    diagnostics["v2_source_side"] = v2.source_side
    base.update(refined_seam=refined, v2_mosaic=v2.image)
    status = "PASS" if v1.status == v2.status == "OK" else (
        "NUMERICAL_INVALID" if "NUMERICAL_INVALID" in (v1.status, v2.status)
        else "AMBIGUOUS_SOURCE_SIDE")
    return _failed(status, v1_status=v1.status, v2_status=v2.status, **base) if status != "PASS" else PairResult(
        status="PASS", v1_status="PASS", v2_status="PASS", **base)
