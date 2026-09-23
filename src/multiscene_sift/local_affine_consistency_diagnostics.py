"""Diagnosis-only local affine consistency probes.

This module consumes saved global-RANSAC evidence and never changes the
production registration, network, radiometric, or mosaic paths.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


LOCAL_AFFINE_EDGES = {
    (0, 6): "GOOD_REFERENCE",
    (2, 5): "HIGH_INLIER_FALSE_GOOD",
    (0, 5): "LOW_SUPPORT_CONTROL",
}
MIN_POINTS_FOR_LOCAL_AFFINE = 12
ALLOWED_GRIDS = (2, 3)
BASELINE_ARTIFACTS = (
    "00_edge_reliability_baseline.json",
    "01_inlier_reproduction_summary.json",
    "01_inlier_points.csv",
    "02_inlier_spatial_metrics.json",
    "07_direct_residual_summary.json",
    "08_residual_vs_control_support_summary.json",
    "11_good_vs_false_good_comparison.json",
    "12_edge_diagnoses.json",
    "13_edge_reliability_conclusion.json",
)


def _edge_key(edge: str | tuple[int, int] | list[int]) -> str:
    if isinstance(edge, str):
        parts = edge.split("-")
        if len(parts) != 2:
            raise ValueError(f"Invalid edge key: {edge!r}")
        i, j = (int(parts[0]), int(parts[1]))
    else:
        if len(edge) != 2:
            raise ValueError(f"Invalid edge: {edge!r}")
        i, j = int(edge[0]), int(edge[1])
    return f"{i}-{j}"


def _load_json(path: Path, default: Any) -> Any:
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _rows_by_edge(value: Any) -> dict[str, dict]:
    if isinstance(value, dict):
        if isinstance(value.get("edges"), list):
            value = value["edges"]
        elif all(isinstance(v, dict) for v in value.values()):
            return {_edge_key(k): v for k, v in value.items() if "-" in str(k)}
        else:
            return {}
    if not isinstance(value, list):
        return {}
    out = {}
    for row in value:
        if not isinstance(row, dict):
            continue
        edge = row.get("edge")
        if edge is None:
            edge = row.get("edge_key")
        if edge is not None:
            out[_edge_key(edge)] = row
    return out


def load_local_affine_baseline(edge_reliability_dir: str | Path) -> dict:
    """Load the frozen evidence needed by the three-edge local probe.

    The loader is intentionally strict about the two high-support edges and
    does not reconstruct missing inliers or rerun a matcher.  Each edge record
    contains the original coordinates and any saved residual/direct-overlap
    context that is available in the prior diagnostic artifacts.
    """
    root = Path(edge_reliability_dir)
    missing = [name for name in BASELINE_ARTIFACTS if not (root / name).is_file()]
    baseline = _rows_by_edge(_load_json(root / "00_edge_reliability_baseline.json", {}))
    points: dict[str, list[dict]] = {}
    point_path = root / "01_inlier_points.csv"
    if point_path.is_file():
        with point_path.open("r", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                key = _edge_key((int(row["edge_i"]), int(row["edge_j"])))
                points.setdefault(key, []).append(row)

    contexts = {
        name: _rows_by_edge(_load_json(root / name, {}))
        for name in BASELINE_ARTIFACTS
        if name.endswith(".json")
    }
    critical_missing = [
        name for name in ("00_edge_reliability_baseline.json", "01_inlier_points.csv")
        if name in missing
    ]
    edges: dict[str, dict] = {}
    for edge, group in LOCAL_AFFINE_EDGES.items():
        key = _edge_key(edge)
        rows = points.get(key, [])
        ref_xy = np.asarray(
            [[float(r["ref_x"]), float(r["ref_y"])] for r in rows], dtype=np.float64
        ).reshape((-1, 2))
        tgt_xy = np.asarray(
            [[float(r["tgt_x"]), float(r["tgt_y"])] for r in rows], dtype=np.float64
        ).reshape((-1, 2))
        edge_context = dict(baseline.get(key, {}))
        edge_context.update({name: values.get(key) for name, values in contexts.items()})
        edges[key] = {
            "edge": [edge[0], edge[1]],
            "group": group,
            "ref_xy": ref_xy,
            "tgt_xy": tgt_xy,
            "residual_px": np.asarray(
                [float(r.get("residual_px", "nan")) for r in rows], dtype=np.float64
            ),
            "global_matrix": edge_context.get("pixel_matrix"),
            "direct_overlap": edge_context.get("07_direct_residual_summary.json"),
            "context": edge_context,
        }

    return {
        "edge_keys": [_edge_key(edge) for edge in LOCAL_AFFINE_EDGES],
        "edges": edges,
        "artifact_paths": {name: str(root / name) for name in BASELINE_ARTIFACTS},
        "missing_artifacts": missing,
        "missing_critical_artifacts": critical_missing,
    }


def _as_points(points: np.ndarray) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("points must have shape (N, 2)")
    return arr


def _normalization(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.mean(points, axis=0)
    scale = float(np.sqrt(np.mean(np.sum((points - center) ** 2, axis=1))))
    if not np.isfinite(scale) or scale < 1e-12:
        scale = 1.0
    transform = np.array(
        [[1.0 / scale, 0.0, -center[0] / scale],
         [0.0, 1.0 / scale, -center[1] / scale],
         [0.0, 0.0, 1.0]], dtype=np.float64
    )
    return transform, (points - center) / scale


def fit_affine_least_squares(src_xy: np.ndarray, dst_xy: np.ndarray) -> dict:
    """Fit ``src -> dst`` with normalized-coordinate deterministic LS."""
    src = _as_points(src_xy)
    dst = _as_points(dst_xy)
    if len(src) != len(dst):
        raise ValueError("src_xy and dst_xy must contain the same number of points")
    if len(src) < MIN_POINTS_FOR_LOCAL_AFFINE:
        return {
            "status": "INSUFFICIENT_POINTS", "matrix_3x3": None,
            "n_points": int(len(src)), "rank": 0,
            "condition_number": float("inf"), "rmse_px": None, "p95_px": None,
        }
    if not np.isfinite(src).all() or not np.isfinite(dst).all():
        return {
            "status": "DEGENERATE", "matrix_3x3": None,
            "n_points": int(len(src)), "rank": 0,
            "condition_number": float("inf"), "rmse_px": None, "p95_px": None,
        }
    src_t, src_n = _normalization(src)
    dst_t, dst_n = _normalization(dst)
    design = np.column_stack((src_n, np.ones(len(src_n))))
    rank = int(np.linalg.matrix_rank(design))
    condition_number = float(np.linalg.cond(design))
    if rank < 3 or not np.isfinite(condition_number) or condition_number > 1e12:
        return {
            "status": "DEGENERATE", "matrix_3x3": None,
            "n_points": int(len(src)), "rank": rank,
            "condition_number": condition_number, "rmse_px": None, "p95_px": None,
        }
    coeff, _, _, _ = np.linalg.lstsq(design, dst_n, rcond=None)
    normalized_matrix = np.array(
        [[coeff[0, 0], coeff[1, 0], coeff[2, 0]],
         [coeff[0, 1], coeff[1, 1], coeff[2, 1]],
         [0.0, 0.0, 1.0]], dtype=np.float64
    )
    matrix = np.linalg.inv(dst_t) @ normalized_matrix @ src_t
    predicted = _apply_matrix(matrix, src)
    residual = np.linalg.norm(predicted - dst, axis=1)
    return {
        "status": "OK", "matrix_3x3": matrix,
        "n_points": int(len(src)), "rank": rank,
        "condition_number": condition_number,
        "rmse_px": float(np.sqrt(np.mean(residual ** 2))),
        "p95_px": float(np.percentile(residual, 95)),
    }


def _apply_matrix(matrix: np.ndarray, xy: np.ndarray) -> np.ndarray:
    pts = _as_points(xy)
    m = np.asarray(matrix, dtype=np.float64)
    homogeneous = np.column_stack((pts, np.ones(len(pts))))
    out = (m @ homogeneous.T).T
    denom = out[:, 2:3]
    return out[:, :2] / np.where(np.abs(denom) < 1e-12, 1.0, denom)


def assign_points_to_overlap_regions(
    xy: np.ndarray,
    overlap_bounds_px,
    grid_n: int,
) -> np.ndarray:
    """Assign overlap-frame points to row-major 2×2 or 3×3 cells."""
    if grid_n not in ALLOWED_GRIDS:
        raise ValueError(f"grid_n must be one of {ALLOWED_GRIDS}, got {grid_n}")
    points = _as_points(xy)
    x0, y0, x1, y1 = map(float, overlap_bounds_px)
    if not x1 > x0 or not y1 > y0:
        raise ValueError("overlap_bounds_px must be (left, top, right, bottom)")
    u = np.clip((points[:, 0] - x0) / (x1 - x0), 0.0, 1.0)
    v = np.clip((points[:, 1] - y0) / (y1 - y0), 0.0, 1.0)
    col = np.minimum(np.floor(u * grid_n).astype(int), grid_n - 1)
    row = np.minimum(np.floor(v * grid_n).astype(int), grid_n - 1)
    return row * grid_n + col


def _region_bounds(bounds, grid_n: int, row: int, col: int) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = map(float, bounds)
    dx = (x1 - x0) / grid_n
    dy = (y1 - y0) / grid_n
    return (x0 + col * dx, y0 + row * dy,
            x1 if col == grid_n - 1 else x0 + (col + 1) * dx,
            y1 if row == grid_n - 1 else y0 + (row + 1) * dy)


def _decompose_affine(matrix: np.ndarray) -> dict:
    m = np.asarray(matrix, dtype=np.float64)
    linear = m[:2, :2]
    try:
        u, singular, vt = np.linalg.svd(linear)
        rotation = u @ vt
        if np.linalg.det(rotation) < 0:
            u[:, -1] *= -1
            singular[-1] *= -1
            rotation = u @ vt
        stretch = rotation.T @ linear
        rotation_deg = float(np.degrees(np.arctan2(rotation[1, 0], rotation[0, 0])))
        scale_x = float(stretch[0, 0])
        scale_y = float(stretch[1, 1])
        shear_deg = float(np.degrees(np.arctan2(stretch[0, 1], max(abs(scale_y), 1e-12))))
    except np.linalg.LinAlgError:
        rotation_deg = scale_x = scale_y = shear_deg = float("nan")
    return {
        "translation_x": float(m[0, 2]),
        "translation_y": float(m[1, 2]),
        "rotation_deg": rotation_deg,
        "scale_x": scale_x,
        "scale_y": scale_y,
        "shear_deg": shear_deg,
    }


def compare_local_to_global_at_points(
    local_matrix: np.ndarray,
    global_matrix: np.ndarray,
    eval_xy: np.ndarray,
) -> dict:
    """Compare predictions in the same pixel frame, not raw translations."""
    points = _as_points(eval_xy)
    local = _apply_matrix(local_matrix, points)
    global_ = _apply_matrix(global_matrix, points)
    delta = local - global_
    magnitudes = np.linalg.norm(delta, axis=1)
    center_delta = delta[0]
    centroid_delta = np.mean(delta, axis=0)
    local_parts = _decompose_affine(local_matrix)
    global_parts = _decompose_affine(global_matrix)
    return {
        "center_delta_dx_px": float(center_delta[0]),
        "center_delta_dy_px": float(center_delta[1]),
        "center_delta_mag_px": float(magnitudes[0]),
        "inlier_centroid_delta_dx_px": float(centroid_delta[0]),
        "inlier_centroid_delta_dy_px": float(centroid_delta[1]),
        "inlier_centroid_delta_mag_px": float(np.linalg.norm(centroid_delta)),
        "rotation_delta_deg": float(local_parts["rotation_deg"] - global_parts["rotation_deg"]),
        "scale_delta_x": float(local_parts["scale_x"] - global_parts["scale_x"]),
        "scale_delta_y": float(local_parts["scale_y"] - global_parts["scale_y"]),
        "shear_delta_deg": float(local_parts["shear_deg"] - global_parts["shear_deg"]),
        "delta_vectors": delta,
        "delta_magnitudes": magnitudes,
    }


def fit_region_local_affines(
    ref_xy: np.ndarray,
    tgt_xy: np.ndarray,
    global_matrix: np.ndarray,
    overlap_bounds,
    grid_n: int,
) -> list[dict]:
    """Fit deterministic local models using only the supplied global inliers."""
    ref = _as_points(ref_xy)
    tgt = _as_points(tgt_xy)
    if len(ref) != len(tgt):
        raise ValueError("ref_xy and tgt_xy must contain the same number of points")
    labels = assign_points_to_overlap_regions(ref, overlap_bounds, grid_n)
    models: list[dict] = []
    for region_id in range(grid_n * grid_n):
        row, col = divmod(region_id, grid_n)
        mask = labels == region_id
        region_ref = ref[mask]
        region_tgt = tgt[mask]
        bounds = _region_bounds(overlap_bounds, grid_n, row, col)
        center = np.array([[(bounds[0] + bounds[2]) / 2.0, (bounds[1] + bounds[3]) / 2.0]])
        base = {
            "grid_n": grid_n, "region_id": region_id, "region_row": row,
            "region_col": col, "bounds_px": bounds, "center_xy": center[0],
            "n_points": int(len(region_ref)), "status": "INSUFFICIENT_POINTS",
            "local_matrix": None, "global_matrix": np.asarray(global_matrix, dtype=float),
            "condition_number": float("inf"), "local_rmse": None, "local_p95": None,
        }
        fit = fit_affine_least_squares(region_tgt, region_ref)
        base.update({
            "status": fit["status"], "local_matrix": fit["matrix_3x3"],
            "condition_number": fit["condition_number"],
            "local_rmse": fit["rmse_px"], "local_p95": fit["p95_px"],
        })
        if fit["status"] == "OK":
            base.update(_decompose_affine(fit["matrix_3x3"]))
            comparison = compare_local_to_global_at_points(
                fit["matrix_3x3"], global_matrix, np.vstack((center, np.mean(region_ref, axis=0)))
            )
            base.update(comparison)
            base["inlier_centroid_xy"] = np.mean(region_ref, axis=0)
        else:
            base.update({
                "translation_x": None, "translation_y": None,
                "rotation_deg": None, "scale_x": None, "scale_y": None,
                "shear_deg": None,
            })
        models.append(base)
    return models


def summarize_local_affine_variation(region_models, grid_n: int) -> dict:
    """Summarize spatial variation without treating it as absolute geolocation error."""
    models = list(region_models)
    fitted = [m for m in models if m.get("status") == "OK"]
    center_deltas = np.asarray([m["center_delta_mag_px"] for m in fitted], dtype=float)
    translations = np.asarray([
        [m["center_delta_dx_px"], m["center_delta_dy_px"]] for m in fitted
    ], dtype=float)
    rotations = np.asarray([m["rotation_delta_deg"] for m in fitted], dtype=float)
    scales = np.asarray([[m["scale_delta_x"], m["scale_delta_y"]] for m in fitted], dtype=float)
    shears = np.asarray([m["shear_delta_deg"] for m in fitted], dtype=float)

    def _range(values):
        if np.asarray(values).size == 0:
            return [None, None]
        a = np.asarray(values, dtype=float)
        return [float(np.nanmin(a)), float(np.nanmax(a))]

    result = {
        "grid_n": int(grid_n), "n_regions_total": int(len(models)),
        "n_regions_fittable": int(len(fitted)),
        "median_center_delta_mag_px": float(np.median(center_deltas)) if len(center_deltas) else None,
        "p95_center_delta_mag_px": float(np.percentile(center_deltas, 95)) if len(center_deltas) else None,
        "max_center_delta_mag_px": float(np.max(center_deltas)) if len(center_deltas) else None,
        "translation_delta_range_px": _range(np.linalg.norm(translations, axis=1) if len(translations) else []),
        "rotation_delta_range_deg": _range(rotations),
        "scale_delta_range": _range(scales),
        "shear_delta_range_deg": _range(shears),
        "dx_r2": None, "dy_r2": None, "predicted_delta_range_px": [None, None],
    }
    if len(fitted) >= 3:
        centers = np.asarray([m["center_xy"] for m in fitted], dtype=float)
        bounds_min = centers.min(axis=0)
        bounds_max = centers.max(axis=0)
        span = np.where(bounds_max > bounds_min, bounds_max - bounds_min, 1.0)
        uv = (centers - bounds_min) / span
        design = np.column_stack((np.ones(len(uv)), uv))
        predictions = []
        r2s = []
        for axis in (0, 1):
            target = translations[:, axis]
            coef, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
            pred = design @ coef
            ss_tot = float(np.sum((target - target.mean()) ** 2))
            r2s.append(1.0 - float(np.sum((target - pred) ** 2)) / ss_tot if ss_tot > 1e-12 else 1.0)
            predictions.append(pred)
        pred_mag = np.linalg.norm(np.column_stack(predictions), axis=1)
        result["dx_r2"], result["dy_r2"] = map(float, r2s)
        result["predicted_delta_range_px"] = [float(pred_mag.min()), float(pred_mag.max())]
    return result


def cross_validate_local_affine(
    src_xy: np.ndarray,
    dst_xy: np.ndarray,
    n_splits: int = 5,
    seed: int = 0,
) -> dict:
    """Deterministic held-out comparison of global LS vs spatial-neighbour LS."""
    src = _as_points(src_xy)
    dst = _as_points(dst_xy)
    if len(src) < max(n_splits * 3, MIN_POINTS_FOR_LOCAL_AFFINE + 1):
        return {"status": "INSUFFICIENT_POINTS", "folds": [], "summary": {"n_folds": 0}}
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(src))
    folds = np.array_split(order, n_splits)
    rows = []
    for fold_id, heldout in enumerate(folds):
        train = np.setdiff1d(order, heldout, assume_unique=False)
        global_fit = fit_affine_least_squares(src[train], dst[train])
        if global_fit["status"] != "OK":
            continue
        global_pred = _apply_matrix(global_fit["matrix_3x3"], src[heldout])
        global_error = np.linalg.norm(global_pred - dst[heldout], axis=1)
        local_pred = np.empty_like(global_pred)
        local_ok = 0
        for idx, point in enumerate(src[heldout]):
            distances = np.linalg.norm(src[train] - point, axis=1)
            neighbours = train[np.argsort(distances)[:min(32, len(train))]]
            local_fit = fit_affine_least_squares(src[neighbours], dst[neighbours])
            if local_fit["status"] == "OK":
                local_pred[idx] = _apply_matrix(local_fit["matrix_3x3"], point.reshape(1, 2))[0]
                local_ok += 1
            else:
                local_pred[idx] = global_pred[idx]
        local_error = np.linalg.norm(local_pred - dst[heldout], axis=1)
        global_rmse = float(np.sqrt(np.mean(global_error ** 2)))
        local_rmse = float(np.sqrt(np.mean(local_error ** 2)))
        rows.append({
            "fold": fold_id, "n_train": int(len(train)), "n_test": int(len(heldout)),
            "n_local_predictions": int(local_ok), "global_affine_rmse": global_rmse,
            "local_affine_rmse": local_rmse, "improvement_px": global_rmse - local_rmse,
            "improvement_ratio": (global_rmse - local_rmse) / global_rmse if global_rmse else 0.0,
        })
    improvements = np.asarray([r["improvement_px"] for r in rows], dtype=float)
    return {
        "status": "OK" if rows else "UNAVAILABLE", "folds": rows,
        "summary": {
            "n_folds": len(rows),
            "global_affine_rmse": float(np.mean([r["global_affine_rmse"] for r in rows])) if rows else None,
            "local_affine_rmse": float(np.mean([r["local_affine_rmse"] for r in rows])) if rows else None,
            "improvement_px": float(np.mean(improvements)) if len(improvements) else None,
            "improvement_ratio": float(np.mean([r["improvement_ratio"] for r in rows])) if rows else None,
        },
    }


def _scene_array(scene) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(scene, dict):
        array = scene.get("array", scene.get("data"))
        valid = scene.get("valid_mask")
    else:
        array = getattr(scene, "array", getattr(scene, "data", None))
        valid = getattr(scene, "valid_mask", None)
    if array is None:
        raise ValueError("scene must provide array/data")
    arr = np.asarray(array)
    mask = np.ones(arr.shape, dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    if arr.shape != mask.shape:
        raise ValueError("scene array and valid_mask must have the same shape")
    return arr.astype(np.float32, copy=False), mask


def build_region_validation_crops(
    scene_ref,
    scene_tgt,
    global_matrix: np.ndarray,
    local_matrix: np.ndarray,
    region: dict,
) -> dict:
    """Create global/local target crops in one reference-pixel frame."""
    import cv2

    reference, reference_valid = _scene_array(scene_ref)
    target, target_valid = _scene_array(scene_tgt)
    x0, y0, x1, y1 = (int(round(v)) for v in region["bounds_px"])
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(reference.shape[1], x1), min(reference.shape[0], y1)
    if x1 <= x0 or y1 <= y0:
        return {"status": "PIXEL_VALIDATION_UNAVAILABLE", "reason": "empty_region"}
    width, height = x1 - x0, y1 - y0
    shift = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], dtype=float)

    def warp(matrix):
        crop_matrix = shift @ np.asarray(matrix, dtype=float)
        image = cv2.warpAffine(target, crop_matrix[:2], (width, height), flags=cv2.INTER_LINEAR)
        mask = cv2.warpAffine(target_valid.astype(np.uint8), crop_matrix[:2], (width, height), flags=cv2.INTER_NEAREST) > 0
        return image, mask

    global_warp, global_valid = warp(global_matrix)
    local_warp, local_valid = warp(local_matrix)
    reference_crop = reference[y0:y1, x0:x1]
    reference_mask = reference_valid[y0:y1, x0:x1]
    joint = (
        reference_mask
        & global_valid
        & local_valid
        & np.isfinite(reference_crop)
        & np.isfinite(global_warp)
        & np.isfinite(local_warp)
    )
    if int(joint.sum()) < 20 or float(np.std(reference_crop[joint])) < 1e-6:
        status = "PIXEL_VALIDATION_UNAVAILABLE"
    else:
        status = "OK"
    return {
        "status": status, "reason": None if status == "OK" else "insufficient_valid_or_texture",
        "reference": reference_crop, "global_warp": global_warp, "local_warp": local_warp,
        "reference_valid": reference_mask, "global_valid": global_valid, "local_valid": local_valid,
        "joint_valid": joint, "bounds_px": (x0, y0, x1, y1),
    }


def compare_global_vs_local_pixel_alignment(
    ref_crop: np.ndarray,
    global_warp_crop: np.ndarray,
    local_warp_crop: np.ndarray,
    masks,
) -> dict:
    """Measure independent phase/NCC evidence on identical pixels."""
    from src.multiscene_sift.loop_diagnostics import (
        _ncc_between_overlays,
        _phase_cross_correlation_shift,
    )
    if isinstance(masks, dict):
        ref_valid = np.asarray(masks["reference_valid"], dtype=bool)
        global_valid = np.asarray(masks["global_valid"], dtype=bool)
        local_valid = np.asarray(masks["local_valid"], dtype=bool)
    else:
        ref_valid, global_valid, local_valid = masks
    joint_global = (
        ref_valid
        & global_valid
        & np.isfinite(ref_crop)
        & np.isfinite(global_warp_crop)
    )
    joint_local = (
        ref_valid
        & local_valid
        & np.isfinite(ref_crop)
        & np.isfinite(local_warp_crop)
    )
    if int(joint_global.sum()) < 20 or int(joint_local.sum()) < 20:
        return {"status": "PIXEL_VALIDATION_UNAVAILABLE", "reason": "insufficient_joint_valid"}

    def measure(moving, valid):
        a = np.where(valid, ref_crop, float(np.mean(ref_crop[valid])))
        b = np.where(valid, moving, float(np.mean(moving[valid])))
        dx, dy, score = _phase_cross_correlation_shift(a, b, upsample=5)
        return float(dx), float(dy), float(np.hypot(dx, dy)), float(score), float(_ncc_between_overlays(a, b, valid, valid))

    g = measure(global_warp_crop, joint_global)
    l = measure(local_warp_crop, joint_local)
    return {
        "status": "OK", "global_phase_dx": g[0], "global_phase_dy": g[1], "global_phase_mag": g[2],
        "local_phase_dx": l[0], "local_phase_dy": l[1], "local_phase_mag": l[2],
        "phase_improvement_px": g[2] - l[2],
        "phase_improvement_ratio": (g[2] - l[2]) / g[2] if g[2] else 0.0,
        "global_ncc": g[4], "local_ncc": l[4], "ncc_improvement": l[4] - g[4],
    }


def compare_partition_stability(grid2_summary: dict, grid3_summary: dict) -> dict:
    """Classify whether 2×2 and 3×3 evidence tells the same story."""
    summaries = (grid2_summary, grid3_summary)
    if any(s.get("n_regions_fittable", 0) < 2 for s in summaries):
        state = "LOW_SUPPORT"
    else:
        deltas = [float(s.get("p95_center_delta_mag_px") or 0.0) for s in summaries]
        phases = [float(s.get("global_phase_median") or 0.0) for s in summaries]
        improvements = [float(s.get("median_phase_improvement_px") or 0.0) for s in summaries]
        if max(deltas) < 1.0 and max(improvements) < 1.0 and max(phases) < 1.0:
            state = "STABLE_GLOBAL_CONSISTENCY"
        elif min(deltas) >= 1.0 and min(improvements) >= 1.0:
            state = "STABLE_LOCAL_VARIATION"
        else:
            state = "MIXED_OR_UNSTABLE"
    return {"state": state, "grid2": grid2_summary, "grid3": grid3_summary}


def classify_local_geometry_consistency(evidence: dict) -> str:
    """Apply the plan's evidence-gated final classifier."""
    if evidence.get("low_support"):
        return "LOW_SUPPORT_EDGE"
    if evidence.get("phase_validation_status") not in (None, "OK"):
        return "PHASE_VALIDATION_UNCERTAIN"
    state = evidence.get("partition_stability")
    differs = bool(evidence.get("local_models_differ"))
    improves = bool(evidence.get("multiple_region_phase_improvement"))
    if state == "STABLE_LOCAL_VARIATION" and differs and improves:
        return "SPATIALLY_VARYING_LOCAL_GEOMETRY_SUPPORTED"
    if state == "STABLE_GLOBAL_CONSISTENCY" and not differs and not improves:
        return "GLOBAL_AFFINE_CONSISTENT"
    if differs and not improves:
        return "LOCAL_MODEL_DOES_NOT_EXPLAIN_RESIDUAL"
    return "MIXED_OR_UNDERDETERMINED"


def build_diagnostic_evidence(
    edge: str,
    edge_role: str,
    grid2_summary: dict,
    grid3_summary: dict,
    stability: dict,
    cross_validation: dict,
    phase_summary: dict,
) -> dict:
    """Build explicit evidence and limits for one fixed edge."""
    phase = phase_summary.get("summary", phase_summary)
    fittable = [
        int(grid2_summary.get("n_regions_fittable", 0)),
        int(grid3_summary.get("n_regions_fittable", 0)),
    ]
    delta = [
        float(grid2_summary.get("p95_center_delta_mag_px") or 0.0),
        float(grid3_summary.get("p95_center_delta_mag_px") or 0.0),
    ]
    improvement = float(phase.get("median_phase_improvement_px") or 0.0)
    validated = int(phase.get("n_regions_pixel_validated", 0) or 0)
    phase_ok = validated > 0 or not phase_summary
    local_models_differ = max(delta) >= 1.0
    multiple_region_improvement = bool(
        validated >= 2 and (
            improvement >= 1.0 or float(phase.get("fraction_regions_improved_gt_1px") or 0.0) >= 0.5
        )
    )
    evidence = {
        "edge": edge, "edge_role": edge_role,
        "n_regions_fittable_2x2": fittable[0], "n_regions_fittable_3x3": fittable[1],
        "local_models_differ": local_models_differ,
        "multiple_region_phase_improvement": multiple_region_improvement,
        "partition_stability": stability.get("state", "MIXED_OR_UNSTABLE"),
        "phase_validation_status": "OK" if phase_ok else "UNCERTAIN",
        "cross_validation": cross_validation.get("summary", cross_validation),
        "phase_summary": phase,
        "low_support": edge_role == "LOW_SUPPORT_CONTROL" or max(fittable) == 0,
    }
    evidence["diagnosis"] = classify_local_geometry_consistency(evidence)
    evidence["can_conclude"] = [
        "Local-vs-global affine variation was tested in both 2x2 and 3x3 overlap partitions.",
        "Independent pixel validation is required before interpreting local model variation as geometry.",
    ]
    evidence["cannot_conclude"] = [
        "Cannot generalize one edge to universal remote-sensing affine failure.",
        "Cannot promote this diagnostic local affine model into the production algorithm.",
        "Cannot use this run alone to prove Homography/TPS is superior or absolute geolocation is correct.",
    ]
    return evidence


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(_json_safe(value), indent=2, ensure_ascii=False), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(_json_safe(rows))


def _phase_rows_for_result(result: dict, edge: str) -> list[dict]:
    phase = result.get("phase", {})
    rows = phase.get("rows", phase.get("regions", [])) if isinstance(phase, dict) else []
    return [dict(row, edge=edge) for row in rows]


def write_diagnostic_artifacts(output_dir: str | Path, results: dict[str, dict]) -> None:
    """Write the plan's machine-readable summaries and figure entry points."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "07_region_validation").mkdir(exist_ok=True)
    support_rows, model_rows, displacement_rows, cv_rows, phase_rows = [], [], [], [], []
    variations, cv_summaries, phase_summaries, stability = {}, {}, {}, {}
    evidence = {}
    for edge, result in results.items():
        models = result.get("models", [])
        for model in models:
            model_rows.append(dict(model, edge=edge))
            support_rows.append({
                "edge": edge, "grid_n": model.get("grid_n"),
                "region_row": model.get("region_row"), "region_col": model.get("region_col"),
                "n_inliers": model.get("n_points"),
                "fraction_of_edge_inliers": model.get("n_points", 0) / max(
                    sum(m.get("n_points", 0) for m in models if m.get("grid_n") == model.get("grid_n")), 1
                ), "status": model.get("status"),
            })
            if model.get("status") == "OK":
                displacement_rows.append({
                    "edge": edge, "grid_n": model.get("grid_n"), "region_id": model.get("region_id"),
                    "center_delta_dx_px": model.get("center_delta_dx_px"),
                    "center_delta_dy_px": model.get("center_delta_dy_px"),
                    "center_delta_mag_px": model.get("center_delta_mag_px"),
                    "inlier_centroid_delta_mag_px": model.get("inlier_centroid_delta_mag_px"),
                    "rotation_delta_deg": model.get("rotation_delta_deg"),
                    "scale_delta_x": model.get("scale_delta_x"), "scale_delta_y": model.get("scale_delta_y"),
                    "shear_delta_deg": model.get("shear_delta_deg"),
                })
        variations[edge] = result.get("variation", {})
        cv_summaries[edge] = result.get("cross_validation", {}).get("summary", result.get("cross_validation", {}))
        cv_rows.extend([dict(row, edge=edge) for row in result.get("cross_validation", {}).get("folds", [])])
        phase_rows.extend(_phase_rows_for_result(result, edge))
        phase_summaries[edge] = result.get("phase", {}).get("summary", result.get("phase", {}))
        stability[edge] = result.get("stability", {})
        evidence[edge] = result.get("evidence", {})

    _write_csv(out / "01_region_support.csv", support_rows,
               ["edge", "grid_n", "region_row", "region_col", "n_inliers", "fraction_of_edge_inliers", "status"])
    _write_csv(out / "02_local_affine_models.csv", model_rows)
    _write_json(out / "02_local_affine_models.json", model_rows)
    _write_csv(out / "03_local_vs_global_displacement.csv", displacement_rows)
    _write_json(out / "03_local_vs_global_summary.json", {k: v for k, v in variations.items()})
    _write_json(out / "04_local_affine_variation_summary.json", variations)
    _write_csv(out / "05_local_affine_cross_validation.csv", cv_rows)
    _write_json(out / "05_local_affine_cross_validation_summary.json", cv_summaries)
    _write_csv(out / "06_global_vs_local_phase.csv", phase_rows)
    _write_json(out / "06_global_vs_local_phase_summary.json", phase_summaries)
    _write_json(out / "09_partition_stability.json", stability)
    comparison = {edge: value for edge, value in evidence.items() if edge in ("0-6", "2-5")}
    _write_json(out / "10_good_vs_false_good_local_geometry.json", comparison)
    (out / "10_good_vs_false_good_local_geometry.txt").write_text(
        _format_evidence_text(comparison), encoding="utf-8"
    )
    low_support = {edge: value for edge, value in evidence.items() if edge == "0-5"}
    _write_json(out / "11_low_support_control.json", low_support)
    (out / "11_low_support_control.txt").write_text(_format_evidence_text(low_support), encoding="utf-8")
    plot_good_vs_false_good_comparison(results, out / "10_good_vs_false_good_local_geometry.png")
    conclusion = {
        "edges": evidence,
        "can_conclude": ["The result is a fixed-edge diagnostic, not a production registration decision."],
        "cannot_conclude": [
            "Cannot generalize one edge to universal failure.",
            "Cannot promote diagnostic local affine into production.",
            "Cannot prove Homography/TPS superiority from this run.",
            "Cannot infer absolute geolocation correctness from local improvement.",
        ],
    }
    _write_json(out / "12_local_affine_consistency_conclusion.json", conclusion)
    (out / "12_local_affine_consistency_conclusion.txt").write_text(_format_evidence_text(conclusion), encoding="utf-8")
    write_region_validation_figures(results, out / "07_region_validation")
    plot_local_affine_field_map(results, out / "08_local_affine_field_map.png")
    plot_local_affine_dashboard(results, out / "13_local_affine_consistency_dashboard.png")


def _format_evidence_text(value: dict) -> str:
    lines = ["Local affine consistency diagnostic evidence", ""]
    for key, item in value.items():
        lines.append(f"{key}: {json.dumps(_json_safe(item), ensure_ascii=False, sort_keys=True)}")
    return "\n".join(lines) + "\n"


def _validation_image_stretch(crop: dict) -> tuple[float, float]:
    values = []
    for key in ("reference", "global_warp", "local_warp"):
        array = np.asarray(crop[key], dtype=float)
        values.append(array[np.isfinite(array)])
    finite = np.concatenate([value for value in values if value.size]) if any(value.size for value in values) else np.array([])
    if finite.size == 0:
        return 0.0, 1.0
    low, high = np.percentile(finite, [2.0, 98.0])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low, high = float(np.nanmin(finite)), float(np.nanmax(finite))
    if high <= low:
        high = low + 1.0
    return float(low), float(high)


def _checkerboard(reference: np.ndarray, moving: np.ndarray, tile: int = 24) -> np.ndarray:
    result = np.asarray(reference, dtype=np.float32).copy()
    yy, xx = np.indices(result.shape)
    choose_moving = ((yy // max(tile, 1) + xx // max(tile, 1)) % 2) == 1
    result[choose_moving] = moving[choose_moving]
    return result


def _select_region_validation_crops(crops: list[dict], limit: int = 4) -> list[dict]:
    usable = [item for item in crops if item.get("crop", {}).get("reference") is not None]
    if not usable:
        return []

    def phase_value(item: dict, key: str, default: float) -> float:
        value = item.get("phase", {}).get(key)
        return float(value) if value is not None and np.isfinite(value) else default

    ordered = [
        min(usable, key=lambda item: phase_value(item, "global_phase_mag", float("inf"))),
        sorted(usable, key=lambda item: phase_value(item, "global_phase_mag", float("inf")))[len(usable) // 2],
        max(usable, key=lambda item: phase_value(item, "global_phase_mag", float("-inf"))),
        max(usable, key=lambda item: phase_value(item, "phase_improvement_px", float("-inf"))),
    ]
    selected = []
    seen = set()
    for item in ordered:
        key = (item.get("grid_n"), item.get("region_id"))
        if key in seen:
            continue
        seen.add(key)
        selected.append(item)
        if len(selected) >= limit:
            break
    return selected


def write_region_validation_figures(results: dict[str, dict], output_dir: str | Path) -> None:
    """Write same-stretch reference/global/local checkerboards for selected regions."""
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for edge in ("0-6", "2-5", "0-5"):
        selected = _select_region_validation_crops(results.get(edge, {}).get("validation_crops", []))
        for item in selected:
            crop = item["crop"]
            reference = np.asarray(crop["reference"], dtype=np.float32)
            global_warp = np.asarray(crop["global_warp"], dtype=np.float32)
            local_warp = np.asarray(crop["local_warp"], dtype=np.float32)
            vmin, vmax = _validation_image_stretch(crop)
            global_overlay = _checkerboard(reference, global_warp)
            local_overlay = _checkerboard(reference, local_warp)
            phase = item.get("phase", {})
            model = item.get("model", {})
            fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
            panels = (
                (reference, "A. reference"),
                (global_overlay, "B. global-affine checkerboard"),
                (local_overlay, "C. local-affine checkerboard"),
            )
            for ax, (image, title) in zip(axes.flat[:3], panels):
                ax.imshow(image, cmap="gray", vmin=vmin, vmax=vmax)
                ax.set_title(title)
                ax.set_axis_off()
            annotation = {
                "edge": edge,
                "grid": item.get("grid_n"),
                "region": item.get("region_id"),
                "n_inliers": model.get("n_points"),
                "global_phase_px": phase.get("global_phase_mag"),
                "local_phase_px": phase.get("local_phase_mag"),
                "global_ncc": phase.get("global_ncc"),
                "local_ncc": phase.get("local_ncc"),
                "center_delta_px": model.get("center_delta_mag_px"),
                "phase_improvement_px": phase.get("phase_improvement_px"),
            }
            axes.flat[3].text(0.02, 0.98, json.dumps(_json_safe(annotation), indent=2),
                              va="top", ha="left", family="monospace", fontsize=10,
                              transform=axes.flat[3].transAxes)
            axes.flat[3].set_title("D. annotations")
            axes.flat[3].set_axis_off()
            name = f"edge_{edge.replace('-', '_')}_grid{item.get('grid_n')}_region{item.get('region_id')}.png"
            fig.savefig(out / name, dpi=160)
            plt.close(fig)


def plot_local_affine_field_map(results: dict[str, dict], output_path: str | Path) -> None:
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for ax, edge in zip(axes, ("0-6", "2-5")):
        result = results.get(edge, {})
        models = [m for m in result.get("models", []) if m.get("status") == "OK"]
        for model in models:
            center = np.asarray(model.get("center_xy", [0, 0]), dtype=float)
            ax.quiver(center[0], center[1], model.get("center_delta_dx_px", 0),
                      model.get("center_delta_dy_px", 0), angles="xy", scale_units="xy",
                      scale=1, color="tab:red")
        ax.set_title(f"{edge}: local - global prediction")
        ax.set_xlabel("reference pixel x")
        ax.set_ylabel("reference pixel y")
        ax.invert_yaxis()
        ax.grid(alpha=0.25)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_good_vs_false_good_comparison(results: dict[str, dict], output_path: str | Path) -> None:
    """Plot the fixed 0-6/2-5 diagnostic comparison without changing evidence."""
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    edges = ("0-6", "2-5")
    labels = ("2x2", "3x3")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    colors = ("tab:blue", "tab:orange")
    for edge, color in zip(edges, colors):
        result = results.get(edge, {})
        variation = result.get("variation", {})
        cv_summary = result.get("cross_validation", {}).get("summary", {})
        phase = result.get("phase", {}).get("summary", {})
        fittable = [variation.get(str(grid), {}).get("n_regions_fittable", 0) for grid in (2, 3)]
        p95 = [variation.get(str(grid), {}).get("p95_center_delta_mag_px") for grid in (2, 3)]
        axes[0, 0].plot(labels, fittable, marker="o", color=color, label=edge)
        axes[0, 1].plot(labels, p95, marker="o", color=color, label=edge)
        axes[1, 0].bar(edge, cv_summary.get("improvement_px") or 0.0, color=color)
        axes[1, 1].bar(edge, phase.get("median_phase_improvement_px") or 0.0, color=color)
    axes[0, 0].set_title("Fittable local regions")
    axes[0, 1].set_title("P95 local-vs-global center delta (px)")
    axes[1, 0].set_title("Held-out CV improvement (px)")
    axes[1, 1].set_title("Independent phase improvement (px)")
    axes[0, 0].legend()
    for ax in axes.flat:
        ax.grid(alpha=0.25)
    fig.suptitle("0-6 GOOD_REFERENCE vs 2-5 HIGH_INLIER_FALSE_GOOD")
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_local_affine_dashboard(results: dict[str, dict], output_path: str | Path) -> None:
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 4, figsize=(16, 10), constrained_layout=True)
    for row, edge in enumerate(("0-6", "2-5", "0-5")):
        models = [m for m in results.get(edge, {}).get("models", []) if m.get("status") == "OK"]
        x = [m.get("center_xy", [0, 0])[0] for m in models]
        y = [m.get("center_xy", [0, 0])[1] for m in models]
        mags = [m.get("center_delta_mag_px", 0) for m in models]
        if models:
            axes[row, 0].scatter(x, y, c=mags, cmap="viridis")
        else:
            axes[row, 0].text(0.5, 0.5, "no fittable regions", ha="center", va="center")
        axes[row, 0].set_title(f"{edge} inlier/local support")
        axes[row, 1].quiver(x, y, [m.get("center_delta_dx_px", 0) for m in models],
                            [m.get("center_delta_dy_px", 0) for m in models], angles="xy", scale_units="xy", scale=1)
        axes[row, 1].set_title("local - global vectors")
        axes[row, 2].bar(range(len(mags)), mags)
        axes[row, 2].set_title("center delta magnitude")
        axes[row, 3].text(0.02, 0.98, json.dumps(_json_safe(results.get(edge, {}).get("evidence", {})), indent=2),
                          va="top", ha="left", fontsize=7, transform=axes[row, 3].transAxes)
        axes[row, 3].set_axis_off()
        for ax in axes[row, :3]:
            ax.grid(alpha=0.25)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
