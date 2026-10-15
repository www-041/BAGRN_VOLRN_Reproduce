"""Edge-reliability / false-good pairwise registration diagnostics.

For the 7 representative edges (0-6 good, 4-6/1-8 borderline, 2-5
high-inlier false-good candidate, 0-5/5-6 low-support failures, 0-1 old
anomaly) this module answers *why* a pairwise edge with excellent RANSAC
metrics can still misalign over the full overlap.  It computes spatial-support
metrics (occupancy, convex hull, quadrant balance, nearest-control distance),
RANSAC residual spatial patterns, direct-warp local phase residual fields,
their coupling to control-point support, and an evidence-based diagnostic
classification.  No SIFT/RANSAC/Affine/MST semantics are modified, no ML model
is trained, and no global adjustment is run.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
from rasterio.transform import Affine
from scipy.spatial import cKDTree, ConvexHull

from src.multiscene_sift.models import Scene

logger = logging.getLogger(__name__)

DIAGNOSTIC_EDGES = [
    (0, 6),
    (4, 6),
    (1, 8),
    (2, 5),
    (0, 5),
    (5, 6),
    (0, 1),
]

EDGE_GROUPS = {
    (0, 6): "GOOD_REFERENCE",
    (4, 6): "BORDERLINE",
    (1, 8): "BORDERLINE",
    (2, 5): "HIGH_INLIER_FALSE_GOOD_CANDIDATE",
    (0, 5): "LOW_SUPPORT_FAILURE",
    (5, 6): "LOW_SUPPORT_FAILURE",
    (0, 1): "OLD_ANOMALY",
}

DISTANCE_THRESHOLDS_PX = {"near": 64.0, "far": 128.0}


# ---------------------------------------------------------------------------
# Task 1 — baseline
# ---------------------------------------------------------------------------


def load_edge_reliability_baseline(
    selected_pair_dir: str | Path,
    five_scene_run_dir: str | Path,
    dense_01_dir: str | Path | None = None,
) -> list[dict]:
    """Per-edge baseline metrics from the previous run artifacts."""
    import json

    sp = Path(selected_pair_dir)
    fr = Path(five_scene_run_dir)

    new_rows: dict[tuple[int, int], dict] = {}
    p02 = sp / "02_new_pairwise_registration_results.json"
    if p02.is_file():
        with open(p02, encoding="utf-8") as f:
            for r in json.load(f)["results"]:
                new_rows[(int(r["idx_i"]), int(r["idx_j"]))] = r
    q03 = sp / "03_new_pair_quality_checks.json"
    qual: dict[tuple[int, int], dict] = {}
    if q03.is_file():
        with open(q03, encoding="utf-8") as f:
            for r in json.load(f)["checks"]:
                qual[(int(r["idx_i"]), int(r["idx_j"]))] = r

    saved: list[dict] = []
    pw = fr / "pairwise_summary.json"
    if pw.is_file():
        with open(pw, encoding="utf-8") as f:
            saved = json.load(f)["results"]

    edges = []
    for (i, j) in DIAGNOSTIC_EDGES:
        saved_row = next((r for r in saved
                          if {int(r["idx_i"]), int(r["idx_j"])} == {i, j}), None)
        new_row = new_rows.get((i, j)) or new_rows.get((j, i))
        qual_row = qual.get((i, j)) or qual.get((j, i))
        base = new_row or saved_row or {}
        edges.append({
            "edge": [i, j],
            "group": EDGE_GROUPS[(i, j)],
            "source": "new_registration" if new_row else "five_scene_saved",
            "raw_matches": int(base.get("raw_matches", 0)),
            "inliers": int(base.get("inliers", 0)),
            "inlier_ratio": float(base.get("inlier_ratio", 0.0)),
            "coverage": float(base.get("coverage", 0.0)),
            "rmse_px": float(base.get("residual_rmse", float("nan"))),
            "p95_px": float(base.get("residual_p95", float("nan"))),
            "full_overlap_phase_residual_px": (
                qual_row.get("phase_mag_px") if qual_row else None),
            "ncc": (qual_row.get("ncc") if qual_row else None),
            "previous_validation": (
                qual_row.get("validation_status") if qual_row else
                ("saved" if saved_row else "missing")),
        })
    return edges


# ---------------------------------------------------------------------------
# Task 2 — reproduction gate
# ---------------------------------------------------------------------------


def reproduce_gate_status(saved: dict | None, reg) -> str:
    """EXACT / CLOSE / MISMATCH vs the previously saved summary."""
    if saved is None:
        return "NO_SAVED_BASELINE"
    same_core = (
        int(saved.get("inliers", -1)) == reg.inliers
        and int(saved.get("raw_matches", -1)) == reg.raw_matches
    )
    tol = 1e-3
    res_close = (
        abs(float(saved.get("residual_rmse", 0.0)) - reg.residual_rmse) < tol
        and abs(float(saved.get("residual_p95", 0.0)) - reg.residual_p95) < tol
    )
    if same_core and res_close:
        return "EXACT"
    if same_core:
        return "CLOSE"
    return "MISMATCH"


def match_view_points_to_pixels(points: np.ndarray, match_view: dict) -> np.ndarray:
    """match-view pixels -> common-grid pixels (MatchView semantics)."""
    from src.multiscene_sift.frame_diagnostics import (
        match_view_to_common_affine, transform_points,
    )

    T = match_view_to_common_affine(
        match_view["origin_x"], match_view["origin_y"],
        match_view["scale_x"], match_view["scale_y"],
    )
    return transform_points(T, np.asarray(points, dtype=np.float64))


# ---------------------------------------------------------------------------
# Task 3 — inlier spatial support metrics
# ---------------------------------------------------------------------------


def grid_occupancy(inlier_xy_px: np.ndarray, shape: tuple[int, int], n: int) -> dict:
    h, w = shape
    occupied = np.zeros((n, n), dtype=bool)
    xs = np.clip(np.floor(np.asarray(inlier_xy_px)[:, 0] / max(w, 1) * n),
                 0, n - 1).astype(int)
    ys = np.clip(np.floor(np.asarray(inlier_xy_px)[:, 1] / max(h, 1) * n),
                 0, n - 1).astype(int)
    occupied[ys, xs] = True
    return {
        "grid": n,
        "occupied_cells": int(occupied.sum()),
        "total_cells": int(n * n),
        "occupancy_ratio": float(occupied.sum() / (n * n)),
    }


def convex_hull_coverage(inlier_xy_px: np.ndarray, valid_area_px2: float) -> float:
    pts = np.asarray(inlier_xy_px, dtype=np.float64)
    if len(pts) < 3 or valid_area_px2 <= 0:
        return 0.0
    try:
        hull = ConvexHull(pts)
        area = float(hull.volume)  # 2-D hull "volume" == area
    except Exception:  # noqa: BLE001 collinear points
        return 0.0
    return float(np.clip(area / valid_area_px2, 0.0, 1.0))


def quadrant_balance(inlier_xy_px: np.ndarray, shape: tuple[int, int]) -> dict:
    h, w = shape
    x = np.asarray(inlier_xy_px)[:, 0]
    y = np.asarray(inlier_xy_px)[:, 1]
    q = [((x <= w / 2) & (y <= h / 2)).sum(),
         ((x > w / 2) & (y <= h / 2)).sum(),
         ((x <= w / 2) & (y > h / 2)).sum(),
         ((x > w / 2) & (y > h / 2)).sum()]
    n = int(sum(q))
    if n == 0:
        return {"count_per_quadrant": [0, 0, 0, 0], "max_min_ratio": None,
                "entropy": None}
    p = np.array(q) / n
    entropy = float(-np.sum(p[p > 0] * np.log(p[p > 0])) / math.log(4))
    return {"count_per_quadrant": [int(v) for v in q],
            "max_min_ratio": float(max(q) / max(min(q), 1)),
            "entropy": entropy}


def centroid_bias(inlier_xy_px: np.ndarray, valid_mask: np.ndarray,
                  shape: tuple[int, int]) -> dict:
    h, w = shape
    inlier_cent = np.mean(np.asarray(inlier_xy_px), axis=0)
    if valid_mask is not None and valid_mask.any():
        ys, xs = np.nonzero(valid_mask)
        valid_cent = np.array([float(xs.mean()), float(ys.mean())])
    else:
        valid_cent = np.array([w / 2, h / 2])
    diag = math.hypot(w, h)
    return {
        "inlier_centroid_px": [float(inlier_cent[0]), float(inlier_cent[1])],
        "valid_centroid_px": [float(valid_cent[0]), float(valid_cent[1])],
        "centroid_offset_normalized": float(
            np.linalg.norm(inlier_cent - valid_cent) / max(diag, 1e-9)),
    }


def compute_inlier_spatial_metrics(
    inlier_xy_px: np.ndarray, valid_mask: np.ndarray, shape: tuple[int, int]
) -> dict:
    valid_area_px2 = float(valid_mask.sum()) if valid_mask is not None else \
        float(shape[0] * shape[1])
    x = np.asarray(inlier_xy_px)[:, 0]
    y = np.asarray(inlier_xy_px)[:, 1]
    cx, cy = float(x.mean()), float(y.mean())
    boundary = ((np.abs(x - cx) > 0.6 * shape[1] / 2) |
                (np.abs(y - cy) > 0.6 * shape[0] / 2)).sum()
    return {
        "n_inliers": int(len(x)),
        "occupancy_4x4": grid_occupancy(inlier_xy_px, shape, 4),
        "occupancy_8x8": grid_occupancy(inlier_xy_px, shape, 8),
        "hull_coverage": convex_hull_coverage(inlier_xy_px, valid_area_px2),
        "quadrants": quadrant_balance(inlier_xy_px, shape),
        "centroid": centroid_bias(inlier_xy_px, valid_mask, shape),
        "boundary_support_fraction_outer60": float(boundary / max(len(x), 1)),
        "diagnostic_note": ("hull/occupancy are candidate indicators, not "
                            "absolute coverage truth; nearest-control distance "
                            "adds locality."),
    }


# ---------------------------------------------------------------------------
# Task 4 — nearest-control distance map
# ---------------------------------------------------------------------------


def sample_grid_points(shape: tuple[int, int], step_px: int) -> np.ndarray:
    h, w = shape
    ys, xs = np.mgrid[step_px // 2:h:step_px, step_px // 2:w:step_px]
    return np.column_stack([xs.ravel(), ys.ravel()])


def nearest_control_statistics(
    inlier_xy_px: np.ndarray, sample_points: np.ndarray
) -> dict:
    pts = np.asarray(inlier_xy_px, dtype=np.float64)
    if len(pts) == 0 or len(sample_points) == 0:
        return {"n_samples": 0, "median_nearest_control_px": None,
                "p90_nearest_control_px": None, "p95_nearest_control_px": None,
                "max_nearest_control_px": None}
    d, _ = cKDTree(pts).query(sample_points)
    return {
        "n_samples": int(len(d)),
        "median_nearest_control_px": float(np.median(d)),
        "p90_nearest_control_px": float(np.percentile(d, 90)),
        "p95_nearest_control_px": float(np.percentile(d, 95)),
        "max_nearest_control_px": float(np.max(d)),
        "fraction_farther_than_64px": float((d > DISTANCE_THRESHOLDS_PX["near"]).mean()),
        "fraction_farther_than_128px": float((d > DISTANCE_THRESHOLDS_PX["far"]).mean()),
    }


# ---------------------------------------------------------------------------
# Task 6 — residual spatial pattern
# ---------------------------------------------------------------------------


def summarize_residual_spatial_pattern(
    inlier_xy_px: np.ndarray, residual_px: np.ndarray, shape: tuple[int, int]
) -> dict:
    h, w = shape
    x = np.asarray(inlier_xy_px)[:, 0]
    y = np.asarray(inlier_xy_px)[:, 1]
    r = np.asarray(residual_px, dtype=np.float64)

    def _med(mask):
        return float(np.median(r[mask])) if mask.sum() else float("nan")

    stats = {
        "overall_median": _med(np.ones_like(r, dtype=bool)),
        "overall_p95": float(np.percentile(r, 95)),
        "left_median": _med(x <= w / 2),
        "right_median": _med(x > w / 2),
        "top_median": _med(y <= h / 2),
        "bottom_median": _med(y > h / 2),
        "quadrant_medians": [
            _med((x <= w / 2) & (y <= h / 2)),
            _med((x > w / 2) & (y <= h / 2)),
            _med((x <= w / 2) & (y > h / 2)),
            _med((x > w / 2) & (y > h / 2)),
        ],
        "corr_residual_x": float(np.corrcoef(x, r)[0, 1]) if len(x) > 1 else None,
        "corr_residual_y": float(np.corrcoef(y, r)[0, 1]) if len(y) > 1 else None,
    }
    try:
        X = np.column_stack([np.ones_like(x), x, y])
        coef, *_ = np.linalg.lstsq(X, r, rcond=None)
        pred = X @ coef
        ss_res = float(np.sum((r - pred) ** 2))
        ss_tot = float(np.sum((r - r.mean()) ** 2))
        stats["linear_fit"] = {
            "coefficients": [float(c) for c in coef],
            "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0,
        }
    except np.linalg.LinAlgError:  # noqa: BLE001
        stats["linear_fit"] = {"coefficients": None, "r2": None}
    return stats


# ---------------------------------------------------------------------------
# Task 7/8 — direct-warp overlap + local residual field
# ---------------------------------------------------------------------------


def corrected_footprint_bounds(transform: Affine, shape: tuple[int, int]):
    rows, cols = int(shape[0]), int(shape[1])
    corners = [transform * (0, 0), transform * (cols, 0),
               transform * (0, rows), transform * (cols, rows)]
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    return min(xs), min(ys), max(xs), max(ys)


def build_direct_warp_overlap(
    scene_i: Scene,
    scene_j: Scene,
    band: str,
    t_world_i_from_j: np.ndarray,
    max_side: int = 4096,
) -> dict:
    """Overlap raster after applying the direct world transform (no MST)."""
    from src.multiscene_sift.loop_diagnostics import _north_up_grid, _reproject_band_to_grid
    from src.multiscene_sift.frame_diagnostics import (
        affine_to_matrix, matrix_to_affine,
    )

    A_i = scene_i.transforms[band]
    M_world = np.asarray(t_world_i_from_j, dtype=np.float64)
    A_j_corr = matrix_to_affine(M_world @ affine_to_matrix(scene_j.transforms[band]))
    res = abs(A_i.a)
    b_i = scene_i.bounds[band]
    bx, by, bxr, byt = corrected_footprint_bounds(A_j_corr, scene_j.shapes[band])
    left = max(b_i.left, bx)
    bottom = max(b_i.bottom, by)
    right = min(b_i.right, bxr)
    top = min(b_i.top, byt)
    if right <= left or top <= bottom:
        raise ValueError(
            f"no direct-warp overlap for scenes {scene_i.index}/{scene_j.index}")
    grid_transform, gw, gh = _north_up_grid((left, bottom, right, top), res, max_side)
    shape = (gh, gw)
    a_i, v_i = _reproject_band_to_grid(
        scene_i.band_paths[band], A_i, scene_i.crs, grid_transform, shape)
    a_j, v_j = _reproject_band_to_grid(
        scene_j.band_paths[band], A_j_corr, scene_j.crs, grid_transform, shape)
    joint = v_i & v_j
    return {
        "image0": a_i, "image1": a_j,
        "mask0": v_i, "mask1": v_j, "joint_mask": joint,
        "transform": grid_transform, "crs": str(scene_i.crs),
        "width": gw, "height": gh,
        "resolution": float(grid_transform.a),
        "joint_valid_pixels": int(joint.sum()),
        "joint_valid_fraction": float(joint.mean()) if joint.size else 0.0,
        "frame_note": "direct world transform applied to scene_j only (no MST)",
    }


def measure_direct_residual_field(
    overlap: dict, grids: tuple[int, ...] = (4, 6), config: dict | None = None
) -> dict:
    """Per-tile local phase residual after the direct warp (reuses dense machinery)."""
    from src.multiscene_sift.dense_phase_diagnostics import (
        measure_dense_shift_field, summarize_dense_shift_field,
    )

    fields: dict[str, list[dict]] = {}
    summaries: dict[str, dict] = {}
    rows_all: list[dict] = []
    for g in grids:
        rows = measure_dense_shift_field(overlap, g, config)
        fields[f"grid_{g}"] = rows
        summaries[f"grid_{g}"] = summarize_dense_shift_field(rows)
        rows_all.extend(rows)
    return {"fields": fields, "summaries": summaries, "tiles": rows_all}


def summarize_direct_residual_field(result: dict) -> dict:
    tiles = result["tiles"]
    acc = [t for t in tiles if t.get("accepted")
           and np.isfinite(t.get("phase_mag_px"))]
    if len(acc) < 3:
        return {"n_accepted": len(acc), "status": "INSUFFICIENT_TILES"}
    mag = np.array([t["phase_mag_px"] for t in acc])
    med = float(np.median(mag))
    mad = float(np.median(np.abs(mag - med)))
    keep = [m for m in mag if abs(m - med) <= 3 * max(1.4826 * mad, 0.5)]
    return {
        "n_accepted": len(acc),
        "median_px": med,
        "robust_mad_px": float(mad),
        "p90_px": float(np.percentile(mag, 90)),
        "p95_px": float(np.percentile(mag, 95)),
        "max_px": float(np.max(mag)),
        "max_after_outlier_filter_px": float(max(keep)) if keep else float("nan"),
        "fraction_tiles_le1px": float((mag <= 1.0).mean()),
        "fraction_tiles_le2px": float((mag <= 2.0).mean()),
        "fraction_tiles_gt5px": float((mag > 5.0).mean()),
        "status": "OK",
    }


# ---------------------------------------------------------------------------
# Task 9 — residual vs control support
# ---------------------------------------------------------------------------


def relate_residual_to_control_support(
    tiles: list[dict], inlier_xy_px: np.ndarray, grid_transform: Affine
) -> dict:
    from scipy import stats as _stats

    pts = np.asarray(inlier_xy_px, dtype=np.float64)
    inv_grid = ~grid_transform
    rows = []
    for t in tiles:
        if not t.get("accepted") or not np.isfinite(t.get("phase_mag_px")):
            continue
        gx, gy = inv_grid * (t["center_world_x"], t["center_world_y"])
        d = float(cKDTree(pts).query([gx, gy])[0]) if len(pts) else 0.0
        inside = int((((pts[:, 0] - gx) ** 2 + (pts[:, 1] - gy) ** 2) < 64.0).sum())
        rows.append({"nearest_control_distance_px": d,
                     "inliers_inside_tile": inside,
                     "phase_mag_px": float(t["phase_mag_px"])})
    if len(rows) < 5:
        return {"n_tiles": len(rows), "corr_spearman": None,
                "near_median": None, "far_median": None}
    d = np.array([r["nearest_control_distance_px"] for r in rows])
    p = np.array([r["phase_mag_px"] for r in rows])
    corr, pval = _stats.spearmanr(d, p)
    near = p[d <= 64]
    far = p[d > 64]
    return {
        "n_tiles": len(rows),
        "corr_spearman_phase_vs_distance": float(corr),
        "spearman_p": float(pval),
        "near_control_median_residual_px": (
            float(np.median(near)) if len(near) else None),
        "far_control_median_residual_px": (
            float(np.median(far)) if len(far) else None),
        "far_minus_near_median_px": (
            float(np.median(far) - np.median(near))
            if len(near) and len(far) else None),
    }


# ---------------------------------------------------------------------------
# Task 13 — evidence-based classifier
# ---------------------------------------------------------------------------

RELIABILITY_THRESHOLDS = {
    "low_support_inliers": 100,
    "low_support_occupancy": 0.15,
    "reliable_phase_median_px": 2.0,
    "reliable_bad_tile_frac": 0.15,
    "extrapolation_corr": 0.4,
    "uneven_far_frac": 0.25,
}


def classify_edge_reliability_pattern(evidence: dict) -> str:
    th = RELIABILITY_THRESHOLDS
    n_in = evidence.get("inliers", 0)
    occ = evidence.get("occupancy_8x8", 1.0)
    phase_med = evidence.get("direct_phase_median_px")
    bad_frac = evidence.get("fraction_tiles_gt5px", 0.0)
    corr = evidence.get("corr_spearman_phase_vs_distance")
    far_frac = evidence.get("fraction_farther_than_128px", 0.0)
    if phase_med is None:
        return "MIXED_OR_UNDERDETERMINED"
    if n_in < th["low_support_inliers"] or occ < th["low_support_occupancy"]:
        return "LOW_SUPPORT_EDGE"
    if phase_med <= th["reliable_phase_median_px"] and \
            bad_frac <= th["reliable_bad_tile_frac"]:
        return "DIRECT_GEOMETRY_RELIABLE"
    if corr is not None and corr >= th["extrapolation_corr"] and \
            far_frac >= th["uneven_far_frac"]:
        return "COVERAGE_LIMITED_EXTRAPOLATION_SUSPECT"
    if phase_med > th["reliable_phase_median_px"] * 2.0 and \
            corr is not None and corr < 0.3:
        return "LOCAL_RESIDUAL_FIELD_INCONSISTENT"
    return "MIXED_OR_UNDERDETERMINED"