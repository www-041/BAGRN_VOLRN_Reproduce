"""Dense multiscale local phase-field diagnostics for the 0-1 overlap.

The geolocation round established that the 0-1 pair has a ~40-45 px
metadata-vs-content offset while both tree edges are ~0, with the multiscale
behaviour unresolved.  This module measures the displacement field of the
metadata-only 0-1 overlap at 4×4 / 6×6 / 8×8 tile grids, applies explicit
quality gates (valid fraction, texture, phase confidence), fits spatial
trends per scale, and decides whether the offset is an approximately constant
shift, a stable spatial gradient, or an unstable/ambiguous measurement.

Diagnosis only: no SIFT/LoFTR/RANSAC/Affine/MST/BAGRN/VOLRN semantics are
modified, and no direct or MST transform is applied before phase correlation.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
from rasterio.transform import Affine

from src.multiscene_sift.loop_diagnostics import _phase_cross_correlation_shift  # noqa: F401
from src.multiscene_sift.models import Scene

logger = logging.getLogger(__name__)

DEFAULT_QUALITY_CONFIG = {
    "min_joint_valid_fraction": 0.60,
    "min_tile_size_px": 128,
    "min_texture_std": 1.0,
    "max_phase_error_px": 1.0,
}

DEFAULT_GRIDS = (4, 6, 8)


def load_dense_phase_baseline(
    geolocation_diag_dir: str | Path,
) -> dict:
    """Load the 0-1 geolocation facts established by the previous round."""
    import json

    gd = Path(geolocation_diag_dir)
    baseline: dict[str, Any] = {
        "geolocation_diagnostics_dir": str(gd),
    }
    for name, key in (
        ("13_revised_state.json", "revised_state"),
        ("05_metadata_only_shift_summary.json", "metadata_only_shift"),
        ("08_within_overlap_spatial_variation.json", "within_overlap_variation"),
    ):
        path = gd / name
        baseline[key] = json.loads(path.read_text(encoding="utf-8")) \
            if path.is_file() else None
    return baseline


# ---------------------------------------------------------------------------
# Task 2 — metadata-only 0-1 overlap raster (shared by all scales)
# ---------------------------------------------------------------------------


def build_metadata_only_overlap_01(
    scene0: Scene,
    scene1: Scene,
    band: str = "B14",
    resolution_m: float = 14.0,
) -> dict:
    """Metadata-only north-up overlap raster using the original geotransforms.

    Reuses the geolocation module's reprojection so all tile grids see the
    exact same raster.  No direct/MST transform is applied.
    """
    from src.multiscene_sift.geolocation_diagnostics import metadata_only_overlap_grid

    grid_transform, shape, a_i, a_j, v_i, v_j = metadata_only_overlap_grid(
        scene0, scene1, band
    )
    if abs(grid_transform.a - resolution_m) > 1e-6:
        logger.warning(
            "overlap grid resolution %.3f != requested %.3f",
            grid_transform.a, resolution_m,
        )
    joint = v_i & v_j
    return {
        "image0": a_i,
        "image1": a_j,
        "mask0": v_i,
        "mask1": v_j,
        "joint_mask": joint,
        "transform": grid_transform,
        "crs": str(scene0.crs),
        "bounds": {
            "left": grid_transform.c,
            "bottom": grid_transform.f - grid_transform.e * shape[0],
            "right": grid_transform.c + grid_transform.a * shape[1],
            "top": grid_transform.f,
        },
        "width": shape[1],
        "height": shape[0],
        "resolution": float(grid_transform.a),
        "joint_valid_pixels": int(joint.sum()),
        "joint_valid_fraction": float(joint.mean()) if joint.size else 0.0,
    }


def overlap_grid_metadata(overlap: dict) -> dict:
    """JSON-serialisable description of the overlap raster."""
    return {
        "crs": overlap["crs"],
        "bounds": overlap["bounds"],
        "width": int(overlap["width"]),
        "height": int(overlap["height"]),
        "resolution": float(overlap["resolution"]),
        "joint_valid_pixels": overlap["joint_valid_pixels"],
        "joint_valid_fraction": overlap["joint_valid_fraction"],
        "transforms_used": "scene0/scene1 original geotransforms only",
        "applied_corrections": "none",
    }


# ---------------------------------------------------------------------------
# Task 3 — deterministic tile windows
# ---------------------------------------------------------------------------


def make_tile_windows(
    height: int, width: int, grid_n: int, min_tile_size_px: int = 128
) -> list[dict]:
    """Split a raster into grid_n×grid_n non-overlapping covering tiles.

    Coverage is exact: edges are placed by rounding and the final edge clamps
    to the raster edge, so every pixel belongs to exactly one tile.
    """
    def _edges(size: int, parts: int) -> list[int]:
        edges = [int(round(size * k / parts)) for k in range(parts)]
        edges.append(size)
        out = []
        for k in range(parts):
            a = max(edges[k], 0)
            b = max(edges[k + 1], a + 1)
            out.append((a, min(b, size)))
        return out

    rows_e = _edges(height, grid_n)
    cols_e = _edges(width, grid_n)
    windows = []
    for ri, (r0, r1) in enumerate(rows_e):
        for ci, (c0, c1) in enumerate(cols_e):
            tile_h = r1 - r0
            tile_w = c1 - c0
            too_small = min(tile_h, tile_w) < min_tile_size_px
            windows.append({
                "grid_n": int(grid_n),
                "tile_row": ri,
                "tile_col": ci,
                "row0": int(r0),
                "row1": int(r1),
                "col0": int(c0),
                "col1": int(c1),
                "tile_height": tile_h,
                "tile_width": tile_w,
                "scale_too_small": bool(too_small),
            })
    return windows


def assert_full_coverage(windows: list[dict], height: int, width: int) -> bool:
    """Sanity check: windows tile [0, height) × [0, width) exactly once."""
    grid = np.zeros((height, width), dtype=np.int8)
    for w in windows:
        grid[w["row0"]:w["row1"], w["col0"]:w["col1"]] += 1
    return bool((grid == 1).all())


# ---------------------------------------------------------------------------
# Task 4 — tile quality gate
# ---------------------------------------------------------------------------


def evaluate_tile_quality(
    img0: np.ndarray,
    img1: np.ndarray,
    mask0: np.ndarray,
    mask1: np.ndarray,
    config: dict | None = None,
) -> dict:
    """Quality of one tile pair: validity, texture, and accept decision."""
    cfg = {**DEFAULT_QUALITY_CONFIG, **(config or {})}
    v0 = float(mask0.mean()) if mask0.size else 0.0
    v1 = float(mask1.mean()) if mask1.size else 0.0
    joint = mask0 & mask1
    jv = float(joint.mean()) if joint.size else 0.0

    def _texture(arr: np.ndarray, mask: np.ndarray) -> dict:
        if not mask.any():
            return {"std": 0.0, "gradient_energy": 0.0}
        vals = arr[mask]
        std = float(np.std(vals))
        gy, gx = np.gradient(arr)
        gyv = gy[mask]
        gxv = gx[mask]
        grad_energy = float(np.mean(np.abs(gxv) + np.abs(gyv)))
        return {"std": std, "gradient_energy": grad_energy}

    tex0 = _texture(img0, mask0)
    tex1 = _texture(img1, mask1)

    reason = None
    if jv < cfg["min_joint_valid_fraction"]:
        reason = "LOW_VALID_FRACTION"
    elif tex0["std"] < cfg["min_texture_std"] or \
            tex1["std"] < cfg["min_texture_std"]:
        reason = "LOW_TEXTURE"

    return {
        "valid_fraction_0": v0,
        "valid_fraction_1": v1,
        "joint_valid_fraction": jv,
        "std_0": tex0["std"],
        "std_1": tex1["std"],
        "gradient_energy_0": tex0["gradient_energy"],
        "gradient_energy_1": tex1["gradient_energy"],
        "accepted_for_phase": reason is None,
        "reject_reason": reason,
    }


# ---------------------------------------------------------------------------
# Task 5 — single-tile phase adapter (reuses the existing phase helper)
# ---------------------------------------------------------------------------


def measure_tile_phase_shift(
    img0_tile: np.ndarray,
    img1_tile: np.ndarray,
    joint_mask: np.ndarray,
) -> dict:
    """Phase shift of tile 1 relative to tile 0 (via the reused helper)."""
    if not joint_mask.any():
        return {"dx_px": float("nan"), "dy_px": float("nan"),
                "confidence": float("nan"), "status": "NO_VALID_PIXELS"}
    mi = float(np.mean(img0_tile[joint_mask]))
    mj = float(np.mean(img1_tile[joint_mask]))
    if not np.isfinite(mi) or not np.isfinite(mj):
        return {"dx_px": float("nan"), "dy_px": float("nan"),
                "confidence": float("nan"), "status": "NO_VALID_PIXELS"}
    ti = np.where(joint_mask, img0_tile, mi)
    tj = np.where(joint_mask, img1_tile, mj)
    if float(np.std(ti)) < 1e-12 or float(np.std(tj)) < 1e-12:
        return {"dx_px": float("nan"), "dy_px": float("nan"),
                "confidence": float("nan"), "status": "LOW_TEXTURE"}
    try:
        dx, dy, score = _phase_cross_correlation_shift(ti, tj, upsample=5)
    except Exception as exc:  # noqa: BLE001
        return {"dx_px": float("nan"), "dy_px": float("nan"),
                "confidence": float("nan"), "status": f"ERROR:{type(exc).__name__}"}
    if not math.isfinite(dx) or not math.isfinite(dy):
        return {"dx_px": float("nan"), "dy_px": float("nan"),
                "confidence": float("nan"), "status": "PHASE_FAILED"}
    return {"dx_px": float(dx), "dy_px": float(dy),
            "confidence": float(score), "status": "OK"}


# ---------------------------------------------------------------------------
# Task 6 — single-scale dense shift field
# ---------------------------------------------------------------------------


def measure_dense_shift_field(
    overlap: dict,
    grid_n: int,
    config: dict | None = None,
) -> list[dict]:
    """Phase-shift rows for one tile grid (metadata-only alignment)."""
    cfg = {**DEFAULT_QUALITY_CONFIG, **(config or {})}
    img0, img1 = overlap["image0"], overlap["image1"]
    joint = overlap["joint_mask"]
    windows = make_tile_windows(
        overlap["height"], overlap["width"], grid_n,
        cfg["min_tile_size_px"],
    )
    rows = []
    for w in windows:
        r0, r1 = w["row0"], w["row1"]
        c0, c1 = w["col0"], w["col1"]
        t0 = img0[r0:r1, c0:c1]
        t1 = img1[r0:r1, c0:c1]
        m0 = overlap["mask0"][r0:r1, c0:c1]
        m1 = overlap["mask1"][r0:r1, c0:c1]
        tj = joint[r0:r1, c0:c1]

        quality = evaluate_tile_quality(t0, t1, m0, m1, cfg)
        phase = {"dx_px": float("nan"), "dy_px": float("nan"),
                 "confidence": float("nan"), "status": "NOT_MEASURED"}
        accepted = False
        reject_reason = quality["reject_reason"]
        if w["scale_too_small"]:
            reject_reason = reject_reason or "TILE_TOO_SMALL"
        if reject_reason is None:
            phase = measure_tile_phase_shift(t0, t1, tj)
            if (phase["status"] == "OK"
                    and (not np.isfinite(phase["confidence"])
                         or phase["confidence"] <= cfg["max_phase_error_px"])):
                accepted = True
            else:
                reject_reason = {
                    "OK": f"PHASE_ERROR_{phase['confidence']:.3f}",
                    "LOW_TEXTURE": "LOW_TEXTURE",
                    "NO_VALID_PIXELS": "LOW_VALID_FRACTION",
                }.get(phase["status"], f"PHASE_{phase['status']}")

        cy = (r0 + r1) / 2.0
        cx = (c0 + c1) / 2.0
        tx = overlap["transform"]
        wx, wy = tx * (cx, cy)
        rows.append({
            "grid_n": grid_n,
            "tile_row": w["tile_row"],
            "tile_col": w["tile_col"],
            "row0": r0, "row1": r1,
            "col0": c0, "col1": c1,
            "center_pixel_x": float(cx),
            "center_pixel_y": float(cy),
            "center_world_x": float(wx),
            "center_world_y": float(wy),
            "valid_fraction_0": quality["valid_fraction_0"],
            "valid_fraction_1": quality["valid_fraction_1"],
            "joint_valid_fraction": quality["joint_valid_fraction"],
            "texture_0": quality["std_0"],
            "texture_1": quality["std_1"],
            "phase_dx_px": phase["dx_px"],
            "phase_dy_px": phase["dy_px"],
            "phase_mag_px": (
                float(math.hypot(phase["dx_px"], phase["dy_px"]))
                if math.isfinite(phase["dx_px"]) else float("nan")
            ),
            "phase_dx_m": (
                float(phase["dx_px"] * overlap["resolution"])
                if math.isfinite(phase["dx_px"]) else None
            ),
            "phase_dy_m": (
                float(phase["dy_px"] * overlap["resolution"])
                if math.isfinite(phase["dy_px"]) else None
            ),
            "phase_mag_m": (
                float(math.hypot(phase["dx_px"], phase["dy_px"])
                      * overlap["resolution"])
                if math.isfinite(phase["dx_px"]) else None
            ),
            "phase_confidence": phase["confidence"],
            "accepted": bool(accepted),
            "reject_reason": reject_reason,
        })
    return rows


# ---------------------------------------------------------------------------
# Task 7 — robust single-scale summary
# ---------------------------------------------------------------------------


def summarize_dense_shift_field(rows: list[dict]) -> dict:
    """Robust summary of an accepted phase-shift set (px units)."""
    ok = [r for r in rows if r.get("accepted") and
          np.isfinite(r.get("phase_dx_px"))]
    if len(ok) < 5:
        return {
            "grid_n": rows[0]["grid_n"] if rows else None,
            "n_total": len(rows),
            "n_quality_valid": sum(
                1 for r in rows if r.get("reject_reason") is None
            ),
            "n_phase_ok": len(ok),
            "acceptance_rate": (len(ok) / len(rows)) if rows else 0.0,
            "summary_status": "INSUFFICIENT_TILES",
        }
    dx = np.array([r["phase_dx_px"] for r in ok])
    dy = np.array([r["phase_dy_px"] for r in ok])
    mag = np.hypot(dx, dy)
    # robust scatter: median absolute deviation of per-tile distance to the
    # median vector, scaled like a standard deviation
    med_v = np.array([float(np.median(dx)), float(np.median(dy))])
    dist = np.hypot(dx - med_v[0], dy - med_v[1])
    mad = float(np.median(dist)) if len(dist) else 0.0
    robust_range_px = float(1.4826 * mad)
    return {
        "grid_n": ok[0].get("grid_n"),
        "n_total": len(rows),
        "n_quality_valid": sum(1 for r in rows
                               if r.get("reject_reason") is None),
        "n_phase_ok": len(ok),
        "acceptance_rate": len(ok) / len(rows),
        "median_dx_px": float(np.median(dx)),
        "median_dy_px": float(np.median(dy)),
        "mean_dx_px": float(np.mean(dx)),
        "mean_dy_px": float(np.mean(dy)),
        "std_dx_px": float(np.std(dx)),
        "std_dy_px": float(np.std(dy)),
        "p05_dx_px": float(np.percentile(dx, 5)),
        "p95_dx_px": float(np.percentile(dx, 95)),
        "p05_dy_px": float(np.percentile(dy, 5)),
        "p95_dy_px": float(np.percentile(dy, 95)),
        "iqr_dx_px": float(np.percentile(dx, 75) - np.percentile(dx, 25)),
        "iqr_dy_px": float(np.percentile(dy, 75) - np.percentile(dy, 25)),
        "mad_magnitude_px": mad,
        "robust_range_px": robust_range_px,
        "median_magnitude_px": float(np.median(mag)),
        "std_magnitude_px": float(np.std(mag)),
        "dx_range_px": float(np.max(dx) - np.min(dx)),
        "dy_range_px": float(np.max(dy) - np.min(dy)),
        "summary_status": "OK",
    }


# ---------------------------------------------------------------------------
# Task 8 — spatial trend fit
# ---------------------------------------------------------------------------


def fit_shift_spatial_trend(rows: list[dict]) -> dict:
    """Fit dx/dy as planes over tile centres normalised to [-1, 1].

    Outlier tiles (phase miss-locks) are removed via a median-absolute-
    deviation gate before fitting so a handful of bad tiles cannot masquerade
    as a spatial gradient.
    """
    ok = [r for r in rows if r.get("accepted")
          and np.isfinite(r.get("phase_dx_px"))]
    if len(ok) < 5:
        return {"trend_status": "INSUFFICIENT_TILES"}
    dx0 = np.array([r["phase_dx_px"] for r in ok])
    dy0 = np.array([r["phase_dy_px"] for r in ok])
    med = np.array([np.median(dx0), np.median(dy0)])
    dist = np.hypot(dx0 - med[0], dy0 - med[1])
    mad = float(np.median(dist))
    keep = dist <= max(3.0 * (1.4826 * mad), 6.0)
    ok = [r for r, k in zip(ok, keep) if k]
    if len(ok) < 5:
        return {"trend_status": "INSUFFICIENT_TILES", "n_before_outlier_reject": len(dx0)}

    cx = np.array([r["center_pixel_x"] for r in ok], dtype=float)
    cy = np.array([r["center_pixel_y"] for r in ok], dtype=float)
    dx = np.array([r["phase_dx_px"] for r in ok])
    dy = np.array([r["phase_dy_px"] for r in ok])
    x = (cx - cx.mean()) / (cx.max() - cx.min()) if cx.max() > cx.min() \
        else np.zeros_like(cx)
    y = (cy - cy.mean()) / (cy.max() - cy.min()) if cy.max() > cy.min() \
        else np.zeros_like(cy)
    X = np.column_stack([np.ones_like(x), x, y])

    def _fit(z):
        coef, *_ = np.linalg.lstsq(X, z, rcond=None)
        pred = X @ coef
        ss_res = float(np.sum((z - pred) ** 2))
        ss_tot = float(np.sum((z - z.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
        return coef, r2

    cdx, r2dx = _fit(dx)
    cdy, r2dy = _fit(dy)

    # predicted range over the tile-centre bounding box
    corners = np.array([[-1, -1], [1, -1], [-1, 1], [1, 1]])
    Xc = np.column_stack([np.ones(len(corners)), corners])
    pdx = Xc @ cdx
    pdy = Xc @ cdy
    span = np.max(np.hypot(pdx, pdy)) - np.min(np.hypot(pdx, pdy))
    return {
        "trend_status": "OK",
        "n_points": len(ok),
        "n_outliers_removed": int((~keep).sum()),
        "dx_coefficients": [float(cdx[0]), float(cdx[1]), float(cdx[2])],
        "dy_coefficients": [float(cdy[0]), float(cdy[1]), float(cdy[2])],
        "dx_r2": float(r2dx),
        "dy_r2": float(r2dy),
        "predicted_dx_min": float(np.min(pdx)),
        "predicted_dx_max": float(np.max(pdx)),
        "predicted_dy_min": float(np.min(pdy)),
        "predicted_dy_max": float(np.max(pdy)),
        "predicted_dx_range": float(np.max(pdx) - np.min(pdx)),
        "predicted_dy_range": float(np.max(pdy) - np.min(pdy)),
        "trend_vector_change_px": float(span),
    }


# ---------------------------------------------------------------------------
# Task 9 — multiscale run
# ---------------------------------------------------------------------------


def run_multiscale_dense_phase(
    overlap: dict,
    grids: tuple[int, ...] = DEFAULT_GRIDS,
    config: dict | None = None,
) -> dict:
    """Run all scales on the shared overlap raster with one quality config."""
    fields: dict[str, list[dict]] = {}
    summaries: dict[str, dict] = {}
    trends: dict[str, dict] = {}
    for g in grids:
        rows = measure_dense_shift_field(overlap, g, config)
        fields[f"grid_{g}"] = rows
        summaries[f"grid_{g}"] = summarize_dense_shift_field(rows)
        trends[f"grid_{g}"] = fit_shift_spatial_trend(rows)
    return {"fields": fields, "summaries": summaries, "trends": trends}


# ---------------------------------------------------------------------------
# Task 10 — multiscale stability classifier
# ---------------------------------------------------------------------------

CLASSIFIER_THRESHOLDS = {
    "min_scales_with_enough_tiles": 2,
    "min_phase_ok_per_scale": 5,
    "between_scale_median_tolerance_px": 3.0,
    "constant_field_range_px": 5.0,
    "gradient_field_range_px": 5.0,
    "gradient_consistency_cos": 0.5,
}


def classify_multiscale_shift_behavior(
    scale_summaries: dict,
    scale_trends: dict,
) -> dict:
    """Classify the 0-1 shift as constant / gradient / unstable / ambiguous."""
    th = CLASSIFIER_THRESHOLDS
    enough = [
        g for g, s in scale_summaries.items()
        if s.get("n_phase_ok", 0) >= th["min_phase_ok_per_scale"]
    ]
    medians = {
        g: (s.get("median_dx_px"), s.get("median_dy_px"))
        for g, s in scale_summaries.items()
        if s.get("n_phase_ok", 0) >= 3
    }
    reasoning = {"scales_with_enough_tiles": enough,
                 "per_scale_median": medians}

    if len(enough) < th["min_scales_with_enough_tiles"]:
        return {"state": "UNSTABLE_OR_INSUFFICIENT",
                "thresholds": th, "reasoning": reasoning,
                "reason": "too few scales have enough accepted tiles"}
    if not medians:
        return {"state": "UNSTABLE_OR_INSUFFICIENT",
                "thresholds": th, "reasoning": reasoning,
                "reason": "no scale has measurable median"}

    keys = sorted(medians)
    tol = th["between_scale_median_tolerance_px"]
    median_consistent = all(
        math.hypot(medians[a][0] - medians[b][0],
                   medians[a][1] - medians[b][1]) <= tol
        for i, a in enumerate(keys) for b in keys[i + 1:]
    )

    raw_ranges = [
        math.hypot(scale_summaries[g].get("dx_range_px", 0.0),
                   scale_summaries[g].get("dy_range_px", 0.0))
        for g in enough
    ]
    robust_values = [
        float(scale_summaries[g].get(
            "robust_range_px",
            math.hypot(scale_summaries[g].get("dx_range_px", 0.0),
                       scale_summaries[g].get("dy_range_px", 0.0)),
        ))
        for g in enough
    ]
    median_robust = float(np.median(robust_values)) if robust_values else float("nan")
    reasoning["raw_range_px"] = raw_ranges
    reasoning["robust_range_px"] = robust_values

    if median_consistent and median_robust <= th["constant_field_range_px"]:
        return {"state": "STABLE_CONSTANT_SHIFT",
                "thresholds": th, "reasoning": reasoning,
                "reason": (f"medians consistent within {tol} px and robust "
                           f"spread {median_robust:.2f} px <= "
                           f"{th['constant_field_range_px']} px")}

    # gradient: need at least the two finer scales, consistent directions,
    # and a trend range above noise
    fine = [g for g in (6, 8) if f"grid_{g}" in scale_trends and
            scale_trends[f"grid_{g}"].get("trend_status") == "OK" and
            scale_trends[f"grid_{g}"].get("n_points", 0) >= 5]
    gradient_evidence = []
    for g in fine:
        t = scale_trends[f"grid_{g}"]
        rn = float(t.get("trend_vector_change_px", 0.0) or 0.0)
        if rn <= th["gradient_field_range_px"]:
            continue
        gvec = np.array([
            t["dx_coefficients"][1], t["dx_coefficients"][2],
            t["dy_coefficients"][1], t["dy_coefficients"][2],
        ], dtype=float)
        gradient_evidence.append((g, gvec, rn))

    if len(gradient_evidence) >= 2 and median_robust > th["constant_field_range_px"]:
        v1 = gradient_evidence[0][1]
        v2 = gradient_evidence[1][1]
        cos_angle = float(np.dot(v1, v2) /
                          (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12))
        if cos_angle >= th["gradient_consistency_cos"]:
            return {
                "state": "STABLE_SPATIAL_GRADIENT",
                "thresholds": th,
                "reasoning": {
                    **reasoning,
                    "gradient_edges": [
                        (g, x.tolist(), float(rn)) for g, x, rn in gradient_evidence
                    ],
                    "gradient_consistency_cos": float(cos_angle),
                },
                "reason": ("6×6 and 8×8 both show a consistent spatial "
                           "gradient with field range above noise"),
            }

    if not median_consistent:
        return {"state": "UNSTABLE_OR_INSUFFICIENT",
                "thresholds": th, "reasoning": reasoning,
                "reason": "across-scale medians disagree"}
    return {"state": "MIXED_OR_AMBIGUOUS",
            "thresholds": th, "reasoning": reasoning,
            "reason": "constant offset or gradient cannot be pinned down"}


# ---------------------------------------------------------------------------
# Task 11/12 — figures
# ---------------------------------------------------------------------------


def _matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_dense_vector_map(
    fields: dict[str, list[dict]],
    overlap: dict,
    out_path: str | Path,
    arrow_scale: float = 2.0,
) -> Path:
    """Three panels (4×4 / 6×6 / 8×8) on the shared overlap extent."""
    plt = _matplotlib()
    names = [k for k in fields if fields[k]]
    h, w = overlap["height"], overlap["width"]
    fig, axes = plt.subplots(1, len(names), figsize=(6 * len(names), 5),
                             sharex=True, sharey=True)
    if len(names) == 1:
        axes = [axes]

    # shared background: down-sampled scene-0 B14, same stretch for every panel
    full = np.asarray(overlap["image0"], dtype=np.float64)
    step = max(h // 400, w // 400, 1)
    bg = full[::step, ::step]
    valid_bg = np.isfinite(bg)
    vals = bg[valid_bg]
    lo = float(np.percentile(vals, 2)) if vals.size else 0.0
    hi = float(np.percentile(vals, 98)) if vals.size else 1.0
    if hi - lo < 1e-12:
        hi = lo + 1.0
    bg_u = np.clip((bg - lo) / (hi - lo), 0.0, 1.0)
    bg_u[~valid_bg] = 0.0

    for ax, name in zip(axes, names):
        rows = fields[name]
        acc = [r for r in rows if r.get("accepted")]
        rej = [r for r in rows if not r.get("accepted")]
        ax.imshow(bg_u, extent=[0, w, h, 0], cmap="gray",
                  vmin=0, vmax=1, aspect="auto")
        for r in acc:
            ax.plot(r["center_pixel_x"], r["center_pixel_y"], "ko", ms=3)
            ax.arrow(r["center_pixel_x"], r["center_pixel_y"],
                     r["phase_dx_px"] * arrow_scale,
                     r["phase_dy_px"] * arrow_scale, color="red",
                     width=1.0, head_width=6.0, length_includes_head=True)
        for r in rej:
            ax.plot(r["center_pixel_x"], r["center_pixel_y"], "x",
                    color="gray", ms=4)
        acc_n = len(acc)
        med = np.median([
            (r["phase_dx_px"], r["phase_dy_px"]) for r in acc
        ], axis=0) if acc else (float("nan"), float("nan"))
        std = np.std([math.hypot(r["phase_dx_px"], r["phase_dy_px"])
                      for r in acc]) if acc else float("nan")
        ax.set_title(
            f"{name}\n{acc_n}/{len(rows)} accepted | "
            f"median=({med[0]:+.1f},{med[1]:+.1f}) px | std={std:.1f}"
        )
        ax.set_xlabel("column (px)")
        ax.set_ylabel("row (px)")
        ax.text(0.02, 0.98,
                f"Arrows visually scaled by {arrow_scale:.1f}×",
                transform=ax.transAxes, va="top", fontsize=8,
                bbox=dict(facecolor="white", alpha=0.7))
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_tile_quality_map(
    fields: dict[str, list[dict]],
    overlap: dict,
    out_path: str | Path,
) -> Path:
    """Accepted vs rejected (with reason) tiles across all scales."""
    plt = _matplotlib()
    colors = {
        None: ("green", "accepted"),
        "LOW_VALID_FRACTION": ("red", "low valid"),
        "LOW_TEXTURE": ("orange", "low texture"),
        "TILE_TOO_SMALL": ("gray", "tile too small"),
        "LOW_PHASE_CONFIDENCE": ("purple", "low confidence"),
        "PHASE_NOT_OK": ("magenta", "phase not ok"),
    }
    fig, ax = plt.subplots(figsize=(8, 8))
    h, w = overlap["height"], overlap["width"]
    for name, rows in fields.items():
        for r in rows:
            color, label = colors.get(r.get("reject_reason"),
                                      ("black", str(r.get("reject_reason"))))
            ax.add_patch(plt.Rectangle(
                (r["col0"], r["row0"]), r["col1"] - r["col0"],
                r["row1"] - r["row0"], fill=False, edgecolor=color,
                linewidth=0.8))
            ax.text((r["col0"] + r["col1"]) / 2, (r["row0"] + r["row1"]) / 2,
                    label, fontsize=5, ha="center", va="center",
                    color=color)
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_aspect("equal")
    ax.set_title("0-1 metadata-only tile quality by reason")
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Task 13 — conclusion
# ---------------------------------------------------------------------------


def build_dense_phase_conclusion(
    *,
    summaries: dict,
    trends: dict,
    stability: dict,
    output_dir: str,
) -> dict:
    """Assemble the final conclusion answering the 8 questions."""
    def _g(g):
        return summaries.get(f"grid_{g}", {})

    conclusion = {
        "final_state": stability["state"],
        "answers": {
            "1_4x4_accepted_tiles": _g(4).get("n_phase_ok"),
            "2_6x6_accepted_tiles": _g(6).get("n_phase_ok"),
            "3_8x8_accepted_tiles": _g(8).get("n_phase_ok"),
            "4_multiscale_median_dxdy_px": {
                g: (_g(g).get("median_dx_px"), _g(g).get("median_dy_px"))
                for g in (4, 6, 8)
            },
            "5_within_overlap_range_px": {
                g: (float(_g(g).get("dx_range_px", 0) or 0),
                    float(_g(g).get("dy_range_px", 0) or 0))
                for g in (4, 6, 8)
            },
            "6_spatial_gradient_present": {
                g: bool(trends.get(f"grid_{g}", {}).get(
                    "trend_vector_change_px", 0) or 0) > 1.0
                for g in (4, 6, 8)
            },
            "7_gradient_repeats_6x8": (
                stability["reasoning"].get("gradient_consistency_cos")
                is not None
            ),
            "8_final_state": stability["state"],
        },
        "thresholds": CLASSIFIER_THRESHOLDS,
        "can_conclude": [
            "The 0-1 overlap metadata-vs-content offset was measured on a "
            "shared metadata-only raster at three scales.",
            "Rejected tiles carry explicit reasons (valid/texture/confidence).",
        ],
        "cannot_conclude": [
            "No scene (0 or 1) is identified as absolutely mis-located.",
            "The relative shift is not called an absolute geolocation error.",
            "Unless a stable multiscale gradient repeats, no local-geometry "
            "model change is forced.",
        ],
        "diagnostic_artifacts": {
            "baseline": f"{output_dir}/00_dense_phase_baseline.json",
            "grid_meta": f"{output_dir}/01_overlap_grid.json",
            "tiles": f"{output_dir}/01_dense_tile_shifts.csv",
            "summary": f"{output_dir}/02_dense_shift_summary.json",
            "vector_map": f"{output_dir}/03_dense_shift_vector_map.png",
            "stability": f"{output_dir}/04_multiscale_stability.json",
            "quality_map": f"{output_dir}/05_tile_quality_map.png",
        },
    }
    return conclusion


def write_dense_phase_conclusion_text(
    conclusion: dict, path: str | Path
) -> Path:
    lines = [
        "=== Dense 0-1 phase-field conclusion ===",
        f"Final state: {conclusion['final_state']}",
        "Answers:",
    ]
    for k, v in conclusion["answers"].items():
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append("CAN conclude:")
    for item in conclusion["can_conclude"]:
        lines.append(f"  - {item}")
    lines.append("CANNOT conclude:")
    for item in conclusion["cannot_conclude"]:
        lines.append(f"  - {item}")
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out