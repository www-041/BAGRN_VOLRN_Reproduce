"""Scene-geolocation consistency diagnostics.

This module asks: why do three locally healthy pairwise registrations
(0-1, 0-4, 1-4) fail to close globally by ~1 km when expressed in a common
world frame?  It constructs two relations per edge — the *metadata-predicted*
relation from the native geotransforms and the *direct* relation from the
(validated, frame-normalised) pairwise registration — and compares them:

    T_meta_i_from_j   = inv(A_i) @ A_j                (scene j -> i, native px)
    T_direct_i_from_j = <frame-normalised pairwise>   (scene j -> i, native px)
    C_world_i_from_j  = A_i @ T_direct_i_from_j @ inv(A_j)
                                      (canonical world correction, ~= I if
                                       metadata georeferencing matches content)

The plan forbids equating raw geotransform differences with absolute
geolocation error, and requires metadata-only phase correlation (no direct
or MST transform applied) to measure each overlap's displacement field
independently.  Diagnosis only; no production semantics are modified.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.transform import Affine

from src.multiscene_sift.frame_diagnostics import (
    affine_to_matrix,
    matrix_to_affine,
    match_view_to_common_affine,
    match_view_frame_params,
    rebuild_pair_common_grids,
    transform_points,
    world_transform_from_saved_pair,
    write_json,
)
from src.multiscene_sift.loop_diagnostics import (
    find_pair_row,
    load_pairwise_results,
    load_run_config,
    load_scenes_for_run,
    _north_up_grid,
    _phase_cross_correlation_shift,
    _reproject_band_to_grid,
)
from src.multiscene_sift.models import Scene

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Task 1 — baseline
# ---------------------------------------------------------------------------


def load_geolocation_baseline(run_dir: str | Path, frame_diag_dir: str | Path) -> dict:
    """Read the frame-diagnostics conclusion into a baseline dict."""
    run = Path(run_dir)
    fr = Path(frame_diag_dir)
    baseline: dict[str, Any] = {
        "run_dir": str(run),
        "frame_diagnostics_dir": str(fr),
    }
    for name, key in (
        ("12_frame_conclusion.json", "frame_conclusion"),
        ("07_global_composition_comparison.json", "global_composition"),
        ("09_tree_edge_independent_checks.json", "tree_checks"),
        ("01_frame_semantics.json", "frame_semantics"),
    ):
        path = fr / name
        baseline[key] = (
            json_load(path) if path.is_file() else None
        )
    return baseline


def json_load(path: str | Path) -> Any:
    import json

    return json.loads(Path(path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Task 2/3 — canonical relations
# ---------------------------------------------------------------------------


def get_native_pixel_to_world(scene: Scene, band: str) -> np.ndarray:
    """``A_i``: scene-native pixel -> world (3×3 homogeneous)."""
    return affine_to_matrix(scene.transforms[band])


def metadata_native_relation(A_i: np.ndarray, A_j: np.ndarray) -> np.ndarray:
    """Metadata-predicted ``scene j -> scene i`` native-pixel relation.

    ``inv(A_i) @ A_j`` applies A_j (native j -> world), then inv(A_i)
    (world -> native i).  All signs come from the matrices.
    """
    return np.linalg.inv(A_i) @ A_j


def direct_native_relation(
    T_direct_world_i_from_j: np.ndarray,
    A_i: np.ndarray,
    A_j: np.ndarray,
) -> np.ndarray:
    """Content-based ``scene j -> scene i`` native-pixel relation.

    ``inv(A_i) @ T_world @ A_j`` maps native-j pixels into native-i pixels
    through the world relation.
    """
    return np.linalg.inv(A_i) @ T_direct_world_i_from_j @ A_j


def compute_world_correction(
    A_i: np.ndarray,
    T_direct_native_i_from_j: np.ndarray,
    A_j: np.ndarray,
) -> np.ndarray:
    """Canonical world correction ``C_world_i_from_j``.

    ``A_i @ T_direct_native @ inv(A_j)`` acts on scene-j world coordinates and
    yields the world location where scene-i content actually is.  If the
    metadata relation equalled the direct relation, this is identity.
    """
    return A_i @ T_direct_native_i_from_j @ np.linalg.inv(A_j)


def edge_world_corrections(
    scenes: list[Scene],
    band: str,
    pair_rows: list[dict],
    grids: dict,
    match_views: dict,
    edges: list[tuple[int, int]],
) -> dict:
    """FRAME-NORMALISED canonical relations for the given edges.

    Returns per edge: ``T_meta_i_from_j``, ``T_direct_i_from_j`` (native px),
    ``C_world_i_from_j``, decomposed translation / rotation / scale / shear.
    """
    out: dict[str, Any] = {}
    for i, j in edges:
        key = f"edge_{i}_{j}"
        gname = f"pair_{min(i, j)}_{max(i, j)}"
        if gname not in grids or gname not in match_views:
            out[key] = {"available": False}
            continue
        A_i = get_native_pixel_to_world(scenes[i], band)
        A_j = get_native_pixel_to_world(scenes[j], band)
        T_meta = metadata_native_relation(A_i, A_j)
        T_world = world_transform_from_saved_pair(
            i, j, pair_rows,
            Affine(*grids[gname]["transform"]),
            match_views[gname],
            adjust_frame=True,
        )
        if T_world is None:
            out[key] = {"available": False}
            continue
        T_direct = direct_native_relation(T_world, A_i, A_j)
        C_world = compute_world_correction(A_i, T_direct, A_j)
        A = np.linalg.inv(T_meta) @ T_direct
        dec = _decompose(A)
        out[key] = {
            "edge": [i, j],
            "T_meta_i_from_j": T_meta.tolist(),
            "T_direct_i_from_j": T_direct.tolist(),
            "C_world_i_from_j": C_world.tolist(),
            "translation_m": [float(C_world[0, 2]), float(C_world[1, 2])],
            "translation_equiv_px14": [
                float(C_world[0, 2] / 14.0), float(C_world[1, 2] / 14.0)
            ],
            "rotation_deg": dec["rotation_deg"],
            "scale_x": dec["scale_x"],
            "scale_y": dec["scale_y"],
            "shear_deg": dec["shear_deg"],
            "meta_to_direct_difference_m": [
                float(A[0, 2]), float(A[1, 2])
            ],
        }
    return out


def _decompose(matrix: np.ndarray) -> dict:
    a, b = matrix[0, 0], matrix[0, 1]
    c, d = matrix[1, 0], matrix[1, 1]
    sx = float(math.hypot(a, c))
    sy = float(math.hypot(b, d))
    rot = math.atan2(c, a)
    rot2 = math.atan2(-b, d)
    rot = math.atan2(math.sin(rot) + math.sin(rot2),
                     math.cos(rot) + math.cos(rot2))
    shear = math.atan2(-b * a - d * c, a * d - b * c)
    return {
        "rotation_deg": float(math.degrees(rot)),
        "scale_x": sx,
        "scale_y": sy,
        "shear_deg": float(math.degrees(shear)),
    }


# ---------------------------------------------------------------------------
# Task 4 — correction field sampling (pure matrix, no imagery)
# ---------------------------------------------------------------------------

CORRECTION_FIELD_GATES = {
    "samples_per_edge": 9,
    "variation_gate_px": 2.0,
    "spatial_gate_px": 5.0,
}


def sample_correction_field(
    T_meta_i_from_j: np.ndarray,
    T_direct_i_from_j: np.ndarray,
    shape_j: tuple[int, int],
) -> list[dict]:
    """Sample metadata/direct positions for a 3×3 grid of scene-j pixels.

    ``correction_dx/dy`` are native-i pixel displacements (direct minus meta).
    """
    h, w = shape_j
    fracs = (0.1, 0.5, 0.9)
    rows: list[dict] = []
    sid = 0
    for fy in fracs:
        for fx in fracs:
            col_j = w * fx
            row_j = h * fy
            p = np.array([[col_j, row_j]])
            meta = transform_points(T_meta_i_from_j, p)[0]
            direct = transform_points(T_direct_i_from_j, p)[0]
            dx = direct[0] - meta[0]
            dy = direct[1] - meta[1]
            rows.append({
                "sample_id": sid,
                "sample_fraction_x": fx,
                "sample_fraction_y": fy,
                "j_col": float(col_j),
                "j_row": float(row_j),
                "meta_i_col": float(meta[0]),
                "meta_i_row": float(meta[1]),
                "direct_i_col": float(direct[0]),
                "direct_i_row": float(direct[1]),
                "correction_dx_px": float(dx),
                "correction_dy_px": float(dy),
                "correction_dx_m": float(dx * 14.0),
                "correction_dy_m": float(dy * 14.0),
                "correction_mag_m": float(math.hypot(dx, dy) * 14.0),
                "valid": True,
            })
            sid += 1
    return rows


def correction_field_summary(samples: list[dict]) -> dict:
    """Statistics of a correction field over the sampled points (px units)."""
    valid = [s for s in samples if s.get("valid")]
    if len(valid) < 4:
        return {"n_valid": len(samples), "classification": "INSUFFICIENT_VALID_SAMPLES"}
    dx = np.array([s["correction_dx_px"] for s in valid])
    dy = np.array([s["correction_dy_px"] for s in valid])
    x = np.array([s["sample_fraction_x"] for s in valid])
    y = np.array([s["sample_fraction_y"] for s in valid])
    grad_dx = _plane_gradient(x, y, dx)
    grad_dy = _plane_gradient(x, y, dy)
    total_std = float(np.hypot(np.std(dx), np.std(dy)))
    summary = {
        "n_valid": len(valid),
        "mean_dx_px": float(np.mean(dx)),
        "mean_dy_px": float(np.mean(dy)),
        "std_dx_px": float(np.std(dx)),
        "std_dy_px": float(np.std(dy)),
        "max_min_dx_px": float(np.max(dx) - np.min(dx)),
        "max_min_dy_px": float(np.max(dy) - np.min(dy)),
        "translation_magnitude_mean_m": float(
            np.mean(np.hypot(dx, dy)) * 14.0
        ),
        "translation_magnitude_std_m": float(np.std(np.hypot(dx, dy)) * 14.0),
        "gradient_dx_per_span": list(grad_dx),
        "gradient_dy_per_span": list(grad_dy),
        "total_std_px": total_std,
        "classification": classify_correction_field(
            total_std, grads=(grad_dx[0], grad_dx[1], grad_dy[0], grad_dy[1])
        ),
        "gates": CORRECTION_FIELD_GATES,
    }
    return summary


def _plane_gradient(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[float, float]:
    """Fit ``z = c0 + c1 x + c2 y`` over [0,1] fractions; return (c1, c2)."""
    try:
        X = np.column_stack([np.ones_like(x), x, y])
        coef, *_ = np.linalg.lstsq(X, z, rcond=None)
        return float(coef[1]), float(coef[2])
    except np.linalg.LinAlgError:
        return 0.0, 0.0


def classify_correction_field(
    total_std_px: float, grads: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
) -> str:
    """Describe whether a correction is constant or varying within an overlap."""
    if total_std_px < CORRECTION_FIELD_GATES["variation_gate_px"]:
        return "NEAR_CONSTANT_WITHIN_OVERLAP"
    gx1, gx2, gy1, gy2 = grads
    grad_mag = math.hypot(gx1, gx2, gy1, gy2)
    if total_std_px > CORRECTION_FIELD_GATES["spatial_gate_px"] and grad_mag > 1.0:
        return "SPATIALLY_VARYING_WITHIN_OVERLAP"
    return "AMBIGUOUS"


# ---------------------------------------------------------------------------
# Task 5 — metadata-only local displacement field (imagery, no transforms)
# ---------------------------------------------------------------------------

METADATA_ONLY_GATES = {
    "tile_grid": 4,
    "min_joint_valid_fraction": 0.6,
    "min_tile": 256,
}


def metadata_only_overlap_grid(
    scene_i: Scene,
    scene_j: Scene,
    band: str,
) -> tuple[Affine, tuple[int, int], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Place both scenes on one metadata-only 14 m grid (no corrections).

    Returns ``(grid_transform, grid_shape, a_i, a_j, v_i, v_j)``.
    """
    b_i = scene_i.bounds[band]
    b_j = scene_j.bounds[band]
    left = max(b_i.left, b_j.left)
    bottom = max(b_i.bottom, b_j.bottom)
    right = min(b_i.right, b_j.right)
    top = min(b_i.top, b_j.top)
    if right <= left or top <= bottom:
        raise ValueError(f"Scenes {scene_i.index}/{scene_j.index} do not overlap")
    res = abs(scene_i.transforms[band].a)
    grid_transform, gw, gh = _north_up_grid(
        (left, bottom, right, top), res, max_side=10 ** 9
    )
    shape = (gh, gw)
    a_i, v_i = _reproject_band_to_grid(
        scene_i.band_paths[band], scene_i.transforms[band], scene_i.crs,
        grid_transform, shape,
    )
    a_j, v_j = _reproject_band_to_grid(
        scene_j.band_paths[band], scene_j.transforms[band], scene_j.crs,
        grid_transform, shape,
    )
    return grid_transform, shape, a_i, a_j, v_i, v_j


def measure_metadata_only_tile_shifts(
    scene_i: Scene,
    scene_j: Scene,
    band: str,
    tile_grid: int | None = None,
    min_joint_valid_fraction: float | None = None,
) -> list[dict]:
    """Metadata-only local phase shift between the two scenes' content.

    No direct or MST transform is applied: the two bands are placed by their
    native geotransforms and each tile's phase shift measures how much the
    scene-j content still needs to move to align with scene i.
    """
    tile_grid = int(tile_grid or METADATA_ONLY_GATES["tile_grid"])
    min_valid = float(
        min_joint_valid_fraction or METADATA_ONLY_GATES["min_joint_valid_fraction"]
    )
    grid_transform, shape, a_i, a_j, v_i, v_j = metadata_only_overlap_grid(
        scene_i, scene_j, band
    )
    gh, gw = shape
    inv_grid = ~grid_transform

    results: list[dict] = []
    for r_i in range(tile_grid):
        for c_i in range(tile_grid):
            r0 = int(gh * r_i / tile_grid)
            r1 = int(gh * (r_i + 1) / tile_grid)
            c0 = int(gw * c_i / tile_grid)
            c1 = int(gw * (c_i + 1) / tile_grid)
            if r1 - r0 < METADATA_ONLY_GATES["min_tile"] // 2 or \
               c1 - c0 < METADATA_ONLY_GATES["min_tile"] // 2:
                continue
            joint = v_i[r0:r1, c0:c1] & v_j[r0:r1, c0:c1]
            joint_frac = float(joint.mean()) if joint.size else 0.0
            tile_i = a_i[r0:r1, c0:c1]
            tile_j = a_j[r0:r1, c0:c1]
            if joint.any():
                std_i = float(np.std(tile_i[joint]))
                std_j = float(np.std(tile_j[joint]))
            else:
                std_i = std_j = 0.0

            cy = (r0 + r1) / 2.0
            cx = (c0 + c1) / 2.0
            center_world = grid_transform * (cx, cy)
            dx = dy = float("nan")
            score = float("nan")
            accepted = False
            if joint_frac >= min_valid and std_i > 0.0 and std_j > 0.0:
                # fill invalid with per-band mean, then phase-correlate
                mi = float(np.mean(tile_i[joint]))
                mj = float(np.mean(tile_j[joint]))
                ti = np.where(v_i[r0:r1, c0:c1], tile_i, mi)
                tj = np.where(v_j[r0:r1, c0:c1], tile_j, mj)
                try:
                    dx, dy, score = _phase_cross_correlation_shift(ti, tj)
                    accepted = True
                except Exception:  # noqa: BLE001
                    dx = dy = float("nan")
            results.append({
                "tile_row": r_i,
                "tile_col": c_i,
                "center_world_x": float(center_world[0]),
                "center_world_y": float(center_world[1]),
                "joint_valid_fraction": joint_frac,
                "texture_i": std_i,
                "texture_j": std_j,
                "phase_dx_px": dx,
                "phase_dy_px": dy,
                "phase_dx_m": float(dx * 14.0) if np.isfinite(dx) else None,
                "phase_dy_m": float(dy * 14.0) if np.isfinite(dy) else None,
                "phase_magnitude_px": (
                    float(math.hypot(dx, dy)) if np.isfinite(dx) else None
                ),
                "phase_confidence": float(score) if np.isfinite(score) else None,
                "accepted": accepted,
            })
    return results


def metadata_only_shift_summary(tiles: list[dict]) -> dict:
    """Per-edge aggregate of the metadata-only phase shifts."""
    acc = [t for t in tiles if t.get("accepted")]
    if len(acc) < 3:
        return {
            "n_tiles_total": len(tiles),
            "n_tiles_accepted": len(acc),
            "classification": "INSUFFICIENT_TILES",
        }
    dx = np.array([t["phase_dx_px"] for t in acc])
    dy = np.array([t["phase_dy_px"] for t in acc])
    # spatial gradient over tile centre fractions
    fx = np.array([t["tile_col"] for t in acc], dtype=float)
    fy = np.array([t["tile_row"] for t in acc], dtype=float)
    g_dx = _plane_gradient(fx / max(fx.max() - fx.min(), 1e-9),
                           fy / max(fy.max() - fy.min(), 1e-9), dx)
    g_dy = _plane_gradient(fx / max(fx.max() - fx.min(), 1e-9),
                           fy / max(fy.max() - fy.min(), 1e-9), dy)
    return {
        "n_tiles_total": len(tiles),
        "n_tiles_accepted": len(acc),
        "mean_dx_px": float(np.mean(dx)),
        "mean_dy_px": float(np.mean(dy)),
        "median_dx_px": float(np.median(dx)),
        "median_dy_px": float(np.median(dy)),
        "std_dx_px": float(np.std(dx)),
        "std_dy_px": float(np.std(dy)),
        "p95_magnitude_px": float(np.percentile(np.hypot(dx, dy), 95)),
        "spatial_gradient_dx": [g_dx[0], g_dx[1]],
        "spatial_gradient_dy": [g_dy[0], g_dy[1]],
    }


# ---------------------------------------------------------------------------
# Task 6 — world-correction triangle closure
# ---------------------------------------------------------------------------


def world_correction_triangle_closure(
    C_0_1: np.ndarray,
    C_0_4: np.ndarray,
    C_4_1: np.ndarray,
    sample_world: np.ndarray,
    pixel_size: float = 14.0,
) -> dict:
    """Close the loop C_01 vs C_04 @ C_41 on actual world points.

    Returns per-sample displacement (px) plus matrix decomposition of
    ``inv(C_01) @ (C_04 @ C_41)``.
    """
    D = np.linalg.inv(C_0_1) @ (C_0_4 @ C_4_1)
    pts = np.asarray(sample_world, dtype=np.float64)
    h = np.hstack([pts, np.ones((len(pts), 1))])
    direct = (C_0_1 @ h.T).T[:, :2]
    composed = ((C_0_4 @ (C_4_1 @ h.T)).T)[:, :2]
    diff_px = (composed - direct) / pixel_size
    mag = np.linalg.norm(diff_px, axis=1)
    classification = (
        "NEAR_CLOSED" if float(np.mean(mag)) <= 2.0 else
        "NEAR_CONSTANT_TRANSLATION_CONFLICT"
        if float(np.std(mag)) < 1.5 else
        "SPATIALLY_VARYING_CLOSURE_CONFLICT"
    )
    return {
        "difference_matrix": D.tolist(),
        "decomposition": _decompose(D),
        "translation_px14": [float(D[0, 2] / pixel_size), float(D[1, 2] / pixel_size)],
        "sample_mean_px": float(np.mean(mag)),
        "sample_median_px": float(np.median(mag)),
        "sample_std_px": float(np.std(mag)),
        "sample_p95_px": float(np.percentile(mag, 95)),
        "sample_max_px": float(np.max(mag)),
        "classification": classification,
    }


# ---------------------------------------------------------------------------
# Task 7 — scene-level constant-correction model (H_i)
# ---------------------------------------------------------------------------


def scene_level_constant_correction_test(
    C_0_1: np.ndarray,
    C_0_4: np.ndarray,
    C_4_1: np.ndarray,
    sample_world: np.ndarray,
    pixel_size: float = 14.0,
) -> dict:
    """Check whether one fixed per-scene world correction explains all three.

    Set H4 = I; then H0 = inv(C_0_4) and H1 = C_4_1 follow from the two tree
    edges; the model predicts ``C_pred_0_from_1 = inv(H0) @ H1`` (== C_04 @ C_41).
    """
    H4 = np.eye(3)
    H0 = np.linalg.inv(C_0_4)
    H1 = C_4_1
    C_pred = np.linalg.inv(H0) @ H1

    pts = np.asarray(sample_world, dtype=np.float64)
    h = np.hstack([pts, np.ones((len(pts), 1))])
    pred_pts = (C_pred @ h.T).T[:, :2]
    direct_pts = (C_0_1 @ h.T).T[:, :2]
    mag = np.linalg.norm((pred_pts - direct_pts) / pixel_size, axis=1)
    mean_px = float(np.mean(mag))
    status = (
        "CONSISTENT_WITH_SCENE_LEVEL_CONSTANT_CORRECTIONS"
        if mean_px <= 2.0 else
        "INCONSISTENT_WITH_SCENE_LEVEL_CONSTANT_CORRECTIONS"
    )
    return {
        "gauge": "H4 = I",
        "H0": H0.tolist(),
        "H1": H1.tolist(),
        "C_predicted_0_from_1": C_pred.tolist(),
        "C_direct_0_from_1": C_0_1.tolist(),
        "prediction_error_mean_px": mean_px,
        "prediction_error_median_px": float(np.median(mag)),
        "prediction_error_p95_px": float(np.percentile(mag, 95)),
        "status": status,
        "note": (
            "A single scene-wide constant correction cannot satisfy all three "
            "pairwise local alignments simultaneously when status is INCONSISTENT; "
            "this does NOT ascribe absolute geolocation error to any scene."
        ),
    }


# ---------------------------------------------------------------------------
# Task 8 — within-overlap displacement variation from metadata-only tiles
# ---------------------------------------------------------------------------


def within_overlap_spatial_variation(tiles: list[dict]) -> dict:
    """Compare a constant vs planar displacement model over accepted tiles."""
    acc = [t for t in tiles if t.get("accepted")]
    if len(acc) < 4:
        return {
            "n_tiles": len(acc),
            "classification": "INSUFFICIENT_TILES",
            "gates": {"improvement_threshold": 0.3, "field_range_threshold_px": 5.0},
        }
    fx = np.array([t["tile_col"] for t in acc], dtype=float)
    fy = np.array([t["tile_row"] for t in acc], dtype=float)
    dx = np.array([t["phase_dx_px"] for t in acc])
    dy = np.array([t["phase_dy_px"] for t in acc])
    X = np.column_stack([np.ones_like(fx), fx, fy])

    def _rmse(z, coef):
        pred = X @ coef
        return float(np.sqrt(np.mean((z - pred) ** 2)))

    c_const_dx = np.array([np.mean(dx), 0.0, 0.0])
    c_const_dy = np.array([np.mean(dy), 0.0, 0.0])
    c_sp_dx, *_ = np.linalg.lstsq(X, dx, rcond=None)
    c_sp_dy, *_ = np.linalg.lstsq(X, dy, rcond=None)
    rmse_const = (np.hypot(_rmse(dx, c_const_dx), _rmse(dy, c_const_dy)))
    rmse_spatial = np.hypot(_rmse(dx, c_sp_dx), _rmse(dy, c_sp_dy))
    improvement = 1.0 - rmse_spatial / rmse_const if rmse_const > 1e-9 else 0.0

    # predicted field range across the tile span
    x0, x1 = fx.min(), fx.max()
    y0, y1 = fy.min(), fy.max()
    corners = np.array([[x0, y0], [x1, y0], [x0, y1], [x1, y1]])
    Xc = np.column_stack([np.ones(len(corners)), corners])
    pdx = Xc @ c_sp_dx
    pdy = Xc @ c_sp_dy
    field_range = float(np.max(np.hypot(pdx, pdy)) - np.min(np.hypot(pdx, pdy)))

    improvement_th, range_th = 0.30, 5.0
    if improvement >= improvement_th and field_range >= range_th:
        classification = "SPATIALLY_VARYING"
    elif improvement >= improvement_th and improvement < 0.20:
        classification = "AMBIGUOUS"
    elif improvement < 0.20 and field_range < 2.0:
        classification = "CONSTANT_LIKE"
    else:
        classification = "AMBIGUOUS"

    return {
        "n_tiles": len(acc),
        "rmse_constant_px": rmse_const,
        "rmse_spatial_px": rmse_spatial,
        "improvement_ratio": float(improvement),
        "field_range_px": field_range,
        "gradient_xy_dx": [float(c_sp_dx[1]), float(c_sp_dx[2])],
        "gradient_xy_dy": [float(c_sp_dy[1]), float(c_sp_dy[2])],
        "classification": classification,
        "gates": {"improvement_threshold": improvement_th,
                  "field_range_threshold_px": range_th},
    }


# ---------------------------------------------------------------------------
# Task 10 — final classifier
# ---------------------------------------------------------------------------


def classify_geolocation_inconsistency(evidence: dict) -> str:
    """Pick the state best supported by the collected evidence."""
    closure_mean = evidence.get("triangle_closure_mean_px")
    if closure_mean is not None and closure_mean <= 2.0:
        return "PAIRWISE_RELATIONS_GLOBALLY_CONSISTENT"
    if evidence.get("direct_edge_phase_conflict_with_metadata", False):
        return "PAIRWISE_DIRECT_GEOMETRY_SUSPECT"
    if evidence.get("spatial_variation_supported", False):
        return "WITHIN_OVERLAP_SPATIAL_VARIATION_SUPPORTED"
    scene_model_status = evidence.get("scene_level_constant_model")
    if scene_model_status == "INCONSISTENT_WITH_SCENE_LEVEL_CONSTANT_CORRECTIONS":
        return (
            "SCENE_LEVEL_CONSTANT_CORRECTION_INSUFFICIENT"
            if (closure_mean or 0.0) > 10.0
            else "LOCAL_RELATIONS_CONFLICT_BUT_SOURCE_UNDERDETERMINED"
        )
    if evidence.get("within_overlap_displacement_constant", False) and \
            (closure_mean or 0.0) > 10.0:
        return "LOCAL_RELATIONS_CONFLICT_BUT_SOURCE_UNDERDETERMINED"
    return "INSUFFICIENT_EVIDENCE"


# ---------------------------------------------------------------------------
# Task 9 — figures
# ---------------------------------------------------------------------------


def _matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_metadata_only_shift_vectors(
    edge_tables: dict[str, list[dict]],
    out_path: str | Path,
    arrow_scale: float = 4.0,
) -> Path:
    """One panel per edge: tile centre + metadata-only phase-shift arrow."""
    plt = _matplotlib()
    names = list(edge_tables.keys())
    fig, axes = plt.subplots(1, len(names), figsize=(6 * len(names), 5))
    if len(names) == 1:
        axes = [axes]
    for ax, name in zip(axes, names):
        tiles = edge_tables[name]
        ax.set_title(f"{name}: metadata-only shift")
        ax.set_aspect("equal")
        for t in tiles:
            if not t.get("accepted"):
                continue
            c = t["tile_col"], t["tile_row"]
            dx, dy = t["phase_dx_px"], t["phase_dy_px"]
            ax.plot(c[0], c[1], "ko", ms=3)
            ax.arrow(c[0], c[1], dx * arrow_scale, dy * arrow_scale,
                     color="red", width=0.02, head_width=0.25,
                     length_includes_head=True)
        ax.text(0.02, 0.98, f"Visual arrow scale = {arrow_scale:g}x",
                transform=ax.transAxes, va="top", color="black", fontsize=9,
                bbox=dict(facecolor="white", alpha=0.7))
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_metadata_vs_direct_corrections(
    edge_tiles: dict[str, list[dict]],
    edge_models: dict[str, np.ndarray],
    edge_scenes: dict[str, tuple[np.ndarray, np.ndarray]],
    out_path: str | Path,
    arrow_scale: float = 2.0,
) -> Path:
    """Per edge: phase-measured (red) vs direct-affine-predicted (blue) arrows.

    ``edge_models[edge]`` is ``T_direct_i_from_j`` (native j -> i px);
    ``edge_scenes[edge]`` is ``(A_i, A_j)`` native->world matrices.
    """
    plt = _matplotlib()
    names = list(edge_tiles.keys())
    fig, axes = plt.subplots(1, len(names), figsize=(6 * len(names), 5))
    if len(names) == 1:
        axes = [axes]
    for ax, name in zip(axes, names):
        ax.set_title(f"{name}: phase (red) vs direct (blue)")
        ax.set_aspect("equal")
        T_native = edge_models.get(name)
        A_i, A_j = edge_scenes.get(name, (np.eye(3), np.eye(3)))
        inv_A_i = np.linalg.inv(A_i)
        inv_A_j = np.linalg.inv(A_j)
        for t in edge_tiles.get(name, []):
            if not t.get("accepted"):
                continue
            c = (t["tile_col"], t["tile_row"])
            world = np.array([t["center_world_x"], t["center_world_y"], 1.0])
            i_center = (inv_A_i @ world)[:2]
            j_center = (inv_A_j @ world)[:2]
            if T_native is not None:
                pred = (T_native @ np.append(j_center, 1.0))[:2]
                dx_p = pred[0] - i_center[0]
                dy_p = pred[1] - i_center[1]
            else:
                dx_p = dy_p = 0.0
            ax.plot(c[0], c[1], "ko", ms=3)
            ax.arrow(c[0], c[1], t["phase_dx_px"] * arrow_scale,
                     t["phase_dy_px"] * arrow_scale, color="red", width=0.02,
                     head_width=0.25, length_includes_head=True)
            ax.arrow(c[0], c[1], dx_p * arrow_scale, dy_p * arrow_scale,
                     color="blue", width=0.02, head_width=0.25,
                     length_includes_head=True, linestyle="--")
        ax.text(0.02, 0.98,
                f"arrow scale = {arrow_scale:g}x\nred=metadata phase | "
                "blue=direct-predicted", transform=ax.transAxes, va="top",
                fontsize=8, bbox=dict(facecolor="white", alpha=0.7))
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_triangle_correction_map(
    edge_summaries: dict[str, dict],
    out_path: str | Path,
    arrow_scale: float = 20.0,
) -> Path:
    """Geographic map of the three overlap corrections (pairwise only)."""
    plt = _matplotlib()
    fig, ax = plt.subplots(figsize=(8, 7))
    for name, summary in edge_summaries.items():
        center = summary["overlap_center_world"]
        dx_m, dy_m = summary["mean_correction_m"]
        ax.plot(center[0], center[1], "ko", ms=5)
        ax.annotate("", xy=(center[0] + dx_m * arrow_scale,
                            center[1] + dy_m * arrow_scale),
                    xytext=(center[0], center[1]),
                    arrowprops=dict(arrowstyle="->", color="red", lw=1.6))
        ax.text(center[0], center[1], f" {name}\n"
                f"({dx_m:+.0f}, {dy_m:+.0f}) m  std={summary['std_mag_m']:.0f} m",
                fontsize=8, va="bottom")
    ax.set_aspect("equal")
    ax.set_xlabel("easting (m)")
    ax.set_ylabel("northing (m)")
    ax.set_title("Pairwise metadata->direct world corrections "
                 "(NOT absolute scene errors)")
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Task 11 — summary
# ---------------------------------------------------------------------------


def build_geolocation_summary(
    *,
    transforms: dict,
    correction_fields: dict,
    triangle_closure: dict,
    scene_model: dict,
    spatial_variation: dict,
    metadata_shifts: dict,
    final_state: str,
    output_dir: str,
) -> dict:
    """Assemble the final diagnosis summary answering the 8 questions."""
    q1 = transforms.get("edge_0_1", {}).get("translation_equiv_px14")
    q2 = transforms.get("edge_0_4", {}).get("translation_equiv_px14")
    q3 = transforms.get("edge_1_4", {}).get("translation_equiv_px14")

    field_classes = {
        k: v.get("classification", "?") for k, v in correction_fields.items()
    }
    spatial = {
        k: v.get("classification", "?") for k, v in spatial_variation.items()
    }
    summary = {
        "final_state": final_state,
        "metadata_to_direct_world_correction_px14": {
            "0_1": q1, "0_4": q2, "1_4": q3,
        },
        "correction_field_classifications": field_classes,
        "within_overlap_spatial_variation": spatial,
        "metadata_only_phase_fields": {
            k: {"accepted_tiles": v.get("n_tiles_accepted"),
                "mean_px": [v.get("mean_dx_px"), v.get("mean_dy_px")]
                if v.get("mean_dx_px") is not None else None}
            for k, v in metadata_shifts.items()
        },
        "triangle_closure": triangle_closure,
        "scene_level_constant_model": scene_model,
        "answers": {
            "1_correction_0_1_px14": q1,
            "2_correction_0_4_px14": q2,
            "3_correction_1_4_px14": q3,
            "4_correction_constant_within_overlap": field_classes,
            "5_metadata_only_phase_supports_direct": (
                _phase_supports_direct(metadata_shifts)
            ),
            "6_world_correction_triangle_closure_px": (
                triangle_closure.get("sample_mean_px")
            ),
            "7_scene_level_constant_model_status": scene_model.get("status"),
            "8_evidence_quality": final_state,
        },
        "can_conclude": [
            "Three locally valid pairwise alignments exist (all ~1 px on their "
            "own overlaps).",
            "metadata->direct world corrections and their consistency were "
            "quantified per overlap.",
            "closure of the world-correction triangle was measured on real "
            "sample points.",
        ],
        "cannot_conclude": [
            "No scene is identified as absolutely mis-located (no external "
            "ground truth).",
            "No absolute DZ01V geolocation-error magnitude is claimed.",
            "No algorithm choice is forced by this round alone.",
        ],
        "diagnostic_artifacts": {
            "baseline": f"{output_dir}/00_baseline.json",
            "transforms": f"{output_dir}/01_metadata_vs_direct_transforms.json",
            "field_samples": f"{output_dir}/02_correction_field_samples.csv",
            "field_summary": f"{output_dir}/03_correction_field_summary.json",
            "metadata_tiles": f"{output_dir}/04_metadata_only_tile_shifts.csv",
            "metadata_shift_summary": f"{output_dir}/05_metadata_only_shift_summary.json",
            "triangle_closure": f"{output_dir}/06_world_correction_triangle_closure.json",
            "triangle_samples": f"{output_dir}/06_world_correction_triangle_samples.csv",
            "scene_model": f"{output_dir}/07_scene_level_constant_correction_test.json",
            "spatial_variation": f"{output_dir}/08_within_overlap_spatial_variation.json",
            "shift_vectors": f"{output_dir}/09_metadata_only_shift_vectors.png",
            "vs_direct": f"{output_dir}/10_metadata_vs_direct_corrections.png",
            "triangle_map": f"{output_dir}/11_triangle_correction_map.png",
        },
    }
    return summary


def _phase_supports_direct(metadata_shifts: dict) -> bool | None:
    """True if every measured edge's metadata-only phase is (near) zero.

    A near-zero metadata-only shift means the raw geotransforms already align
    the overlap content — i.e. the direct correction is unneeded/small.  If
    the field is large, the metadata does not support the direct affine there.
    """
    values = []
    for summary in metadata_shifts.values():
        if summary.get("mean_dx_px") is None:
            continue
        values.append(math.hypot(
            summary["mean_dx_px"], summary["mean_dy_px"]))
    if not values:
        return None
    return float(np.mean(values)) <= 2.0


def write_geolocation_summary_text(summary: dict, path: str | Path) -> Path:
    """Human-readable rendering of the geolocation summary."""
    lines = []
    lines.append("=== Geolocation consistency summary ===")
    lines.append(f"Final state: {summary['final_state']}")
    lines.append("Answers:")
    for key, value in summary["answers"].items():
        lines.append(f"  {key}: {value}")
    lines.append("")
    lines.append("CAN conclude:")
    for item in summary["can_conclude"]:
        lines.append(f"  - {item}")
    lines.append("CANNOT conclude:")
    for item in summary["cannot_conclude"]:
        lines.append(f"  - {item}")
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out