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
