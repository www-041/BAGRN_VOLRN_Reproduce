"""Diagnostic-only audit of the existing phase-validation chain.

This module never changes registration, phase-helper, radiometric, or mosaic
semantics.  It consumes saved two-edge artifacts and provides independent
counterfactual measurements for the exact validation inputs.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


AUDIT_EDGES = {
    (0, 6): "WORKING_CONTROL",
    (2, 5): "SUSPECT_PHASE_FALSE_ALARM",
}


def _edge_key(edge: str | tuple[int, int]) -> str:
    if isinstance(edge, str):
        i, j = (int(part) for part in edge.split("-"))
    else:
        i, j = int(edge[0]), int(edge[1])
    return f"{i}-{j}"


def _read_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _edge_rows(payload: Any) -> dict[str, dict]:
    if isinstance(payload, dict) and isinstance(payload.get("edges"), list):
        payload = payload["edges"]
    if isinstance(payload, dict):
        return {_edge_key(key): value for key, value in payload.items() if "-" in str(key)}
    if not isinstance(payload, list):
        return {}
    rows = {}
    for row in payload:
        edge = row.get("edge", row.get("edge_key"))
        if edge is not None:
            rows[_edge_key(edge)] = row
    return rows


def _load_tiles(path: Path) -> dict[str, list[dict]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    out: dict[str, list[dict]] = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = _edge_key(row["edge"])
            row["grid_n"] = int(row["grid_n"])
            row["tile_row"] = int(row["tile_row"])
            row["tile_col"] = int(row["tile_col"])
            row["phase_mag_px"] = float(row["phase_mag_px"])
            row["accepted"] = str(row.get("accepted", "")).strip().lower() in {"1", "true", "yes"}
            out.setdefault(key, []).append(row)
    return out


def load_phase_audit_baseline(
    edge_reliability_dir: str | Path,
    local_affine_dir: str | Path,
    selected_pair_dir: str | Path | None = None,
) -> dict:
    """Load the frozen two-edge phase evidence without rerunning registration."""
    edge_root = Path(edge_reliability_dir)
    local_root = Path(local_affine_dir)
    selected_root = Path(selected_pair_dir) if selected_pair_dir is not None else edge_root.parent / "nine_scene_selected_pair_consistency"
    required = (
        edge_root / "00_edge_reliability_baseline.json",
        edge_root / "07_direct_residual_summary.json",
        edge_root / "07_direct_residual_tiles.csv",
        local_root / "00_local_affine_baseline.json",
        local_root / "06_global_vs_local_phase_summary.json",
        selected_root / "06_canonical_world_edge_transforms.json",
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f"required phase audit artifact missing: {path}")

    reliability = _edge_rows(_read_json(required[0]))
    direct_summary = _edge_rows(_read_json(required[1]))
    local_summary = _edge_rows(_read_json(required[4]))
    canonical = _read_json(required[5]).get("edges", {})
    tiles = _load_tiles(required[2])
    edges: dict[str, dict] = {}
    for edge, role in AUDIT_EDGES.items():
        key = _edge_key(edge)
        if key not in reliability or key not in direct_summary:
            raise ValueError(f"missing baseline edge {key}")
        matrix_row = canonical.get(key, canonical.get(f"{edge[1]}-{edge[0]}"))
        if matrix_row is None or matrix_row.get("matrix") is None:
            raise ValueError(f"missing canonical world matrix for {key}")
        edges[key] = {
            "edge": [edge[0], edge[1]],
            "role": role,
            "reliability": reliability[key],
            "local_affine_phase": local_summary.get(key),
            "old_phase_median_px": float(direct_summary[key]["median_px"]),
            "old_phase_summary": direct_summary[key],
            "direct_tiles": tiles.get(key, []),
            "world_matrix": np.asarray(matrix_row["matrix"], dtype=float).tolist(),
            "pixel_matrix": None,
        }

    selected_json = selected_root / "02_new_pairwise_registration_results.json"
    if selected_json.is_file():
        for row in _read_json(selected_json).get("results", []):
            key = _edge_key((int(row["idx_i"]), int(row["idx_j"])))
            if key in edges:
                edges[key]["pixel_matrix"] = row.get("pixel_matrix")

    return {"edge_keys": [_edge_key(edge) for edge in AUDIT_EDGES], "edges": edges}


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.integer):
        return int(value)
    return value


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(value), indent=2, ensure_ascii=False), encoding="utf-8")


def sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _phase_arrays(inputs: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ref = np.asarray(inputs["ref_crop"], dtype=np.float64)
    tgt = np.asarray(inputs["warped_target_crop"], dtype=np.float64)
    mask = np.asarray(inputs.get("joint_valid_mask"), dtype=bool)
    if ref.shape != tgt.shape or ref.shape != mask.shape:
        raise ValueError("phase inputs must have identical ref/tgt/mask shapes")
    mask = mask & np.isfinite(ref) & np.isfinite(tgt)
    return ref, tgt, mask


def phase_from_inputs(inputs: dict) -> dict:
    """Run the existing phase helper on frozen arrays using its current fill policy."""
    from src.multiscene_sift.loop_diagnostics import _phase_cross_correlation_shift

    ref, tgt, mask = _phase_arrays(inputs)
    if int(mask.sum()) < 20:
        return {"status": "INSUFFICIENT_VALID_AREA", "dx_px": None, "dy_px": None,
                "magnitude_px": None, "response": None}
    ref_mean = float(np.mean(ref[mask]))
    tgt_mean = float(np.mean(tgt[mask]))
    ref_filled = np.where(mask, ref, ref_mean)
    tgt_filled = np.where(mask, tgt, tgt_mean)
    try:
        dx, dy, response = _phase_cross_correlation_shift(ref_filled, tgt_filled, upsample=5)
    except Exception as exc:  # noqa: BLE001 - diagnostic record, not a production gate
        return {"status": f"ERROR:{type(exc).__name__}", "dx_px": None, "dy_px": None,
                "magnitude_px": None, "response": None}
    if not np.isfinite(dx) or not np.isfinite(dy):
        return {"status": "PHASE_FAILED", "dx_px": None, "dy_px": None,
                "magnitude_px": None, "response": float(response) if np.isfinite(response) else None}
    return {"status": "OK", "dx_px": float(dx), "dy_px": float(dy),
            "magnitude_px": float(np.hypot(dx, dy)),
            "response": float(response) if np.isfinite(response) else None,
            "semantics": "shift required to move moving/target toward reference"}


def _array_stats(array: np.ndarray, mask: np.ndarray | None = None) -> dict:
    arr = np.asarray(array)
    finite = np.isfinite(arr)
    if mask is not None:
        finite &= np.asarray(mask, dtype=bool)
    values = arr[finite]
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "min": float(np.min(values)) if values.size else None,
        "max": float(np.max(values)) if values.size else None,
        "mean": float(np.mean(values)) if values.size else None,
        "std": float(np.std(values)) if values.size else None,
    }


def trace_phase_validation_flow(inputs: dict, phase: dict | None = None) -> dict:
    """Trace the exact frozen-array path from crop inputs to phase output."""
    ref = np.asarray(inputs["ref_crop"])
    tgt = np.asarray(inputs["warped_target_crop"])
    ref_mask = np.asarray(inputs.get("ref_valid_mask", inputs["joint_valid_mask"]), dtype=bool)
    tgt_mask = np.asarray(inputs.get("target_valid_mask", inputs["joint_valid_mask"]), dtype=bool)
    joint = np.asarray(inputs["joint_valid_mask"], dtype=bool) & np.isfinite(ref) & np.isfinite(tgt)
    meta = dict(inputs.get("metadata", {}))
    phase = phase or phase_from_inputs(inputs)
    return {
        "scene_ref": meta.get("scene_ref"),
        "scene_tgt": meta.get("scene_tgt"),
        "matrix_direction": meta.get("matrix_direction", "target -> reference"),
        "matrix_values": meta.get("matrix_values"),
        "ref_tgt_order": meta.get("ref_tgt_order", "reference, target"),
        "source_crs": meta.get("source_crs"),
        "source_transforms": meta.get("source_transforms"),
        "warp_output": meta.get("warp_output", {}),
        "crop_origin": {"row": meta.get("crop_origin_row"), "col": meta.get("crop_origin_col")},
        "crop_world_bounds": meta.get("crop_world_bounds"),
        "pixel_center_semantics": meta.get("pixel_center_semantics", "pixel centers"),
        "coordinate_invariance": meta.get("coordinate_invariance", {"status": "NOT_PROVIDED"}),
        "ref_array": _array_stats(ref),
        "target_array": _array_stats(tgt),
        "ref_valid": {"count": int(ref_mask.sum()), "fraction": float(ref_mask.mean())},
        "target_valid": {"count": int(tgt_mask.sum()), "fraction": float(tgt_mask.mean())},
        "joint_valid": {"count": int(joint.sum()), "fraction": float(joint.mean())},
        "phase_input_shape": list(ref.shape),
        "phase": phase,
    }


def coordinate_invariance_check(
    grid_transform,
    ref_transform,
    tgt_transform,
    bounds_px: tuple[int, int, int, int],
) -> dict:
    """Check crop pixel -> world -> source pixel -> world round trips."""
    x0, y0, x1, y1 = (float(v) for v in bounds_px)
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    world = grid_transform * (cx, cy)
    ref_px = ~ref_transform * world
    tgt_px = ~tgt_transform * world
    ref_world = ref_transform * ref_px
    tgt_world = tgt_transform * tgt_px
    return {
        "status": "OK",
        "crop_pixel": [cx, cy],
        "world": [float(world[0]), float(world[1])],
        "ref_source_pixel": [float(ref_px[0]), float(ref_px[1])],
        "tgt_source_pixel": [float(tgt_px[0]), float(tgt_px[1])],
        "ref_world_roundtrip_error": float(np.hypot(ref_world[0] - world[0], ref_world[1] - world[1])),
        "tgt_world_roundtrip_error": float(np.hypot(tgt_world[0] - world[0], tgt_world[1] - world[1])),
    }


def build_exact_phase_inputs(
    scene_ref,
    scene_tgt,
    world_matrix: np.ndarray,
    direct_tiles: list[dict],
    edge_key: str,
    band: str = "B14",
    max_side: int = 4096,
) -> dict:
    """Rebuild the exact accepted direct-warp tile used by the old residual field."""
    from src.multiscene_sift.edge_reliability_diagnostics import build_direct_warp_overlap
    from src.multiscene_sift.dense_phase_diagnostics import make_tile_windows
    from src.multiscene_sift.frame_diagnostics import affine_to_matrix, matrix_to_affine

    overlap = build_direct_warp_overlap(
        scene_ref, scene_tgt, band, np.asarray(world_matrix, dtype=float), max_side=max_side
    )
    accepted = [row for row in direct_tiles if row.get("accepted") and np.isfinite(row.get("phase_mag_px", np.nan))]
    if not accepted:
        raise ValueError(f"no accepted direct residual tile for {edge_key}")
    old_median = float(np.median([row["phase_mag_px"] for row in accepted]))
    chosen = min(accepted, key=lambda row: abs(row["phase_mag_px"] - old_median))
    windows = make_tile_windows(overlap["height"], overlap["width"], int(chosen["grid_n"]), 128)
    window = next(
        item for item in windows
        if item["row0"] <= chosen["tile_row"] * overlap["height"] / int(chosen["grid_n"]) + 1
        and item["tile_row"] == chosen["tile_row"]
        and item["tile_col"] == chosen["tile_col"]
    )
    r0, r1, c0, c1 = window["row0"], window["row1"], window["col0"], window["col1"]
    ref = overlap["image0"][r0:r1, c0:c1].copy()
    tgt = overlap["image1"][r0:r1, c0:c1].copy()
    ref_valid = overlap["mask0"][r0:r1, c0:c1].copy()
    tgt_valid = overlap["mask1"][r0:r1, c0:c1].copy()
    joint = ref_valid & tgt_valid & np.isfinite(ref) & np.isfinite(tgt)
    grid_transform = overlap["transform"]
    ref_transform = scene_ref.transforms[band]
    tgt_corrected = matrix_to_affine(np.asarray(world_matrix) @ affine_to_matrix(scene_tgt.transforms[band]))
    from rasterio.transform import Affine
    transform = grid_transform * Affine.translation(c0, r0)
    west, south = grid_transform * (c0, r1)
    east, north = grid_transform * (c1, r0)
    crop_bounds_world = (
        float(west), float(south), float(east), float(north),
    )
    metadata = {
        "edge": edge_key,
        "scene_ref": scene_ref.name,
        "scene_tgt": scene_tgt.name,
        "matrix_direction": "target world -> reference world",
        "matrix_values": np.asarray(world_matrix, dtype=float).tolist(),
        "ref_tgt_order": "reference, target",
        "source_crs": str(scene_ref.crs),
        "source_transforms": {"ref": list(ref_transform), "tgt_corrected": list(tgt_corrected)},
        "warp_output": {"shape": [overlap["height"], overlap["width"]], "resolution": overlap["resolution"],
                         "bounds": {"left": float(overlap["transform"].c), "top": float(overlap["transform"].f)}},
        "crop_origin_row": r0,
        "crop_origin_col": c0,
        "crop_world_bounds": crop_bounds_world,
        "pixel_center_semantics": "pixel centers; x=column, y=row",
        "selected_tile": {key: chosen[key] for key in ("grid_n", "tile_row", "tile_col", "phase_mag_px")},
        "coordinate_invariance": coordinate_invariance_check(transform, ref_transform, tgt_corrected, (0, 0, c1 - c0, r1 - r0)),
    }
    return {
        "ref_crop": ref,
        "warped_target_crop": tgt,
        "ref_valid_mask": ref_valid,
        "target_valid_mask": tgt_valid,
        "joint_valid_mask": joint,
        "metadata": metadata,
    }
