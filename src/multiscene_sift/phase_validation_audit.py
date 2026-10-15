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
    selected_json = selected_root / "02_new_pairwise_registration_results.json"
    selected_rows = {}
    if selected_json.is_file():
        for row in _read_json(selected_json).get("results", []):
            key = _edge_key((int(row["idx_i"]), int(row["idx_j"])))
            selected_rows[key] = row
    tiles = _load_tiles(required[2])
    edges: dict[str, dict] = {}
    for edge, role in AUDIT_EDGES.items():
        key = _edge_key(edge)
        if key not in reliability or key not in direct_summary:
            raise ValueError(f"missing baseline edge {key}")
        matrix_row = canonical.get(key, canonical.get(f"{edge[1]}-{edge[0]}"))
        selected_row = selected_rows.get(key, selected_rows.get(f"{edge[1]}-{edge[0]}"))
        if (matrix_row is None or matrix_row.get("matrix") is None) and not selected_row:
            raise ValueError(f"missing canonical world matrix and pixel matrix for {key}")
        edges[key] = {
            "edge": [edge[0], edge[1]],
            "role": role,
            "reliability": reliability[key],
            "local_affine_phase": local_summary.get(key),
            "old_phase_median_px": float(direct_summary[key]["median_px"]),
            "old_phase_summary": direct_summary[key],
            "direct_tiles": tiles.get(key, []),
            "world_matrix": (
                np.asarray(matrix_row["matrix"], dtype=float).tolist()
                if matrix_row is not None and matrix_row.get("matrix") is not None else None
            ),
            "pixel_matrix": selected_row.get("pixel_matrix") if selected_row else None,
        }

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


def export_phase_inputs(inputs: dict, output_dir: str | Path, edge_key: str) -> Path:
    """Persist exact arrays and a machine-readable checksum manifest."""
    out = Path(output_dir)
    input_dir = out / "02_phase_inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    npz_path = input_dir / f"{edge_key.replace('-', '_')}_inputs.npz"
    np.savez_compressed(
        npz_path,
        ref_crop=np.asarray(inputs["ref_crop"]),
        warped_target_crop=np.asarray(inputs["warped_target_crop"]),
        ref_valid_mask=np.asarray(inputs["ref_valid_mask"], dtype=bool),
        target_valid_mask=np.asarray(inputs["target_valid_mask"], dtype=bool),
        joint_valid_mask=np.asarray(inputs["joint_valid_mask"], dtype=bool),
    )
    ref = np.asarray(inputs["ref_crop"])
    joint = np.asarray(inputs["joint_valid_mask"], dtype=bool)
    manifest_path = out / "02_phase_input_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    manifest[edge_key] = {
        "shape": list(ref.shape),
        "dtype": str(ref.dtype),
        "min": float(np.nanmin(ref)) if np.isfinite(ref).any() else None,
        "max": float(np.nanmax(ref)) if np.isfinite(ref).any() else None,
        "mean": float(np.nanmean(ref)) if np.isfinite(ref).any() else None,
        "std": float(np.nanstd(ref)) if np.isfinite(ref).any() else None,
        "joint_valid_fraction": float(joint.mean()),
        "sha256": sha256_array(ref),
        "npz": str(npz_path.name),
        "metadata": _json_safe(inputs.get("metadata", {})),
    }
    write_json(manifest_path, manifest)
    return npz_path


def reload_phase_inputs(output_dir: str | Path, edge_key: str) -> dict:
    input_dir = Path(output_dir) / "02_phase_inputs"
    path = input_dir / f"{edge_key.replace('-', '_')}_inputs.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        result = {key: data[key] for key in data.files}
    manifest_path = Path(output_dir) / "02_phase_input_manifest.json"
    if manifest_path.is_file():
        result["metadata"] = json.loads(manifest_path.read_text(encoding="utf-8")).get(edge_key, {}).get("metadata", {})
    return result


def _raw_ncc(ref: np.ndarray, tgt: np.ndarray, mask: np.ndarray) -> float:
    valid = np.asarray(mask, dtype=bool) & np.isfinite(ref) & np.isfinite(tgt)
    if int(valid.sum()) < 2:
        return float("nan")
    a, b = np.asarray(ref, dtype=float)[valid], np.asarray(tgt, dtype=float)[valid]
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.sqrt(np.sum(a * a) * np.sum(b * b)))
    return float(np.sum(a * b) / denom) if denom > 1e-12 else float("nan")


def phase_input_visual_stats(inputs: dict) -> dict:
    from src.registration_benchmark.metrics import gradient_ncc

    ref, tgt, joint = _phase_arrays(inputs)
    valid = joint & np.isfinite(ref) & np.isfinite(tgt)
    return {
        "raw_ncc": _raw_ncc(ref, tgt, valid),
        "gradient_ncc": float(gradient_ncc(ref, tgt, valid)),
        "masked_mae": float(np.mean(np.abs(ref[valid] - tgt[valid]))) if valid.any() else None,
        "valid_fraction": float(valid.mean()),
    }


def _same_stretch(images: list[np.ndarray], mask: np.ndarray) -> tuple[float, float]:
    values = np.concatenate([np.asarray(image)[mask & np.isfinite(image)] for image in images])
    if values.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(values, [2, 98])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.min(values)), float(np.max(values))
    return float(lo), float(hi if hi > lo else lo + 1.0)


def _gradient_magnitude(image: np.ndarray) -> np.ndarray:
    gy, gx = np.gradient(np.nan_to_num(np.asarray(image, dtype=float), nan=0.0))
    return np.hypot(gx, gy)


def _checkerboard(ref: np.ndarray, tgt: np.ndarray, tile: int = 32) -> np.ndarray:
    yy, xx = np.indices(ref.shape)
    return np.where(((yy // tile + xx // tile) % 2) == 0, ref, tgt)


def write_phase_input_visual_audit(
    inputs_by_edge: dict[str, dict], output_path: str | Path, stats_path: str | Path
) -> Path:
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    stats = {edge: phase_input_visual_stats(inputs) for edge, inputs in inputs_by_edge.items()}
    write_json(stats_path, stats)
    edges = tuple(inputs_by_edge)
    fig, axes = plt.subplots(len(edges), 6, figsize=(18, 5 * len(edges)), squeeze=False, constrained_layout=True)
    titles = ("reference", "global-affine target", "joint mask", "checkerboard", "gradient edge overlay", "absolute difference")
    for row, edge in enumerate(edges):
        inputs = inputs_by_edge[edge]
        ref, tgt, joint = _phase_arrays(inputs)
        lo, hi = _same_stretch([ref, tgt], joint)
        grad_ref, grad_tgt = _gradient_magnitude(ref), _gradient_magnitude(tgt)
        edge_overlay = np.zeros((*ref.shape, 3), dtype=np.float32)
        edge_overlay[..., 0] = grad_ref / max(float(np.nanpercentile(grad_ref, 99)), 1e-9)
        edge_overlay[..., 1] = grad_tgt / max(float(np.nanpercentile(grad_tgt, 99)), 1e-9)
        diff = np.abs(ref - tgt)
        panels = (ref, tgt, joint.astype(float), _checkerboard(ref, tgt), np.clip(edge_overlay, 0, 1), diff)
        for ax, image, title in zip(axes[row], panels, titles):
            cmap = "gray" if title != "gradient edge overlay" else None
            ax.imshow(image, cmap=cmap, vmin=None if title in ("joint mask", "gradient edge overlay") else lo,
                      vmax=None if title in ("joint mask", "gradient edge overlay") else hi)
            ax.set_title(f"{edge}: {title}")
            ax.set_axis_off()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def inject_translation_nonwrapping(
    image: np.ndarray, mask: np.ndarray, dx_px: int, dy_px: int
) -> tuple[np.ndarray, np.ndarray]:
    """Translate an image/mask into a fresh frame without circular wrapping."""
    dx, dy = int(dx_px), int(dy_px)
    if dx != dx_px or dy != dy_px:
        raise ValueError("non-wrapping injection accepts integer pixel shifts")
    source = np.asarray(image)
    source_mask = np.asarray(mask, dtype=bool)
    if source.shape != source_mask.shape:
        raise ValueError("image and mask must have the same shape")
    height, width = source.shape
    out = np.full(source.shape, np.nan, dtype=np.float64)
    out_mask = np.zeros(source.shape, dtype=bool)
    src_y0, src_y1 = max(0, -dy), min(height, height - dy)
    src_x0, src_x1 = max(0, -dx), min(width, width - dx)
    if src_y1 <= src_y0 or src_x1 <= src_x0:
        return out, out_mask
    dst_y0, dst_y1 = src_y0 + dy, src_y1 + dy
    dst_x0, dst_x1 = src_x0 + dx, src_x1 + dx
    out[dst_y0:dst_y1, dst_x0:dst_x1] = source[src_y0:src_y1, src_x0:src_x1]
    out_mask[dst_y0:dst_y1, dst_x0:dst_x1] = source_mask[src_y0:src_y1, src_x0:src_x1]
    out[~out_mask] = np.nan
    return out, out_mask


def run_phase_sign_convention(
    reference: np.ndarray,
    reference_mask: np.ndarray,
    moving: np.ndarray,
    moving_mask: np.ndarray,
    injected_dx: int,
    injected_dy: int,
) -> dict:
    inputs = {
        "ref_crop": reference,
        "warped_target_crop": moving,
        "ref_valid_mask": reference_mask,
        "target_valid_mask": moving_mask,
        "joint_valid_mask": np.asarray(reference_mask, dtype=bool) & np.asarray(moving_mask, dtype=bool),
    }
    phase = phase_from_inputs(inputs)
    expected_dx, expected_dy = -float(injected_dx), -float(injected_dy)
    if phase["dx_px"] is None:
        error = float("inf")
    else:
        error = float(np.hypot(phase["dx_px"] - expected_dx, phase["dy_px"] - expected_dy))
    return {
        "injected_dx": int(injected_dx), "injected_dy": int(injected_dy),
        "expected_phase_dx": expected_dx, "expected_phase_dy": expected_dy,
        "recovered_dx": phase["dx_px"], "recovered_dy": phase["dy_px"],
        "error_mag_px": error, "phase": phase,
        "semantics": "phase (dx,dy) is the shift required to move moving/target toward reference",
    }


def evaluate_shift_counterfactual(
    reference: np.ndarray,
    moving: np.ndarray,
    moving_mask: np.ndarray,
    dx_px: int,
    dy_px: int,
) -> dict:
    """Apply a candidate correction and score it with independent metrics."""
    shifted, shifted_mask = inject_translation_nonwrapping(moving, moving_mask, dx_px, dy_px)
    ref_mask = np.isfinite(reference)
    joint = ref_mask & shifted_mask & np.isfinite(shifted)
    metrics = phase_input_visual_stats({
        "ref_crop": reference,
        "warped_target_crop": shifted,
        "ref_valid_mask": ref_mask,
        "target_valid_mask": shifted_mask,
        "joint_valid_mask": joint,
    })
    metrics.update({"applied_dx": int(dx_px), "applied_dy": int(dy_px),
                   "joint_valid_count": int(joint.sum())})
    return metrics


def translation_metric_surface(
    reference: np.ndarray,
    moving: np.ndarray,
    moving_mask: np.ndarray,
    radius_px: int = 12,
    phase_shift: tuple[float, float] | None = None,
) -> dict:
    """Search integer non-wrapping translations using independent metrics."""
    rows = []
    for dy in range(-int(radius_px), int(radius_px) + 1):
        for dx in range(-int(radius_px), int(radius_px) + 1):
            score = evaluate_shift_counterfactual(reference, moving, moving_mask, dx, dy)
            rows.append({"dx": dx, "dy": dy, "gradient_ncc": score["gradient_ncc"],
                         "raw_ncc": score["raw_ncc"], "masked_mae": score["masked_mae"],
                         "valid_count": score["joint_valid_count"]})
    finite = [row for row in rows if row["gradient_ncc"] is not None and np.isfinite(row["gradient_ncc"])]
    finite.sort(key=lambda row: row["gradient_ncc"], reverse=True)
    if not finite:
        return {"rows": rows, "best_dx": None, "best_dy": None, "best_score": None,
                "score_at_zero": None, "best_minus_second_best": None}
    best = finite[0]
    second = finite[1] if len(finite) > 1 else None
    zero = next(row for row in rows if row["dx"] == 0 and row["dy"] == 0)
    result = {
        "rows": rows,
        "best_dx": int(best["dx"]), "best_dy": int(best["dy"]),
        "best_score": float(best["gradient_ncc"]),
        "best_raw_ncc": best["raw_ncc"], "best_masked_mae": best["masked_mae"],
        "score_at_zero": zero["gradient_ncc"], "raw_ncc_at_zero": zero["raw_ncc"],
        "valid_count_at_best": int(best["valid_count"]),
        "valid_count_at_zero": int(zero["valid_count"]),
        "best_minus_second_best": float(best["gradient_ncc"] - second["gradient_ncc"]) if second else None,
    }
    if phase_shift is not None:
        phase_dx, phase_dy = (int(round(phase_shift[0])), int(round(phase_shift[1])))
        near = min(rows, key=lambda row: (row["dx"] - phase_dx) ** 2 + (row["dy"] - phase_dy) ** 2)
        result["phase_candidate"] = near
    return result


def mask_boundary_stress_test(
    reference: np.ndarray,
    moving: np.ndarray,
    reference_mask: np.ndarray,
    moving_mask: np.ndarray,
    min_valid_fraction: float = 0.30,
) -> dict:
    """Measure phase under fixed erosion and central-valid-mask variants."""
    from scipy.ndimage import binary_erosion

    joint = np.asarray(reference_mask, dtype=bool) & np.asarray(moving_mask, dtype=bool)
    variants = {"V0_original": joint}
    for radius, name in ((16, "V1_erode_16"), (32, "V2_erode_32"), (64, "V3_erode_64")):
        variants[name] = binary_erosion(joint, iterations=radius, border_value=0)
    central = np.zeros_like(joint)
    if joint.any():
        ys, xs = np.nonzero(joint)
        central[ys.min():ys.max() + 1, xs.min():xs.max() + 1] = True
        central &= joint
    variants["V4_central_valid_bbox"] = central
    out = {}
    for name, variant in variants.items():
        fraction = float(variant.mean())
        if fraction < min_valid_fraction:
            result = {"status": "INSUFFICIENT_VALID_AREA", "valid_fraction": fraction,
                      "dx_px": None, "dy_px": None, "magnitude_px": None, "response": None}
        else:
            result = phase_from_inputs({
                "ref_crop": reference, "warped_target_crop": moving,
                "ref_valid_mask": reference_mask, "target_valid_mask": moving_mask,
                "joint_valid_mask": variant,
            })
            result["valid_fraction"] = fraction
        out[name] = result
    return out


def _standardize_masked(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = np.asarray(image, dtype=np.float64).copy()
    values = out[mask & np.isfinite(out)]
    if values.size == 0:
        return np.zeros_like(out)
    std = float(np.std(values))
    out = (out - float(np.mean(values))) / max(std, 1e-12)
    out[~np.isfinite(out)] = 0.0
    return out


def _representation_images(image: np.ndarray, mask: np.ndarray) -> dict[str, np.ndarray]:
    from scipy.ndimage import sobel

    finite_image = np.nan_to_num(np.asarray(image, dtype=np.float64), nan=0.0)
    gy, gx = np.gradient(finite_image)
    gradient = np.hypot(gx, gy)
    sobel_gradient = np.hypot(sobel(finite_image, axis=1), sobel(finite_image, axis=0))
    threshold = float(np.percentile(gradient[mask], 75)) if mask.any() else 0.0
    return {
        "R0_raw": finite_image,
        "R1_standardized": _standardize_masked(finite_image, mask),
        "R2_gradient_magnitude": gradient,
        "R3_sobel_magnitude": sobel_gradient,
        "R4_edge_map": (gradient >= threshold).astype(np.float64),
    }


def representation_stability(
    reference: np.ndarray,
    moving: np.ndarray,
    joint_mask: np.ndarray,
    include_eroded: bool = True,
) -> list[dict]:
    from scipy.ndimage import binary_erosion

    masks = [("original", np.asarray(joint_mask, dtype=bool))]
    if include_eroded:
        masks.append(("erode_32", binary_erosion(np.asarray(joint_mask, dtype=bool), iterations=32)))
    rows = []
    for mask_name, mask in masks:
        ref_images = _representation_images(reference, mask)
        moving_images = _representation_images(moving, mask)
        for name in ref_images:
            valid_fraction = float(mask.mean())
            if valid_fraction < 0.30:
                phase = {"status": "INSUFFICIENT_VALID_AREA"}
            else:
                phase = phase_from_inputs({
                    "ref_crop": ref_images[name], "warped_target_crop": moving_images[name],
                    "ref_valid_mask": mask, "target_valid_mask": mask, "joint_valid_mask": mask,
                })
            rows.append({"representation": name, "mask_variant": mask_name,
                         "valid_fraction": valid_fraction, "phase": phase,
                         "raw_ncc": _raw_ncc(ref_images[name], moving_images[name], mask)})
    return rows


def real_crop_injection_recovery(inputs: dict, shifts: list[tuple[int, int]]) -> list[dict]:
    """Inject known translations into the frozen target crop and recover them."""
    base_phase = phase_from_inputs(inputs)
    reference = np.asarray(inputs["ref_crop"])
    moving = np.asarray(inputs["warped_target_crop"])
    target_mask = np.asarray(inputs["target_valid_mask"], dtype=bool)
    ref_mask = np.asarray(inputs["ref_valid_mask"], dtype=bool)
    rows = []
    for dx, dy in shifts:
        injected, injected_mask = inject_translation_nonwrapping(moving, target_mask, dx, dy)
        phase = phase_from_inputs({
            "ref_crop": reference, "warped_target_crop": injected,
            "ref_valid_mask": ref_mask, "target_valid_mask": injected_mask,
            "joint_valid_mask": ref_mask & injected_mask,
        })
        expected_dx = None if base_phase["dx_px"] is None else float(base_phase["dx_px"] - dx)
        expected_dy = None if base_phase["dy_px"] is None else float(base_phase["dy_px"] - dy)
        error = None if expected_dx is None or phase["dx_px"] is None else float(
            np.hypot(phase["dx_px"] - expected_dx, phase["dy_px"] - expected_dy)
        )
        rows.append({
            "injected_dx": int(dx), "injected_dy": int(dy),
            "expected_phase_dx": expected_dx, "expected_phase_dy": expected_dy,
            "recovered_dx": phase["dx_px"], "recovered_dy": phase["dy_px"],
            "error_mag_px": error, "response": phase.get("response"),
            "status": phase.get("status"),
            "joint_valid_fraction": float((ref_mask & injected_mask).mean()),
        })
    return rows


def phase_peak_diagnostics(existing_helper_result: dict) -> dict:
    """Summarize an exposed correlation surface, without implementing FFT again."""
    surface = existing_helper_result.get("correlation_surface")
    if surface is None:
        return {"status": "NOT_AVAILABLE_IN_CURRENT_HELPER"}
    values = np.asarray(surface, dtype=float)
    if values.ndim != 2 or values.size < 2:
        return {"status": "NOT_AVAILABLE_IN_CURRENT_HELPER"}
    flat = np.sort(values[np.isfinite(values)].ravel())[::-1]
    if flat.size < 2:
        return {"status": "NOT_AVAILABLE_IN_CURRENT_HELPER"}
    return {
        "status": "OK",
        "top1_peak": float(flat[0]), "top2_peak": float(flat[1]),
        "peak_ratio": float(flat[0] / max(abs(flat[1]), 1e-12)),
        "peak_separation": existing_helper_result.get("peak_separation"),
    }


def compare_working_vs_suspect(edge_result: dict) -> dict:
    """Compare independent evidence for one audit edge."""
    counter = edge_result.get("counterfactual", {})
    zero = counter.get("zero", {})
    reported = counter.get("reported", {})
    inverse = counter.get("inverse", {})
    sweep = edge_result.get("sweep", {})
    reported_improves = (
        reported.get("gradient_ncc") is not None
        and zero.get("gradient_ncc") is not None
        and reported["gradient_ncc"] > zero["gradient_ncc"]
        and reported.get("raw_ncc", -np.inf) > zero.get("raw_ncc", -np.inf)
    )
    sweep_near_reported = False
    phase = edge_result.get("phase", {})
    if sweep.get("best_dx") is not None and phase.get("dx_px") is not None:
        sweep_near_reported = bool(np.hypot(
            sweep["best_dx"] - phase["dx_px"], sweep["best_dy"] - phase["dy_px"]
        ) <= 2.0)
    zero_bias = any(
        row.get("injected_dx") == 0 and row.get("injected_dy") == 0
        and row.get("error_mag_px") is not None and row["error_mag_px"] > 3.0
        for row in edge_result.get("injection", [])
    )
    mask = edge_result.get("mask", {})
    mask_collapses = (
        mask.get("V0_original", {}).get("magnitude_px") is not None
        and mask.get("V1_erode_16", {}).get("magnitude_px") is not None
        and mask["V0_original"]["magnitude_px"] > 5.0
        and mask["V1_erode_16"]["magnitude_px"] < 1.0
    )
    repr_rows = edge_result.get("representations", [])
    raw_mag = next((r["phase"].get("magnitude_px") for r in repr_rows if r.get("representation") == "R0_raw"), None)
    grad_mag = next((r["phase"].get("magnitude_px") for r in repr_rows if r.get("representation") in {"R2_gradient_magnitude", "R4_edge_map"}), None)
    representation_sensitive = raw_mag is not None and grad_mag is not None and raw_mag > 5.0 and grad_mag < 1.0
    if reported_improves and sweep_near_reported:
        answer = "SUPPORTED_AS_REAL_TRANSLATION"
    elif (
        not reported_improves
        and sweep.get("best_dx") == 0 and sweep.get("best_dy") == 0
        and (zero_bias or mask_collapses or representation_sensitive)
    ):
        answer = "NOT_SUPPORTED_PHASE_ARTIFACT"
    else:
        answer = "MIXED"
    return {
        "edge": edge_result.get("edge"),
        "answer": answer,
        "reported_improves_independent_metrics": reported_improves,
        "sweep_near_reported": sweep_near_reported,
        "zero_injection_bias": zero_bias,
        "mask_boundary_collapse": mask_collapses,
        "representation_sensitive": representation_sensitive,
        "counterfactual": {"zero": zero, "reported": reported, "inverse": inverse},
        "sweep": sweep,
    }


def classify_phase_validation_root_cause(evidence: dict) -> str:
    """Classify the audit evidence without changing any production gate."""
    yes = lambda *keys: all(bool(evidence.get(key)) for key in keys)
    if evidence.get("frame_bug"):
        return "FRAME_OR_DIRECTION_BUG_SUPPORTED"
    if yes("reported_improves", "sweep_near_reported", "mask_stable", "representation_agrees", "injection_correct"):
        return "REAL_TRANSLATION_RESIDUAL_SUPPORTED"
    if yes("mask_boundary_collapse", "sweep_prefers_zero") and not evidence.get("reported_improves"):
        return "MASK_BOUNDARY_DOMINATED"
    if evidence.get("representation_sensitive"):
        return "RADIOMETRIC_REPRESENTATION_SENSITIVE"
    if evidence.get("multipeak"):
        return "TEXTURE_AMBIGUITY_OR_MULTIPEAK"
    if evidence.get("zero_injection_bias"):
        return "PHASE_HELPER_REAL_CROP_BIAS"
    if evidence.get("reported_improves") is False and evidence.get("sweep_prefers_zero"):
        return "PHASE_VALIDATION_ARTIFACT_SUPPORTED"
    return "MIXED_OR_UNDERDETERMINED"


def build_phase_validation_conclusion(baseline: dict, results: dict[str, dict]) -> dict:
    """Build the bounded final conclusion required by the audit plan."""
    answers = {
        "old_phase_values_reproduced": {
            edge: result.get("baseline_reproduction_status")
            for edge, result in results.items()
        },
        "phase_dx_dy_semantics": "phase (dx,dy) is the shift required to move moving/target toward reference",
        "frame_or_crop_bug_found": any(result.get("frame_bug") for result in results.values()),
        "edge_answers": {
            edge: result.get("comparison", {}).get("answer") for edge, result in results.items()
        },
        "root_cause_states": {edge: result.get("root_cause") for edge, result in results.items()},
        "re_audit_next": ["2-5", "4-6", "1-8"],
    }
    return {
        "answers": answers,
        "can_conclude": [
            "The audit compares only the fixed 0-6 working control and 2-5 suspect edge.",
            "Independent counterfactual, integer sweep, mask, representation, and injection evidence is diagnostic evidence only.",
        ],
        "cannot_conclude": [
            "do not directly modify production phase helper in this run",
            "do not claim SIFT/Affine is always correct",
            "do not automatically restore all previously rejected edges",
            "do not claim a new phase method generalizes from two edges",
        ],
    }


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


def write_phase_validation_dashboard(results: dict[str, dict], output_path) -> Path:
    """Write the plan's two-edge, five-column diagnostic dashboard."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    edges = [("0-6", "WORKING_CONTROL"), ("2-5", "SUSPECT_PHASE_FALSE_ALARM")]
    fig, axes = plt.subplots(2, 5, figsize=(19, 7.5), squeeze=False)
    column_titles = [
        "direct-warp checkerboard",
        "original phase vector",
        "translation-sweep surface",
        "mask-erosion stability",
        "real-crop injection recovery",
    ]
    for col, title in enumerate(column_titles):
        axes[0, col].set_title(title, fontsize=10)
    for row, (edge, role) in enumerate(edges):
        result = results.get(edge, {})
        inputs = result.get("inputs", {})
        ref = np.asarray(inputs.get("ref_crop", np.zeros((2, 2))), dtype=float)
        tgt = np.asarray(inputs.get("warped_target_crop", ref), dtype=float)
        joint = np.asarray(inputs.get("joint_valid_mask", np.ones_like(ref, dtype=bool)), dtype=bool)
        finite = np.isfinite(ref) & np.isfinite(tgt) & joint
        checker = np.where(finite, np.where((np.indices(ref.shape).sum(axis=0) % 2) == 0, ref, tgt), np.nan)
        ax = axes[row, 0]
        ax.imshow(checker, cmap="gray")
        ax.set_axis_off()

        phase = result.get("phase", {})
        dx = float(phase.get("dx_px") or 0.0)
        dy = float(phase.get("dy_px") or 0.0)
        ax = axes[row, 1]
        ax.axhline(0, color="0.75", linewidth=0.8)
        ax.axvline(0, color="0.75", linewidth=0.8)
        ax.arrow(0, 0, dx, dy, width=0.03, length_includes_head=True, color="tab:red")
        extent = max(1.0, abs(dx), abs(dy)) * 1.25
        ax.set_xlim(-extent, extent)
        ax.set_ylim(extent, -extent)
        ax.set_aspect("equal")
        ax.set_xlabel(f"dx={dx:.2f}")
        ax.set_ylabel(f"dy={dy:.2f}")
        ax.grid(True, alpha=0.25)

        ax = axes[row, 2]
        rows = result.get("sweep", {}).get("rows", [])
        if rows:
            scatter = ax.scatter(
                [float(item["dx"]) for item in rows],
                [float(item["dy"]) for item in rows],
                c=[float(item.get("score", item.get("gradient_ncc", np.nan))) for item in rows],
                cmap="viridis",
                s=38,
            )
            fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
            ax.scatter([0], [0], marker="x", color="red", s=45)
        ax.set_xlabel("dx")
        ax.set_ylabel("dy")
        ax.set_title(f"best=({result.get('sweep', {}).get('best_dx')}, {result.get('sweep', {}).get('best_dy')})", fontsize=8)
        ax.grid(True, alpha=0.2)

        ax = axes[row, 3]
        mask_rows = result.get("mask", {})
        labels = list(mask_rows)
        values = [float(mask_rows[label].get("magnitude_px", np.nan)) for label in labels]
        if labels:
            ax.plot(range(len(labels)), values, marker="o")
            ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right", fontsize=7)
        ax.set_ylabel("phase px")
        ax.grid(True, alpha=0.2)

        ax = axes[row, 4]
        injections = result.get("injection", [])
        if injections:
            x = [float(np.hypot(item.get("injected_dx", 0), item.get("injected_dy", 0))) for item in injections]
            y = [float(item.get("error_mag_px", np.nan)) for item in injections]
            ax.plot(x, y, marker="o")
        ax.set_xlabel("injected magnitude px")
        ax.set_ylabel("recovery error px")
        ax.grid(True, alpha=0.2)
        axes[row, 0].set_ylabel(f"{edge}\n{role}", fontsize=9)
    fig.suptitle("Phase validation root-cause audit", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _write_rows_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({str(key) for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _json_safe(row.get(key)) for key in fieldnames})


def write_phase_audit_artifacts(output_dir: str | Path, baseline: dict, results: dict[str, dict]) -> dict:
    """Write the complete artifact tree required by the phase audit plan."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "00_phase_audit_baseline.json", baseline)
    write_json(out / "01_phase_dataflow_trace.json", {edge: result.get("flow", {}) for edge, result in results.items()})
    inputs_by_edge = {edge: result["inputs"] for edge, result in results.items()}
    for edge, inputs in inputs_by_edge.items():
        export_phase_inputs(inputs, out, edge)
    write_phase_input_visual_audit(
        inputs_by_edge,
        out / "03_phase_input_visual_audit.png",
        out / "03_phase_input_visual_stats.json",
    )
    write_json(out / "04_phase_sign_convention.json", {edge: result.get("sign_convention", {}) for edge, result in results.items()})

    counterfactual = {edge: result.get("counterfactual", {}) for edge, result in results.items()}
    write_json(out / "05_phase_shift_counterfactual.json", counterfactual)
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = ["zero", "reported", "inverse"]
    for edge, result in results.items():
        values = [float(result.get("counterfactual", {}).get(label, {}).get("gradient_ncc", np.nan)) for label in labels]
        ax.plot(labels, values, marker="o", label=edge)
    ax.set_ylabel("gradient NCC")
    ax.set_title("phase shift counterfactual")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "05_phase_shift_counterfactual.png", dpi=150)
    plt.close(fig)

    sweep_rows, sweep_summary = [], {}
    for edge, result in results.items():
        rows = [dict(row, edge=edge) for row in result.get("sweep", {}).get("rows", [])]
        sweep_rows.extend(rows)
        sweep_summary[edge] = {key: value for key, value in result.get("sweep", {}).items() if key != "rows"}
    _write_rows_csv(out / "06_translation_sweep.csv", sweep_rows)
    write_json(out / "06_translation_sweep_summary.json", sweep_summary)
    fig, axes = plt.subplots(1, len(results), figsize=(9, 4), squeeze=False)
    for index, (edge, result) in enumerate(results.items()):
        rows = result.get("sweep", {}).get("rows", [])
        ax = axes[0, index]
        if rows:
            image = ax.scatter(
                [row["dx"] for row in rows],
                [row["dy"] for row in rows],
                c=[row.get("score", row.get("gradient_ncc", np.nan)) for row in rows],
                cmap="viridis",
            )
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(edge)
        ax.set_xlabel("dx")
        ax.set_ylabel("dy")
        ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out / "06_translation_sweep_surface.png", dpi=150)
    plt.close(fig)

    mask_rows, mask_json = [], {}
    for edge, result in results.items():
        mask_json[edge] = result.get("mask", {})
        mask_rows.extend([dict(value, edge=edge, variant=key) for key, value in result.get("mask", {}).items()])
    _write_rows_csv(out / "07_mask_boundary_stress_test.csv", mask_rows)
    write_json(out / "07_mask_boundary_stress_test.json", mask_json)

    representation_rows, representation_json = [], {}
    for edge, result in results.items():
        representation_json[edge] = result.get("representations", [])
        representation_rows.extend([dict(row, edge=edge) for row in result.get("representations", [])])
    _write_rows_csv(out / "08_representation_stability.csv", representation_rows)
    write_json(out / "08_representation_stability.json", representation_json)

    injection_rows, injection_json = [], {}
    for edge, result in results.items():
        injection_json[edge] = result.get("injection", [])
        injection_rows.extend([dict(row, edge=edge) for row in result.get("injection", [])])
    _write_rows_csv(out / "09_real_crop_injection_recovery.csv", injection_rows)
    write_json(out / "09_real_crop_injection_recovery.json", injection_json)
    write_json(out / "10_phase_peak_diagnostics.json", {edge: result.get("peak_diagnostics", {}) for edge, result in results.items()})

    comparison = {edge: {"comparison": result.get("comparison", {}), "root_cause": result.get("root_cause")} for edge, result in results.items()}
    write_json(out / "11_working_vs_suspect_phase_audit.json", comparison)
    (out / "11_working_vs_suspect_phase_audit.txt").write_text(
        "\n".join(f"{edge}: {result.get('comparison', {}).get('answer')} / {result.get('root_cause')}" for edge, result in results.items()) + "\n",
        encoding="utf-8",
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axis("off")
    ax.text(0.02, 0.95, "Working vs suspect phase audit", va="top", fontsize=13)
    ax.text(0.02, 0.72, "\n".join(f"{edge}: {result.get('comparison', {}).get('answer')}" for edge, result in results.items()), va="top")
    fig.savefig(out / "11_working_vs_suspect_phase_audit.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    conclusion = build_phase_validation_conclusion(baseline, results)
    write_json(out / "12_phase_validation_root_cause.json", conclusion)
    (out / "12_phase_validation_root_cause.txt").write_text(
        "CAN conclude\n" + "\n".join(f"- {item}" for item in conclusion["can_conclude"])
        + "\n\nCANNOT conclude\n" + "\n".join(f"- {item}" for item in conclusion["cannot_conclude"]) + "\n",
        encoding="utf-8",
    )
    write_phase_validation_dashboard(results, out / "13_phase_validation_audit_dashboard.png")
    files = sorted(path for path in out.rglob("*") if path.is_file())
    return {"artifact_count": len(files), "files": [str(path.relative_to(out)) for path in files]}
