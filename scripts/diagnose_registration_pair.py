#!/usr/bin/env python3
"""Registration-only diagnostic for manual validation.

Usage:
    python scripts/diagnose_registration_pair.py \
        --config configs/dz01_mosaic_series_b14.yaml \
        --scene-i 0 --scene-j 1

This script runs ONLY registration (no BAGRN/VOLRN normalization) to diagnose
geometric alignment quality between two scenes. It writes registration
diagnostics and visual artifacts, including an optional diagnostic mosaic.
"""

import os
import sys
import argparse
import json
import csv
import logging
import shutil
import subprocess
from pathlib import Path
import numpy as np
from rasterio.warp import reproject, Resampling

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.experiment_config import load_config
from src.io_utils import write_geotiff
from src.mosaic import create_mosaic
from src.multiband_pipeline import (
    MultibandPipeline,
    registration_quality_meets_requirement,
)
from src.registration_band_ab import (
    build_holdout_manifest,
    load_holdout_manifest,
    manifest_to_pair_overrides,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def _json_safe(value):
    """Convert registration diagnostics to values accepted by JSON."""
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _metric_text(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "N/A"
    return f"{value:.3f}" if np.isfinite(value) else "N/A"


def _log_stage_validation_summary(registration):
    """Log paired HOLDOUT metrics with the fixed before-minus-after meaning."""
    global_quality = registration.get("global_only_quality", {}) or {}
    final_quality = registration.get("quality", {}) or {}
    comparison = registration.get("stage_validation_comparison", {}) or {}
    logger.info(
        "global-only holdout: quality=%s, median=%s, rmse=%s, p95=%s",
        global_quality.get("quality", "unknown"),
        _metric_text(global_quality.get("median")),
        _metric_text(global_quality.get("rmse")),
        _metric_text(global_quality.get("p95")),
    )
    logger.info(
        "final holdout: quality=%s, median=%s, rmse=%s, p95=%s; "
        "delta_rmse=%s, delta_p95=%s (global-only minus final; positive means improvement)",
        final_quality.get("quality", "unknown"),
        _metric_text(final_quality.get("median")),
        _metric_text(final_quality.get("rmse")),
        _metric_text(final_quality.get("p95")),
        _metric_text(comparison.get("rmse_improvement")),
        _metric_text(comparison.get("p95_improvement")),
    )


_HOLDOUT_LOCAL_FIELD_COLUMNS = [
    "idx_i", "idx_j", "validation_row", "validation_col",
    "reference_center_x", "reference_center_y", "field_center_x",
    "field_center_y", "scene_idx", "coordinate_mapping_available",
    "coordinate_mapping_reason", "predicted_local_dx", "predicted_local_dy",
    "predicted_local_magnitude", "fade_value", "inside_control_hull",
    "distance_outside_hull_px", "nearest_local_control_distance_px",
    "final_residual_dx", "final_residual_dy", "final_residual_magnitude",
    "accepted", "reject_reason", "selected_smoothing",
]


def _iter_holdout_local_field_samples(registration):
    """Yield one flattened row per reserved HOLDOUT validation block."""
    samples = (registration.get("holdout_local_field_samples", {}) or {})
    for edge in samples.get("edges", []) or []:
        for block in edge.get("blocks", []) or []:
            row = {
                "idx_i": edge.get("idx_i"),
                "idx_j": edge.get("idx_j"),
            }
            row.update({key: block.get(key) for key in _HOLDOUT_LOCAL_FIELD_COLUMNS
                        if key not in row})
            yield row


def _write_holdout_local_field_csv(registration, output_dir):
    """Write mapped local-field samples without changing registration state."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    csv_path = output_path / "holdout_local_field_samples.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_HOLDOUT_LOCAL_FIELD_COLUMNS)
        writer.writeheader()
        for row in _iter_holdout_local_field_samples(registration):
            writer.writerow({
                key: _json_safe(row.get(key))
                for key in _HOLDOUT_LOCAL_FIELD_COLUMNS
            })
    return str(csv_path)


def _log_holdout_local_field_summary(registration):
    """Log mapped-field coverage and applied correction magnitudes."""
    rows = list(_iter_holdout_local_field_samples(registration))
    mapped = [row for row in rows if row.get("coordinate_mapping_available")]
    available = [row for row in rows if row.get("field_available")]
    inside = [row for row in rows if row.get("inside_control_hull")]
    fade_positive = [
        row for row in rows
        if row.get("fade_value") is not None
        and float(row["fade_value"]) > 0.0
    ]
    logger.info(
        "local-field HOLDOUT mapping: total=%d, mapped=%d, field_available=%d, "
        "inside_hull=%d, fade_positive=%d",
        len(rows), len(mapped), len(available), len(inside), len(fade_positive),
    )
    magnitudes = np.asarray([
        row.get("predicted_local_magnitude") for row in rows
        if row.get("predicted_local_magnitude") is not None
    ], dtype=float)
    magnitudes = magnitudes[np.isfinite(magnitudes)]
    median = float(np.median(magnitudes)) if len(magnitudes) else None
    p95 = float(np.percentile(magnitudes, 95)) if len(magnitudes) else None
    maximum = float(np.max(magnitudes)) if len(magnitudes) else None
    logger.info(
        "local-field HOLDOUT correction magnitude: median=%s, p95=%s, max=%s",
        _metric_text(median), _metric_text(p95), _metric_text(maximum),
    )


_HULL_CAUSAL_STAGE_COLUMNS = [
    "stage", "available", "quality", "median", "rmse", "p95",
    "confidence", "n_blocks", "rmse_vs_global", "p95_vs_global",
    "median_vs_global", "rmse_vs_legacy", "p95_vs_legacy",
    "median_vs_legacy",
]
_HULL_CAUSAL_WINDOW_COLUMNS = [
    "validation_row", "validation_col", "block_size", "scene_idx",
    "reference_center_x", "reference_center_y", "field_center_x",
    "field_center_y", "center_inside_hull", "center_legacy_weight",
    "center_strict_weight", "window_valid_sample_count",
    "window_inside_hull_fraction", "window_legacy_support_nonzero_fraction",
    "window_strict_support_nonzero_fraction", "window_support_difference_fraction",
    "window_legacy_support_mean", "window_strict_support_mean",
    "window_raw_rbf_magnitude_mean", "window_raw_rbf_magnitude_p95",
    "window_legacy_rbf_magnitude_mean", "window_legacy_rbf_magnitude_p95",
    "window_strict_rbf_magnitude_mean", "window_strict_rbf_magnitude_p95",
    "global_residual_magnitude", "legacy_residual_magnitude",
    "strict_residual_magnitude", "legacy_minus_strict", "recovery_fraction",
    "negative_control_candidate",
]


def _hull_causal_quality(registration, stage):
    causal = registration.get("hull_causal", {}) or {}
    if stage == "global_only":
        return registration.get("global_only_quality", {}) or {}
    if stage == "legacy_rbf":
        return registration.get("quality", {}) or {}
    return causal.get("strict_counterfactual_quality", {}) or {}


def _hull_causal_comparison(registration, name):
    return ((registration.get("hull_causal", {}) or {}).get(
        "comparisons", {}
    ) or {}).get(name, {}) or {}


def _write_hull_causal_stage_metrics_csv(registration, output_dir):
    """Write the fixed three-stage HULL-C1 comparison table."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    csv_path = output_path / "hull_causal_stage_metrics.csv"
    global_to_legacy = registration.get("stage_validation_comparison", {}) or {}
    global_to_strict = _hull_causal_comparison(registration, "global_to_strict")
    legacy_to_strict = _hull_causal_comparison(registration, "legacy_to_strict")
    rows = []
    for stage, comparison, legacy_comparison in (
        ("global_only", {}, {}),
        ("legacy_rbf", global_to_legacy, {}),
        ("strict_rbf", global_to_strict, legacy_to_strict),
    ):
        quality = _hull_causal_quality(registration, stage)
        rows.append({
            "stage": stage,
            "available": bool(quality),
            "quality": quality.get("quality"),
            "median": quality.get("median"),
            "rmse": quality.get("rmse"),
            "p95": quality.get("p95"),
            "confidence": quality.get("confidence", quality.get("mean_confidence")),
            "n_blocks": quality.get("n_blocks", quality.get("n_accepted")),
            "rmse_vs_global": comparison.get("rmse_improvement"),
            "p95_vs_global": comparison.get("p95_improvement"),
            "median_vs_global": comparison.get("median_improvement"),
            "rmse_vs_legacy": legacy_comparison.get("rmse_improvement"),
            "p95_vs_legacy": legacy_comparison.get("p95_improvement"),
            "median_vs_legacy": legacy_comparison.get("median_improvement"),
        })
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_HULL_CAUSAL_STAGE_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _json_safe(row.get(key)) for key in writer.fieldnames})
    return csv_path


def _write_hull_causal_window_stats_csv(registration, output_dir):
    """Write compact full-window HULL-C1 support statistics."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    csv_path = output_path / "hull_causal_window_stats.csv"
    causal = registration.get("hull_causal", {}) or {}
    blocks = ((causal.get("window_stats", {}) or {}).get("blocks", []) or [])
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_HULL_CAUSAL_WINDOW_COLUMNS)
        writer.writeheader()
        for block in blocks:
            row = dict(block)
            global_magnitude = row.get("global_residual_magnitude")
            legacy_magnitude = row.get("legacy_residual_magnitude")
            strict_magnitude = row.get("strict_residual_magnitude")
            try:
                legacy_minus_strict = float(legacy_magnitude) - float(strict_magnitude)
                recovery_fraction = legacy_minus_strict / (
                    float(legacy_magnitude) - float(global_magnitude)
                )
                if not np.isfinite(recovery_fraction):
                    recovery_fraction = None
            except (TypeError, ValueError, ZeroDivisionError):
                legacy_minus_strict = None
                recovery_fraction = None
            row.update({
                "legacy_minus_strict": legacy_minus_strict,
                "recovery_fraction": recovery_fraction,
            })
            writer.writerow({
                key: _json_safe(row.get(key)) for key in writer.fieldnames
            })
    return csv_path


def _log_hull_causal_summary(registration):
    """Log HULL-C1 integrity and the three paired validation stages."""
    causal = registration.get("hull_causal", {}) or {}
    if not causal.get("available"):
        return
    integrity = causal.get("integrity", {}) or {}
    logger.info(
        "HULL-C1 integrity: pass=%s, same_holdout=%s, raw_fit_count=%s, "
        "strict_outside_nonzero=%s",
        integrity.get("integrity_pass"),
        integrity.get("same_reserved_holdout"),
        integrity.get("raw_rbf_fit_count_by_scene"),
        integrity.get("strict_outside_hull_nonzero_pixels"),
    )
    for stage in ("global_only", "legacy_rbf", "strict_rbf"):
        quality = _hull_causal_quality(registration, stage)
        logger.info(
            "HULL-C1 %s: median=%s, rmse=%s, p95=%s",
            stage, _metric_text(quality.get("median")),
            _metric_text(quality.get("rmse")), _metric_text(quality.get("p95")),
        )
    comparison = _hull_causal_comparison(registration, "legacy_to_strict")
    logger.info(
        "HULL-C1 legacy->strict: delta_rmse=%s, delta_p95=%s, "
        "delta_median=%s (positive=strict improvement)",
        _metric_text(comparison.get("rmse_improvement")),
        _metric_text(comparison.get("p95_improvement")),
        _metric_text(comparison.get("median_improvement")),
    )
def _initial_global_shifts(registration, scene_ids):
    """Return pre-refinement shifts when the schema records them."""
    diagnostics = registration.get("diagnostics", {}) or {}
    for container in (registration, diagnostics):
        for key in ("initial_global_shifts", "global_shifts_initial", "initial_shifts"):
            if key in container and container[key] is not None:
                return container[key]

    # Task 6 keeps the initial robust pair measurements in pair_matches. Rebuild
    # the network estimate when possible so both registration stages are visible.
    pair_matches = registration.get("pair_matches", []) or []
    if pair_matches and all(
        "idx_i" in pair and "idx_j" in pair
        and "shift_dx" in pair and "shift_dy" in pair
        for pair in pair_matches
    ):
        try:
            from src.coregistration import multi_image_network_adjustment

            initial = multi_image_network_adjustment(
                pair_matches, len(scene_ids), reference_idx=0
            ).get("global_shifts")
            if initial is not None:
                return initial
        except (KeyError, TypeError, ValueError, IndexError):
            logger.debug("Unable to reconstruct initial network shifts", exc_info=True)

    return registration.get("global_shifts", [])


def _raw_block_matches(registration):
    """Extract raw block samples while retaining the producing edge."""
    diagnostics = registration.get("diagnostics", {}) or {}
    if "raw_block_matches" in diagnostics:
        return diagnostics["raw_block_matches"]

    return [
        {
            "idx_i": pair.get("idx_i"),
            "idx_j": pair.get("idx_j"),
            "matches": pair.get("raw_matches", []),
        }
        for pair in registration.get("pair_matches", []) or []
    ]


def _local_field_diagnostics(registration):
    """Collect local CV and field statistics without serializing full fields."""
    diagnostics = registration.get("diagnostics", {}) or {}
    local = registration.get("local_refinement", {}) or {}
    if "local_field_diagnostics" in diagnostics:
        return diagnostics["local_field_diagnostics"]

    fields = {}
    for scene_id, scene_result in (local.get("scenes", {}) or {}).items():
        if "field_stats" in scene_result:
            fields[str(scene_id)] = scene_result["field_stats"]
    return fields


def build_diagnostic_payload(registration, scene_ids, output_dir):
    """Build and write ``registration_diagnostics.json`` from actual schema fields."""
    diagnostics = registration.get("diagnostics", {}) or {}
    local_refinement = registration.get("local_refinement", {}) or {}
    quality = registration.get("quality", {})
    final_validation = registration.get("final_validation", {})
    classification = quality.get("quality") if isinstance(quality, dict) else None
    classification = str(classification).upper() if classification is not None else None

    payload = {
        "scene_ids": list(scene_ids),
        "registration_band_name": registration.get("registration_band_name"),
        "validation_band_name": registration.get("validation_band_name"),
        "registration_params_sha256": registration.get("registration_params_sha256"),
        "output_dir": str(output_dir),
        "status": registration.get("status"),
        "failure": registration.get("failure", {}),
        "connected": registration.get("connected", False),
        "diagnostics": diagnostics,
        "pair_matches": registration.get("pair_matches", []),
        "raw_block_matches": _raw_block_matches(registration),
        "robust_pair_measurements": registration.get("pair_matches", []),
        "initial_global_shifts": _initial_global_shifts(registration, scene_ids),
        "global_shifts": registration.get("global_shifts", []),
        "final_global_shifts": registration.get("global_shifts", []),
        "global_refinement_history": diagnostics.get(
            "global_refinement_history", diagnostics.get("global_refinement", [])
        ),
        "global_refinement_warnings": diagnostics.get("global_refinement_warnings", []),
        "local_refinement": local_refinement,
        "local_cv_results": local_refinement.get("cv_results", {}),
        "local_field_diagnostics": _local_field_diagnostics(registration),
        "overlap": diagnostics.get("overlap", {}),
        "holdout": diagnostics.get("holdout", {}),
        "local_controls": diagnostics.get("local_controls", {}),
        "local_field": diagnostics.get("local_field", {}),
        "global_only_quality": registration.get("global_only_quality"),
        "global_only_validation": registration.get("global_only_validation"),
        "stage_validation_comparison": registration.get(
            "stage_validation_comparison"
        ),
        "holdout_local_field_samples": registration.get(
            "holdout_local_field_samples", {}
        ),
        "hull_causal": registration.get("hull_causal"),
        "final_validation": final_validation,
        "quality": quality,
        "quality_classification": classification,
        "classification": classification,
        "artifacts": registration.get("diagnostic_artifacts", {}),
    }

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    json_path = output_path / "registration_diagnostics.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(
            _json_safe(payload), handle, indent=2, ensure_ascii=False, allow_nan=False
        )
    return payload


def _build_no_overlap_registration(scene_ids, reason, local_enabled=True):
    """Build a JSON-safe registration failure record when no overlap exists."""
    quality = {
        "quality": "fail",
        "rmse": float("inf"),
        "p95": float("inf"),
        "median": float("inf"),
        "confidence": 0.0,
        "n_blocks": 0,
    }
    final_validation = {"edges": [], "overall": quality}
    failure = {"code": "no_overlap", "reason": reason}
    return {
        "status": "fail",
        "failure": failure,
        "registered_arrays": [],
        "global_shifts": np.zeros((len(scene_ids), 2), dtype=float),
        "pair_matches": [],
        "connected": False,
        "spanning_tree": [],
        "geometric_edges": [],
        "matching_edges": [],
        "rejected_edges": [],
        "connected_components": [[index] for index in range(len(scene_ids))],
        "unreachable_scenes": list(scene_ids[1:]),
        "quality": quality,
        "final_validation": final_validation,
        "local_refinement": {
            "enabled": bool(local_enabled),
            "used_for_scenes": [],
            "fallback_scenes": [],
            "cv_results": {},
        },
        "diagnostics": {
            "registration_blocked": True,
            "no_overlap": True,
            "reason": reason,
            "final_validation": final_validation,
            "quality": quality,
            "raw_block_matches": [],
        },
    }


def _valid_pixels(array, nodata):
    values = np.asarray(array, dtype=float)
    valid = np.isfinite(values)
    if nodata is not None:
        valid &= values != nodata
    return values, valid


def _stretch_for_overlay(array, nodata):
    """Scale one registered band to uint8 without changing registration data."""
    values, valid = _valid_pixels(array, nodata)
    output = np.zeros(values.shape, dtype=np.uint8)
    if not valid.any():
        return output
    low, high = np.percentile(values[valid], [2, 98])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low = float(values[valid].min())
        high = float(values[valid].max())
    if high > low:
        scaled = np.clip((values - low) / (high - low) * 255.0, 0.0, 255.0)
        output[valid] = scaled[valid].astype(np.uint8)
    else:
        output[valid] = 128
    return output


def _registration_band(array, band_index):
    """Select a two-dimensional band from a registered scene array."""
    values = np.asarray(array)
    if values.ndim == 2:
        return values
    if values.ndim != 3:
        raise ValueError(f"registered scene array must be 2D or 3D, got {values.ndim}D")
    if not 0 <= band_index < values.shape[0]:
        raise IndexError(f"registration band {band_index} is outside {values.shape[0]} bands")
    return values[band_index]


def _reproject_to_reference(array, src_transform, src_crs, nodata, shape,
                            dst_transform, dst_crs):
    """Put a registered target band on the registered reference grid for overlay."""
    destination = np.full(shape, np.nan, dtype=np.float64)
    reproject(
        source=np.asarray(array, dtype=np.float64),
        destination=destination,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        src_nodata=nodata,
        dst_nodata=np.nan,
        resampling=Resampling.nearest,
    )
    return destination


def _save_mask_png(path, common_mask, train_mask, holdout_mask):
    """Save a compact RGB map of common-valid, TRAIN and HOLDOUT regions."""
    import matplotlib.pyplot as plt

    common = np.asarray(common_mask, dtype=bool)
    train = np.asarray(train_mask, dtype=bool)
    holdout = np.asarray(holdout_mask, dtype=bool)
    rgb = np.zeros((*common.shape, 3), dtype=float)
    rgb[..., 2] = common.astype(float) * 0.35
    rgb[..., 1] = train.astype(float)
    rgb[..., 0] = holdout.astype(float)
    plt.imsave(path, np.clip(rgb, 0.0, 1.0))


def _save_residual_vectors(path, matches, title):
    """Save residual vectors without changing registration data."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5), dpi=140)
    points = [m for m in matches if all(k in m for k in ("ref_x", "ref_y", "shift_dx", "shift_dy"))]
    if points:
        x = np.asarray([m["ref_x"] for m in points])
        y = np.asarray([m["ref_y"] for m in points])
        u = np.asarray([m["shift_dx"] for m in points])
        v = np.asarray([m["shift_dy"] for m in points])
        ax.quiver(x, y, u, v, angles="xy", scale_units="xy", scale=1, width=0.003)
        ax.scatter(x, y, s=8, c="black")
    ax.set_title(title)
    ax.set_xlabel("reference pixel x")
    ax.set_ylabel("reference pixel y")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _save_field_png(path, field, title):
    """Save one local displacement field as a heatmap."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5), dpi=140)
    image = ax.imshow(np.asarray(field, dtype=float), cmap="coolwarm")
    ax.set_title(title)
    fig.colorbar(image, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _save_validation_holdout_png(path, validation, shape, title=None):
    """Draw final validation blocks and their rejection/acceptance state."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots(figsize=(7, 5), dpi=140)
    ax.set_xlim(0, shape[1])
    ax.set_ylim(shape[0], 0)
    for edge in validation.get("edges", []) or []:
        for block in edge.get("blocks", []) or []:
            color = "lime" if block.get("accepted") else "red"
            size = int(validation.get("validation_block_size_selected") or 192)
            ax.add_patch(Rectangle(
                (block.get("validation_col", 0), block.get("validation_row", 0)),
                size, size, fill=False, edgecolor=color, linewidth=0.5,
            ))
    ax.set_title(title or "Final holdout validation blocks")
    ax.set_xlabel("reference pixel x")
    ax.set_ylabel("reference pixel y")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_diagnostic_artifacts(registration, scene_data, scene_ids, output_dir,
                               registration_band_idx=0, mosaic_mode="weighted",
                               *, allow_quality_fail_for_diagnostics=False,
                               visualization_band_idx=None):
    """Write registered images, a red-green overlay, and a diagnostic mosaic."""
    quality = registration.get("quality", {}) or {}
    registered = registration.get("registered_arrays")
    diagnostics = registration.get("diagnostics", {}) or {}
    blocked = bool(diagnostics.get("registration_blocked"))
    connected = registration.get("connected") is not False
    quality_fail = quality.get("quality") == "fail"
    has_registered = bool(registered) and len(registered) >= 2
    if blocked or not connected or not has_registered:
        raise ValueError(
            "cannot write diagnostic registration rasters for blocked/disconnected result"
        )
    if quality_fail and not allow_quality_fail_for_diagnostics:
        raise ValueError("quality-failed registration artifacts require diagnostic opt-in")

    global_only = registration.get("global_only_arrays") or registered
    holdout_local_field_csv = _write_holdout_local_field_csv(
        registration, output_dir
    )
    _log_holdout_local_field_summary(registration)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    transforms = scene_data["transforms"]
    nodatas = scene_data["nodata_values"]
    crs = scene_data["crs"]

    visualization_band_idx = (
        registration_band_idx if visualization_band_idx is None
        else int(visualization_band_idx)
    )

    def write_pair(prefix, arrays):
        reference_path = output_path / f"{prefix}_reference.tif"
        target_path = output_path / f"{prefix}_target.tif"
        write_geotiff(str(reference_path), arrays[0], transforms[0], crs,
                      nodata=nodatas[0], dtype="float32")
        write_geotiff(str(target_path), arrays[1], transforms[1], crs,
                      nodata=nodatas[1], dtype="float32")

        reference_band = _registration_band(arrays[0], visualization_band_idx)
        target_band = _registration_band(arrays[1], visualization_band_idx)
        target_on_reference = _reproject_to_reference(
            target_band, transforms[1], crs, nodatas[1], reference_band.shape,
            transforms[0], crs,
        )
        reference_green = _stretch_for_overlay(reference_band, nodatas[0])
        target_red = _stretch_for_overlay(target_on_reference, None)
        overlay = np.stack([
            target_red, reference_green, np.zeros_like(reference_green)
        ], axis=0)
        overlay_path = output_path / f"{prefix}_red_green_overlay.tif"
        write_geotiff(str(overlay_path), overlay, transforms[0], crs,
                      nodata=0, dtype="uint8")
        return reference_path, target_path, overlay_path, reference_band, overlay

    global_reference_path, global_target_path, global_overlay_path, _, _ = write_pair(
        "registered_global_only", global_only
    )
    final_reference_path, final_target_path, final_overlay_path, reference_band, final_overlay = write_pair(
        "registered_final", registered
    )
    strict = registration.get("strict_counterfactual_arrays")
    strict_paths = {}
    if (registration.get("hull_causal") or {}).get("available") and strict:
        strict_reference_path, strict_target_path, strict_overlay_path, _, _ = write_pair(
            "registered_strict_hull", strict
        )
        strict_paths = {
            "registered_strict_hull_reference": str(strict_reference_path),
            "registered_strict_hull_target": str(strict_target_path),
            "registered_strict_hull_red_green_overlay": str(strict_overlay_path),
        }
    # Keep the original artifact names as compatibility aliases for existing
    # consumers while making the two validation stages explicit.
    reference_path = output_path / "registered_reference.tif"
    target_path = output_path / "registered_target.tif"
    overlay_path = output_path / "registered_red_green_overlay.tif"
    write_geotiff(str(reference_path), registered[0], transforms[0], crs,
                  nodata=nodatas[0], dtype="float32")
    write_geotiff(str(target_path), registered[1], transforms[1], crs,
                  nodata=nodatas[1], dtype="float32")
    write_geotiff(str(overlay_path), final_overlay, transforms[0], crs,
                  nodata=0, dtype="uint8")

    mosaic_path = output_path / f"diagnostic_mosaic_{mosaic_mode}.tif"
    mosaic_arrays = [
        array if np.asarray(array).ndim == 3 else np.asarray(array)[np.newaxis, ...]
        for array in registered[:2]
    ]
    create_mosaic(
        mosaic_arrays,
        [transforms[0], transforms[1]],
        crs,
        [nodatas[0], nodatas[1]],
        str(mosaic_path),
        mode=mosaic_mode,
    )
    b14_global_overlay = output_path / "b14_global_overlay.tif"
    b14_final_overlay = output_path / "b14_final_overlay.tif"
    b14_mosaic = output_path / f"b14_diagnostic_mosaic_{mosaic_mode}.tif"
    if global_overlay_path.exists():
        shutil.copyfile(global_overlay_path, b14_global_overlay)
    if final_overlay_path.exists():
        shutil.copyfile(final_overlay_path, b14_final_overlay)
    if mosaic_path.exists():
        shutil.copyfile(mosaic_path, b14_mosaic)

    masks_by_edge = registration.get("diagnostic_masks", {}) or {}
    edge_key = next(iter(masks_by_edge), None)
    png_paths = {}
    if edge_key:
        masks = masks_by_edge[edge_key]
        mask_path = output_path / "train_holdout_map.png"
        _save_mask_png(
            str(mask_path), masks["common_valid_mask"],
            masks["train_sampling_mask"], masks["holdout_region_mask"],
        )
        png_paths["train_holdout_map"] = str(mask_path)
        raw = (diagnostics.get("raw_block_matches", [{}])[0] or {}).get("matches", [])
        raw_path = output_path / "raw_residual_vectors.png"
        _save_residual_vectors(str(raw_path), raw, "Raw residual vectors")
        png_paths["raw_residual_vectors"] = str(raw_path)
        robust = (registration.get("pair_matches", [{}])[0] or {}).get("matches", [])
        filtered_path = output_path / "filtered_residual_vectors.png"
        _save_residual_vectors(str(filtered_path), robust, "Filtered residual vectors")
        png_paths["filtered_residual_vectors"] = str(filtered_path)
        validation_path = output_path / "final_holdout_validation_blocks.png"
        comparison = registration.get("stage_validation_comparison", {}) or {}
        delta_rmse = comparison.get("rmse_improvement")
        title = "Final holdout validation blocks"
        try:
            if np.isfinite(float(delta_rmse)):
                title += f" (global to final delta RMSE={float(delta_rmse):+.3f})"
        except (TypeError, ValueError):
            pass
        _save_validation_holdout_png(
            str(validation_path), registration.get("final_validation", {}),
            reference_band.shape, title=title,
        )
        png_paths["validation_holdout_blocks"] = str(validation_path)

    if "validation_holdout_blocks" not in png_paths:
        validation_path = output_path / "final_holdout_validation_blocks.png"
        _save_validation_holdout_png(
            str(validation_path), registration.get("final_validation", {}),
            reference_band.shape,
        )
        png_paths["validation_holdout_blocks"] = str(validation_path)

    local_dx_fields = registration.get("local_dx_fields", []) or []
    local_dy_fields = registration.get("local_dy_fields", []) or []
    if len(local_dx_fields) > 1:
        for key, field, title in (
            ("local_displacement_dx", local_dx_fields[1], "Local displacement dx"),
            ("local_displacement_dy", local_dy_fields[1], "Local displacement dy"),
        ):
            field_path = output_path / f"{key}.png"
            _save_field_png(str(field_path), field, title)
            png_paths[key] = str(field_path)
        magnitude_path = output_path / "local_displacement_magnitude.png"
        _save_field_png(
            str(magnitude_path), np.hypot(local_dx_fields[1], local_dy_fields[1]),
            "Local displacement magnitude",
        )
        png_paths["local_displacement_magnitude"] = str(magnitude_path)
    png_paths["global_registered_overlay"] = str(global_overlay_path)
    png_paths["final_registered_overlay"] = str(final_overlay_path)

    hull_stage_csv = None
    hull_window_csv = None
    if (registration.get("hull_causal") or {}).get("available"):
        hull_stage_csv = _write_hull_causal_stage_metrics_csv(
            registration, output_dir
        )
        hull_window_csv = _write_hull_causal_window_stats_csv(
            registration, output_dir
        )

    return {
        "registered_reference": str(reference_path),
        "registered_target": str(target_path),
        "red_green_overlay": str(overlay_path),
        "registered_global_only_reference": str(global_reference_path),
        "registered_global_only_target": str(global_target_path),
        "registered_global_only_red_green_overlay": str(global_overlay_path),
        "registered_final_reference": str(final_reference_path),
        "registered_final_target": str(final_target_path),
        "registered_final_red_green_overlay": str(final_overlay_path),
        "final_holdout_validation_blocks": str(output_path / "final_holdout_validation_blocks.png"),
        "holdout_local_field_samples": holdout_local_field_csv,
        "hull_causal_stage_metrics": str(hull_stage_csv) if hull_stage_csv else None,
        "hull_causal_window_stats": str(hull_window_csv) if hull_window_csv else None,
        "diagnostic_mosaic": str(mosaic_path),
        "b14_global_overlay": str(b14_global_overlay),
        "b14_final_overlay": str(b14_final_overlay),
        "b14_diagnostic_mosaic": str(b14_mosaic),
        "registration_band_name": registration.get("registration_band_name"),
        "visualization_band": registration.get("validation_band_name", "B14"),
        "scene_ids": [scene_ids[0], scene_ids[1]],
        "mosaic_mode": mosaic_mode,
        **strict_paths,
        **png_paths,
    }


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Registration-only diagnostic")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--scene-i", type=int, default=0, help="First scene index")
    parser.add_argument("--scene-j", type=int, default=1, help="Second scene index")
    parser.add_argument("--output-dir", default=None, help="Output directory for diagnostics")
    parser.add_argument(
        "--mosaic-mode",
        choices=("weighted", "source_selection", "narrow_feather"),
        default="weighted",
        help="Diagnostic mosaic mode; source_selection is used only when explicitly selected",
    )
    parser.add_argument(
        "--validation-band", default=None,
        help="Fixed band used only for independent validation and visual artifacts",
    )
    parser.add_argument(
        "--export-holdout-manifest", default=None,
        help="Write the exact reserved HOLDOUT windows used by this run",
    )
    parser.add_argument(
        "--holdout-manifest", default=None,
        help="Reuse an existing reserved HOLDOUT manifest without resampling",
    )
    parser.add_argument(
        "--hull-causal-test",
        action="store_true",
        help=(
            "Run diagnostic-only legacy-vs-strict hull RBF counterfactual; "
            "does not change production model selection."
        ),
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    config = load_config(args.config)

    if not (0 <= args.scene_i < len(config.scenes)):
        raise IndexError(f"scene-i {args.scene_i} is outside the configured scene list")
    if not (0 <= args.scene_j < len(config.scenes)):
        raise IndexError(f"scene-j {args.scene_j} is outside the configured scene list")
    if args.scene_i == args.scene_j:
        raise ValueError("scene-i and scene-j must select different scenes")

    selected = [config.scenes[args.scene_i], config.scenes[args.scene_j]]
    scene_ids = [
        scene.get("id", f"scene_{index}")
        for index, scene in zip((args.scene_i, args.scene_j), selected)
    ]

    # The diagnostic pair is reindexed to [0, 1]; its first scene must be the
    # control scene before MultibandPipeline resolves control_idx.
    config.control_scene = scene_ids[0]
    config.scenes = selected
    config.output_root = args.output_dir or os.path.join(
        config.output_root, "registration_diagnostic"
    )
    output_dir = Path(config.output_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Registration diagnostic: %s -> %s", scene_ids[0], scene_ids[1])
    pipeline = MultibandPipeline(config)
    holdout_overrides = None
    if args.holdout_manifest:
        holdout_overrides = manifest_to_pair_overrides(
            load_holdout_manifest(args.holdout_manifest), scene_ids
        )
    scene_data = pipeline.load_scenes()
    overlaps = pipeline.detect_overlaps(scene_data)
    logger.info("Detected %d overlap pair(s)", len(overlaps))
    if not overlaps:
        reason = "no geographic overlap detected between selected scenes"
        logger.error("%s", reason)
        registration = _build_no_overlap_registration(
            scene_ids,
            reason,
            local_enabled=(getattr(config, "registration_params", {}) or {}).get(
                "enable_local_refinement", True
            ),
        )
        build_diagnostic_payload(registration, scene_ids, output_dir)
        return 1

    try:
        registration = pipeline.register_scenes(
            scene_data,
            overlaps,
            registration_band_idx=pipeline.registration_band_idx,
            hull_causal_diagnostic=args.hull_causal_test,
            diagnostic_validation_band=args.validation_band,
            holdout_reservation_overrides=holdout_overrides,
        )
    except TypeError as exc:
        # Keep compatibility with lightweight external pipeline doubles that
        # predate the keyword-only diagnostic extension. Real pipeline calls
        # accept both keywords and do not take this path.
        if "unexpected keyword argument" not in str(exc):
            raise
        registration = pipeline.register_scenes(scene_data, overlaps)
    if args.export_holdout_manifest:
        manifest_registration = dict(registration)
        manifest_registration["scene_ids"] = scene_ids
        try:
            source_commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            source_commit = "unknown"
        manifest = build_holdout_manifest(
            manifest_registration, source_commit=source_commit,
            source_registration_band=registration.get(
                "registration_band_name", getattr(config, "registration_band", "B14")
            ), validation_band=registration.get(
                "validation_band_name", args.validation_band or getattr(config, "registration_band", "B14")
            ),
        )
        manifest_path = Path(args.export_holdout_manifest)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(_json_safe(manifest), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    quality = registration.get("quality", {}) or {}
    required_quality = (getattr(config, "registration_params", {}) or {}).get(
        "required_quality", "pass"
    )
    connected = bool(registration.get("connected", False))
    quality_name = str(quality.get("quality", "unknown")).lower()
    quality_ok = (
        connected
        and quality_name != "fail"
        and registration_quality_meets_requirement(quality, required_quality)
    )
    status_failed = str(registration.get("status", "")).lower() == "fail"
    if status_failed or not quality_ok:
        failure = registration.get("failure", {}) or {}
        if not failure:
            if not connected:
                code = "registration_connectivity_failed"
                reason = "registration graph is not connected"
            else:
                code = "registration_quality_failed"
                if quality_name == "fail":
                    reason = "final registration quality is classified as fail"
                else:
                    reason = (
                        f"final registration quality {quality.get('quality', 'unknown')!r} "
                        f"is below required quality {required_quality!r}"
                    )
            failure = {"code": code, "reason": reason}
        blocked = bool((registration.get("diagnostics") or {}).get(
            "registration_blocked"
        ))
        has_registered = (
            registration.get("registered_arrays") is not None
            and len(registration.get("registered_arrays", [])) >= 2
        )
        if connected and not blocked and has_registered and quality_name == "fail":
            artifacts = write_diagnostic_artifacts(
                registration, scene_data, scene_ids, output_dir,
                registration_band_idx=pipeline.registration_band_idx,
                visualization_band_idx=getattr(
                    pipeline, "common_bands", [registration.get(
                        "validation_band_name", getattr(config, "registration_band", "B14")
                    )]
                ).index(
                        registration.get("validation_band_name", getattr(config, "registration_band", "B14"))
                ),
                mosaic_mode=args.mosaic_mode,
                allow_quality_fail_for_diagnostics=True,
            )
        else:
            artifacts = {}
        registration_for_payload = {
            **registration,
            "status": "fail",
            "failure": failure,
            "diagnostic_artifacts": artifacts,
        }
        build_diagnostic_payload(registration_for_payload, scene_ids, output_dir)
        _log_stage_validation_summary(registration)
        _log_hull_causal_summary(registration)
        logger.error(
            "Registration diagnostic failed: connected=%s, quality=%s, required=%s",
            connected, quality.get("quality", "unknown"), required_quality,
        )
        return 1

    artifacts = write_diagnostic_artifacts(
        registration, scene_data, scene_ids, output_dir,
        registration_band_idx=pipeline.registration_band_idx,
        visualization_band_idx=getattr(
            pipeline, "common_bands", [registration.get(
                "validation_band_name", getattr(config, "registration_band", "B14")
            )]
        ).index(
            registration.get("validation_band_name", getattr(config, "registration_band", "B14"))
        ),
        mosaic_mode=args.mosaic_mode,
    )
    registration_for_payload = {
        **registration,
        "status": registration.get("status") or "pass",
        "failure": registration.get("failure", {}) or {},
        "diagnostic_artifacts": artifacts,
    }
    build_diagnostic_payload(registration_for_payload, scene_ids, output_dir)
    _log_stage_validation_summary(registration)
    _log_hull_causal_summary(registration)

    logger.info("Connected: %s", registration.get("connected", False))
    logger.info("Quality: %s", quality.get("quality", "unknown"))
    logger.info("RMSE: %s px; P95: %s px; confidence: %s",
                quality.get("rmse", "N/A"), quality.get("p95", "N/A"),
                quality.get("confidence", "N/A"))
    logger.info("Wrote registration diagnostics to %s", output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
