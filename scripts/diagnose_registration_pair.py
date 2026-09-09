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
import logging
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


def _save_validation_holdout_png(path, validation, shape):
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
    ax.set_title("Final holdout validation blocks")
    ax.set_xlabel("reference pixel x")
    ax.set_ylabel("reference pixel y")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_diagnostic_artifacts(registration, scene_data, scene_ids, output_dir,
                               registration_band_idx=0, mosaic_mode="weighted"):
    """Write registered images, a red-green overlay, and a diagnostic mosaic."""
    quality = registration.get("quality", {}) or {}
    if (
        str(registration.get("status", "")).lower() == "fail"
        or registration.get("connected") is False
        or quality.get("quality") == "fail"
    ):
        raise ValueError("cannot write registered artifacts for a failed registration")
    registered = registration.get("registered_arrays")
    if not registered or len(registered) < 2:
        raise ValueError("registration result does not contain two registered arrays")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    transforms = scene_data["transforms"]
    nodatas = scene_data["nodata_values"]
    crs = scene_data["crs"]

    reference_path = output_path / "registered_reference.tif"
    target_path = output_path / "registered_target.tif"
    write_geotiff(str(reference_path), registered[0], transforms[0], crs,
                  nodata=nodatas[0], dtype="float32")
    write_geotiff(str(target_path), registered[1], transforms[1], crs,
                  nodata=nodatas[1], dtype="float32")

    reference_band = _registration_band(registered[0], registration_band_idx)
    target_band = _registration_band(registered[1], registration_band_idx)
    target_on_reference = _reproject_to_reference(
        target_band, transforms[1], crs, nodatas[1], reference_band.shape,
        transforms[0], crs,
    )
    reference_green = _stretch_for_overlay(reference_band, nodatas[0])
    target_red = _stretch_for_overlay(target_on_reference, None)
    overlay = np.stack([target_red, reference_green, np.zeros_like(reference_green)], axis=0)
    overlay_path = output_path / "registered_red_green_overlay.tif"
    write_geotiff(str(overlay_path), overlay, transforms[0], crs,
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

    diagnostics = registration.get("diagnostics", {}) or {}
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
        validation_path = output_path / "validation_holdout_blocks.png"
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
    png_paths["global_registered_overlay"] = str(overlay_path)
    png_paths["final_registered_overlay"] = str(overlay_path)

    return {
        "registered_reference": str(reference_path),
        "registered_target": str(target_path),
        "red_green_overlay": str(overlay_path),
        "diagnostic_mosaic": str(mosaic_path),
        "scene_ids": [scene_ids[0], scene_ids[1]],
        "mosaic_mode": mosaic_mode,
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

    registration = pipeline.register_scenes(scene_data, overlaps)
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
        registration_for_payload = {
            **registration,
            "status": "fail",
            "failure": failure,
            "diagnostic_artifacts": {},
        }
        build_diagnostic_payload(registration_for_payload, scene_ids, output_dir)
        logger.error(
            "Registration diagnostic failed: connected=%s, quality=%s, required=%s",
            connected, quality.get("quality", "unknown"), required_quality,
        )
        return 1

    artifacts = write_diagnostic_artifacts(
        registration, scene_data, scene_ids, output_dir,
        registration_band_idx=pipeline.registration_band_idx,
        mosaic_mode=args.mosaic_mode,
    )
    registration_for_payload = {
        **registration,
        "status": registration.get("status") or "pass",
        "failure": registration.get("failure", {}) or {},
        "diagnostic_artifacts": artifacts,
    }
    build_diagnostic_payload(registration_for_payload, scene_ids, output_dir)

    logger.info("Connected: %s", registration.get("connected", False))
    logger.info("Quality: %s", quality.get("quality", "unknown"))
    logger.info("RMSE: %s px; P95: %s px; confidence: %s",
                quality.get("rmse", "N/A"), quality.get("p95", "N/A"),
                quality.get("confidence", "N/A"))
    logger.info("Wrote registration diagnostics to %s", output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
