"""Blockwise audit that hard-source mosaic pixels come from their tagged scene."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

from .freeze import verify_lock


def _resolve(path, base):
    if path is None or not str(path).strip():
        return None
    candidate = Path(path)
    return candidate if candidate.is_absolute() else (base / candidate).resolve()


def _windows(height, width, block_shape):
    block_h, block_w = map(int, block_shape)
    if block_h <= 0 or block_w <= 0:
        raise ValueError("INVALID_BLOCK_SHAPE")
    for row in range(0, height, block_h):
        for col in range(0, width, block_w):
            yield Window(col, row, min(block_w, width - col), min(block_h, height - row))


def _same_grid(dataset, reference):
    return (dataset.width == reference.width and dataset.height == reference.height
            and dataset.transform == reference.transform and dataset.crs == reference.crs)


def _value_tolerance(source_dtype, mosaic_dtype, values):
    source_type = np.dtype(source_dtype)
    mosaic_type = np.dtype(mosaic_dtype)
    if source_type.kind not in "fc" and mosaic_type.kind not in "fc":
        return 0.0
    eps = max(np.finfo(t).eps for t in (source_type, mosaic_type) if t.kind in "fc")
    scale = max(1.0, float(np.max(np.abs(values))) if values.size else 1.0)
    return float(2 * eps * scale)


def audit_mosaic_provenance(submission, freeze_root, *, submission_dir=None, block_shape=(512, 512),
                            freeze_verifier=None):
    """Verify HARD source labels and pixel values against frozen inputs and outputs.

    SOFT blending has no single source value to compare and is deliberately not
    inferred from a hard-looking label map.
    """
    if submission.get("seam_type", "NONE") == "SOFT":
        return {"status": "NOT_EVALUABLE_SOFT_PROVENANCE", "checked_pixels": 0,
                "mismatch_pixels": 0, "invalid_label_pixels": 0,
                "invalid_source_support_pixels": 0}

    freeze = Path(freeze_root).resolve()
    base = Path(submission_dir or ".").resolve()
    lock_result = (freeze_verifier or verify_lock)(freeze)
    if not lock_result.valid:
        return {"status": "FREEZE_INVALID", "errors": list(lock_result.errors),
                "checked_pixels": 0, "mismatch_pixels": 0}
    if submission.get("seam_type", "NONE") != "HARD":
        return {"status": "NOT_EVALUABLE_NO_HARD_SOURCE_LABELS", "checked_pixels": 0,
                "mismatch_pixels": 0}

    pair_path = freeze / "pair_overlap_inventory.csv"
    frozen = json.loads((freeze / "frozen_manifest.json").read_text(encoding="utf-8"))
    with pair_path.open("r", newline="", encoding="utf-8-sig") as stream:
        pairs = list(csv.DictReader(stream))
    scenes = []
    for pair in pairs:
        for key in ("scene_i", "scene_j"):
            if pair[key] not in scenes:
                scenes.append(pair[key])
    corrected_paths = submission.get("corrected_scene_paths") or {}
    if set(corrected_paths) != set(scenes):
        return {"status": "SCENE_OUTPUT_IDENTITY_MISMATCH", "checked_pixels": 0,
                "mismatch_pixels": 0,
                "missing_scene_ids": sorted(set(scenes) - set(corrected_paths)),
                "extra_scene_ids": sorted(set(corrected_paths) - set(scenes))}

    mosaic_path = _resolve(submission.get("output_mosaic_path"), base)
    mask_path = _resolve(submission.get("output_mosaic_mask_path"), base)
    labels_path = _resolve(submission.get("source_label_path"), base)
    if not all(path and path.is_file() for path in (mosaic_path, mask_path, labels_path)):
        return {"status": "MISSING_ARTIFACT", "checked_pixels": 0, "mismatch_pixels": 0,
                "missing_paths": [str(path) for path in (mosaic_path, mask_path, labels_path)
                                  if not path or not path.is_file()]}

    counts = {"checked_pixels": 0, "mismatch_pixels": 0,
              "invalid_label_pixels": 0, "invalid_source_support_pixels": 0}
    max_abs_error = 0.0
    max_tolerance = 0.0
    try:
        with rasterio.open(mosaic_path) as mosaic, rasterio.open(mask_path) as explicit_mask, rasterio.open(labels_path) as labels:
            for ds, name in ((explicit_mask, "mosaic_mask"), (labels, "source_labels")):
                if not _same_grid(ds, mosaic):
                    return {"status": "GRID_MISMATCH", "artifact": name, **counts}
            if mosaic.count != 1 or explicit_mask.count != 1 or labels.count != 1:
                return {"status": "INVALID_BAND_COUNT", **counts}
            input_datasets = []
            output_datasets = []
            try:
                for index, scene in enumerate(scenes):
                    input_path = freeze / "aligned_inputs" / f"scene_{index:03d}_B9_input.tif"
                    mask_input_path = freeze / "aligned_inputs" / f"scene_{index:03d}_input_mask.tif"
                    output_path = _resolve(corrected_paths[scene], base)
                    if not input_path.is_file() or not mask_input_path.is_file() or not output_path or not output_path.is_file():
                        return {"status": "MISSING_ARTIFACT", "scene_id": scene, **counts}
                    input_ds = rasterio.open(input_path)
                    mask_ds = rasterio.open(mask_input_path)
                    output_ds = rasterio.open(output_path)
                    for ds in (input_ds, mask_ds, output_ds):
                        if not _same_grid(ds, mosaic) or ds.count != 1:
                            ds.close()
                            return {"status": "GRID_MISMATCH", "scene_id": scene, **counts}
                    input_datasets.append((input_ds, mask_ds))
                    output_datasets.append(output_ds)
                for window in _windows(mosaic.height, mosaic.width, block_shape):
                    m = mosaic.read(1, window=window)
                    m_valid = (mosaic.read_masks(1, window=window) > 0) & (explicit_mask.read(1, window=window) > 0)
                    m_valid &= np.isfinite(m)
                    source_ids = labels.read(1, window=window)
                    label_valid = labels.read_masks(1, window=window) > 0
                    bad_label = m_valid & (~label_valid | (source_ids < 0) | (source_ids >= len(scenes)))
                    counts["invalid_label_pixels"] += int(bad_label.sum())
                    for source_id, ((_, frozen_mask), output_ds) in enumerate(zip(input_datasets, output_datasets)):
                        selected = m_valid & label_valid & (source_ids == source_id)
                        if not selected.any():
                            continue
                        raw_support = frozen_mask.read(1, window=window) > 0
                        output = output_ds.read(1, window=window)
                        output_support = (output_ds.read_masks(1, window=window) > 0) & np.isfinite(output)
                        unsupported = selected & (~raw_support | ~output_support)
                        counts["invalid_source_support_pixels"] += int(unsupported.sum())
                        valid_selected = selected & raw_support & output_support
                        if valid_selected.any():
                            src_values = output[valid_selected].astype(np.float64)
                            mosaic_values = m[valid_selected].astype(np.float64)
                            tolerance = _value_tolerance(output_ds.dtypes[0], mosaic.dtypes[0], src_values)
                            differences = np.abs(mosaic_values - src_values)
                            counts["checked_pixels"] += int(valid_selected.sum())
                            counts["mismatch_pixels"] += int(np.count_nonzero(differences > tolerance))
                            max_abs_error = max(max_abs_error, float(differences.max(initial=0.0)))
                            max_tolerance = max(max_tolerance, tolerance)
            finally:
                for input_ds, mask_ds in input_datasets:
                    input_ds.close(); mask_ds.close()
                for output_ds in output_datasets:
                    output_ds.close()
    except (OSError, ValueError) as exc:
        return {"status": "INVALID_ARTIFACT", "error": f"{type(exc).__name__}:{exc}", **counts}

    passed = (counts["checked_pixels"] > 0 and counts["mismatch_pixels"] == 0
              and counts["invalid_label_pixels"] == 0
              and counts["invalid_source_support_pixels"] == 0)
    return {"status": "PASS" if passed else "INVALID_SOURCE_VALUE_PROVENANCE",
            **counts, "max_abs_error": max_abs_error,
            "max_allowed_tolerance": max_tolerance,
            "value_comparison": "HARD_SOURCE_LABEL_TO_CORRECTED_SCENE"}
