"""Read-only access to the frozen Unified Benchmark B9 grid."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import json

import numpy as np
import rasterio
from affine import Affine
from rasterio.windows import Window

from src.benchmark_v1_unified.freeze import verify_lock
from src.benchmark_v1_unified.identity import sha256_file


@dataclass(frozen=True)
class CropSpec:
    pair_id: str
    anchor_window_id: str
    row_off: int
    col_off: int
    height: int
    width: int


@dataclass(frozen=True)
class RasterCrop:
    values: np.ndarray
    valid: np.ndarray
    transform: Affine
    crs: str
    scene_index: int


def verify_frozen_b9(freeze_root: Path) -> dict:
    """Verify the immutable lock plus the 38 locked-code manifest entries."""
    root = Path(freeze_root)
    lock = verify_lock(root)
    errors = list(lock.errors)
    manifest_path = root / "code_manifest.json"
    code_count = 0
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        entries = manifest.get("files", []) if isinstance(manifest, dict) else manifest
        code_count = len(entries)
        if code_count != 38:
            errors.append(f"CODE_MANIFEST_COUNT:{code_count}")
        # The freeze directory can be passed as a repository-relative path.
        # Resolve repository-relative manifest entries from this source module,
        # not from the runtime spelling/depth of the freeze path.
        repo = Path(__file__).resolve().parents[2]
        for entry in entries:
            rel = Path(entry["path"])
            path = rel if rel.is_absolute() else repo / rel
            if not path.is_file():
                errors.append(f"MISSING_LOCKED_CODE:{entry['path']}")
            elif sha256_file(path) != entry["sha256"]:
                errors.append(f"LOCKED_CODE_HASH_MISMATCH:{entry['path']}")
    except Exception as exc:
        errors.append(f"CODE_MANIFEST_ERROR:{type(exc).__name__}:{exc}")
    return {"status": "PASS" if not errors else "BLOCKED_FREEZE", "lock_valid": lock.valid,
            "code_manifest_count": code_count, "errors": errors}


def select_pair_crop(freeze_root: Path, pair_id: str, side_px: int = 1024) -> CropSpec:
    """Center the fixed crop on the first frozen B2 window for the pair."""
    if pair_id not in {"00_01", "00_10"}:
        raise ValueError(f"unsupported diagnostic pair: {pair_id}")
    root = Path(freeze_root)
    with (root / "fixed_windows.csv").open(encoding="utf-8-sig", newline="") as stream:
        row = next((r for r in csv.DictReader(stream) if r["pair_id"] == pair_id), None)
    if row is None:
        raise ValueError(f"NO_FROZEN_B2_ANCHOR:{pair_id}")
    anchor = Window(int(row["col_off"]), int(row["row_off"]), int(row["width"]), int(row["height"]))
    reference_index = int(pair_id[:2])
    ref_path = root / "aligned_inputs" / f"scene_{reference_index:03d}_B9_input.tif"
    with rasterio.open(ref_path) as src:
        if side_px <= 0 or side_px > src.width or side_px > src.height:
            raise ValueError("INVALID_CROP_SIZE")
        row_center = int(anchor.row_off) + int(anchor.height) // 2
        col_center = int(anchor.col_off) + int(anchor.width) // 2
        row_off = min(max(0, row_center - side_px // 2), src.height - side_px)
        col_off = min(max(0, col_center - side_px // 2), src.width - side_px)
    return CropSpec(pair_id, row["window_id"], row_off, col_off, side_px, side_px)


def read_frozen_crop(freeze_root: Path, scene_index: int, spec: CropSpec) -> RasterCrop:
    """Read frozen float pixels and the separately frozen mask without resampling."""
    root = Path(freeze_root) / "aligned_inputs"
    image_path = root / f"scene_{scene_index:03d}_B9_input.tif"
    mask_path = root / f"scene_{scene_index:03d}_input_mask.tif"
    window = Window(spec.col_off, spec.row_off, spec.width, spec.height)
    with rasterio.open(image_path) as ds, rasterio.open(mask_path) as mask_ds:
        values = ds.read(1, window=window).astype(np.float64, copy=False)
        valid = mask_ds.read(1, window=window).astype(bool, copy=False)
        crop = RasterCrop(values.copy(), valid.copy(), ds.window_transform(window),
                          ds.crs.to_string() if ds.crs else "", scene_index)
    if crop.values.shape != (spec.height, spec.width) or crop.valid.shape != crop.values.shape:
        raise ValueError("FROZEN_CROP_SHAPE_MISMATCH")
    if np.any(crop.valid & ~np.isfinite(crop.values)):
        raise ValueError("NONFINITE_VALID_INPUT")
    if np.any(crop.valid & (crop.values < 0)):
        raise ValueError("NEGATIVE_VALID_INPUT_BLOCKED")
    return crop


def assert_same_crop_grid(ref: RasterCrop, sub: RasterCrop) -> None:
    if ref.values.shape != sub.values.shape or ref.transform != sub.transform or ref.crs != sub.crs:
        raise ValueError("INPUT_CROP_GRID_MISMATCH")
