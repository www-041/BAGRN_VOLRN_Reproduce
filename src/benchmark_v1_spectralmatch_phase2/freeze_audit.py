from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path

import rasterio

from src.benchmark_v1_unified.freeze import verify_lock


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_head() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def audit_frozen_inputs(freeze_root, *, expected_head=None, check_rasters=True) -> dict:
    """Read-only audit of the frozen Phase-1 artifacts used by the smoke adapter."""
    root = Path(freeze_root).resolve()
    errors: list[str] = []
    lock = verify_lock(root)
    if not lock.valid:
        errors.extend(f"LOCK:{error}" for error in lock.errors)

    manifest_path = root / "frozen_manifest.json"
    code_manifest_path = root / "code_manifest.json"
    inventory_path = root / "pair_overlap_inventory.csv"
    try:
        frozen = json.loads(manifest_path.read_text(encoding="utf-8"))
        code_manifest = json.loads(code_manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"status": "FREEZE_INVALID", "errors": [*errors, f"MANIFEST_READ:{exc}"]}

    if frozen.get("status") != "FROZEN":
        errors.append("MANIFEST_NOT_FROZEN")
    if frozen.get("protocol_id") != "DZ01V_UNIFIED_A_B_V1_20261013":
        errors.append("PROTOCOL_ID_MISMATCH")
    if int(frozen.get("scene_count", -1)) != 13:
        errors.append("SCENE_COUNT_MISMATCH")

    head = _git_head()
    locked_head = frozen.get("git_head")
    current_head = expected_head if expected_head is not None else head
    if locked_head and current_head != locked_head:
        errors.append("GIT_HEAD_MISMATCH")
    if code_manifest.get("git_head") != locked_head:
        errors.append("CODE_MANIFEST_HEAD_MISMATCH")
    if frozen.get("code_manifest_sha256") and frozen.get("code_manifest_sha256") != _sha256(code_manifest_path):
        errors.append("CODE_MANIFEST_SHA256_MISMATCH")

    code_files = code_manifest.get("files", [])
    code_mismatches = []
    for item in code_files:
        path = Path(item["path"])
        if not path.is_absolute():
            path = Path.cwd() / path
        if not path.is_file() or _sha256(path) != item.get("sha256"):
            code_mismatches.append(item.get("path", "UNKNOWN"))
    if code_mismatches:
        errors.append(f"CODE_FILE_HASH_MISMATCH:{len(code_mismatches)}")

    pair_rows = []
    try:
        with inventory_path.open(encoding="utf-8-sig", newline="") as stream:
            pair_rows = list(csv.DictReader(stream))
    except OSError as exc:
        errors.append(f"PAIR_INVENTORY_READ:{exc}")
    positive = [row for row in pair_rows if row.get("status") == "OVERLAP" and int(row.get("aligned_valid_overlap_pixels") or 0) > 0]
    zero = [row for row in pair_rows if row.get("status") == "NO_VALID_OVERLAP" and int(row.get("aligned_valid_overlap_pixels") or 0) == 0]
    expected_counts = frozen.get("pair_counts", {})
    observed = {"total": len(pair_rows), "positive_overlap": len(positive), "no_valid_overlap": len(zero)}
    if observed != {
        "total": int(expected_counts.get("total_pairs", -1)),
        "positive_overlap": int(expected_counts.get("positive_overlap_pairs", -1)),
        "no_valid_overlap": int(expected_counts.get("no_valid_overlap_pairs", -1)),
    }:
        errors.append("PAIR_INVENTORY_COUNT_MISMATCH")

    requested_pairs = {}
    for pair_id in ("00_01", "00_10"):
        row = next((r for r in positive if r.get("pair_id") == pair_id), None)
        if row is None:
            errors.append(f"REQUESTED_PAIR_NOT_POSITIVE:{pair_id}")
        else:
            requested_pairs[pair_id] = {
                "scene_i": row["scene_i"], "scene_j": row["scene_j"],
                "aligned_valid_overlap_pixels": int(row["aligned_valid_overlap_pixels"]),
                "days_apart": float(row["days_apart"]) if row.get("days_apart") else None,
                "time_stratum": row.get("time_stratum"),
            }

    raster_audit = []
    if check_rasters:
        grid = frozen.get("grid", {})
        aligned_dir = root / "aligned_inputs"
        transform = tuple(grid.get("transform", []))
        for i in range(13):
            image = aligned_dir / f"scene_{i:03d}_B9_input.tif"
            mask = aligned_dir / f"scene_{i:03d}_input_mask.tif"
            for path in (image, mask):
                if not path.is_file():
                    errors.append(f"MISSING_FROZEN_RASTER:{path.name}")
            if not image.is_file() or not mask.is_file():
                continue
            try:
                with rasterio.open(image) as src, rasterio.open(mask) as msk:
                    if (src.width, src.height) != (grid.get("width"), grid.get("height")):
                        errors.append(f"GRID_SHAPE_MISMATCH:{image.name}")
                    if src.crs is None or src.crs.to_string() != grid.get("crs"):
                        errors.append(f"GRID_CRS_MISMATCH:{image.name}")
                    if tuple(src.transform)[:6] != transform[:6] or tuple(msk.transform)[:6] != transform[:6]:
                        errors.append(f"GRID_TRANSFORM_MISMATCH:{image.name}")
                    if src.count != 1 or src.dtypes[0] != "float32":
                        errors.append(f"INPUT_DTYPE_MISMATCH:{image.name}")
                    if (src.width, src.height, src.transform, src.crs) != (msk.width, msk.height, msk.transform, msk.crs):
                        errors.append(f"MASK_GRID_MISMATCH:{image.name}")
                raster_audit.append({"scene_index": i, "input": str(image), "mask": str(mask), "status": "PASS"})
            except Exception as exc:  # record damaged/unreadable frozen inputs without modifying them
                errors.append(f"RASTER_READ:{image.name}:{type(exc).__name__}:{exc}")

    return {
        "status": "PASS" if not errors else "FREEZE_INVALID",
        "errors": errors,
        "freeze_root": str(root),
        "protocol_id": frozen.get("protocol_id"),
        "locked_git_head": locked_head,
        "current_git_head": current_head,
        "code_manifest_file_count": len(code_files),
        "code_manifest_hash_mismatch_count": len(code_mismatches),
        "pair_counts": observed,
        "positive_pair_ids": sorted(row["pair_id"] for row in positive),
        "requested_pairs": requested_pairs,
        "raster_count_checked": len(raster_audit),
        "rasters": raster_audit,
        "input_source": "freeze_root/aligned_inputs only; Stage06 outputs are not referenced",
    }
