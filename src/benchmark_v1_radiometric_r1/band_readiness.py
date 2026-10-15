from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _read(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def summarize_v_bands(inventory_csv: Path, scene_inventory_csv: Path, identity_manifest: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Summarize existing metadata and reuse only hashes already frozen in an identity manifest."""
    rows = _read(inventory_csv)
    scenes = _read(scene_inventory_csv)
    identity = _read(identity_manifest)
    scene_ids = {str(row["scene_id"]) for row in scenes}
    by_scene_bands: dict[str, set[str]] = defaultdict(set)
    seen: set[tuple[str, str]] = set()
    hashes: dict[tuple[str, str], str] = {}
    for row in identity:
        band = str(row.get("band", "")).upper().removeprefix("B")
        value = str(row.get("source_file_sha256", ""))
        key = (str(row.get("scene_id", "")), band)
        if value:
            hashes[key] = value

    output = []
    wavelengths = 0
    units = Counter()
    risk_scenes = set()
    scene_signature_sets: dict[str, set[tuple[str, ...]]] = defaultdict(set)
    for source in rows:
        row = dict(source)
        scene_id = str(row.get("scene_id", ""))
        band = str(row.get("band", ""))
        key = (scene_id, band)
        if scene_id not in scene_ids:
            raise ValueError(f"band inventory references unknown scene: {scene_id}")
        if key in seen:
            raise ValueError(f"duplicate scene/band record: {key}")
        seen.add(key); by_scene_bands[scene_id].add(band)
        center = str(row.get("wavelength_center_nm", "")).strip()
        width = str(row.get("bandwidth_nm", "")).strip()
        row["wavelength_status"] = "METADATA_PRESENT" if center else "UNKNOWN"
        row["bandwidth_status"] = "METADATA_PRESENT" if width else "UNKNOWN"
        wavelengths += bool(center)
        unit = str(row.get("data_unit", "UNKNOWN")) or "UNKNOWN"
        row["data_unit"] = unit; units[unit] += 1
        if row.get("quality_status_tag") == "RISK":
            risk_scenes.add(scene_id)
        signature = tuple(str(row.get(k, "")) for k in (
            "crs", "pixel_size_x", "pixel_size_y", "width_px", "height_px",
            "affine_a", "affine_b", "affine_c", "affine_d", "affine_e", "affine_f",
        ))
        scene_signature_sets[scene_id].add(signature)
        product = str(row.get("product_id", ""))
        digest = hashes.get((product, band), hashes.get(key, ""))
        row["source_file_sha256"] = digest
        row["sha256_status"] = "FROZEN_IN_EXISTING_MANIFEST" if digest else "NOT_FROZEN_NO_FULL_FILE_HASH"
        output.append(row)

    for row in output:
        row["same_scene_grid_status"] = "METADATA_GRID_MATCH" if len(scene_signature_sets[row["scene_id"]]) == 1 else "GRID_MISMATCH_WITHIN_SCENE"

    missing = {scene: [str(b) for b in range(1, 17) if str(b) not in by_scene_bands.get(scene, set())] for scene in sorted(scene_ids)}
    missing = {scene: bands for scene, bands in missing.items() if bands}
    complete_count = len(scene_ids) - len(missing)
    mismatched_scenes = sorted(scene for scene, values in scene_signature_sets.items() if len(values) > 1)
    b9_scene_signatures = {tuple(str(r.get(k, "")) for k in ("crs", "pixel_size_x", "pixel_size_y", "width_px", "height_px", "affine_a", "affine_b", "affine_c", "affine_d", "affine_e", "affine_f")) for r in rows if str(r.get("band")) == "9"}
    summary = {
        "scene_count": len(scene_ids), "band_rows": len(output), "expected_band_count_per_scene": 16,
        "complete_scene_count": complete_count, "missing_bands": missing,
        "wavelength_metadata_rows": int(wavelengths), "data_units": dict(sorted(units.items())),
        "quality_risk_scene_count": len(risk_scenes), "quality_risk_scenes": sorted(risk_scenes),
        "within_scene_grid_mismatch_scenes": mismatched_scenes,
        "cross_scene_b9_grid_uniform": len(b9_scene_signatures) == 1,
        "cross_scene_b9_grid_signature_count": len(b9_scene_signatures),
        "sha256_available_rows": sum(bool(r["source_file_sha256"]) for r in output),
        "sha256_missing_rows": sum(not r["source_file_sha256"] for r in output),
        "sha256_scope_note": "No new whole-image hashing was performed; only hashes already present in the identity manifest are copied.",
        "same_affine_is_not_subpixel_coregistration_proof": True,
    }
    return output, summary
