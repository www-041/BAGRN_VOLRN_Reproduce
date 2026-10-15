from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import rasterio


ALLOWED_PAIRS = frozenset({"00_01", "02_03", "05_07", "08_10"})
BANDS = tuple(f"B{i}" for i in range(1, 17))


@dataclass(frozen=True)
class BandInput:
    band: str
    path: Path | None
    usable: bool
    exclusion_reason: str
    data_unit: str
    wavelength_center_nm: float | None
    manifest_sha256: str
    header_fingerprint: dict[str, Any]


@dataclass(frozen=True)
class SceneInputs:
    scene_id: str
    product_id: str
    acquisition_start: str
    quality_status: str
    bands: dict[str, BandInput]


@dataclass(frozen=True)
class PairInputs:
    pair_id: str
    scenes: tuple[SceneInputs, SceneInputs]
    valid_overlap_pixels: int
    days_apart: float | None
    source_inventory_paths: tuple[Path, ...]


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise ValueError(f"required inventory missing: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _index_unique(rows: list[dict[str, str]], key: str, label: str) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        value = str(row.get(key, "")).strip()
        if not value:
            raise ValueError(f"{label} has empty {key}")
        if value in result:
            raise ValueError(f"duplicate {label} {key}: {value}")
        result[value] = row
    return result


def _float_or_none(value: str) -> float | None:
    try:
        parsed = float(value)
        return parsed if parsed == parsed else None
    except (TypeError, ValueError):
        return None


def _header(path: Path) -> dict[str, Any]:
    """Read raster metadata only; never read pixel arrays or masks."""
    with rasterio.open(path) as ds:
        transform = tuple(float(v) for v in ds.transform)[:6]
        crs = ds.crs.to_string() if ds.crs else "UNKNOWN"
        stat = path.stat()
        payload = {
            "width": int(ds.width), "height": int(ds.height), "count": int(ds.count),
            "dtype": tuple(ds.dtypes), "nodata": ds.nodata, "crs": crs,
            "transform": transform, "bounds": tuple(float(v) for v in ds.bounds),
            "file_size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
        }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    payload["fingerprint_sha256"] = hashlib.sha256(canonical).hexdigest()
    return payload


def load_pair_inputs(pair_id: str, repo_root: str | Path) -> PairInputs:
    """Load an allowlisted pair via scene IDs, then join inventory rows by product and band.

    Missing auxiliary bands/files are retained as unavailable records. B9 is mandatory.
    Validation reads only CSV metadata and GeoTIFF headers, not image pixels.
    """
    if pair_id not in ALLOWED_PAIRS:
        raise ValueError(f"pair {pair_id!r} is not in frozen R1.5 pair allowlist")
    root = Path(repo_root).resolve()
    base = root / "docs" / "benchmark" / "v1"
    scene_path = base / "scene_inventory.csv"
    bands_path = base / "v_band_inventory.csv"
    pairs_path = base / "temporal_pair_inventory.csv"
    manifest_path = base / "phase2" / "input_identity_manifest.csv"
    pair_rows = [r for r in _read_csv(pairs_path) if r.get("pair_id") == pair_id]
    if len(pair_rows) != 1:
        raise ValueError(f"expected exactly one temporal pair row for {pair_id}, found {len(pair_rows)}")
    pair_row = pair_rows[0]
    try:
        overlap = int(float(pair_row.get("valid_overlap_pixels", "0")))
    except ValueError:
        overlap = 0
    if overlap <= 0:
        raise ValueError(f"pair {pair_id} has no verified B9 overlap")

    scenes_by_id = _index_unique(_read_csv(scene_path), "scene_id", "scene inventory")
    band_rows = _read_csv(bands_path)
    manifest_rows = _read_csv(manifest_path)
    manifest_by_key: dict[tuple[str, str], dict[str, str]] = {}
    for row in manifest_rows:
        key = (str(row.get("scene_id", "")), str(row.get("band", "")))
        if key in manifest_by_key:
            raise ValueError(f"duplicate frozen input manifest row: {key}")
        manifest_by_key[key] = row

    result_scenes: list[SceneInputs] = []
    for suffix in ("i", "j"):
        scene_id = str(pair_row.get(f"scene_id_{suffix}", ""))
        scene = scenes_by_id.get(scene_id)
        if not scene:
            raise ValueError(f"pair {pair_id} references unknown scene_id {scene_id}")
        product_id = str(scene.get("product_id", ""))
        rows_by_band: dict[str, dict[str, str]] = {}
        for row in band_rows:
            if str(row.get("scene_id", "")) == scene_id or str(row.get("product_id", "")) == product_id:
                band = str(row.get("band", "")).strip().upper()
                if band.isdigit():
                    band = f"B{int(band)}"
                if band in rows_by_band:
                    raise ValueError(f"duplicate band {band} for scene {scene_id}")
                rows_by_band[band] = row
        if "B9" not in rows_by_band:
            raise ValueError(f"B9 missing for scene {scene_id}")
        bands: dict[str, BandInput] = {}
        for band in BANDS:
            row = rows_by_band.get(band)
            if row is None:
                bands[band] = BandInput(band, None, False, "BAND_MISSING_FROM_INVENTORY", "UNKNOWN", None, "", {})
                continue
            path = Path(row.get("file_path", "")).expanduser()
            unit = str(row.get("data_unit") or "UNKNOWN")
            wavelength = _float_or_none(row.get("wavelength_center_nm", ""))
            manifest = manifest_by_key.get((product_id, band), {})
            frozen_hash = str(manifest.get("recorded_source_sha256_task15", ""))
            if not path.is_file():
                if band == "B9":
                    raise ValueError(f"B9 input file missing for scene {scene_id}: {path}")
                bands[band] = BandInput(band, path, False, "FILE_MISSING", unit, wavelength, frozen_hash, {})
                continue
            try:
                header = _header(path)
            except Exception as exc:
                if band == "B9":
                    raise ValueError(f"cannot read B9 header for scene {scene_id}: {exc}") from exc
                bands[band] = BandInput(band, path, False, f"HEADER_UNREADABLE:{type(exc).__name__}", unit, wavelength, frozen_hash, {})
                continue
            if band == "B9" and (header["crs"] != "EPSG:32650" or header["width"] <= 0 or header["height"] <= 0):
                raise ValueError(f"B9 header invalid for scene {scene_id}: expected projected non-empty EPSG:32650 raster")
            bands[band] = BandInput(band, path, True, "", unit, wavelength, frozen_hash, header)
        b9 = bands["B9"].header_fingerprint
        left, bottom, right, top = b9["bounds"]
        if right <= left or top <= bottom:
            raise ValueError(f"B9 has invalid bounds for scene {scene_id}")
        result_scenes.append(SceneInputs(
            scene_id=scene_id, product_id=product_id,
            acquisition_start=str(scene.get("acquisition_start", "UNKNOWN")),
            quality_status=str(scene.get("b9_quality_status_tag", "UNKNOWN")), bands=bands,
        ))
    first, second = result_scenes
    a = first.bands["B9"].header_fingerprint["bounds"]
    b = second.bands["B9"].header_fingerprint["bounds"]
    if min(a[2], b[2]) <= max(a[0], b[0]) or min(a[3], b[3]) <= max(a[1], b[1]):
        raise ValueError(f"pair {pair_id} B9 raster headers have no geometric overlap")
    return PairInputs(pair_id, (first, second), overlap, _float_or_none(pair_row.get("days_apart", "")),
                      (scene_path, bands_path, pairs_path, manifest_path))
