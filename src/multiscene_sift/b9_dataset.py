"""Metadata-only discovery for the two-level B9 reference dataset."""

from __future__ import annotations

from pathlib import Path

import rasterio


def discover_b9_scenes(root: str | Path) -> list[dict]:
    """Discover B9 scenes below ``root/<time>/<scene>/``.

    Directories without either B9 or MTL files are treated as non-scene
    directories. Once a directory contains one required artifact, both
    artifacts must exist exactly once.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"B9 dataset root does not exist: {root}")

    records: list[dict] = []
    seen_scene_ids: set[str] = set()
    for time_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        for scene_dir in sorted(path for path in time_dir.iterdir() if path.is_dir()):
            b9_files = _files_with_suffix(scene_dir, "_B9.TIF")
            mtl_files = _files_with_suffix(scene_dir, "_MTL.TXT")
            if not b9_files and not mtl_files:
                continue
            if not b9_files:
                raise FileNotFoundError(f"B9 file missing in scene directory: {scene_dir}")
            if not mtl_files:
                raise FileNotFoundError(f"MTL file missing in scene directory: {scene_dir}")
            if len(b9_files) > 1:
                raise ValueError(f"Multiple B9 files found in {scene_dir}: {b9_files}")
            if len(mtl_files) > 1:
                raise ValueError(f"Multiple MTL files found in {scene_dir}: {mtl_files}")

            scene_id = scene_dir.name
            if scene_id in seen_scene_ids:
                raise ValueError(f"Duplicate scene_id discovered: {scene_id}")
            seen_scene_ids.add(scene_id)
            records.append(_scene_record(time_dir, scene_dir, b9_files[0], mtl_files[0]))

    if not records:
        raise FileNotFoundError(f"No B9 scenes found below dataset root: {root}")
    return records


def _files_with_suffix(directory: Path, suffix: str) -> list[Path]:
    suffix = suffix.upper()
    return sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.name.upper().endswith(suffix)
    )


def _scene_record(
    time_dir: Path,
    scene_dir: Path,
    b9_path: Path,
    mtl_path: Path,
) -> dict:
    mtl = _parse_mtl(mtl_path)
    with rasterio.open(b9_path) as src:
        transform = src.transform
        record = {
            "scene_id": scene_dir.name,
            "product_id": mtl.get("PRODUCT_ID"),
            "time_dir": time_dir.name,
            "scene_dir": str(scene_dir),
            "b9_path": str(b9_path),
            "mtl_path": str(mtl_path),
            "acquisition_time": _acquisition_time(mtl),
            "cloud_cover": _float_or_none(mtl.get("CLOUD_COVER")),
            "crs": src.crs.to_string() if src.crs else None,
            "transform": [
                float(transform.a), float(transform.b), float(transform.c),
                float(transform.d), float(transform.e), float(transform.f),
            ],
            "resolution_x_m": abs(float(transform.a)),
            "resolution_y_m": abs(float(transform.e)),
            "width": int(src.width),
            "height": int(src.height),
            "left": float(src.bounds.left),
            "bottom": float(src.bounds.bottom),
            "right": float(src.bounds.right),
            "top": float(src.bounds.top),
        }
    return record


def _parse_mtl(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"')
    return values


def _acquisition_time(mtl: dict[str, str]) -> str | None:
    acquired = mtl.get("DATE_ACQUIRED")
    center = mtl.get("SCENE_CENTER_TIME")
    if acquired and center:
        return f"{acquired}T{center}"
    return acquired or center


def _float_or_none(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None
