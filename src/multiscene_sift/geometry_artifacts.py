"""Atomic point-level geometry artifacts for the B9 Global-ready pipeline."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from rasterio.transform import Affine


SCHEMA_VERSION = 1
PAIR_COMMON_GRID_FRAME = "pair_common_grid"
TRANSFORM_DIRECTION = "target_to_reference"


def _as_points(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(f"{name} must have shape (N, 2)")
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"{name} must be numeric")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _as_matrix(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3, 3):
        raise ValueError(f"{name} must have shape (3, 3)")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _transform_matrix(value: Any) -> np.ndarray:
    if isinstance(value, Affine) or all(hasattr(value, attr) for attr in "abcdef"):
        return np.array(
            [
                [value.a, value.b, value.c],
                [value.d, value.e, value.f],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
    return _as_matrix(value, "pair_common_transform")


def _affine_from_matrix(matrix: np.ndarray) -> Affine:
    return Affine(
        float(matrix[0, 0]),
        float(matrix[0, 1]),
        float(matrix[0, 2]),
        float(matrix[1, 0]),
        float(matrix[1, 1]),
        float(matrix[1, 2]),
    )


def _atomic_json_write(payload: dict[str, Any], path: Path) -> None:
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _atomic_npz_write(
    path: Path,
    *,
    inlier_ref_xy: np.ndarray,
    inlier_tgt_xy: np.ndarray,
) -> None:
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    os.close(fd)
    try:
        with open(temporary, "wb") as handle:
            np.savez(
                handle,
                inlier_ref_xy=inlier_ref_xy,
                inlier_tgt_xy=inlier_tgt_xy,
            )
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def save_pair_geometry_bundle(
    output_dir: str | Path,
    *,
    matcher: str,
    idx_i: int,
    idx_j: int,
    scene_i: str,
    scene_j: str,
    accepted: bool,
    status: str,
    inlier_ref_xy: Any,
    inlier_tgt_xy: Any,
    pair_pixel_matrix: Any,
    pair_common_transform: Any,
    crs: str | None,
    pixel_size_x_m: float,
    pixel_size_y_m: float,
    coordinate_frame: str,
    transform_direction: str,
    config_path: str,
    config_sha256: str,
    match_max_side: int,
) -> Path:
    """Atomically save one pair's real inlier geometry and JSON sidecar."""
    if coordinate_frame != PAIR_COMMON_GRID_FRAME:
        raise ValueError("geometry bundle coordinate frame must be pair_common_grid")
    if transform_direction != TRANSFORM_DIRECTION:
        raise ValueError("geometry bundle transform direction must be target_to_reference")
    if pixel_size_x_m <= 0 or pixel_size_y_m <= 0:
        raise ValueError("pixel sizes must be positive")

    ref = _as_points(inlier_ref_xy, "inlier_ref_xy")
    tgt = _as_points(inlier_tgt_xy, "inlier_tgt_xy")
    if len(ref) != len(tgt):
        raise ValueError("inlier_ref_xy and inlier_tgt_xy must have the same number of rows")
    pixel_matrix = _as_matrix(pair_pixel_matrix, "pair_pixel_matrix")
    common_matrix = _transform_matrix(pair_common_transform)
    if not np.isfinite(common_matrix).all():
        raise ValueError("pair_common_transform must contain only finite values")

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"pair_{int(idx_i):02d}_{int(idx_j):02d}"
    npz_path = directory / f"{stem}.npz"
    sidecar_path = directory / f"{stem}.json"
    if npz_path.exists() or sidecar_path.exists():
        raise FileExistsError(f"geometry bundle already exists for pair {idx_i},{idx_j}")

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "matcher": str(matcher),
        "pair": {
            "idx_i": int(idx_i),
            "idx_j": int(idx_j),
            "scene_i": str(scene_i),
            "scene_j": str(scene_j),
        },
        "accepted": bool(accepted),
        "status": str(status),
        "coordinate_frame": coordinate_frame,
        "transform_direction": transform_direction,
        "crs": None if crs is None else str(crs),
        "pixel_size_m": {
            "x": float(pixel_size_x_m),
            "y": float(pixel_size_y_m),
        },
        "protocol_config_path": str(config_path),
        "config_sha256": str(config_sha256),
        "match_max_side": int(match_max_side),
        "npz_path": npz_path.name,
        "point_count": int(len(ref)),
        "arrays": {
            "inlier_ref_xy": {"shape": list(ref.shape), "dtype": str(ref.dtype)},
            "inlier_tgt_xy": {"shape": list(tgt.shape), "dtype": str(tgt.dtype)},
        },
        "pair_pixel_matrix": {
            "shape": list(pixel_matrix.shape),
            "dtype": str(pixel_matrix.dtype),
            "matrix": pixel_matrix.tolist(),
        },
        "pair_common_transform": {
            "shape": list(common_matrix.shape),
            "dtype": str(common_matrix.dtype),
            "matrix": common_matrix.tolist(),
        },
    }
    _atomic_npz_write(
        npz_path,
        inlier_ref_xy=ref,
        inlier_tgt_xy=tgt,
    )
    try:
        _atomic_json_write(metadata, sidecar_path)
    except Exception:
        npz_path.unlink(missing_ok=True)
        raise
    return sidecar_path


def validate_pair_geometry_bundle(sidecar_path: str | Path) -> dict[str, Any]:
    """Validate sidecar, NPZ arrays, metadata shapes, and geometry invariants."""
    sidecar = Path(sidecar_path)
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported geometry bundle schema_version")
    if metadata.get("coordinate_frame") != PAIR_COMMON_GRID_FRAME:
        raise ValueError("geometry bundle must use pair_common_grid")
    if metadata.get("transform_direction") != TRANSFORM_DIRECTION:
        raise ValueError("geometry bundle must use target_to_reference")
    if int(metadata.get("match_max_side", 0)) <= 0:
        raise ValueError("geometry bundle match_max_side must be positive")

    pair_matrix = _as_matrix(metadata.get("pair_pixel_matrix", {}).get("matrix"), "pair_pixel_matrix")
    common_matrix = _as_matrix(
        metadata.get("pair_common_transform", {}).get("matrix"),
        "pair_common_transform",
    )
    npz_path = sidecar.parent / str(metadata.get("npz_path", ""))
    if not npz_path.is_file():
        raise FileNotFoundError(f"NPZ geometry bundle not found: {npz_path}")

    with np.load(npz_path, allow_pickle=False) as arrays:
        required = {"inlier_ref_xy", "inlier_tgt_xy"}
        if set(arrays.files) != required:
            raise ValueError("NPZ geometry bundle has unexpected array keys")
        ref = _as_points(arrays["inlier_ref_xy"], "inlier_ref_xy")
        tgt = _as_points(arrays["inlier_tgt_xy"], "inlier_tgt_xy")
        if len(ref) != len(tgt):
            raise ValueError("inlier arrays must have the same number of rows")
        if int(metadata.get("point_count", -1)) != len(ref):
            raise ValueError("geometry metadata point_count mismatch")
        for name, array in (("inlier_ref_xy", ref), ("inlier_tgt_xy", tgt)):
            declared = metadata.get("arrays", {}).get(name, {})
            if declared.get("shape") != list(array.shape):
                raise ValueError(f"geometry metadata shape mismatch for {name}")
            if declared.get("dtype") != str(array.dtype):
                raise ValueError(f"geometry metadata dtype mismatch for {name}")

    if metadata.get("pair_pixel_matrix", {}).get("shape") != [3, 3]:
        raise ValueError("pair_pixel_matrix metadata shape mismatch")
    if metadata.get("pair_common_transform", {}).get("shape") != [3, 3]:
        raise ValueError("pair_common_transform metadata shape mismatch")
    return {
        "valid": True,
        "metadata": metadata,
        "npz_path": npz_path,
        "pair_pixel_matrix": pair_matrix,
        "pair_common_transform": common_matrix,
    }


def load_pair_geometry_bundle(sidecar_path: str | Path) -> dict[str, Any]:
    """Load a validated bundle without changing point-array dtypes."""
    validation = validate_pair_geometry_bundle(sidecar_path)
    with np.load(validation["npz_path"], allow_pickle=False) as arrays:
        ref = arrays["inlier_ref_xy"].copy()
        tgt = arrays["inlier_tgt_xy"].copy()
    result = dict(validation)
    result["inlier_ref_xy"] = ref
    result["inlier_tgt_xy"] = tgt
    result["pair_common_transform"] = _affine_from_matrix(
        validation["pair_common_transform"]
    )
    return result
