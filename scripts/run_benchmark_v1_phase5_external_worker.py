"""Isolated KARIOS/AROSICS process for the Phase5 controller.

This file is launched with the explicitly selected external interpreter. It
does not install packages and writes only beneath the caller's new run folder.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import traceback

import numpy as np


def _respond(path: Path, result: dict) -> None:
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")


def _karios(request: dict) -> dict:
    try:
        import karios  # noqa: F401
        from importlib.util import find_spec
    except ImportError as exc:
        return {"status": "ENV_UNAVAILABLE", "reason": f"KARIOS_IMPORT: {exc}"}
    spec = find_spec("karios")
    package_dir = Path(spec.origin).resolve().parent if spec and spec.origin else None
    processing = (package_dir / "configuration" / "processing_configuration.json"
                  if package_dir else None)
    if processing is None or not processing.is_file():
        return {"status": "ENV_UNAVAILABLE",
                "reason": "KARIOS_DEFAULT_PROCESSING_CONFIG_NOT_FOUND"}
    out = Path(request["output_dir"]) / "karios_results"
    out.mkdir()
    command = [
        sys.executable, "-m", "karios", "process",
        request["target_path"], request["reference_path"],
        request["valid_mask_path"], "-",
        "--conf", str(processing),
        "--input-pixel-size", str(request["pixel_size_m"]),
        "--out", str(out),
        "--no-log-file",
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode:
        return {"status": "ALGORITHM_FAIL",
                "reason": f"KARIOS_EXIT_{completed.returncode}: {completed.stderr[-2000:]}",
                "command": command}
    files = list(out.glob("KLT_matcher_*.csv"))
    if len(files) != 1:
        return {"status": "ALGORITHM_FAIL",
                "reason": f"KARIOS_MATCH_CSV_COUNT_{len(files)}",
                "command": command}
    return {"status": "SUCCESS", "matches_csv": str(files[0]),
            "library_version": importlib.metadata.version("karios"),
            "processing_config": str(processing),
            "processing_config_sha256": _sha256(processing),
            "command": command,
            "stdout_tail": completed.stdout[-2000:]}


def _sha256(path: Path) -> str:
    import hashlib
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _arosics(request: dict) -> dict:
    try:
        from arosics import COREG
        from geoarray import GeoArray
        import rasterio
    except ImportError as exc:
        return {"status": "ENV_UNAVAILABLE", "reason": f"AROSICS_IMPORT: {exc}"}
    try:
        version = importlib.metadata.version("arosics")
        if not _verify_arosics_sign(COREG, GeoArray, rasterio, request["crs"]):
            return {"status": "ALGORITHM_FAIL",
                    "reason": "AROSICS_SYNTHETIC_SIGN_CHECK_FAILED",
                    "library_version": version}
        with rasterio.open(request["reference_path"]) as ref:
            height, width = ref.height, ref.width
            transform, projection = ref.transform, ref.crs.to_wkt()
        if height < 256 or width < 256:
            return {"status": "ALGORITHM_FAIL",
                    "reason": "FIXED_AROSICS_WINDOW_256_EXCEEDS_VIEW",
                    "library_version": version}
        center = transform * (width / 2, height / 2)
        coreg = COREG(
            request["reference_path"], request["target_path"],
            path_out=None, wp=center, ws=(256, 256),
            max_iter=5, max_shift=5, align_grids=False, match_gsd=False,
            nodata=(None, None),
            mask_baddata_ref=request["bad_mask_path"],
            mask_baddata_tgt=request["bad_mask_path"],
            CPUs=1, progress=False, q=True, ignore_errors=True)
        coreg.calculate_spatial_shifts()
        if coreg.success is not True or coreg.x_shift_px is None or coreg.y_shift_px is None:
            return {"status": "ALGORITHM_FAIL", "reason": "AROSICS_COREG_FAILED",
                    "library_version": version,
                    "tracked_errors": [str(error) for error in coreg.tracked_errors]}
        return {
            "status": "SUCCESS",
            "success": True,
            "x_shift_px": float(coreg.x_shift_px),
            "y_shift_px": float(coreg.y_shift_px),
            "x_shift_map": float(coreg.x_shift_map),
            "y_shift_map": float(coreg.y_shift_map),
            "shift_convention": "target_geocoding_correction",
            "sign_verified": True,
            "library_version": version,
            "window_position_xy_map": [float(center[0]), float(center[1])],
            "window_size_xy_px": [256, 256],
            "max_iter": 5,
            "max_shift_px": 5,
            "solver": "native AROSICS COREG global Fourier phase correlation",
        }
    except Exception as exc:
        return {"status": "ALGORITHM_FAIL",
                "reason": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc()[-4000:]}


def _verify_arosics_sign(COREG, GeoArray, rasterio, crs: str) -> bool:
    """Check raw COREG correction sign on a known (+row,+col) target shift."""
    rng = np.random.default_rng(20261011)
    reference = rng.normal(0, 1, size=(256, 256)).astype(np.float32)
    # A target feature displaced by +2 rows and +3 columns requires a
    # target-geocoding correction of -2 rows and -3 columns.
    target = np.roll(reference, shift=(2, 3), axis=(0, 1))
    gt = (500000.0, 14.0, 0.0, 4500000.0, 0.0, -14.0)
    projection = rasterio.crs.CRS.from_string(crs).to_wkt()
    ref = GeoArray(reference, gt, projection)
    tgt = GeoArray(target, gt, projection)
    center = (gt[0] + 128 * gt[1], gt[3] + 128 * gt[5])
    coreg = COREG(ref, tgt, path_out=None, wp=center, ws=(256, 256),
                  max_iter=5, max_shift=5, align_grids=False,
                  match_gsd=False, nodata=(None, None), CPUs=1,
                  progress=False, q=True, ignore_errors=True)
    coreg.calculate_spatial_shifts()
    return (coreg.success is True and coreg.x_shift_px is not None
            and coreg.y_shift_px is not None
            and np.allclose([coreg.x_shift_px, coreg.y_shift_px], [-3.0, -2.0],
                            atol=0.5, rtol=0))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--response", required=True, type=Path)
    args = parser.parse_args()
    try:
        request = json.loads(args.request.read_text(encoding="utf-8"))
        result = (_karios(request) if request["method"] == "KARIOS"
                  else _arosics(request) if request["method"] == "AROSICS_GLOBAL"
                  else {"status": "ALGORITHM_FAIL", "reason": "UNKNOWN_METHOD"})
    except Exception as exc:
        result = {"status": "ALGORITHM_FAIL", "reason": f"{type(exc).__name__}: {exc}",
                  "traceback": traceback.format_exc()[-4000:]}
    _respond(args.response, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
