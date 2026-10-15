"""Strict parsers and isolated-process bridge for KARIOS/AROSICS adapters."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import rasterio
from affine import Affine

from .coordinates import NativeMatches, georef_only_translation, view_xy_to_native_rc
from .inputs import PairInput
from .protocol import Phase5Config


@dataclass(frozen=True)
class DirectTranslation:
    status: str
    delta_row_col: tuple[float, float] | None
    raw_shift_xy: tuple[float, float]
    shift_convention: str
    library_version: str
    sign_verified: bool


@dataclass(frozen=True)
class ExternalResult:
    status: str
    method: str
    matches: NativeMatches | None = None
    direct_translation: DirectTranslation | None = None
    reason: str | None = None
    provenance: dict[str, Any] | None = None


def parse_karios_csv(path: Path, pair: PairInput) -> NativeMatches:
    """Parse official KARIOS ref-point + monitored dx/dy into native points.

    Official convention: x0/y0 are reference pixel coordinates, dx/dy are
    target-minus-reference pixels; positive dx is east/column and positive dy
    is south/row. Confidence is retained for provenance but never weights fit.
    Rows outside their respective scene-valid masks are discarded.
    """
    if pair.status != "READY" or pair.view is None:
        raise ValueError("KARIOS input pair is not ready")
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream, delimiter=";")
        try:
            headers = next(reader)
        except StopIteration as exc:
            raise ValueError("KARIOS CSV is empty") from exc
        if len(headers) != len(set(headers)):
            raise ValueError("KARIOS CSV has duplicate columns")
        required = {"x0", "y0", "dx", "dy"}
        if not required.issubset(headers):
            raise ValueError(f"KARIOS CSV missing columns: {sorted(required - set(headers))}")
        positions = {name: headers.index(name) for name in headers}
        left, right, scores = [], [], []
        view_i, view_j = [], []
        for line_no, row in enumerate(reader, start=2):
            if len(row) != len(headers):
                raise ValueError(f"KARIOS CSV malformed row {line_no}")
            try:
                x0, y0, dx, dy = (
                    float(row[positions[key]]) for key in ("x0", "y0", "dx", "dy")
                )
                score = float(row[positions["score"]]) if "score" in positions else 1.0
            except (TypeError, ValueError) as exc:
                raise ValueError(f"KARIOS CSV has nonnumeric row {line_no}") from exc
            if not np.isfinite([x0, y0, dx, dy, score]).all():
                raise ValueError(f"KARIOS CSV has nonfinite row {line_no}")
            xy_i = np.array([[x0, y0]], dtype=float)
            xy_j = np.array([[x0 + dx, y0 + dy]], dtype=float)
            try:
                rc_i = view_xy_to_native_rc(xy_i, pair, "i")
                rc_j = view_xy_to_native_rc(xy_j, pair, "j")
            except ValueError as exc:
                if any(token in str(exc) for token in ("outside", "coordinate")):
                    continue
                raise
            left.append(rc_i[0]); right.append(rc_j[0])
            view_i.append(xy_i[0]); view_j.append(xy_j[0]); scores.append(score)
    n = len(left)
    shape = (n, 2)
    return NativeMatches(
        np.asarray(left, dtype=float).reshape(shape), np.asarray(right, dtype=float).reshape(shape),
        np.asarray(view_i, dtype=float).reshape(shape), np.asarray(view_j, dtype=float).reshape(shape),
        np.asarray(scores, dtype=float), "KARIOS", pair.view.joint_mask.shape)


def parse_arosics_result(result: dict[str, Any] | Path, pair: PairInput) -> DirectTranslation:
    """Convert COREG's target-geocoding correction to observed i->j shift.

    AROSICS COREG reports x/y shifts in target-image pixel units and applies
    them to target geocoding. Therefore the observed target feature motion is
    the negative of that correction. Its common-view pixel motion is mapped
    through the shared affine, then combined with the native georeference-only
    origin offset. A caller must supply a successful synthetic sign check.
    """
    if isinstance(result, Path):
        result = json.loads(result.read_text(encoding="utf-8"))
    if not isinstance(result, dict) or result.get("success") is not True:
        raise ValueError("AROSICS COREG did not succeed")
    if result.get("shift_convention") != "target_geocoding_correction":
        raise ValueError("AROSICS shift convention is unrecognized")
    if result.get("sign_verified") is not True:
        raise ValueError("AROSICS sign has not passed the synthetic check")
    if pair.status != "READY" or pair.view is None:
        raise ValueError("AROSICS input pair is not ready")
    try:
        x_shift, y_shift = float(result["x_shift_px"]), float(result["y_shift_px"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("AROSICS result missing finite x/y shifts") from exc
    if not np.isfinite([x_shift, y_shift]).all():
        raise ValueError("AROSICS shift is nonfinite")
    base = georef_only_translation(pair.native_affine_i, pair.native_affine_j)
    overlap = _staged_view_affine(pair)
    native = pair.native_affine_i
    view_dx, view_dy = -x_shift, -y_shift
    map_dx = overlap.a * view_dx + overlap.b * view_dy
    map_dy = overlap.d * view_dx + overlap.e * view_dy
    det = native.a * native.e - native.b * native.d
    if not np.isfinite(det) or abs(det) < 1e-12:
        raise ValueError("AROSICS cannot map through a singular native affine")
    delta_col = (native.e * map_dx - native.b * map_dy) / det
    delta_row = (-native.d * map_dx + native.a * map_dy) / det
    delta = (float(base[0] + delta_row), float(base[1] + delta_col))
    return DirectTranslation("SUCCESS", delta, (x_shift, y_shift),
                             "target_geocoding_correction",
                             str(result.get("library_version", "UNKNOWN")), True)


def _write_input(path: Path, image: np.ndarray, mask: np.ndarray, affine: Affine,
                 crs: str, *, karios: bool = False) -> None:
    if karios:
        array = np.rint(np.clip(image, 0, 1) * 255).astype("uint8")
        dtype = "uint8"
    else:
        array = np.asarray(image, dtype="float32")
        dtype = "float32"
    with rasterio.open(path, "w", driver="GTiff", height=array.shape[0],
                       width=array.shape[1], count=1, dtype=dtype,
                       crs=crs, transform=affine) as dst:
        dst.write(array, 1)
        dst.write_mask(mask.astype("uint8") * 255)


def _staged_view_affine(pair: PairInput) -> Affine:
    """Affine actually written to external inputs, including view resizing."""
    provenance = pair.view.provenance
    return provenance.overlap_affine * Affine.scale(
        1 / provenance.resize_x, 1 / provenance.resize_y)


def run_external(method: str, pair: PairInput, config: Phase5Config,
                 python_exe: Path, output_dir: Path) -> ExternalResult:
    """Stage only the bounded shared view and invoke the isolated worker.

    The output directory must be new. Optional algorithm dependencies are
    imported only in the child environment; no installation is attempted.
    """
    if method not in {"KARIOS", "AROSICS_GLOBAL"}:
        raise ValueError("run_external accepts KARIOS or AROSICS_GLOBAL")
    if pair.status != "READY" or pair.view is None:
        return ExternalResult(pair.status, method, reason=pair.reason,
                              provenance={"worker_invoked": False})
    configured = dict(config.external_python).get(method)
    if not configured:
        return ExternalResult("ENV_UNAVAILABLE", method,
                              reason="EXTERNAL_ENV_NOT_CONFIGURED",
                              provenance={"worker_invoked": False})
    try:
        configured_path = Path(configured).expanduser()
        if not configured_path.is_absolute():
            return ExternalResult("ENV_UNAVAILABLE", method,
                                  reason="EXTERNAL_ENV_PATH_MUST_BE_ABSOLUTE",
                                  provenance={"worker_invoked": False})
        executable = Path(python_exe).expanduser().resolve()
        configured_executable = configured_path.resolve()
    except (TypeError, ValueError, OSError, RuntimeError) as exc:
        return ExternalResult("ENV_UNAVAILABLE", method,
                              reason=f"INVALID_PYTHON_EXECUTABLE: {exc}",
                              provenance={"worker_invoked": False})
    if executable != configured_executable:
        raise ValueError("External Python executable does not match frozen config")
    project_root = Path(__file__).resolve().parents[2]
    registration_env = (project_root / ".venv-registration").resolve()
    if (executable == Path(sys.executable).resolve()
            or executable == registration_env / "Scripts" / "python.exe"
            or executable == registration_env / "bin" / "python"):
        return ExternalResult("ENV_UNAVAILABLE", method,
                              reason="EXTERNAL_ENV_MUST_NOT_USE_REGISTRATION_ENV",
                              provenance={"worker_invoked": False})
    if not executable.is_file():
        return ExternalResult("ENV_UNAVAILABLE", method,
                              reason=f"PYTHON_EXECUTABLE_MISSING: {executable}",
                              provenance={"worker_invoked": False})
    out = Path(output_dir)
    if out.exists():
        raise FileExistsError(f"External output directory already exists: {out}")
    out.mkdir(parents=True)
    view = pair.view
    suffix = "karios" if method == "KARIOS" else "arosics"
    ref_path, tgt_path = out / f"{suffix}_scene_i.tif", out / f"{suffix}_scene_j.tif"
    mask_path = out / f"{suffix}_valid_mask.tif"
    bad_mask_path = out / f"{suffix}_bad_mask.tif"
    affine = _staged_view_affine(pair)
    _write_input(ref_path, view.left, view.mask_i, affine, config.crs, karios=method == "KARIOS")
    _write_input(tgt_path, view.right, view.mask_j, affine, config.crs, karios=method == "KARIOS")
    with rasterio.open(mask_path, "w", driver="GTiff", height=view.joint_mask.shape[0],
                       width=view.joint_mask.shape[1], count=1, dtype="uint8",
                       crs=config.crs, transform=affine) as dst:
        dst.write(view.joint_mask.astype("uint8"), 1)
    with rasterio.open(bad_mask_path, "w", driver="GTiff", height=view.joint_mask.shape[0],
                       width=view.joint_mask.shape[1], count=1, dtype="uint8",
                       crs=config.crs, transform=affine) as dst:
        dst.write((~view.joint_mask).astype("uint8"), 1)
    request = {"method": method, "pair_id": pair.pair_id,
               "reference_path": str(ref_path), "target_path": str(tgt_path),
               "valid_mask_path": str(mask_path), "bad_mask_path": str(bad_mask_path),
               "output_dir": str(out), "crs": config.crs,
               "transform": list(affine)[:6], "pixel_size_m": config.native_pixel_m,
               "parameters": _external_parameters(config, method)}
    request_path, response_path = out / "request.json", out / "response.json"
    request_path.write_text(json.dumps(request, indent=2), encoding="utf-8")
    worker = Path(__file__).resolve().parents[2] / "scripts" / "run_benchmark_v1_phase5_external_worker.py"
    try:
        completed = subprocess.run(
            [str(executable), str(worker), "--request", str(request_path),
             "--response", str(response_path)],
            capture_output=True, text=True, timeout=config.smoke_seconds, check=False)
    except subprocess.TimeoutExpired as exc:
        (out / "worker_timeout.txt").write_text(str(exc), encoding="utf-8")
        return ExternalResult("ALGORITHM_FAIL", method, reason="RESOURCE_LIMIT: timeout",
                              provenance={"worker_invoked": True, "request": request,
                                          "external_python": str(executable)})
    (out / "worker_stdout.log").write_text(completed.stdout or "", encoding="utf-8")
    (out / "worker_stderr.log").write_text(completed.stderr or "", encoding="utf-8")
    if not response_path.is_file():
        status = "ENV_UNAVAILABLE" if completed.returncode != 0 else "ALGORITHM_FAIL"
        return ExternalResult(status, method,
                              reason=f"WORKER_NO_RESPONSE: exit={completed.returncode}",
                              provenance={"worker_invoked": True, "request": request,
                                          "external_python": str(executable)})
    try:
        response = json.loads(response_path.read_text(encoding="utf-8"))
        if not isinstance(response, dict):
            raise ValueError("response root is not an object")
        status = response.get("status")
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        return ExternalResult("ALGORITHM_FAIL", method,
                              reason=f"INVALID_WORKER_RESPONSE: {type(exc).__name__}: {exc}",
                              provenance={"worker_invoked": True,
                                          "worker_exit_code": completed.returncode,
                                          "external_python": str(executable)})
    provenance = {"request": request, "response": response,
                  "worker_exit_code": completed.returncode,
                  "external_python": str(executable)}
    if status != "SUCCESS":
        if status not in {"ENV_UNAVAILABLE", "WEIGHTS_UNAVAILABLE", "ALGORITHM_FAIL"}:
            status = "ALGORITHM_FAIL"
        return ExternalResult(status, method,
                              reason=response.get("reason", "external worker failed"),
                              provenance=provenance)
    try:
        if method == "KARIOS":
            matches = parse_karios_csv(Path(response["matches_csv"]), pair)
            return ExternalResult("SUCCESS", method, matches=matches, provenance=provenance)
        direct = parse_arosics_result(response, pair)
        return ExternalResult("SUCCESS", method, direct_translation=direct, provenance=provenance)
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as exc:
        return ExternalResult("ALGORITHM_FAIL", method,
                              reason=f"INVALID_WORKER_RESULT: {type(exc).__name__}: {exc}",
                              provenance=provenance)


def _external_parameters(config: Phase5Config, method: str) -> dict[str, Any]:
    options = {key: value for name, fields in config.method_settings if name == method
               for key, value in fields}
    if method == "KARIOS":
        return {**options, "reference_is": "scene_i", "monitored_is": "scene_j",
                "large_shift_detection": False, "input_scale": "uint8_round_0_255"}
    return {**options, "reference_is": "scene_i", "target_is": "scene_j",
            "align_grids": False, "match_gsd": False,
            "window_position": "shared_view_overlap_center"}
