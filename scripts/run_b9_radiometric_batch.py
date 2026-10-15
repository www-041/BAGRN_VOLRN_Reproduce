"""Run and validate the six frozen Task 10 B9 radiometric experiments."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import rasterio

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_b9_radiometric import run_fixed_geometry_radiometric
from src.multiscene_sift.radiometric_protocol import (
    GEOMETRY_SPECS,
    RADIOMETRIC_METHODS,
    SCIENCE_STATUSES,
)


RUN_KEYS = tuple(
    f"{geometry}/{method}"
    for geometry in GEOMETRY_SPECS
    for method in RADIOMETRIC_METHODS
)
REQUIRED_OUTPUTS = (
    "mosaic.tif", "preview.png", "radiometric_summary.json",
    "run_config.json", "geometry_source.json", "radiometric_method.json",
    "valid_mask.tif", "contributor_count.tif", "weight_sum.tif",
)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve_repo_path(repo_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _verify_protocol(protocol_path: str | Path, source_config: str | Path, repo_root: Path = REPO_ROOT) -> dict:
    protocol = _load_json(Path(protocol_path))
    if protocol.get("dataset") != "B9" or protocol.get("manifest_indices") != [2, 3, 5, 8, 10]:
        raise ValueError("protocol is not the frozen B9 five-scene selection")
    if float(protocol.get("pixel_size_m")) != 14.0:
        raise ValueError("protocol pixel size changed")
    source_hash = protocol.get("source_config_sha256")
    source_ref = protocol.get("source_config")
    if not source_hash or not source_ref:
        raise ValueError("protocol is missing frozen source-config SHA-256")
    frozen_source = _resolve_repo_path(repo_root, source_ref)
    actual_source = Path(source_config)
    if not frozen_source.is_file() or _sha256(frozen_source) != source_hash:
        raise ValueError(f"frozen source config hash mismatch: {frozen_source}")
    if not actual_source.is_file() or _sha256(actual_source) != source_hash:
        raise ValueError(f"source config hash mismatch: {actual_source}")
    required = {
        (spec["matcher"], spec["global_method"])
        for spec in GEOMETRY_SPECS.values()
    }
    observed = set()
    for item in protocol.get("global_transform_hashes", []):
        key = (item.get("matcher"), item.get("global_method"))
        if key not in required:
            continue
        path = _resolve_repo_path(repo_root, item["path"])
        if not path.is_file() or _sha256(path) != item.get("sha256"):
            raise ValueError(f"global transform hash mismatch: {path}")
        observed.add(key)
    if observed != required:
        raise ValueError("protocol does not freeze both Task 10 geometry transform hashes")
    return protocol


def _merge_success(row: dict, validation: dict) -> None:
    """Merge a successful validation and remove stale failure state."""
    row.pop("error", None)
    row.update(validation)


def resume_provenance_matches(
    run_dir: str | Path,
    source_config: str | Path,
    output_grid: str | Path,
    global_transforms: str | Path,
    protocol: str | Path,
    geometry_run: str,
    method: str,
) -> bool:
    """Return whether an existing run belongs to the exact frozen inputs."""
    try:
        run_dir = Path(run_dir)
        geometry = _load_json(run_dir / "geometry_source.json")
        config = _load_json(run_dir / "run_config.json")
        radiometric = _load_json(run_dir / "radiometric_method.json")
        return (
            geometry.get("source_config_sha256") == _sha256(Path(source_config))
            and geometry.get("output_grid_sha256") == _sha256(Path(output_grid))
            and geometry.get("global_transforms_sha256") == _sha256(Path(global_transforms))
            and config.get("protocol_sha256") == _sha256(Path(protocol))
            and geometry.get("geometry_run") == geometry_run
            and radiometric.get("method") == method
        )
    except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError):
        return False


def _validate_strict_runtime(protocol: dict, *, block_size_pixels: int, lambda_param: float,
                             rho: float, max_iter: int, tol: float) -> None:
    params = protocol.get("volrn_params")
    if not isinstance(params, dict):
        raise ValueError("strict protocol is missing volrn_params")
    expected = {
        "block_size_pixels": 400,
        "lambda": 0.5,
        "rho": 1.0,
        "max_iter": 200,
        "tol": 1e-4,
    }
    actual = {
        "block_size_pixels": block_size_pixels,
        "lambda": lambda_param,
        "rho": rho,
        "max_iter": max_iter,
        "tol": tol,
    }
    for key, value in expected.items():
        if float(params.get(key)) != float(value) or float(actual[key]) != float(value):
            raise ValueError(f"strict protocol mismatch for volrn_params.{key}")
    if int(protocol.get("radiometric_control_idx", 0)) != 0:
        raise ValueError("strict protocol radiometric_control_idx must be 0")


def prepare_output_dir(output_dir: str | Path) -> Path:
    """Rename a non-empty partial directory and preserve it for inspection."""
    output_dir = Path(output_dir)
    if not output_dir.exists() or not any(output_dir.iterdir()):
        return output_dir
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    renamed = output_dir.with_name(f"{output_dir.name}_incomplete_{stamp}")
    output_dir.rename(renamed)
    return renamed


def validate_run_outputs(output_dir: str | Path, grid: dict) -> dict:
    output_dir = Path(output_dir)
    missing = [name for name in REQUIRED_OUTPUTS if not (output_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing radiometric outputs: {missing}")
    transform = tuple(float(value) for value in grid["transform"])
    width, height = int(grid["width"]), int(grid["height"])
    with rasterio.open(output_dir / "mosaic.tif") as mosaic:
        if (mosaic.width, mosaic.height) != (width, height):
            raise ValueError("radiometric mosaic dimensions differ from canonical grid")
        if not np.allclose(tuple(mosaic.transform)[:6], transform, atol=1e-9):
            raise ValueError("radiometric mosaic transform differs from canonical grid")
        if mosaic.crs is None or mosaic.crs.to_string() != str(grid["crs"]):
            raise ValueError("radiometric mosaic CRS differs from canonical grid")
        data = mosaic.read(1)
    with rasterio.open(output_dir / "valid_mask.tif") as valid:
        valid_mask = valid.read(1).astype(bool)
    if not np.isfinite(data[valid_mask]).all():
        raise ValueError("radiometric mosaic contains non-finite valid pixels")
    summary = _load_json(output_dir / "radiometric_summary.json")
    method = _load_json(output_dir / "radiometric_method.json")
    if summary.get("radiometric_method") != method.get("method"):
        raise ValueError("radiometric method provenance mismatch")
    science_status = summary.get("science_status")
    if science_status not in SCIENCE_STATUSES:
        raise ValueError(f"invalid or missing science_status: {science_status}")
    return {
        "status": science_status,
        "radiometric_method": method.get("method"),
        "valid_pixels": int(_load_json(output_dir / "radiometric_summary.json").get("mosaic", {}).get("diagnostics", {}).get("valid_pixels", 0)),
    }


def run_batch(
    source_config: str | Path,
    global_root: str | Path,
    output_grid: str | Path,
    output_root: str | Path,
    protocol_path: str | Path,
    *,
    status_path: str | Path | None = None,
    block_size_pixels: int = 400,
    lambda_param: float = 0.5,
    rho: float = 1.0,
    max_iter: int = 200,
    tol: float = 1e-4,
) -> dict:
    source_config = Path(source_config)
    global_root = Path(global_root)
    output_grid = Path(output_grid)
    output_root = Path(output_root)
    status_path = Path(status_path) if status_path is not None else output_root / "radiometric_status.json"
    protocol = _verify_protocol(protocol_path, source_config)
    _validate_strict_runtime(
        protocol,
        block_size_pixels=block_size_pixels,
        lambda_param=lambda_param,
        rho=rho,
        max_iter=max_iter,
        tol=tol,
    )
    grid = _load_json(output_grid)
    output_root.mkdir(parents=True, exist_ok=True)
    logs = output_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    if status_path.is_file():
        status = _load_json(status_path)
    else:
        status = {"schema_version": 2, "rows": {key: {"status": "PENDING"} for key in RUN_KEYS}}
    if list(status.get("rows", {})) != list(RUN_KEYS):
        raise ValueError("status ledger does not contain the six Task 10 runs in frozen order")

    for run_key in RUN_KEYS:
        geometry_run, method = run_key.split("/", 1)
        spec = GEOMETRY_SPECS[geometry_run]
        run_dir = output_root / geometry_run / method
        row = status["rows"][run_key]
        row["output_dir"] = str(run_dir)
        if row.get("status") in set(SCIENCE_STATUSES) | {"PASS"}:
            global_transforms = global_root / spec["matcher"] / spec["global_method"] / "global_transforms.json"
            if resume_provenance_matches(
                run_dir, source_config, output_grid, global_transforms,
                protocol_path, geometry_run, method,
            ):
                try:
                    validation = validate_run_outputs(run_dir, grid)
                except Exception:
                    row["status"] = "PENDING"
                else:
                    _merge_success(row, validation)
                    continue
            else:
                row.update({"status": "PENDING", "error": "FAILED_PROVENANCE: frozen run identity mismatch"})
        row["status"] = "RUNNING"
        status_path.write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")
        log_path = logs / f"{geometry_run}_{method}.log"
        try:
            incomplete = prepare_output_dir(run_dir)
            with log_path.open("w", encoding="utf-8") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                print(f"run={run_key}")
                if incomplete != run_dir:
                    print(f"renamed_partial_output={incomplete}")
                result = run_fixed_geometry_radiometric(
                    source_config, global_root / spec["matcher"] / spec["global_method"],
                    output_grid, run_dir, method=method, geometry_run=geometry_run,
                    block_size_pixels=block_size_pixels, lambda_param=lambda_param,
                    rho=rho, max_iter=max_iter, tol=tol,
                    protocol_path=protocol_path, strict_protocol=True,
                )
                validation = validate_run_outputs(run_dir, grid)
                print(json.dumps({"result": result, "validation": validation}, indent=2, ensure_ascii=False, default=str))
            _merge_success(row, validation)
        except Exception as exc:
            with log_path.open("a", encoding="utf-8") as log:
                traceback.print_exc(file=log)
            row.update({"status": "FAILED_RUNTIME", "error": f"{type(exc).__name__}: {exc}"})
        status_path.write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")
    return status


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-config", required=True, type=Path)
    parser.add_argument("--global-root", required=True, type=Path)
    parser.add_argument("--output-grid", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--status", default=None, type=Path)
    parser.add_argument("--block-size-pixels", type=int, default=400)
    parser.add_argument("--lambda-param", type=float, default=0.5)
    parser.add_argument("--rho", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--tol", type=float, default=1e-4)
    args = parser.parse_args(argv)
    run_batch(
        args.source_config, args.global_root, args.output_grid, args.output_root,
        args.protocol, status_path=args.status, block_size_pixels=args.block_size_pixels,
        lambda_param=args.lambda_param, rho=args.rho, max_iter=args.max_iter, tol=args.tol,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
