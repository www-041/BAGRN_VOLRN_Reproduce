"""Sequentially execute and validate the eight frozen B9 mosaic runs."""

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

from scripts.run_b9_weighted_mosaic import run_weighted_mosaic
from src.multiscene_sift.mosaic_protocol import RUN_KEYS, grid_transform


REQUIRED_OUTPUTS = (
    "mosaic.tif",
    "valid_mask.tif",
    "contributor_count.tif",
    "weight_sum.tif",
    "run_summary.json",
    "preview.png",
)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_run_outputs(
    output_dir: str | Path,
    grid: dict | str | Path,
    *,
    expected_mosaic_dtype: str,
) -> dict:
    output_dir = Path(output_dir)
    if not isinstance(grid, dict):
        grid = _load_json(Path(grid))
    missing = [name for name in REQUIRED_OUTPUTS if not (output_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"missing mosaic outputs: {missing}")

    expected_transform = grid_transform(grid)
    expected_width = int(grid["width"])
    expected_height = int(grid["height"])
    expected_crs = str(grid["crs"])
    expected_resolution = float(grid.get("resolution", grid["pixel_size"]))
    with rasterio.open(output_dir / "mosaic.tif") as mosaic:
        if (mosaic.width, mosaic.height) != (expected_width, expected_height):
            raise ValueError("mosaic dimensions differ from canonical grid")
        if not np.allclose(tuple(mosaic.transform), tuple(expected_transform), atol=1e-9):
            raise ValueError("mosaic transform differs from canonical grid")
        if mosaic.crs is None or mosaic.crs.to_string() != expected_crs:
            raise ValueError("mosaic CRS differs from canonical grid")
        if abs(mosaic.res[0] - expected_resolution) > 1e-9:
            raise ValueError("mosaic resolution differs from canonical grid")
        if mosaic.dtypes[0] != expected_mosaic_dtype:
            raise ValueError(f"unexpected mosaic dtype: {mosaic.dtypes[0]}")
        mosaic_data = mosaic.read(1)
        mosaic_nodata = mosaic.nodata

    with rasterio.open(output_dir / "valid_mask.tif") as valid, rasterio.open(output_dir / "contributor_count.tif") as contributors, rasterio.open(output_dir / "weight_sum.tif") as weights:
        for dataset in (valid, contributors, weights):
            if (dataset.width, dataset.height) != (expected_width, expected_height):
                raise ValueError("diagnostic raster dimensions differ from canonical grid")
            if not np.allclose(tuple(dataset.transform), tuple(expected_transform), atol=1e-9):
                raise ValueError("diagnostic raster transform differs from canonical grid")
            if dataset.crs is None or dataset.crs.to_string() != expected_crs:
                raise ValueError("diagnostic raster CRS differs from canonical grid")
        valid_mask = valid.read(1).astype(bool)
        contributor_count = contributors.read(1)
        weight_sum = weights.read(1)

    if not np.isfinite(mosaic_data[valid_mask]).all():
        raise ValueError("mosaic has non-finite pixels inside valid mask")
    if not np.isfinite(weight_sum).all():
        raise ValueError("weight_sum contains non-finite pixels")
    if not np.array_equal(valid_mask, mosaic_data != mosaic_nodata):
        raise ValueError("valid_mask does not match mosaic nodata footprint")
    if contributor_count[valid_mask].size and int(contributor_count[valid_mask].max()) < 1:
        raise ValueError("valid mask contains no contributors")
    return {
        "status": "PASS",
        "valid_pixels": int(valid_mask.sum()),
        "mosaic_dtype": expected_mosaic_dtype,
    }


def _source_dtype(source_config: Path) -> str:
    config = _load_json(source_config)
    with rasterio.open(config["scenes"][0].get("b9_path") or config["scenes"][0]["path"]) as src:
        return src.dtypes[0]


def _verify_protocol(
    protocol_path: Path,
    repo_root: Path,
    source_config_path: str | Path | None = None,
) -> None:
    protocol = _load_json(protocol_path)
    if protocol.get("dataset") != "B9" or protocol.get("manifest_indices") != [2, 3, 5, 8, 10]:
        raise ValueError("protocol is not the frozen B9 five-scene selection")
    if float(protocol.get("pixel_size_m")) != 14.0 or protocol.get("radiometric_normalization") != "NONE":
        raise ValueError("protocol pixel size or radiometric policy changed")
    source_ref = protocol.get("source_config")
    source_hash = protocol.get("source_config_sha256")
    if not source_ref or not source_hash:
        raise ValueError("protocol is missing frozen source config hash")
    source_path = Path(source_ref)
    if not source_path.is_absolute():
        source_path = repo_root / source_path
    if not source_path.exists() or _sha256(source_path) != source_hash:
        raise ValueError(f"source config hash mismatch: {source_path}")
    if source_config_path is not None:
        actual_source = Path(source_config_path)
        if not actual_source.exists() or _sha256(actual_source) != source_hash:
            raise ValueError(f"source config hash mismatch: {actual_source}")
    for item in protocol.get("global_transform_hashes", []):
        path = Path(item["path"])
        if not path.is_absolute():
            path = repo_root / path
        if not path.exists() or _sha256(path) != item["sha256"]:
            raise ValueError(f"persisted transform hash mismatch: {path}")


def prepare_output_dir(output_dir: str | Path) -> Path:
    """Move a non-empty partial output aside, preserving it for inspection."""
    output_dir = Path(output_dir)
    if not output_dir.exists() or not any(output_dir.iterdir()):
        return output_dir
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    renamed = output_dir.with_name(f"{output_dir.name}_incomplete_{stamp}")
    output_dir.rename(renamed)
    return renamed


def run_batch(
    source_config: str | Path,
    global_root: str | Path,
    output_grid: str | Path,
    output_root: str | Path,
    status_path: str | Path,
    audit_dir: str | Path,
    protocol_path: str | Path,
) -> dict:
    source_config = Path(source_config)
    global_root = Path(global_root)
    output_grid = Path(output_grid)
    output_root = Path(output_root)
    status_path = Path(status_path)
    audit_dir = Path(audit_dir)
    protocol_path = Path(protocol_path)
    repo_root = Path(__file__).resolve().parents[1]
    _verify_protocol(protocol_path, repo_root, source_config)
    grid = _load_json(output_grid)
    status = _load_json(status_path)
    if list(status.get("rows", {})) != list(RUN_KEYS):
        raise ValueError("status ledger does not contain the eight runs in frozen order")
    expected_dtype = _source_dtype(source_config)
    logs_dir = audit_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    for run_key in RUN_KEYS:
        matcher, method = run_key.split("/")
        row = status["rows"][run_key]
        output_dir = output_root / matcher / method
        log_path = logs_dir / f"{matcher}_{method}.log"
        row.update({"output_dir": str(output_dir), "log": str(log_path)})
        if row.get("status") == "PASS":
            try:
                validate_run_outputs(output_dir, grid, expected_mosaic_dtype=expected_dtype)
            except Exception:
                row["status"] = "PENDING"
            else:
                continue

        row["status"] = "RUNNING"
        status_path.write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")
        try:
            incomplete_dir = prepare_output_dir(output_dir)
            with log_path.open("w", encoding="utf-8") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                print(f"run={run_key}")
                print(f"global_run_dir={global_root / matcher / method}")
                if incomplete_dir != output_dir:
                    print(f"renamed_partial_output={incomplete_dir}")
                run_weighted_mosaic(
                    source_config,
                    global_root / matcher / method,
                    output_grid,
                    output_dir,
                    run_name=run_key,
                )
                validation = validate_run_outputs(output_dir, grid, expected_mosaic_dtype=expected_dtype)
                print(json.dumps(validation, indent=2))
            row.update(validation)
            row["status"] = "PASS"
        except Exception as exc:
            with log_path.open("a", encoding="utf-8") as log:
                traceback.print_exc(file=log)
            row.update({"status": "FAILED", "error": f"{type(exc).__name__}: {exc}"})
        status_path.write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")
    return status


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-config", required=True, type=Path)
    parser.add_argument("--global-root", required=True, type=Path)
    parser.add_argument("--output-grid", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--status", required=True, type=Path)
    parser.add_argument("--audit-dir", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    args = parser.parse_args(argv)
    run_batch(
        args.source_config,
        args.global_root,
        args.output_grid,
        args.output_root,
        args.status,
        args.audit_dir,
        args.protocol,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
