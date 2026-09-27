"""BAGRN-only final-baseline replay over frozen geometry artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import csv
import hashlib
import json
from pathlib import Path
import shutil
from datetime import datetime, timezone

import numpy as np
import rasterio
from rasterio.transform import Affine
from scipy.ndimage import distance_transform_edt

from src.mosaic import create_mosaic
from src.multiscene_sift.mosaic_protocol import grid_transform
from src.multiscene_sift.radiometric_runner import run_fixed_geometry_radiometric, _write_preview
from scripts.run_b9_weighted_mosaic import _scene_path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class BaselineReplaySpec:
    """Immutable inputs for one BAGRN replay with already-frozen geometry."""

    name: str
    geometry_run: str
    source_config: Path
    global_run_dir: Path
    output_grid: Path
    replay_root: Path
    pairwise_summary_csv: Path | None = None


@dataclass(frozen=True)
class ReplayArtifact:
    """Validated BAGRN replay outputs used by final-package construction."""

    spec: BaselineReplaySpec
    cache_status: str
    run_dir: Path
    normalized_scenes_manifest: Path
    mosaic_path: Path
    mosaic_sha256: str
    summary: dict


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _expected_manifest(spec: BaselineReplaySpec, grid: dict) -> dict:
    config = _load_json(spec.source_config)
    source_scene_sha256 = [
        _sha256(_scene_path(spec.source_config, record)) for record in config.get("scenes", [])
    ]
    scene_ids = [str(record.get("scene_id", _scene_path(spec.source_config, record).stem)) for record in config.get("scenes", [])]
    return {
        "method": "BAGRN",
        "geometry_run": spec.geometry_run,
        "radiometric_control_idx": 0,
        "source_config_sha256": _sha256(spec.source_config),
        "global_transforms_sha256": _sha256(spec.global_run_dir / "global_transforms.json"),
        "output_grid_sha256": _sha256(spec.output_grid),
        "scene_ids": scene_ids,
        "source_scene_sha256": source_scene_sha256,
        "grid": {
            "crs": str(grid["crs"]),
            "width": int(grid["width"]),
            "height": int(grid["height"]),
            "transform": list(grid_transform(grid))[:6],
        },
    }


def _validate_scene_cache(spec: BaselineReplaySpec, run_dir: Path) -> tuple[Path, dict] | None:
    """Return a verified cache manifest, or ``None`` if it cannot be reused."""
    manifest_path = run_dir / "normalized_scenes_manifest.json"
    summary_path = run_dir / "radiometric_summary.json"
    if not (manifest_path.is_file() and summary_path.is_file()):
        return None
    grid = _load_json(spec.output_grid)
    manifest = _load_json(manifest_path)
    expected = _expected_manifest(spec, grid)
    if any(manifest.get(key) != value for key, value in expected.items()):
        return None
    scenes = manifest.get("scenes")
    if not isinstance(scenes, list) or len(scenes) != len(expected["scene_ids"]):
        return None
    transform = tuple(float(value) for value in expected["grid"]["transform"])
    for index, scene in enumerate(scenes):
        if scene.get("scene_index") != index or scene.get("scene_id") != expected["scene_ids"][index]:
            return None
        path = run_dir / "normalized_scenes" / str(scene.get("path", ""))
        if not path.is_file() or scene.get("sha256") != _sha256(path):
            return None
        with rasterio.open(path) as dataset:
            if (
                dataset.crs is None
                or dataset.crs.to_string() != expected["grid"]["crs"]
                or (dataset.width, dataset.height) != (expected["grid"]["width"], expected["grid"]["height"])
                or not np.allclose(tuple(dataset.transform)[:6], transform, rtol=0, atol=1e-9)
            ):
                return None
            data = dataset.read(1)
            finite = np.isfinite(data)
            if int(np.count_nonzero(finite)) != int(scene.get("valid_pixels", -1)):
                return None
            if scene.get("source_sha256") != expected["source_scene_sha256"][index]:
                return None
    summary = _load_json(summary_path)
    if summary.get("radiometric_method") != "BAGRN" or not isinstance(summary.get("task10d_primary_metrics"), dict):
        return None
    return manifest_path, summary


def _validate_mosaic(spec: BaselineReplaySpec, mosaic_path: Path, expected_support: np.ndarray | None = None) -> None:
    grid = _load_json(spec.output_grid)
    with rasterio.open(mosaic_path) as dataset:
        if (
            dataset.crs is None
            or dataset.crs.to_string() != str(grid["crs"])
            or (dataset.width, dataset.height) != (int(grid["width"]), int(grid["height"]))
            or not np.allclose(tuple(dataset.transform)[:6], tuple(grid_transform(grid))[:6], rtol=0, atol=1e-9)
        ):
            raise ValueError(f"{mosaic_path}: mosaic grid differs from frozen grid")
        values = dataset.read(1)
        valid = np.isfinite(values)
        if not valid.any() or np.isinf(values).any():
            raise ValueError(f"{mosaic_path}: mosaic has no finite pixels")
        if expected_support is not None and not np.array_equal(valid, expected_support):
            raise ValueError(f"{mosaic_path}: mosaic nodata footprint differs from scene support")


def _cache_support(run_dir: Path, manifest: dict) -> np.ndarray:
    grid = manifest["grid"]
    support = np.zeros((int(grid["height"]), int(grid["width"])), dtype=bool)
    for scene in manifest["scenes"]:
        with rasterio.open(run_dir / "normalized_scenes" / str(scene["path"])) as dataset:
            support |= np.isfinite(dataset.read(1))
    return support


def _rebuild_mosaic_from_scenes(spec: BaselineReplaySpec, run_dir: Path, manifest_path: Path) -> Path:
    """Recreate weighted-feather output solely from persisted normalized scenes."""
    manifest = _load_json(manifest_path)
    grid = _load_json(spec.output_grid)
    output_dir = run_dir / "weighted_feather"
    output_dir.mkdir(exist_ok=True)
    mosaic_path = output_dir / "mosaic.tif"
    arrays = []
    transforms = []
    masks = []
    for scene in manifest["scenes"]:
        with rasterio.open(run_dir / "normalized_scenes" / str(scene["path"])) as dataset:
            data = dataset.read(1).astype(np.float64)
        valid = np.isfinite(data)
        rows, cols = np.where(valid)
        if not rows.size:
            raise ValueError("normalized scene has no finite support")
        r0, r1, c0, c1 = int(rows.min()), int(rows.max()) + 1, int(cols.min()), int(cols.max()) + 1
        arrays.append(data[r0:r1, c0:c1][None, ...])
        transforms.append(grid_transform(grid) * Affine.translation(c0, r0))
        masks.append(valid)
    create_mosaic(
        arrays=arrays, transforms=transforms, crs=str(grid["crs"]), nodata_values=[None] * len(arrays),
        output_path=str(mosaic_path), resolution=float(grid.get("resolution", grid["pixel_size"])),
        mode="weighted", output_transform=grid_transform(grid), output_width=int(grid["width"]),
        output_height=int(grid["height"]), output_dtype="float32",
    )
    support = np.zeros((int(grid["height"]), int(grid["width"])), dtype=bool)
    for mask in masks:
        support |= mask
    with rasterio.open(mosaic_path) as dataset:
        values = dataset.read(1)
    _validate_mosaic(spec, mosaic_path, support)
    _write_preview(output_dir / "preview.png", mosaic_path, support)
    return mosaic_path


def replay_baseline(spec: BaselineReplaySpec) -> ReplayArtifact:
    """Reuse a verified BAGRN cache or regenerate BAGRN only on frozen geometry."""
    spec = BaselineReplaySpec(
        name=spec.name,
        geometry_run=spec.geometry_run,
        source_config=Path(spec.source_config),
        global_run_dir=Path(spec.global_run_dir),
        output_grid=Path(spec.output_grid),
        replay_root=Path(spec.replay_root),
        pairwise_summary_csv=(Path(spec.pairwise_summary_csv) if spec.pairwise_summary_csv else None),
    )
    if not (spec.source_config.is_file() and spec.output_grid.is_file()):
        raise FileNotFoundError("replay source config or canonical output grid is missing")
    if not (spec.global_run_dir / "global_transforms.json").is_file():
        raise FileNotFoundError("frozen global_transforms.json is missing")
    run_dir = spec.replay_root / spec.name
    if run_dir.exists():
        try:
            cache = _validate_scene_cache(spec, run_dir)
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            cache = None
    else:
        cache = None
    if cache is None:
        if run_dir.exists() and any(run_dir.iterdir()):
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            candidate = spec.replay_root / f"{spec.name}__regen_{stamp}"
            suffix = 1
            while candidate.exists():
                candidate = spec.replay_root / f"{spec.name}__regen_{stamp}_{suffix}"
                suffix += 1
            run_dir = candidate
        summary = run_fixed_geometry_radiometric(
            spec.source_config,
            spec.global_run_dir,
            spec.output_grid,
            run_dir,
            method="BAGRN",
            geometry_run=spec.geometry_run,
            radiometric_control_idx=0,
            persist_normalized_scenes=True,
        )
        manifest_path = run_dir / "normalized_scenes_manifest.json"
        cache_status = "REGENERATED"
    else:
        manifest_path, summary = cache
        cache_status = "REUSED"
    mosaic_path = _rebuild_mosaic_from_scenes(spec, run_dir, manifest_path)
    manifest = _load_json(manifest_path)
    _validate_mosaic(spec, mosaic_path, _cache_support(run_dir, manifest))
    return ReplayArtifact(
        spec=spec,
        cache_status=cache_status,
        run_dir=run_dir,
        normalized_scenes_manifest=manifest_path,
        mosaic_path=mosaic_path,
        mosaic_sha256=_sha256(mosaic_path),
        summary=summary,
    )


def _write_pairwise_metrics(artifact: ReplayArtifact, destination: Path) -> None:
    """Copy frozen pairwise measurements, retaining a usable empty schema in tests."""
    source = artifact.spec.pairwise_summary_csv
    if source is not None and source.is_file():
        shutil.copy2(source, destination)
        return
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("scene_i", "scene_j", "matches", "inliers", "rmse", "p95", "coverage", "ncc"),
        )
        writer.writeheader()


def _global_metrics(artifact: ReplayArtifact) -> dict:
    path = artifact.spec.global_run_dir / "global_connection_summary.json"
    if path.is_file():
        return _load_json(path)
    return {
        "status": "UNAVAILABLE_SYNTHETIC",
        "geometry_run": artifact.spec.geometry_run,
        "source_global_transforms_sha256": _sha256(
            artifact.spec.global_run_dir / "global_transforms.json"
        ),
    }


def _write_baseline_package(root: Path, label: str, artifact: ReplayArtifact) -> dict:
    registration = root / "01_registration"
    radiometric = root / "02_radiometric"
    mosaic = root / "03_mosaic"
    for directory in (registration, radiometric, mosaic):
        directory.mkdir(parents=True)
    _write_pairwise_metrics(artifact, registration / "pairwise_metrics.csv")
    global_metrics = _global_metrics(artifact)
    (registration / "global_metrics.json").write_text(
        json.dumps(global_metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (registration / "conclusion.md").write_text(
        f"Frozen geometry replay: `{artifact.spec.geometry_run}`; cache `{artifact.cache_status}`.\n",
        encoding="utf-8",
    )
    metrics = {
        "metric_protocol": "Task10D frozen metrics; paper CD/GL remain unverified",
        "raw": {"status": "SNAPSHOT", "task10d": artifact.summary.get("task10d_raw_primary_metrics")},
        "bagrn": artifact.summary["task10d_primary_metrics"],
        "source_summary": str(artifact.run_dir / "radiometric_summary.json"),
    }
    (radiometric / "radiometric_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (radiometric / "conclusion.md").write_text(
        "BAGRN-only replay; Task10D metrics are descriptive frozen-protocol values.\n",
        encoding="utf-8",
    )
    final_mosaic = mosaic / "final_mosaic.tif"
    shutil.copy2(artifact.mosaic_path, final_mosaic)
    preview = artifact.mosaic_path.parent / "preview.png"
    if not preview.is_file():
        raise FileNotFoundError(f"replay preview is missing: {preview}")
    shutil.copy2(preview, mosaic / "preview.png")
    grid = _load_json(artifact.spec.output_grid)
    mosaic_summary = {
        "blend_method": "weighted_feather",
        "cache_status": artifact.cache_status,
        "source_mosaic": str(artifact.mosaic_path),
        "source_mosaic_sha256": artifact.mosaic_sha256,
        "final_mosaic_sha256": _sha256(final_mosaic),
        "grid": {key: grid[key] for key in ("crs", "width", "height", "transform", "bounds")},
        "scene_count": len(_load_json(artifact.normalized_scenes_manifest)["scenes"]),
        "nodata": "NaN",
        "dtype": "float32",
    }
    (mosaic / "mosaic_summary.json").write_text(
        json.dumps(mosaic_summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (root / "experiment_summary.md").write_text(
        f"# {label}\n\nGeometry: `{artifact.spec.geometry_run}` (frozen).\n\n"
        f"Radiometry: BAGRN cache `{artifact.cache_status}`.\n\n"
        "Mosaic: regenerated weighted feather source artifact from persisted normalized scenes.\n",
        encoding="utf-8",
    )
    return mosaic_summary


def materialize_final_results(
    main: ReplayArtifact, traditional: ReplayArtifact, output_root: Path
) -> dict:
    """Create a non-overwriting paper-baseline package from replay artifacts."""
    output_root = Path(output_root)
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"final results directory is non-empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    main_summary = _write_baseline_package(
        output_root / "EfficientLoFTR_Translation_BAGRN", "EfficientLoFTR + Translation-L2 + BAGRN", main
    )
    traditional_summary = _write_baseline_package(
        output_root / "SIFT_MST_BAGRN", "SIFT + MST + BAGRN", traditional
    )
    for summary, artifact, directory in (
        (main_summary, main, "EfficientLoFTR_Translation_BAGRN"),
        (traditional_summary, traditional, "SIFT_MST_BAGRN"),
    ):
        metric_payload = _load_json(output_root / directory / "02_radiometric" / "radiometric_metrics.json")
        summary["bagrn_mamd"] = metric_payload["bagrn"]["mamd"]["weighted_mean"]
        summary["bagrn_msdd"] = metric_payload["bagrn"]["msdd"]["weighted_mean"]
        summary["bagrn_rdd"] = metric_payload["bagrn"]["rdd"]["weighted_mean"]
    (output_root / "paper_tables.md").write_text(
        "# Paper baseline tables\n\n"
        "| Method | Geometry run | Mosaic SHA-256 | BAGRN MAMD | BAGRN MSDD | BAGRN RDD |\n|---|---|---|---:|---:|---:|\n"
        f"| EfficientLoFTR + Translation-L2 + BAGRN | {main.spec.geometry_run} | {main.mosaic_sha256} | {main_summary['bagrn_mamd']} | {main_summary['bagrn_msdd']} | {main_summary['bagrn_rdd']} |\n"
        f"| SIFT + MST + BAGRN | {traditional.spec.geometry_run} | {traditional.mosaic_sha256} | {traditional_summary['bagrn_mamd']} | {traditional_summary['bagrn_msdd']} | {traditional_summary['bagrn_rdd']} |\n\n"
        "All radiometric values are frozen Task10D snapshots; they do not resolve the separate paper CD/GL formula ambiguity.\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 2,
        "baselines": {
            "main": {"geometry_run": main.spec.geometry_run, "cache_status": main.cache_status, "mosaic_sha256": main_summary["final_mosaic_sha256"], "replay_dir": str(main.run_dir), "source_config_sha256": _sha256(main.spec.source_config), "global_transforms_sha256": _sha256(main.spec.global_run_dir / "global_transforms.json"), "output_grid_sha256": _sha256(main.spec.output_grid)},
            "traditional": {"geometry_run": traditional.spec.geometry_run, "cache_status": traditional.cache_status, "mosaic_sha256": traditional_summary["final_mosaic_sha256"], "replay_dir": str(traditional.run_dir), "source_config_sha256": _sha256(traditional.spec.source_config), "global_transforms_sha256": _sha256(traditional.spec.global_run_dir / "global_transforms.json"), "output_grid_sha256": _sha256(traditional.spec.output_grid)},
        },
    }
    artifacts = {}
    for path in sorted(output_root.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            artifacts[str(path.relative_to(output_root))] = _sha256(path)
    manifest["artifacts_sha256"] = artifacts
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest


def verify_final_results(
    output_root: Path,
    specs: tuple[BaselineReplaySpec, BaselineReplaySpec],
) -> None:
    """Verify provenance, replay caches, grids, support, and every package hash."""
    output_root = Path(output_root)
    manifest = _load_json(output_root / "manifest.json")
    if manifest.get("schema_version") != 2:
        raise ValueError("unsupported final-results manifest schema")
    mapping = (("main", "EfficientLoFTR_Translation_BAGRN"), ("traditional", "SIFT_MST_BAGRN"))
    for key, directory in mapping:
        spec = specs[0 if key == "main" else 1]
        for required in (spec.source_config, spec.output_grid, spec.global_run_dir / "global_transforms.json"):
            if not Path(required).is_file():
                raise FileNotFoundError(f"verification input is missing: {required}")
        info = manifest["baselines"][key]
        if info["source_config_sha256"] != _sha256(spec.source_config) or info["global_transforms_sha256"] != _sha256(spec.global_run_dir / "global_transforms.json") or info["output_grid_sha256"] != _sha256(spec.output_grid):
            raise ValueError(f"{key}: provenance hash mismatch")
        run_dir = Path(info["replay_dir"])
        cache = _validate_scene_cache(spec, run_dir)
        if cache is None:
            raise ValueError(f"{key}: replay cache is invalid")
        manifest_path, _ = cache
        support = _cache_support(run_dir, _load_json(manifest_path))
        mosaic_path = run_dir / "weighted_feather" / "mosaic.tif"
        _validate_mosaic(spec, mosaic_path, support)
        final_mosaic = output_root / directory / "03_mosaic" / "final_mosaic.tif"
        if not final_mosaic.is_file() or _sha256(final_mosaic) != info["mosaic_sha256"]:
            raise ValueError(f"{key}: final mosaic checksum mismatch")
    for relative, expected in manifest.get("artifacts_sha256", {}).items():
        path = output_root / relative
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"artifact checksum mismatch: {relative}")
