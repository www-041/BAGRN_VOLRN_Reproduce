"""Frozen Task 10 radiometric experiment configuration and provenance helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping


RADIOMETRIC_METHODS = ("RAW", "BAGRN", "BAGRN_VOLRN")
GEOMETRY_SPECS = {
    "efficient_loftr_translation_l2": {
        "matcher": "efficient_loftr",
        "global_method": "translation_l2",
        "label": "EfficientLoFTR + Equal-L2 Translation",
    },
    "sift_mst": {
        "matcher": "sift",
        "global_method": "mst",
        "label": "SIFT + MST",
    },
}


def build_task10_config(repo_root: str | Path) -> dict:
    """Return the frozen Task 10 config rooted at ``repo_root``."""
    root = Path(repo_root)
    validation_root = root / "data" / "output" / "b9_five_scene_validation"
    config = {
        "schema_version": 1,
        "task": "Task 10",
        "dataset": "B9",
        "manifest_indices": [2, 3, 5, 8, 10],
        "pixel_size_m": 14.0,
        "band": "B9",
        "geometry_runs": list(GEOMETRY_SPECS),
        "radiometric_methods": list(RADIOMETRIC_METHODS),
        "output_root": str(validation_root / "radiometric_runs_1024"),
        "source_config": str(validation_root / "04_frozen_five_scene_config_1024.json"),
        "output_grid": str(validation_root / "mosaic_runs_1024" / "protocol" / "canonical_output_grid.json"),
        "global_root": str(validation_root / "global_runs_1024"),
        "mosaic_mode": "weighted",
        "cloud_mask_enabled": False,
        "geometry_mutable": False,
        "seamline_optimization": False,
        "volrn_params": {
            "block_size_pixels": 800,
            "lambda": 0.1,
            "rho": 1.0,
            "max_iter": 20,
            "tol": 1e-4,
        },
    }
    validate_task10_config(config)
    return config


def validate_task10_config(config: Mapping) -> None:
    """Validate the non-negotiable Task 10 matrix and invariants."""
    if config.get("dataset") != "B9":
        raise ValueError("dataset must be frozen to B9")
    if list(config.get("manifest_indices", [])) != [2, 3, 5, 8, 10]:
        raise ValueError("manifest_indices must be [2, 3, 5, 8, 10]")
    if float(config.get("pixel_size_m", 0.0)) != 14.0:
        raise ValueError("pixel_size_m must be 14.0")
    if list(config.get("radiometric_methods", [])) != list(RADIOMETRIC_METHODS):
        raise ValueError("radiometric_methods must be RAW, BAGRN, BAGRN_VOLRN")
    if list(config.get("geometry_runs", [])) != list(GEOMETRY_SPECS):
        raise ValueError("geometry_runs must contain the two frozen geometry runs")
    if config.get("mosaic_mode") != "weighted":
        raise ValueError("mosaic_mode must remain weighted")
    for key in ("geometry_mutable", "seamline_optimization"):
        if config.get(key) is not False:
            raise ValueError(f"{key} must remain false for fixed-geometry Task 10")


def write_run_metadata(
    run_dir: str | Path,
    config: Mapping,
    *,
    geometry_source: Mapping,
    radiometric_method: Mapping,
) -> None:
    """Write the three required per-run provenance files."""
    validate_task10_config(config)
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    (run_path / "run_config.json").write_text(
        json.dumps(dict(config), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (run_path / "geometry_source.json").write_text(
        json.dumps(dict(geometry_source), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (run_path / "radiometric_method.json").write_text(
        json.dumps(dict(radiometric_method), indent=2, ensure_ascii=False), encoding="utf-8"
    )
