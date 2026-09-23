"""Offline global geometric-adjustment diagnostics for an existing five-scene run.

This module deliberately consumes saved artifacts only.  It never invokes a
matcher or reconstructs pairwise correspondences from imagery.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def _read_json(path: Path, default: Any) -> Any:
    if not path.is_file():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _edge_key(i: int, j: int) -> tuple[int, int]:
    return min(i, j), max(i, j)


def _point_artifact_edges(run_dir: Path) -> set[tuple[int, int]]:
    """Find edges explicitly represented by stored point-level CSVs.

    A CSV is accepted only when it contains ``edge_i``/``edge_j`` columns.
    A generic residual CSV without edge identity is intentionally not guessed
    to belong to an accepted edge.
    """
    found: set[tuple[int, int]] = set()
    for path in run_dir.rglob("*.csv"):
        try:
            with path.open("r", newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                fields = set(reader.fieldnames or ())
                point_fields = {"point_id", "ref_x", "ref_y", "tgt_x", "tgt_y"}
                if not {"edge_i", "edge_j"}.issubset(fields) or not point_fields.issubset(fields):
                    continue
                for row in reader:
                    if row.get("edge_i") in (None, "") or row.get("edge_j") in (None, ""):
                        continue
                    found.add(_edge_key(int(row["edge_i"]), int(row["edge_j"])))
        except (OSError, ValueError, UnicodeError):
            continue
    return found


def discover_five_scene_adjustment_inputs(five_scene_run_dir: str | Path) -> dict:
    """Audit saved five-scene inputs before any global adjustment.

    The returned manifest is ``READY`` only when every accepted direct edge
    has explicitly recoverable point-level observations.  Missing point data
    is a hard stop; this function has no rematching fallback by design.
    """
    run_dir = Path(five_scene_run_dir)
    dataset = _read_json(run_dir / "dataset_manifest.json", {})
    registration = _read_json(run_dir / "global_registration.json", {})
    spanning = _read_json(run_dir / "spanning_tree.json", {})
    pairwise = _read_json(run_dir / "pairwise_summary.json", {})
    results = pairwise.get("results", []) if isinstance(pairwise, dict) else []
    accepted = [row for row in results if row.get("status") == "OK"]
    point_edges = _point_artifact_edges(run_dir)

    accepted_edges = []
    for row in accepted:
        i, j = int(row["idx_i"]), int(row["idx_j"])
        key = _edge_key(i, j)
        accepted_edges.append(
            {
                "edge": [i, j],
                "status": row.get("status"),
                "inliers": int(row.get("inliers", 0)),
                "point_artifact_status": "READY" if key in point_edges else "MISSING",
            }
        )

    missing = [row["edge"] for row in accepted_edges if row["point_artifact_status"] == "MISSING"]
    return {
        "status": "READY" if not missing else "INSUFFICIENT_EXISTING_ARTIFACTS",
        "rematch_allowed": False,
        "run_dir": str(run_dir),
        "scene_indices": [int(scene["index"]) for scene in dataset.get("scenes", [])],
        "reference_idx": registration.get("reference_index", spanning.get("reference_index")),
        "spanning_tree_edges": spanning.get("edges", []),
        "accepted_edges": accepted_edges,
        "missing_point_edges": missing,
    }


def write_adjustment_input_manifest(manifest: dict, output_dir: str | Path) -> Path:
    """Write the Task 0 audit without changing the source run artifacts."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "00_adjustment_input_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_historical_pair_baselines(five_scene_run_dir: str | Path) -> dict:
    """Freeze the historical pair summary without re-running registration."""
    run_dir = Path(five_scene_run_dir)
    summary_path = run_dir / "pairwise_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    payload = _read_json(summary_path, {})
    rows = payload.get("results", []) if isinstance(payload, dict) else []

    def convert(row: dict) -> dict:
        i, j = int(row["idx_i"]), int(row["idx_j"])
        return {
            "edge": [i, j],
            "status": row.get("status"),
            "raw_matches": int(row.get("raw_matches", 0)),
            "inliers": int(row.get("inliers", 0)),
            "inlier_ratio": float(row.get("inlier_ratio", 0.0)),
            "coverage": float(row.get("coverage", 0.0)),
            "rmse_px": row.get("residual_rmse"),
            "p95_px": row.get("residual_p95"),
            "affine_matrix": row.get("pixel_matrix"),
            "artifact_sources": [str(summary_path)],
        }

    converted = [convert(row) for row in rows]
    return {
        "run_dir": str(run_dir),
        "source": str(summary_path),
        "accepted_edges": [row for row in converted if row["status"] == "OK"],
        "rejected_edges": [row for row in converted if row["status"] != "OK"],
    }


def write_historical_pair_baselines(baselines: dict, output_dir: str | Path) -> Path:
    """Persist the frozen historical pair baseline artifact."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "00_historical_pair_baselines.json"
    path.write_text(json.dumps(baselines, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_frozen_registration_config(five_scene_run_dir: str | Path) -> dict:
    """Recover the exact historical registration configuration.

    Values absent from the run manifest are read only from the production
    call-chain constants and implementations, never from this function's
    defaults.
    """
    run_dir = Path(five_scene_run_dir)
    config_path = run_dir / "run_config.json"
    dataset_path = run_dir / "dataset_manifest.json"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    if not dataset_path.is_file():
        raise ValueError("FROZEN_CONFIG_INCOMPLETE: dataset_manifest.json")
    run_config = _read_json(config_path, {})
    required_run_keys = {
        "matcher", "registration_band", "match_max_side", "random_seed",
        "ransac_threshold",
    }
    missing = sorted(key for key in required_run_keys if key not in run_config)
    if missing:
        raise ValueError(
            "FROZEN_CONFIG_INCOMPLETE: missing historical keys " + ", ".join(missing)
        )

    from src.multiscene_sift.pairwise import (
        LOWE_RATIO,
        MIN_INLIER_RATIO,
        MIN_INLIERS,
        RANSAC_MAX_TRIALS,
        SIFT_NFEATURES,
    )

    dataset = _read_json(dataset_path, {})
    nodata_by_band: dict[str, list[float | None]] = {}
    for scene in dataset.get("scenes", []):
        for band, metadata in scene.get("bands", {}).items():
            if "nodata" not in metadata:
                raise ValueError(f"FROZEN_CONFIG_INCOMPLETE: nodata for {band}")
            value = metadata["nodata"]
            if value not in nodata_by_band.setdefault(band, []):
                nodata_by_band[band].append(value)

    return {
        "matcher": str(run_config["matcher"]).upper(),
        "registration_band": run_config["registration_band"],
        "match_max_side": int(run_config["match_max_side"]),
        "sift_nfeatures": int(SIFT_NFEATURES),
        "lowe_ratio": float(LOWE_RATIO),
        "mutual_check": True,
        "ransac_model": "AffineTransform",
        "ransac_residual_threshold": float(run_config["ransac_threshold"]),
        "ransac_max_trials": int(RANSAC_MAX_TRIALS),
        "minimum_inliers": int(MIN_INLIERS),
        "minimum_inlier_ratio": float(MIN_INLIER_RATIO),
        "seed": int(run_config["random_seed"]),
        "nodata_policy": nodata_by_band,
        "valid_mask_policy": "nodata exclusion plus finite-value mask",
        "pair_common_grid_policy": {
            "resolution": "finer input resolution",
            "extent": "union extent",
            "data_resampling": "bilinear",
            "mask_resampling": "nearest",
            "match_view": "overlap crop, percentile stretch 2-98, shared max-side resize",
        },
        "source_artifacts": [str(config_path), str(dataset_path)],
        "source_code": [
            "src/multiscene_sift/pairwise.py",
            "src/registration_benchmark/matchers/sift.py",
            "src/registration_benchmark/geometry.py",
            "src/registration_benchmark/common_grid.py",
        ],
    }


def write_frozen_registration_config(config: dict, output_dir: str | Path) -> Path:
    """Persist the frozen registration configuration artifact."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "01_frozen_registration_config.json"
    path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
