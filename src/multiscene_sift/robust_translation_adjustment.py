"""Diagnostic robust translation variants over frozen five-scene artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .global_geometric_adjustment import (
    apply_translation_corrections,
    build_translation_adjustment_system,
    evaluate_edge_point_residuals,
    load_frozen_edge_observations,
    load_mst_global_transforms,
    solve_translation_adjustment,
    summarize_network_residuals,
)


def _edge_key(edge: tuple[int, int] | list[int]) -> tuple[int, int]:
    i, j = (int(edge[0]), int(edge[1]))
    return (i, j) if i < j else (j, i)


def _edge_label(edge: tuple[int, int]) -> str:
    return f"{int(edge[0])}-{int(edge[1])}"


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _require(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_pairwise_metrics(path: Path, accepted_edges: set[tuple[int, int]]) -> dict:
    payload = _read_json(path)
    rows = payload.get("results", payload.get("accepted_edges", []))
    metrics = {}
    for row in rows:
        edge = _edge_key(row.get("edge", (row.get("idx_i"), row.get("idx_j"))))
        if edge not in accepted_edges or row.get("status") not in (None, "OK"):
            continue
        rmse = row.get("rmse_px", row.get("residual_rmse"))
        p95 = row.get("p95_px", row.get("residual_p95"))
        metrics[edge] = {
            "inlier_ratio": float(row["inlier_ratio"]),
            "coverage": float(row["coverage"]),
            "rmse_px": float(rmse),
            "p95_px": float(p95),
            "inliers": int(row.get("inliers", 0)),
            "raw_matches": int(row.get("raw_matches", 0)),
        }
    missing = accepted_edges.difference(metrics)
    if missing:
        raise ValueError(f"pairwise metrics missing accepted edges: {sorted(missing)}")
    return metrics


def _normalise_solution(solution: dict) -> dict:
    if "scene_corrections_px" in solution:
        return solution
    corrections = {}
    for scene, values in solution.get("scene_corrections", {}).items():
        corrections[int(scene)] = [float(values["dx_px"]), float(values["dy_px"])]
    return {**solution, "scene_corrections_px": corrections}


def load_robust_translation_inputs(
    five_scene_run_dir: str | Path,
    inlier_recovery_dir: str | Path,
    global_adjustment_dir: str | Path,
) -> dict:
    """Load the frozen five-scene inputs without rerunning matching or solving."""
    run_dir = Path(five_scene_run_dir)
    recovery_dir = Path(inlier_recovery_dir)
    adjustment_dir = Path(global_adjustment_dir)
    paths = {
        "global_transforms": _require(run_dir / "global_transforms.json"),
        "dataset_manifest": _require(run_dir / "dataset_manifest.json"),
        "pairwise_summary": _require(run_dir / "pairwise_summary.json"),
        "spanning_tree": _require(run_dir / "spanning_tree.json"),
        "inliers_csv": _require(recovery_dir / "05_global_adjustment_inliers.csv"),
        "inliers_manifest": _require(recovery_dir / "05_global_adjustment_inliers_manifest.json"),
        "mst_summary": _require(adjustment_dir / "01_mst_network_summary.json"),
        "equal_l2_solution": _require(adjustment_dir / "03_translation_solution.json"),
        "equal_l2_summary": _require(adjustment_dir / "05_translation_network_summary.json"),
        "equal_l2_edge_summary": _require(adjustment_dir / "05_translation_edge_summary.csv"),
    }
    manifest = _read_json(paths["inliers_manifest"])
    accepted_edges = {_edge_key(edge) for edge in manifest["accepted_edges"]}
    reference_idx = int(manifest["reference_idx"])
    observations = load_frozen_edge_observations(
        paths["inliers_csv"], paths["spanning_tree"], paths["dataset_manifest"]
    )
    if set(observations) != accepted_edges:
        raise ValueError(
            f"frozen inlier edges {sorted(observations)} do not match manifest "
            f"{sorted(accepted_edges)}"
        )
    pairwise_metrics = _load_pairwise_metrics(paths["pairwise_summary"], accepted_edges)
    equal_l2_solution = _normalise_solution(_read_json(paths["equal_l2_solution"]))
    equal_l2_summary = _read_json(paths["equal_l2_summary"])
    return {
        "reference_idx": reference_idx,
        "mst_global_transforms": load_mst_global_transforms(paths["global_transforms"]),
        "edge_observations": observations,
        "historical_pair_metrics": pairwise_metrics,
        "equal_l2_summary": equal_l2_summary,
        "equal_l2_solution": equal_l2_solution,
        "accepted_edges": sorted(accepted_edges),
        "candidate_weighting_used_for_baseline": False,
        "source_paths": {key: str(path) for key, path in paths.items()},
        "input_hashes": {key: _sha256(path) for key, path in paths.items()},
        "point_count_per_edge": {
            _edge_label(edge): int(len(observations[edge]["x_i"]))
            for edge in sorted(observations)
        },
        "frozen_manifest": manifest,
    }


def write_frozen_baseline(inputs: dict, output_dir: str | Path) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "input_hashes": inputs["input_hashes"],
        "reference_idx": int(inputs["reference_idx"]),
        "accepted_edges": [list(edge) for edge in inputs["accepted_edges"]],
        "point_count_per_edge": inputs["point_count_per_edge"],
        "equal_l2_summary": inputs["equal_l2_summary"],
        "equal_l2_solution": inputs["equal_l2_solution"],
        "candidate_weighting_used_for_baseline": False,
    }
    path = out / "00_frozen_baseline.json"
    path.write_text(json.dumps(_json_safe(payload), indent=2, ensure_ascii=False), encoding="utf-8")
    return path
