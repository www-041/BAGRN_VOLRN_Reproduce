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


def compute_intrinsic_edge_quality_weights(
    pairwise_metrics: dict[tuple[int, int], dict],
    min_weight: float = 0.5,
    max_weight: float = 2.0,
) -> dict[tuple[int, int], dict]:
    """Compute bounded priors from pairwise metrics available before adjustment."""
    if not pairwise_metrics:
        raise ValueError("pairwise_metrics must contain at least one accepted edge")
    if min_weight <= 0.0 or max_weight < min_weight:
        raise ValueError("invalid quality-weight bounds")
    raw_values = {}
    for edge, metrics in pairwise_metrics.items():
        inlier_ratio = float(metrics["inlier_ratio"])
        coverage = float(metrics["coverage"])
        rmse_px = float(metrics["rmse_px"])
        raw = (
            max(inlier_ratio, 1e-6)
            * max(coverage, 1e-6)
            / max(rmse_px, 0.25)
        )
        if not np.isfinite(raw):
            raise ValueError(f"non-finite intrinsic quality for edge {_edge_key(edge)}")
        raw_values[_edge_key(edge)] = float(raw)
    median_quality = float(np.median(list(raw_values.values())))
    if not np.isfinite(median_quality) or median_quality <= 0.0:
        raise ValueError("intrinsic quality median must be positive")
    result = {}
    for edge in sorted(raw_values):
        metrics = dict(pairwise_metrics[edge])
        result[edge] = {
            "raw_quality": raw_values[edge],
            "median_quality": median_quality,
            "weight": float(np.clip(raw_values[edge] / median_quality, min_weight, max_weight)),
            "input_metrics": metrics,
            "uses_global_residual": False,
            "uses_tree_status": False,
            "uses_inlier_count": False,
        }
    return result


def write_intrinsic_quality_weights(weights: dict[tuple[int, int], dict], output_dir: str | Path) -> dict:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "01_intrinsic_quality_weights.json"
    payload = {
        "edges": {
            _edge_label(edge): {**value, "edge": list(edge)}
            for edge, value in sorted(weights.items())
        },
        "uses_global_residual": False,
        "uses_tree_status": False,
        "uses_inlier_count": False,
    }
    json_path.write_text(json.dumps(_json_safe(payload), indent=2, ensure_ascii=False), encoding="utf-8")
    csv_path = out / "01_intrinsic_quality_weights.csv"
    fields = ["edge_i", "edge_j", "inlier_ratio", "coverage", "pairwise_rmse_px", "raw_quality", "median_quality", "weight"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for edge, value in sorted(weights.items()):
            metrics = value["input_metrics"]
            writer.writerow({
                "edge_i": edge[0], "edge_j": edge[1],
                "inlier_ratio": metrics["inlier_ratio"],
                "coverage": metrics["coverage"],
                "pairwise_rmse_px": metrics["rmse_px"],
                "raw_quality": value["raw_quality"],
                "median_quality": value["median_quality"],
                "weight": value["weight"],
            })
    return {"json": json_path, "csv": csv_path}


def derive_huber_delta_px(pairwise_metrics: dict[tuple[int, int], dict]) -> float:
    """Freeze Huber delta from pairwise P95 values before global optimization."""
    p95_values = [float(metrics["p95_px"]) for metrics in pairwise_metrics.values()]
    if not p95_values or not all(np.isfinite(p95_values)):
        raise ValueError("pairwise metrics must provide finite p95_px values")
    return float(max(2.0, 2.0 * float(np.median(p95_values))))


def huber_edge_factor(
    residual_norm_px: float,
    delta_px: float,
    min_factor: float = 0.1,
) -> float:
    """Return a bounded edge-level Huber IRLS factor."""
    residual = float(residual_norm_px)
    delta = float(delta_px)
    floor = float(min_factor)
    if not np.isfinite(residual) or not np.isfinite(delta) or delta <= 0.0:
        raise ValueError("Huber residual and delta must be finite and delta positive")
    if floor <= 0.0 or floor > 1.0:
        raise ValueError("min_factor must be in (0, 1]")
    factor = 1.0 if residual <= delta else delta / max(residual, 1e-12)
    return float(max(floor, min(1.0, factor)))


def write_huber_configuration(
    pairwise_metrics: dict[tuple[int, int], dict],
    delta_px: float,
    output_dir: str | Path,
    min_factor: float = 0.1,
) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    p95_values = [float(pairwise_metrics[edge]["p95_px"]) for edge in sorted(pairwise_metrics)]
    payload = {
        "pairwise_p95_px": {
            _edge_label(edge): float(pairwise_metrics[edge]["p95_px"])
            for edge in sorted(pairwise_metrics)
        },
        "pairwise_p95_median_px": float(np.median(p95_values)),
        "delta_px": float(delta_px),
        "min_factor": float(min_factor),
        "global_residual_used": False,
    }
    path = out / "02_huber_configuration.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
