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


def _edge_residual_medians(global_transforms: dict[int, np.ndarray], edge_observations: dict) -> dict:
    point_residuals = evaluate_edge_point_residuals(global_transforms, edge_observations)
    grouped = {}
    for row in point_residuals.to_dict(orient="records"):
        grouped.setdefault(_edge_key((row["edge_i"], row["edge_j"])), []).append(float(row["residual_px"]))
    return {edge: float(np.median(values)) for edge, values in grouped.items()}


def solve_edge_weighted_translation_irls(
    mst_global_transforms: dict[int, np.ndarray],
    edge_observations: dict[tuple[int, int], dict],
    reference_idx: int,
    prior_edge_weights: dict[tuple[int, int], float],
    *,
    use_huber: bool,
    huber_delta_px: float,
    min_robust_factor: float = 0.1,
    max_iterations: int = 50,
    convergence_tol_px: float = 1e-6,
) -> dict:
    """Solve edge-balanced translation corrections with optional edge Huber IRLS."""
    base_system = build_translation_adjustment_system(
        mst_global_transforms, edge_observations, reference_idx, weight_mode="equal_edge"
    )
    edges = sorted(_edge_key(edge) for edge in edge_observations)
    missing = set(edges).difference(_edge_key(edge) for edge in prior_edge_weights)
    if missing:
        raise ValueError(f"missing prior weights for edges: {sorted(missing)}")
    prior = {_edge_key(edge): float(value) for edge, value in prior_edge_weights.items()}
    if not all(np.isfinite(value) and value > 0.0 for value in prior.values()):
        raise ValueError("prior edge weights must be finite and positive")

    edge_point_slices = {}
    offset = 0
    for edge in edges:
        n_points = len(edge_observations[edge]["x_i"])
        edge_point_slices[edge] = slice(offset, offset + 2 * n_points)
        offset += 2 * n_points
    expected_rank = int(base_system["rank_expectation"])
    scenes = sorted(int(scene) for scene in mst_global_transforms)
    corrections = {scene: (0.0, 0.0) for scene in scenes}
    objective_history = []
    edge_factor_history = []
    converged = False
    rank = int(base_system["rank"])
    condition_number = base_system["condition_number"]
    final_edge_factors = {edge: 1.0 for edge in edges}
    final_edge_total_weights = dict(prior)

    for iteration in range(1, max_iterations + 1):
        adjusted_transforms = apply_translation_corrections(mst_global_transforms, corrections)
        residual_medians = _edge_residual_medians(adjusted_transforms, edge_observations)
        factors = {
            edge: huber_edge_factor(residual_medians[edge], huber_delta_px, min_robust_factor)
            if use_huber else 1.0
            for edge in edges
        }
        total_weights = {edge: prior[edge] * factors[edge] for edge in edges}
        point_weights = np.zeros(len(base_system["point_weights"]), dtype=np.float64)
        point_offset = 0
        for edge in edges:
            n_points = len(edge_observations[edge]["x_i"])
            point_weights[point_offset:point_offset + n_points] = total_weights[edge] / n_points
            point_offset += n_points
        weighted_rows = np.repeat(point_weights, 2)
        sqrt_weights = np.sqrt(weighted_rows)
        weighted_matrix = base_system["A"] * sqrt_weights[:, None]
        weighted_vector = base_system["b"] * sqrt_weights
        rank = int(np.linalg.matrix_rank(weighted_matrix))
        condition_number = float(np.linalg.cond(weighted_matrix)) if weighted_matrix.size else None
        edge_factor_history.append({_edge_label(edge): float(factors[edge]) for edge in edges})
        if rank < expected_rank:
            return {
                "status": "RANK_DEFICIENT",
                "rank": rank,
                "condition_number": condition_number,
                "iterations": iteration,
                "converged": False,
                "scene_corrections_px": {scene: list(values) for scene, values in corrections.items()},
                "objective_history": objective_history,
                "edge_factor_history": edge_factor_history,
                "final_edge_factors": final_edge_factors,
                "final_edge_total_weights": final_edge_total_weights,
            }
        solution, _, _, _ = np.linalg.lstsq(weighted_matrix, weighted_vector, rcond=None)
        next_corrections = {int(reference_idx): (0.0, 0.0)}
        next_corrections.update({
            int(scene): (float(solution[2 * idx]), float(solution[2 * idx + 1]))
            for idx, scene in enumerate(base_system["unknown_scene_order"])
        })
        objective_history.append(float(np.sum((weighted_matrix @ solution - weighted_vector) ** 2)))
        update = max(
            float(np.hypot(next_corrections[scene][0] - corrections[scene][0],
                           next_corrections[scene][1] - corrections[scene][1]))
            for scene in scenes
        )
        corrections = next_corrections
        final_edge_factors = factors
        final_edge_total_weights = total_weights
        if update < convergence_tol_px:
            converged = True
            break

    final_transforms = apply_translation_corrections(mst_global_transforms, corrections)
    final_residuals = _edge_residual_medians(final_transforms, edge_observations)
    final_edge_factors = {
        edge: huber_edge_factor(final_residuals[edge], huber_delta_px, min_robust_factor)
        if use_huber else 1.0
        for edge in edges
    }
    final_edge_total_weights = {edge: prior[edge] * final_edge_factors[edge] for edge in edges}
    return {
        "status": "OK",
        "rank": rank,
        "condition_number": condition_number,
        "iterations": len(objective_history),
        "converged": converged,
        "scene_corrections_px": {scene: list(values) for scene, values in corrections.items()},
        "objective_history": objective_history,
        "edge_factor_history": edge_factor_history,
        "final_edge_factors": final_edge_factors,
        "final_edge_total_weights": final_edge_total_weights,
    }
