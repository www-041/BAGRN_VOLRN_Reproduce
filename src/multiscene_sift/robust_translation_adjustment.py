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
        "mst_summary": _read_json(paths["mst_summary"]),
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


def _observation_pixel_size(edge_observations: dict) -> float:
    sizes = {
        float(observation.get("pixel_size", 1.0))
        for observation in edge_observations.values()
    }
    if not sizes or not all(np.isfinite(size) and size > 0.0 for size in sizes):
        raise ValueError("edge observations must have a finite positive pixel_size")
    if len(sizes) != 1:
        raise ValueError(f"inconsistent pixel sizes in edge observations: {sorted(sizes)}")
    return sizes.pop()


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
    pixel_size = _observation_pixel_size(edge_observations)

    for iteration in range(1, max_iterations + 1):
        adjusted_transforms = apply_translation_corrections(
            mst_global_transforms, corrections, pixel_size=pixel_size
        )
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

    final_transforms = apply_translation_corrections(
        mst_global_transforms, corrections, pixel_size=pixel_size
    )
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


VARIANTS = {
    "EQUAL_L2": {"prior": "equal", "huber": False},
    "EQUAL_HUBER": {"prior": "equal", "huber": True},
    "QUALITY_L2": {"prior": "intrinsic_quality", "huber": False},
    "QUALITY_HUBER": {"prior": "intrinsic_quality", "huber": True},
}

CYCLE_EDGES = ((0, 1), (0, 4), (1, 4))
LEAF_EDGES = ((0, 2), (3, 4))


def _evaluate_solution(inputs: dict, solution: dict) -> tuple[dict, Any]:
    corrections = {
        int(scene): (float(values[0]), float(values[1]))
        for scene, values in solution["scene_corrections_px"].items()
    }
    adjusted = apply_translation_corrections(
        inputs["mst_global_transforms"], corrections,
        pixel_size=_observation_pixel_size(inputs["edge_observations"]),
    )
    point_residuals = evaluate_edge_point_residuals(adjusted, inputs["edge_observations"])
    return summarize_network_residuals(point_residuals), point_residuals


def _baseline_reproduction_status(inputs: dict, summary: dict) -> dict:
    baseline = inputs["equal_l2_summary"]
    checks = {
        "zero_one_p95_px": (summary["zero_one"]["p95_px"], baseline["zero_one"]["p95_px"]),
        "mean_edge_rmse_px": (
            summary["edge_balanced"]["mean_edge_rmse_px"],
            baseline["edge_balanced"]["mean_edge_rmse_px"],
        ),
        "max_edge_p95_px": (
            summary["edge_balanced"]["max_edge_p95_px"],
            baseline["edge_balanced"]["max_edge_p95_px"],
        ),
    }
    baseline_edges = {
        (int(row["edge_i"]), int(row["edge_j"])): row["p95_px"]
        for row in baseline["per_edge"]
    }
    candidate_edges = {
        (int(row["edge_i"]), int(row["edge_j"])): row["p95_px"]
        for row in summary["per_edge"]
    }
    checks.update({
        f"edge_{_edge_label(edge)}_p95_px": (candidate_edges[edge], baseline_edges[edge])
        for edge in baseline_edges
    })
    passed = all(abs(float(actual) - float(expected)) <= 1e-8 for actual, expected in checks.values())
    return {
        "status": "PASS" if passed else "BASELINE_REPRODUCTION_FAILED",
        "checks": {
            key: {"candidate": float(actual), "baseline": float(expected),
                  "absolute_difference": float(abs(actual - expected))}
            for key, (actual, expected) in checks.items()
        },
    }


def _comparison_row(method: str, summary: dict) -> dict:
    edge_rows = {
        (int(row["edge_i"]), int(row["edge_j"])): row
        for row in summary["per_edge"]
    }
    edge_p95 = {edge: float(row["p95_px"]) for edge, row in edge_rows.items()}
    worst_edge = max(edge_p95, key=edge_p95.get)
    return {
        "method": method,
        "zero_one_rmse_px": summary["zero_one"]["rmse_px"],
        "zero_one_p95_px": summary["zero_one"]["p95_px"],
        "zero_one_max_px": summary["zero_one"]["max_px"],
        "mean_edge_rmse_px": summary["edge_balanced"]["mean_edge_rmse_px"],
        "mean_edge_p95_px": summary["edge_balanced"]["mean_edge_p95_px"],
        "max_edge_p95_px": summary["edge_balanced"]["max_edge_p95_px"],
        "tree_edge_mean_p95_px": summary["tree_edges"]["mean_edge_p95_px"],
        "non_tree_edge_mean_p95_px": summary["non_tree_edges"]["mean_edge_p95_px"],
        "0-4_p95_px": edge_p95.get((0, 4)),
        "1-4_p95_px": edge_p95.get((1, 4)),
        "0-2_p95_px": edge_p95.get((0, 2)),
        "3-4_p95_px": edge_p95.get((3, 4)),
        "worst_edge": _edge_label(worst_edge),
        "worst_edge_p95_px": edge_p95[worst_edge],
        "edge_p95_spread_px": max(edge_p95.values()) - min(edge_p95.values()),
    }


def build_robust_translation_comparison(
    mst_summary: dict,
    equal_l2_summary: dict,
    candidate_summaries: dict[str, dict],
) -> list[dict]:
    """Compare all methods on the same frozen point residuals."""
    rows = [_comparison_row("MST", mst_summary), _comparison_row("EQUAL_L2", equal_l2_summary)]
    for name in ("EQUAL_HUBER", "QUALITY_L2", "QUALITY_HUBER"):
        rows.append(_comparison_row(name, candidate_summaries[name]))
    return rows


def write_robust_translation_comparison(comparison: list[dict], output_dir: str | Path) -> dict:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "04_method_comparison.csv"
    fields = list(comparison[0]) if comparison else ["method"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(comparison)
    json_path = out / "04_method_comparison.json"
    json_path.write_text(json.dumps(_json_safe(comparison), indent=2, ensure_ascii=False), encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = ["0-1", "0-4", "1-4", "0-2", "3-4"]
    edges = [(0, 1), (0, 4), (1, 4), (0, 2), (3, 4)]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.16
    for index, row in enumerate(comparison):
        values = [row["zero_one_p95_px"] if edge == (0, 1) else row[f"{_edge_label(edge)}_p95_px"] for edge in edges]
        ax.bar(x + (index - (len(comparison) - 1) / 2) * width, values, width=width, label=row["method"])
    ax.set_xticks(x, labels)
    ax.set_ylabel("P95 residual (px)")
    ax.set_title("Same-point translation residual comparison")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    png_path = out / "04_method_comparison.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    return {"csv": csv_path, "json": json_path, "png": png_path}


def _write_variant_artifacts(variant: str, result: dict, output_dir: Path) -> None:
    variant_dir = output_dir / "03_variants"
    variant_dir.mkdir(parents=True, exist_ok=True)
    solution_path = variant_dir / f"{variant}_solution.json"
    solution_path.write_text(json.dumps(_json_safe(result["solution"]), indent=2, ensure_ascii=False), encoding="utf-8")
    summary_path = variant_dir / f"{variant}_network_summary.json"
    summary_path.write_text(json.dumps(_json_safe(result["summary"]), indent=2, ensure_ascii=False), encoding="utf-8")
    edge_path = variant_dir / f"{variant}_edge_summary.csv"
    fields = ["edge_i", "edge_j", "is_tree_edge", "n_points", "median_px", "rmse_px", "p95_px", "max_px", "dx_mean", "dy_mean"]
    with edge_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(result["summary"]["per_edge"])
    history_path = variant_dir / f"{variant}_weight_history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["iteration", "edge", "prior_weight", "robust_factor", "total_weight"])
        writer.writeheader()
        for iteration, factors in enumerate(result["solution"].get("edge_factor_history", []), start=1):
            for label, factor in factors.items():
                edge = tuple(int(item) for item in label.split("-"))
                writer.writerow({
                    "iteration": iteration,
                    "edge": label,
                    "prior_weight": result["prior_edge_weights"][edge],
                    "robust_factor": factor,
                    "total_weight": result["prior_edge_weights"][edge] * factor,
                })


def run_translation_variants(inputs: dict, output_dir: str | Path | None = None) -> dict:
    """Run all four translation-only candidates over the same frozen points."""
    quality_weights = compute_intrinsic_edge_quality_weights(inputs["historical_pair_metrics"])
    quality_priors = {edge: value["weight"] for edge, value in quality_weights.items()}
    equal_priors = {edge: 1.0 for edge in inputs["edge_observations"]}
    delta_px = derive_huber_delta_px(inputs["historical_pair_metrics"])
    variants = {}
    for name, config in VARIANTS.items():
        prior = equal_priors if config["prior"] == "equal" else quality_priors
        solution = solve_edge_weighted_translation_irls(
            inputs["mst_global_transforms"],
            inputs["edge_observations"],
            inputs["reference_idx"],
            prior,
            use_huber=bool(config["huber"]),
            huber_delta_px=delta_px,
        )
        summary, point_residuals = _evaluate_solution(inputs, solution)
        variants[name] = {
            "config": config,
            "prior_edge_weights": prior,
            "solution": solution,
            "summary": summary,
            "point_residuals": point_residuals,
        }
    baseline_check = _baseline_reproduction_status(inputs, variants["EQUAL_L2"]["summary"])
    result = {
        "variants": variants,
        "quality_weights": quality_weights,
        "huber_delta_px": delta_px,
        "baseline_reproduction_status": baseline_check["status"],
        "baseline_reproduction": baseline_check,
    }
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        write_intrinsic_quality_weights(quality_weights, out)
        write_huber_configuration(inputs["historical_pair_metrics"], delta_px, out)
        for name, variant in variants.items():
            _write_variant_artifacts(name, variant, out)
        comparison = build_robust_translation_comparison(
            inputs["mst_summary"],
            inputs["equal_l2_summary"],
            {name: value["summary"] for name, value in variants.items()},
        )
        write_robust_translation_comparison(comparison, out)
    return result


def _sensitivity_classification(max_delta_px: float) -> str:
    if max_delta_px <= 3.0:
        return "STABLE"
    if max_delta_px <= 10.0:
        return "MODERATELY_SENSITIVE"
    return "HIGHLY_SENSITIVE"


def evaluate_cycle_edge_sensitivity(
    variant_config: dict,
    mst_global_transforms: dict[int, np.ndarray],
    edge_observations: dict[tuple[int, int], dict],
    reference_idx: int,
    prior_edge_weights: dict[tuple[int, int], float],
    huber_delta_px: float,
) -> dict:
    """Run leave-one-out only on the redundant 0-1-4 triangle."""
    normalized_observations = {_edge_key(edge): value for edge, value in edge_observations.items()}
    normalized_priors = {_edge_key(edge): float(value) for edge, value in prior_edge_weights.items()}
    full_solution = solve_edge_weighted_translation_irls(
        mst_global_transforms, normalized_observations, reference_idx, normalized_priors,
        use_huber=bool(variant_config.get("huber", False)), huber_delta_px=huber_delta_px,
    )
    full_corrections = full_solution["scene_corrections_px"]
    results = []
    cycle_nodes = {0, 1, 4}
    for removed_edge in CYCLE_EDGES:
        if removed_edge not in normalized_observations:
            continue
        subset_transforms = {
            scene: matrix for scene, matrix in mst_global_transforms.items()
            if int(scene) in cycle_nodes
        }
        subset_observations = {
            edge: value for edge, value in normalized_observations.items()
            if edge != removed_edge and set(edge).issubset(cycle_nodes)
        }
        subset_priors = {edge: normalized_priors[edge] for edge in subset_observations}
        subset_solution = solve_edge_weighted_translation_irls(
            subset_transforms, subset_observations, reference_idx, subset_priors,
            use_huber=bool(variant_config.get("huber", False)), huber_delta_px=huber_delta_px,
        )
        subset_inputs = {
            "mst_global_transforms": subset_transforms,
            "edge_observations": subset_observations,
        }
        subset_summary, _ = _evaluate_solution(subset_inputs, subset_solution)
        changed = {}
        for scene in sorted(cycle_nodes):
            before = np.asarray(full_corrections[scene], dtype=np.float64)
            after = np.asarray(subset_solution["scene_corrections_px"][scene], dtype=np.float64)
            changed[scene] = float(np.linalg.norm(after - before))
        max_delta = max(changed.values(), default=0.0)
        results.append({
            "removed_edge": removed_edge,
            "scene_correction_changes_px": changed,
            "max_correction_delta_px": float(max_delta),
            "remaining_edge_p95_px": {
                _edge_label((int(row["edge_i"]), int(row["edge_j"]))): float(row["p95_px"])
                for row in subset_summary["per_edge"]
            },
            "solver_status": subset_solution["status"],
            "converged": bool(subset_solution["converged"]),
            "classification": _sensitivity_classification(max_delta),
        })
    return {
        "variant_config": dict(variant_config),
        "removed_edges": [edge for edge in CYCLE_EDGES if edge in normalized_observations],
        "leaf_edges_excluded": [list(edge) for edge in LEAF_EDGES],
        "thresholds_are_engineering_sensitivity_thresholds": True,
        "results": results,
    }


def run_cycle_sensitivity(inputs: dict, variant_result: dict) -> dict:
    outputs = {}
    for name, variant in variant_result["variants"].items():
        outputs[name] = evaluate_cycle_edge_sensitivity(
            variant["config"],
            inputs["mst_global_transforms"],
            inputs["edge_observations"],
            inputs["reference_idx"],
            variant["prior_edge_weights"],
            variant_result["huber_delta_px"],
        )
    return outputs


def write_cycle_sensitivity(sensitivity: dict, output_dir: str | Path) -> dict:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "05_cycle_edge_sensitivity.json"
    json_path.write_text(json.dumps(_json_safe(sensitivity), indent=2, ensure_ascii=False), encoding="utf-8")
    csv_path = out / "05_cycle_edge_sensitivity.csv"
    fields = ["variant", "removed_edge", "max_correction_delta_px", "classification", "solver_status", "converged"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for variant, payload in sensitivity.items():
            for row in payload["results"]:
                writer.writerow({"variant": variant, **{field: row[field] for field in fields[1:]}})
    return {"json": json_path, "csv": csv_path}
