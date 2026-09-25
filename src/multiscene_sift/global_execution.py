"""Execute the existing MST and Equal-L2 Global backends on Global-ready data."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from src.multiscene_sift.global_geometric_adjustment import (
    apply_translation_corrections,
    build_translation_adjustment_system,
    evaluate_edge_point_residuals,
    solve_translation_adjustment,
    summarize_network_residuals,
    write_global_transforms,
    write_mst_residual_artifacts,
    write_translation_residual_artifacts,
    write_translation_solution,
    write_translation_system_summary,
)
from src.multiscene_sift.global_registration import (
    build_accepted_graph,
    build_spanning_tree,
    compose_global_transforms,
    global_consistency_diagnostics,
    save_consistency_diagnostics,
    save_global_registration_info,
    select_reference_scene,
)


def _affine_matrix(transform: Any) -> np.ndarray:
    return np.asarray([
        [transform.a, transform.b, transform.c],
        [transform.d, transform.e, transform.f],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


def _observations(accepted, tree_edges, pixel_size_m: float) -> dict:
    tree_pairs = {
        tuple(sorted((int(edge["parent"]), int(edge["child"]))))
        for edge in tree_edges
    }
    observations = {}
    for result in accepted:
        if result.inlier_ref_xy is None or result.inlier_tgt_xy is None:
            raise ValueError(f"accepted pair {result.idx_i}-{result.idx_j} has no point arrays")
        observations[(result.idx_i, result.idx_j)] = {
            "x_i": np.asarray(result.inlier_ref_xy, dtype=np.float64),
            "x_j": np.asarray(result.inlier_tgt_xy, dtype=np.float64),
            "pair_common_transform": _affine_matrix(result.pair_common_transform),
            "pixel_size": float(pixel_size_m),
            "is_tree_edge": tuple(sorted((result.idx_i, result.idx_j))) in tree_pairs,
        }
    return observations


def _metrics(summary: dict) -> dict:
    point = summary["point_weighted"]
    balanced = summary["edge_balanced"]
    tree = summary["tree_edges"]
    non_tree = summary["non_tree_edges"]
    return {
        "global_rmse_pixel": point["rmse_px"],
        "global_p95_pixel": point["p95_px"],
        "global_max_pixel": point["max_px"],
        "mean_edge_rmse_pixel": balanced["mean_edge_rmse_px"],
        "mean_edge_p95_pixel": balanced["mean_edge_p95_px"],
        "max_edge_p95_pixel": balanced["max_edge_p95_px"],
        "tree_mean_p95_pixel": tree["mean_edge_p95_px"],
        "non_tree_mean_p95_pixel": non_tree["mean_edge_p95_px"],
        "pixel_size_m": None,
    }


def _write_summary(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def run_global_connections(
    context: dict[str, Any],
    output_dir: str | Path,
    *,
    matcher: str,
    pixel_size_m: float,
) -> dict[str, Any]:
    """Run MST then the existing Equal-L2 Translation backend once."""
    started = time.perf_counter()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    results = list(context["results"])
    scenes = context["scenes"]
    adj, accepted = build_accepted_graph(results, matcher_name=matcher)
    ref_info = select_reference_scene(adj, accepted)
    ref_idx = int(ref_info["reference_index"])
    ref_info["reference_name"] = scenes[ref_idx].name
    tree_edges = build_spanning_tree(adj, accepted, ref_idx)
    if len(tree_edges) != len(scenes) - 1:
        raise RuntimeError(f"MST edge count {len(tree_edges)} != N-1 {len(scenes) - 1}")

    transforms_list = compose_global_transforms(scenes, accepted, tree_edges, ref_idx)
    if not all(np.all(np.isfinite(matrix)) for matrix in transforms_list):
        raise RuntimeError("non-finite MST global transform")
    mst_out = out / "mst"
    save_global_registration_info(ref_info, tree_edges, transforms_list, mst_out)
    mst_consistency = global_consistency_diagnostics(
        accepted, transforms_list, tree_edges, pixel_size_m=pixel_size_m
    )
    save_consistency_diagnostics(mst_consistency, mst_out)
    observations = _observations(accepted, tree_edges, pixel_size_m)
    mst_dict = {index: matrix for index, matrix in enumerate(transforms_list)}
    mst_points = evaluate_edge_point_residuals(mst_dict, observations)
    mst_summary = summarize_network_residuals(mst_points)
    write_mst_residual_artifacts(mst_points, mst_summary, mst_out)
    mst_report = {
        "status": "PASS",
        "reference_index": ref_idx,
        "tree_edge_count": len(tree_edges),
        "accepted_edge_count": len(accepted),
        "metrics": {**_metrics(mst_summary), "pixel_size_m": float(pixel_size_m)},
        "runtime_sec": time.perf_counter() - started,
    }
    _write_summary(mst_out / "global_connection_summary.json", mst_report)

    translation_out = out / "translation_l2"
    translation_started = time.perf_counter()
    system = build_translation_adjustment_system(
        mst_dict, observations, reference_idx=ref_idx, weight_mode="equal_edge"
    )
    write_translation_system_summary(system, translation_out / "04_translation_system_summary.json")
    solution = solve_translation_adjustment(system)
    write_translation_solution(solution, translation_out / "06_translation_solution.json")
    adjusted = apply_translation_corrections(
        mst_dict, solution["scene_corrections_px"], pixel_size=float(pixel_size_m)
    )
    write_global_transforms(adjusted, translation_out / "global_transforms.json")
    adjusted_list = [adjusted[index] for index in range(len(scenes))]
    translation_consistency = global_consistency_diagnostics(
        accepted, adjusted_list, tree_edges, pixel_size_m=pixel_size_m
    )
    save_consistency_diagnostics(translation_consistency, translation_out)
    translation_points = evaluate_edge_point_residuals(adjusted, observations)
    translation_summary = summarize_network_residuals(translation_points)
    write_translation_residual_artifacts(translation_points, translation_summary, translation_out)
    translation_status = "PASS" if solution["status"] == "OK" else "FAILED"
    translation_report = {
        "status": translation_status,
        "reference_index": ref_idx,
        "objective_before": solution.get("objective_before"),
        "objective_after": solution.get("objective_after"),
        "metrics": {**_metrics(translation_summary), "pixel_size_m": float(pixel_size_m)},
        "runtime_sec": time.perf_counter() - translation_started,
    }
    _write_summary(translation_out / "global_connection_summary.json", translation_report)
    return {
        "matcher": matcher,
        "mst": mst_report,
        "translation_l2": translation_report,
        "runtime_sec": time.perf_counter() - started,
    }
