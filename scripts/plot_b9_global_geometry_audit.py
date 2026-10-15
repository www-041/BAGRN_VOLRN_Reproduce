"""Generate analysis-only figures/tables for the frozen B9 Global runs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.multiscene_sift.global_geometry_audit import (
    GLOBAL_METHODS,
    MATCHERS,
    build_delta_rows,
    build_residual_distribution_summary,
    build_tree_non_tree_rows,
    compute_cycle_diagnostics,
    json_safe,
    load_frozen_audit_inputs,
    lightglue_anomaly_audit,
    occupancy_count,
    write_csv,
)


def plot_residual_cdf(residuals_by_method: dict[str, np.ndarray], output_path: str | Path, title: str = "Residual CDF") -> Path:
    output_path = Path(output_path); output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    for label, values in residuals_by_method.items():
        values = np.sort(np.asarray(values, dtype=float))
        if not len(values): continue
        ax.plot(values, np.arange(1, len(values) + 1) / len(values), label=label)
    ax.set_xlabel("Residual (pixel)"); ax.set_ylabel("Cumulative fraction"); ax.set_title(title)
    ax.grid(True, alpha=0.3); ax.legend(); fig.tight_layout(); fig.savefig(output_path, dpi=160); plt.close(fig)
    return output_path


def _bar(path: Path, labels, values, ylabel: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5)); ax.bar(labels, values)
    ax.set_ylabel(ylabel); ax.set_title(title); ax.tick_params(axis="x", rotation=35); ax.grid(axis="y", alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(json_safe(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def _load_existing_world_rows(run_dir: Path) -> dict[tuple[int, int], dict]:
    path = run_dir / "global_edge_consistency.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        return {(int(r["idx_i"]), int(r["idx_j"])): r for r in csv.DictReader(handle)}


def _spatial_plot(run: dict, path: Path, threshold: float | None = None) -> None:
    xs, ys, values = [], [], []
    for pair, residuals in run["residuals"].items():
        ref, tgt = run["global_points"][pair]
        mask = np.ones(len(residuals), dtype=bool) if threshold is None else residuals > threshold
        midpoint = (ref + tgt) / 2.0
        xs.extend(midpoint[mask, 0]); ys.extend(midpoint[mask, 1]); values.extend(residuals[mask])
    if not values: return
    fig, ax = plt.subplots(figsize=(8, 6)); scatter = ax.scatter(xs, ys, c=values, s=4, cmap="gray_r", alpha=.65)
    fig.colorbar(scatter, ax=ax, label="Residual (pixel)")
    ax.set_xlabel("Global evaluator x (world frame)"); ax.set_ylabel("Global evaluator y (world frame)")
    ax.set_title(f"{run['matcher']} {run['global_method']} residual spatial audit" + (f" > {threshold:g} pixel" if threshold is not None else ""))
    ax.set_aspect("equal", adjustable="datalim"); fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def _coverage_plot(runs: list[dict], path: Path, extent: tuple[float, float, float, float]) -> dict:
    points = []
    per_edge = []
    for run in runs:
        for pair, bundle in run["bundles"].items():
            pair_to_world = np.asarray(bundle["pair_common_transform"], dtype=float)
            xy = np.asarray(bundle["ref_xy"], dtype=float)
            homogeneous = np.column_stack([xy, np.ones(len(xy))])
            points.append(((pair_to_world @ homogeneous.T).T)[:, :2])
            per_edge.append({"pair": f"{pair[0]}-{pair[1]}", "coverage": bundle.get("coverage", 0.0), "inliers": len(bundle["ref_xy"])})
    points = np.vstack(points)
    rows, cols = 20, 20
    occupied = occupancy_count(points, extent, (rows, cols))
    xmin, ymin, xmax, ymax = extent
    fig, ax = plt.subplots(figsize=(8, 6)); ax.hexbin(points[:, 0], points[:, 1], gridsize=(cols, rows), cmap="gray_r", mincnt=1)
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax); ax.set_xlabel("Common geographic frame x"); ax.set_ylabel("Common geographic frame y")
    ax.set_title(f"{runs[0]['matcher']} tie-point coverage ({occupied}/{rows*cols} occupied cells)")
    ax.set_aspect("equal", adjustable="box"); fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)
    return {"matcher": runs[0]["matcher"], "total_inliers": int(len(points)), "occupied_cells": occupied, "grid_cells": rows * cols, "per_edge": per_edge}


def run_audit(input_root: str | Path, output_dir: str | Path) -> dict:
    input_root = Path(input_root); output_dir = Path(output_dir); figures = output_dir / "figures"; figures.mkdir(parents=True, exist_ok=True)
    inputs = load_frozen_audit_inputs(input_root)
    all_rows = [row for run in inputs.values() for row in run["edge_rows"]]
    write_csv(output_dir / "01_edge_metrics.csv", all_rows)
    (output_dir / "01_edge_metrics.json").write_text(json.dumps(json_safe(all_rows), indent=2), encoding="utf-8")

    tree_rows = build_tree_non_tree_rows(inputs); write_csv(output_dir / "02_tree_non_tree_summary.csv", tree_rows)
    delta_rows, delta_summary = build_delta_rows(inputs); write_csv(output_dir / "03_translation_edge_delta.csv", delta_rows)

    cycle_rows = []
    cycle_summary = {}
    for matcher in MATCHERS:
        run = inputs[(matcher, "mst")]
        pairs = [(r["pair_i"], r["pair_j"]) for r in run["edge_rows"]]
        current = compute_cycle_diagnostics(run["bundles"], pairs, 5)
        for row in current: cycle_rows.append({"matcher": matcher, "global_method": "raw_pair_transform", **row})
        cycle_summary[matcher] = {"cycle_basis_size": len(current), "worst_translation_closure_px": max((r["translation_closure_px"] for r in current), default=None), "median_translation_closure_px": float(np.median([r["translation_closure_px"] for r in current])) if current else None}
    write_csv(output_dir / "04_cycle_consistency.csv", cycle_rows)

    distribution = build_residual_distribution_summary(inputs); write_csv(output_dir / "05_residual_distribution_summary.csv", distribution)
    lg = lightglue_anomaly_audit(inputs[("lightglue_disk", "mst")], inputs[("lightglue_disk", "translation_l2")])
    # Compare persisted world-unit rows with the same residuals converted from pixel.
    unit_checks = []
    for method in GLOBAL_METHODS:
        run = inputs[("lightglue_disk", method)]; existing = _load_existing_world_rows(input_root / "global_runs_1024" / "lightglue_disk" / method)
        for row in run["edge_rows"]:
            pair = (row["pair_i"], row["pair_j"]); persisted = existing[pair]
            unit_checks.append(abs(float(persisted["global_p95_world_m"]) / 14.0 - float(row["p95_pixel"])))
    lg["unit_check_max_abs_p95_pixel_difference"] = max(unit_checks, default=0.0)
    _write_json(output_dir / "06_lightglue_anomaly_audit.json", lg)

    for matcher in MATCHERS:
        plot_residual_cdf({method: np.concatenate(list(inputs[(matcher, method)]["residuals"].values())) for method in GLOBAL_METHODS}, figures / f"residual_cdf_{matcher}.png", f"{matcher}: MST vs Translation-L2 residual CDF")
        for method in GLOBAL_METHODS:
            run = inputs[(matcher, method)]
            labels = [r["pair"] for r in run["edge_rows"]]; values = [r["p95_pixel"] for r in run["edge_rows"]]
            _bar(figures / f"edge_p95_{matcher}_{method}.png", labels, values, "P95 residual (pixel)", f"{matcher} {method}: per-edge P95")
            _spatial_plot(run, figures / f"residual_spatial_{matcher}_{method}.png")
            for threshold in (.5, 1.0):
                if any(np.any(v > threshold) for v in run["residuals"].values()): _spatial_plot(run, figures / f"residual_spatial_{matcher}_{method}_gt{str(threshold).replace('.', '')}.png", threshold)
        tree = [r["tree_mean_p95_pixel"] for r in tree_rows if r["matcher"] == matcher]
        non = [r["non_tree_mean_p95_pixel"] for r in tree_rows if r["matcher"] == matcher]
        _bar(figures / f"tree_non_tree_{matcher}.png", ["MST tree", "MST non-tree", "Translation tree", "Translation non-tree"], [tree[0], non[0], tree[1], non[1]], "Mean P95 residual (pixel)", f"{matcher}: tree vs non-tree")
        by_matcher = [r for r in delta_rows if r["matcher"] == matcher]
        _bar(figures / f"translation_delta_{matcher}.png", [r["pair"] for r in by_matcher], [r["delta_rmse_pixel"] for r in by_matcher], "Translation-L2 − MST RMSE (pixel)", f"{matcher}: per-edge redistribution")
        _bar(figures / f"cycle_{matcher}.png", [r["cycle_nodes"] for r in cycle_rows if r["matcher"] == matcher], [r["translation_closure_px"] for r in cycle_rows if r["matcher"] == matcher], "Translation closure (pixel)", f"{matcher}: raw pair-transform cycles")
    plot_residual_cdf({f"{m}/{method}": np.concatenate(list(inputs[(m, method)]["residuals"].values())) for m in MATCHERS for method in GLOBAL_METHODS}, figures / "residual_cdf_all.png", "B9 Global geometry: all 8 residual CDFs")
    _bar(figures / "edge_balanced_comparison.png", [f"{r['matcher']}/{r['global_method']}" for r in distribution], [r["mean_edge_rmse_pixel"] for r in distribution], "Mean edge RMSE (pixel)", "Edge-balanced RMSE comparison")
    all_points = []
    for run in inputs.values():
        for bundle in run["bundles"].values():
            pair_to_world = np.asarray(bundle["pair_common_transform"], dtype=float)
            xy = np.asarray(bundle["ref_xy"], dtype=float)
            homogeneous = np.column_stack([xy, np.ones(len(xy))])
            all_points.append(((pair_to_world @ homogeneous.T).T)[:, :2])
    all_points = np.vstack(all_points)
    extent = (float(np.min(all_points[:, 0])), float(np.min(all_points[:, 1])), float(np.max(all_points[:, 0])), float(np.max(all_points[:, 1])))
    coverage_summary = []
    for matcher in MATCHERS:
        coverage_summary.append(_coverage_plot([inputs[(matcher, "mst")]], figures / f"coverage_{matcher}.png", extent))

    summary = {
        "frozen_experiment": {"dataset": "B9", "scene_count": 5, "match_max_side": 1024, "matchers": list(MATCHERS), "global_methods": ["MST", "Translation-L2"], "rerun": False},
        "edge_rows": len(all_rows), "tree_non_tree": tree_rows, "translation_delta": delta_summary,
        "cycle_consistency": {"cycle_metric_source": "global_comparison initializes cycle_metric to null; this audit reports explicit raw pair-transform cycles separately.", "by_matcher": cycle_summary},
        "lightglue_anomaly": lg, "coverage": coverage_summary,
        "scope": {"can_prove": ["measured residuals under the fixed protocol", "spatial tail locations", "constraint redistribution", "raw pair-cycle closure"], "cannot_prove": ["absolute geolocation accuracy", "all-region/sensor generalization", "final algorithm superiority"]},
    }
    _write_json(output_dir / "08_geometry_audit_summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", default="data/output/b9_five_scene_validation")
    parser.add_argument("--output-dir", default="data/output/b9_five_scene_validation/geometry_audit_1024")
    args = parser.parse_args(); run_audit(args.input_root, args.output_dir); print(f"audit complete: {args.output_dir}"); return 0


if __name__ == "__main__":
    raise SystemExit(main())
