"""Matcher-independent summary utilities for five-scene experiments."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

import numpy as np


def _finite_values(values: Iterable[float]) -> list[float]:
    out = []
    for value in values:
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            out.append(value)
    return out


def _mean_or_none(values: Iterable[float]):
    vals = _finite_values(values)
    return float(np.mean(vals)) if vals else None


def _max_or_none(values: Iterable[float]):
    vals = _finite_values(values)
    return float(np.max(vals)) if vals else None


def _round_float(value, digits: int = 6):
    if value is None:
        return None
    return round(float(value), digits)


def _metric_values(rows, key: str, legacy_key: str | None = None):
    values = []
    for row in rows:
        value = row.get(key)
        if value is None and legacy_key is not None:
            value = row.get(legacy_key)
        values.append(value)
    return values


def collect_registration_summary(
    *,
    matcher: str,
    random_seed: int,
    geographic_edges: int,
    pairwise_results,
    consistency: list[dict],
    bagrn_runtime_sec: float,
    volrn_runtime_sec: float,
    total_runtime_sec: float,
    pixel_size_m: float | None = None,
) -> dict:
    """Aggregate detailed pairwise/global outputs into one PPT-friendly row.

    Match-count/coverage/runtime aggregates use all attempted geographic edges so
    failed pairs are not silently removed. Pairwise residual aggregates use only
    finite values. Global aggregates prefer non-tree accepted edges because tree
    edges participate directly in construction of the global transforms; when no
    non-tree edge exists, all consistency edges are used as a documented
    fallback. Global residuals are exposed separately in CRS metres and pixels.
    """
    pairwise_results = list(pairwise_results)
    accepted = [r for r in pairwise_results if r.status == "OK"]

    non_tree = [r for r in consistency if not bool(r.get("in_tree", False))]
    if non_tree:
        global_eval = non_tree
        global_scope = "non_tree"
    else:
        global_eval = list(consistency)
        global_scope = "all_edges_fallback"

    if pixel_size_m is None:
        pixel_sizes = _finite_values(r.get("pixel_size_m") for r in global_eval)
        pixel_size_m = pixel_sizes[0] if pixel_sizes else None

    global_rmse_world = _metric_values(global_eval, "global_rmse_world_m")
    global_p95_world = _metric_values(global_eval, "global_p95_world_m")
    global_max_world = _metric_values(global_eval, "global_max_world_m")
    global_rmse_pixel = _metric_values(
        global_eval, "global_rmse_pixel", "global_rmse_px"
    )
    global_p95_pixel = _metric_values(
        global_eval, "global_p95_pixel", "global_p95_px"
    )
    global_max_pixel = _metric_values(
        global_eval, "global_max_pixel", "global_max_px"
    )

    summary = {
        "matcher": str(matcher).lower(),
        "random_seed": int(random_seed),
        "geographic_edges": int(geographic_edges),
        "accepted_edges": len(accepted),
        "failed_edges": len(pairwise_results) - len(accepted),
        "total_raw_matches": int(sum(int(r.raw_matches) for r in pairwise_results)),
        "mean_raw_matches": _round_float(
            _mean_or_none(r.raw_matches for r in pairwise_results)
        ),
        "total_inliers": int(sum(int(r.inliers) for r in pairwise_results)),
        "mean_inliers": _round_float(
            _mean_or_none(r.inliers for r in pairwise_results)
        ),
        "mean_inlier_ratio": _round_float(
            _mean_or_none(r.inlier_ratio for r in pairwise_results)
        ),
        "mean_coverage": _round_float(
            _mean_or_none(r.coverage for r in pairwise_results)
        ),
        "mean_pairwise_rmse_px": _round_float(
            _mean_or_none(r.residual_rmse for r in pairwise_results)
        ),
        "mean_pairwise_p95_px": _round_float(
            _mean_or_none(r.residual_p95 for r in pairwise_results)
        ),
        "worst_pairwise_p95_px": _round_float(
            _max_or_none(r.residual_p95 for r in pairwise_results)
        ),
        "global_eval_scope": global_scope,
        "global_eval_edges": len(global_eval),
        "global_n_points": int(sum(int(r.get("n_points", 0)) for r in global_eval)),
        "pixel_size_m": _round_float(pixel_size_m),
        "global_rmse_world_m": _round_float(_mean_or_none(global_rmse_world)),
        "global_p95_world_m": _round_float(_mean_or_none(global_p95_world)),
        "global_p95_worst_world_m": _round_float(_max_or_none(global_p95_world)),
        "global_max_world_m": _round_float(_max_or_none(global_max_world)),
        "global_rmse_pixel": _round_float(_mean_or_none(global_rmse_pixel)),
        "global_p95_pixel": _round_float(_mean_or_none(global_p95_pixel)),
        "global_p95_worst_pixel": _round_float(_max_or_none(global_p95_pixel)),
        "global_max_pixel": _round_float(_max_or_none(global_max_pixel)),
        # Deprecated aliases retained for existing report readers.
        "global_rmse_mean_px": _round_float(
            _mean_or_none(global_rmse_pixel)
        ),
        "global_p95_mean_px": _round_float(
            _mean_or_none(global_p95_pixel)
        ),
        "global_p95_worst_px": _round_float(
            _max_or_none(global_p95_pixel)
        ),
        "global_max_px": _round_float(
            _max_or_none(global_max_pixel)
        ),
        "matching_runtime_sec": _round_float(
            sum(float(getattr(r, "matcher_runtime_sec", 0.0)) for r in pairwise_results)
        ),
        "geometry_runtime_sec": _round_float(
            sum(float(getattr(r, "geometry_runtime_sec", 0.0)) for r in pairwise_results)
        ),
        "peak_gpu_memory_mb": _round_float(
            _max_or_none(
                getattr(r, "peak_gpu_memory_mb", None)
                for r in pairwise_results
            )
        ),
        "pairwise_total_runtime_sec": _round_float(
            sum(float(getattr(r, "runtime_sec", 0.0)) for r in pairwise_results)
        ),
        "bagrn_runtime_sec": _round_float(bagrn_runtime_sec),
        "volrn_runtime_sec": _round_float(volrn_runtime_sec),
        "total_runtime_sec": _round_float(total_runtime_sec),
    }
    return summary


def save_registration_summary(summary: dict, out_dir: str | Path) -> None:
    """Save ``registration_summary.json`` and one-row CSV."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    with open(out / "registration_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open(out / "registration_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)


def save_mosaic_coverage_summary(summary: dict, out_dir: str | Path) -> None:
    """Save final mosaic coverage correctness diagnostics."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "mosaic_coverage_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
