"""Global-ready and replay gates for one B9 matcher rerun."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.multiscene_sift.global_ready import (
    GlobalInputIncompleteError,
    ProvenanceMismatchError,
    load_global_ready_run,
)


_REPLAY_METRICS = (
    "raw_matches",
    "inliers",
    "residual_rmse",
    "residual_p95",
    "coverage",
)


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _pair(row: dict[str, Any]) -> tuple[int, int]:
    return tuple(sorted((int(row["idx_i"]), int(row["idx_j"]))))


def _accepted_edges(graph: dict[str, Any]) -> set[tuple[int, int]]:
    return {
        tuple(sorted((int(item["idx_i"]), int(item["idx_j"]))))
        for item in graph.get("accepted_edges", [])
    }


def _finite_metric(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _close_metric(left: Any, right: Any) -> bool:
    try:
        return bool(np.isclose(float(left), float(right), rtol=1e-3, atol=1e-3))
    except (TypeError, ValueError):
        return False


def _replay_report(
    run_dir: Path,
    old_run_dir: Path,
    results: list[Any],
    accepted_graph: dict[str, Any],
) -> dict[str, Any]:
    old_summary_path = old_run_dir / "pairwise_summary.json"
    old_graph_path = old_run_dir / "accepted_graph.json"
    if not old_summary_path.is_file() or not old_graph_path.is_file():
        return {
            "replay": "REPLAY_UNAVAILABLE",
            "old_run_dir": str(old_run_dir),
            "reason": "old 1024 summary or graph is missing",
        }

    old_summary = _read(old_summary_path)
    old_graph = _read(old_graph_path)
    old_rows = {_pair(row): row for row in old_summary.get("results", [])}
    new_rows = {
        (result.idx_i, result.idx_j): result
        for result in results
    }
    new_edges = _accepted_edges(accepted_graph)
    old_edges = _accepted_edges(old_graph)
    pair_set_same = set(new_rows) == set(old_rows)
    graph_same = new_edges == old_edges
    metric_rows: list[dict[str, Any]] = []
    metrics_plausible = pair_set_same
    for pair in sorted(set(new_rows) | set(old_rows)):
        old = old_rows.get(pair)
        new = new_rows.get(pair)
        item: dict[str, Any] = {"pair": list(pair), "present_in_both": old is not None and new is not None}
        if old is None or new is None:
            metrics_plausible = False
            metric_rows.append(item)
            continue
        item["status_same"] = str(old.get("status")) == str(new.status)
        if not item["status_same"]:
            metrics_plausible = False
        values: dict[str, Any] = {}
        for name in _REPLAY_METRICS:
            new_value = getattr(new, "residual_rmse" if name == "residual_rmse" else name, None)
            old_value = old.get("residual_rmse" if name == "residual_rmse" else name)
            if name in {"raw_matches", "inliers"}:
                same = int(old_value or 0) == int(new_value or 0)
            else:
                # Non-OK rows may intentionally carry NaN residuals.
                if old_value is None or not _finite_metric(old_value):
                    same = not _finite_metric(new_value)
                else:
                    same = _finite_metric(new_value) and _close_metric(old_value, new_value)
            values[name] = {
                "old": old_value,
                "new": float(new_value) if isinstance(new_value, (float, np.floating)) else new_value,
                "same_or_close": same,
            }
            if not same and str(old.get("status")) == "OK":
                metrics_plausible = False
        item["metrics"] = values
        metric_rows.append(item)

    if not graph_same:
        replay = "REPLAY_GRAPH_CHANGED"
    elif not pair_set_same or not metrics_plausible:
        replay = "REPLAY_INCONSISTENT"
    else:
        replay = "REPLAY_CONSISTENT"
    return {
        "replay": replay,
        "old_run_dir": str(old_run_dir),
        "processed_pair_count_same": len(new_rows) == len(old_rows),
        "accepted_edge_set_same": graph_same,
        "metric_rows": metric_rows,
    }


def validate_global_ready_run(
    run_dir: str | Path,
    old_run_dir: str | Path,
    frozen_config: dict[str, Any],
    protocol_config_path: str | Path,
) -> dict[str, Any]:
    """Validate one rerun and compare it with the historical 1024 run."""
    run = Path(run_dir)
    try:
        context = load_global_ready_run(run, frozen_config, protocol_config_path)
    except (GlobalInputIncompleteError, ProvenanceMismatchError, OSError, ValueError, KeyError) as exc:
        return {
            "global_ready": False,
            "validation": "FAILED",
            "replay": "NOT_RUN",
            "error": f"{type(exc).__name__}: {exc}",
            "run_dir": str(run),
        }

    results = context["results"]
    graph = context["accepted_graph"]
    accepted = context["accepted_results"]
    replay = _replay_report(run, Path(old_run_dir), results, graph)
    connected_components = graph.get("connected_components", [])
    report: dict[str, Any] = {
        "global_ready": True,
        "validation": "PASS" if replay["replay"] == "REPLAY_CONSISTENT" else "FAILED_FOR_GLOBAL",
        "replay": replay["replay"],
        "run_dir": str(run),
        "processed_pair_count": len(results),
        "successful_count": sum(result.status == "OK" for result in results),
        "rejected_count": sum(result.status not in {"OK", "FAILED"} for result in results),
        "failed_count": sum(result.status == "FAILED" for result in results),
        "accepted_edge_count": len(accepted),
        "connected": graph.get("status") == "CONNECTED" and len(connected_components) == 1,
        "connected_components": connected_components,
        "geometry_bundle_count": len(context["geometry_index"].get("bundles", [])),
        "total_inlier_point_count": int(sum(len(result.inlier_ref_xy) for result in accepted)),
        "coordinate_frame": context["geometry_index"].get("coordinate_frame"),
        "transform_direction": context["geometry_index"].get("transform_direction"),
        "config_sha256": context["config_sha256"],
        "replay_details": replay,
        "error": None,
    }
    return report


def write_global_ready_validation_report(
    run_dir: str | Path,
    report: dict[str, Any],
) -> Path:
    """Write the validator result atomically beside the matcher run."""
    path = Path(run_dir) / "global_ready_validation.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)
    return path
