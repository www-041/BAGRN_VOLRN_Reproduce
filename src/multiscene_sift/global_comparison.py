"""Build the fixed four-matcher by two-global-method comparison table."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


MATCHERS = ("sift", "loftr", "efficient_loftr", "lightglue_disk")
METHODS = (("MST", "mst"), ("Translation-L2", "translation_l2"))


def _read(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _pairwise_metrics(path: Path) -> dict[str, Any]:
    payload = _read(path / "pairwise_summary.json")
    runtime = _read(path / "runtime.json") or {}
    if not payload:
        return {
            "pairwise_mean_rmse_px": None,
            "pairwise_mean_p95_px": None,
            "mean_coverage": None,
            "matcher_runtime_sec": runtime.get("matcher_runtime_sec"),
        }
    rows = [row for row in payload.get("results", []) if row.get("status") == "OK"]
    mean = lambda name: float(np.mean([float(row[name]) for row in rows])) if rows else None
    return {
        "pairwise_mean_rmse_px": mean("residual_rmse"),
        "pairwise_mean_p95_px": mean("residual_p95"),
        "mean_coverage": mean("coverage"),
        "matcher_runtime_sec": runtime.get("matcher_runtime_sec"),
    }


def _graph_metrics(path: Path) -> dict[str, Any]:
    graph = _read(path / "accepted_graph.json") or {}
    return {
        "accepted_edges": graph.get("accepted_edge_count"),
        "connected": graph.get("status") == "CONNECTED",
    }


def _world_metrics(path: Path) -> dict[str, Any]:
    payload = _read(path / "global_edge_consistency.json") or {}
    rows = payload.get("results", [])
    counts = np.asarray([float(row.get("n_points", 0)) for row in rows], dtype=float)
    rmse = np.asarray([float(row["global_rmse_world_m"]) for row in rows if row.get("global_rmse_world_m") is not None], dtype=float)
    p95 = np.asarray([float(row["global_p95_world_m"]) for row in rows if row.get("global_p95_world_m") is not None], dtype=float)
    return {
        "global_rmse_world_m": float(np.sqrt(np.average(rmse ** 2, weights=counts[:len(rmse)]))) if len(rmse) and len(counts) == len(rmse) else None,
        "mean_edge_p95_world_m": float(np.mean(p95)) if len(p95) else None,
        "max_edge_p95_world_m": float(np.max(p95)) if len(p95) else None,
    }


def _status_for(entry: dict[str, Any], method_key: str, global_dir: Path) -> str:
    if entry.get("rerun") == "FAILED":
        return "RERUN_FAILED"
    if entry.get("rerun") != "PASS":
        return "RERUN_NOT_STARTED"
    if entry.get("validation") != "PASS":
        replay = entry.get("replay")
        return replay if replay == "REPLAY_GRAPH_CHANGED" else "GLOBAL_INPUT_INVALID"
    method_status = entry.get(method_key)
    if method_status == "PASS" and (global_dir / "global_connection_summary.json").is_file():
        return "PASS"
    return "TRANSLATION_FAILED" if method_key == "translation_l2" else "MST_FAILED"


def build_global_comparison_rows(global_root: str | Path, ledger: dict[str, Any]) -> list[dict[str, Any]]:
    """Return exactly eight rows, including explicit unavailable statuses."""
    root = Path(global_root)
    matcher_root = root.parent / "matcher_runs_1024_globalready"
    rows: list[dict[str, Any]] = []
    for matcher in MATCHERS:
        entry = dict(ledger.get("matchers", {}).get(matcher, {}))
        matcher_run = matcher_root / matcher
        pairwise = _pairwise_metrics(matcher_run)
        graph = _graph_metrics(matcher_run)
        for label, method_key in METHODS:
            global_dir = root / matcher / method_key
            summary = _read(global_dir / "global_connection_summary.json") or {}
            metrics = summary.get("metrics", {})
            world = _world_metrics(global_dir)
            rows.append({
                "matcher": matcher,
                "global_method": label,
                "status": _status_for(entry, method_key, global_dir),
                "accepted_edges": graph["accepted_edges"],
                "connected": graph["connected"],
                **pairwise,
                "global_rmse_pixel": metrics.get("global_rmse_pixel"),
                "global_p95_pixel": metrics.get("global_p95_pixel"),
                "global_max_pixel": metrics.get("global_max_pixel"),
                "mean_edge_rmse_pixel": metrics.get("mean_edge_rmse_pixel"),
                "mean_edge_p95_pixel": metrics.get("mean_edge_p95_pixel"),
                "max_edge_p95_pixel": metrics.get("max_edge_p95_pixel"),
                "tree_mean_p95_pixel": metrics.get("tree_mean_p95_pixel"),
                "non_tree_mean_p95_pixel": metrics.get("non_tree_mean_p95_pixel"),
                **world,
                "cycle_metric": None,
                "global_runtime_sec": summary.get("runtime_sec"),
                "pixel_size_m": metrics.get("pixel_size_m", 14.0),
            })
    return rows


def write_global_comparison(
    global_root: str | Path,
    ledger: dict[str, Any],
    report_path: str | Path,
) -> list[dict[str, Any]]:
    root = Path(global_root)
    root.mkdir(parents=True, exist_ok=True)
    rows = build_global_comparison_rows(root, ledger)
    json_path = root / "08_global_method_comparison.json"
    csv_path = root / "08_global_method_comparison.csv"
    json_path.write_text(json.dumps({"rows": rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    fields = list(rows[0]) if rows else []
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    completed = [row for row in rows if row["status"] == "PASS"]
    lines = [
        "# B9 1024 Global 4×2 Results",
        "",
        "This report records the fixed comparison outputs; it does not select a winner.",
        "",
        f"Completed rows: {len(completed)}/{len(rows)}.",
        "",
        "| matcher | global method | status | global RMSE (px) | global P95 (px) | global max (px) |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['matcher']} | {row['global_method']} | {row['status']} | "
            f"{row['global_rmse_pixel']} | {row['global_p95_pixel']} | {row['global_max_pixel']} |"
        )
    lines.extend(["", "Interpretation is intentionally left to the next analysis stage."])
    Path(report_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rows
