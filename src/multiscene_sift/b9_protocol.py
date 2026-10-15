"""Validation helpers for the frozen B9 1024 matcher protocol.

This module checks provenance and result completeness only.  It does not rank
matchers, change quality thresholds, or interpret graph connectivity as a
quality failure.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PAIRWISE_FIELDS = {
    "idx_i",
    "idx_j",
    "status",
    "raw_matches",
    "inliers",
    "inlier_ratio",
    "coverage",
    "residual_median",
    "residual_rmse",
    "residual_p95",
    "runtime_sec",
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _expected_local_pairs(config: dict[str, Any]) -> set[tuple[int, int]]:
    manifest_to_local = {
        int(manifest): local
        for local, manifest in enumerate(config["selection"]["manifest_indices"])
    }
    return {
        tuple(sorted((
            manifest_to_local[int(edge["manifest_idx_i"])],
            manifest_to_local[int(edge["manifest_idx_j"])],
        )))
        for edge in config["graph_edges"]
    }


def validate_b9_1024_run(
    run_dir: str | Path,
    frozen_config: dict[str, Any],
    *,
    protocol_config_path: str | Path,
) -> dict[str, Any]:
    """Validate one completed formal B9 1024 run.

    ``PASS`` means the expected candidate rows and provenance are present;
    individual rows may still carry ordinary quality statuses.  The graph
    status is returned as ``network_status`` and is intentionally not used to
    reject a completed run.
    """
    run = Path(run_dir)
    config_path = Path(protocol_config_path)
    run_config_path = run / "run_config.json"
    summary_path = run / "pairwise_summary.json"
    graph_path = run / "accepted_graph.json"
    runtime_path = run / "runtime.json"

    required_files = (run_config_path, summary_path, graph_path, runtime_path)
    missing_files = [str(path.name) for path in required_files if not path.exists()]
    if missing_files:
        return {"status": "RUN_INCOMPLETE", "missing_files": missing_files}

    run_config = _read_json(run_config_path)
    expected_protocol_name = config_path.name
    recorded_protocol = run_config.get("protocol_config_path")
    recorded_protocol_name = Path(str(recorded_protocol)).name if recorded_protocol else None
    config_scale = frozen_config.get("registration", {}).get("match_max_side")
    if (
        config_scale != 1024
        or run_config.get("match_max_side") != 1024
        or recorded_protocol_name != expected_protocol_name
    ):
        return {
            "status": "PROTOCOL_MISMATCH",
            "recorded_match_max_side": run_config.get("match_max_side"),
            "recorded_protocol_config_path": recorded_protocol,
            "expected_protocol_config_path": str(protocol_config_path),
        }

    summary = _read_json(summary_path)
    rows = summary.get("results")
    if not isinstance(rows, list):
        return {"status": "RUN_INCOMPLETE", "reason": "results is not a list"}

    expected_pairs = _expected_local_pairs(frozen_config)
    observed_pairs = {
        tuple(sorted((int(row["idx_i"]), int(row["idx_j"]))))
        for row in rows
        if "idx_i" in row and "idx_j" in row
    }
    missing_pairs = sorted(expected_pairs - observed_pairs)
    unexpected_pairs = sorted(observed_pairs - expected_pairs)
    malformed_pairs = [
        [row.get("idx_i"), row.get("idx_j")]
        for row in rows
        if not PAIRWISE_FIELDS.issubset(row)
    ]
    if missing_pairs or unexpected_pairs or malformed_pairs:
        return {
            "status": "RUN_INCOMPLETE",
            "missing_pairs": [list(pair) for pair in missing_pairs],
            "unexpected_pairs": [list(pair) for pair in unexpected_pairs],
            "malformed_pairs": malformed_pairs,
        }

    graph = _read_json(graph_path)
    _read_json(runtime_path)
    return {
        "status": "PASS",
        "network_status": graph.get("status"),
        "n_pairs": len(rows),
        "quality_statuses": sorted({str(row["status"]) for row in rows}),
    }
