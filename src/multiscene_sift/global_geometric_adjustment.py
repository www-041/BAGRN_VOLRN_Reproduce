"""Offline global geometric-adjustment diagnostics for an existing five-scene run.

This module deliberately consumes saved artifacts only.  It never invokes a
matcher or reconstructs pairwise correspondences from imagery.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def _read_json(path: Path, default: Any) -> Any:
    if not path.is_file():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _edge_key(i: int, j: int) -> tuple[int, int]:
    return min(i, j), max(i, j)


def _point_artifact_edges(run_dir: Path) -> set[tuple[int, int]]:
    """Find edges explicitly represented by stored point-level CSVs.

    A CSV is accepted only when it contains ``edge_i``/``edge_j`` columns.
    A generic residual CSV without edge identity is intentionally not guessed
    to belong to an accepted edge.
    """
    found: set[tuple[int, int]] = set()
    for path in run_dir.rglob("*.csv"):
        try:
            with path.open("r", newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                fields = set(reader.fieldnames or ())
                point_fields = {"point_id", "ref_x", "ref_y", "tgt_x", "tgt_y"}
                if not {"edge_i", "edge_j"}.issubset(fields) or not point_fields.issubset(fields):
                    continue
                for row in reader:
                    if row.get("edge_i") in (None, "") or row.get("edge_j") in (None, ""):
                        continue
                    found.add(_edge_key(int(row["edge_i"]), int(row["edge_j"])))
        except (OSError, ValueError, UnicodeError):
            continue
    return found


def discover_five_scene_adjustment_inputs(five_scene_run_dir: str | Path) -> dict:
    """Audit saved five-scene inputs before any global adjustment.

    The returned manifest is ``READY`` only when every accepted direct edge
    has explicitly recoverable point-level observations.  Missing point data
    is a hard stop; this function has no rematching fallback by design.
    """
    run_dir = Path(five_scene_run_dir)
    dataset = _read_json(run_dir / "dataset_manifest.json", {})
    registration = _read_json(run_dir / "global_registration.json", {})
    spanning = _read_json(run_dir / "spanning_tree.json", {})
    pairwise = _read_json(run_dir / "pairwise_summary.json", {})
    results = pairwise.get("results", []) if isinstance(pairwise, dict) else []
    accepted = [row for row in results if row.get("status") == "OK"]
    point_edges = _point_artifact_edges(run_dir)

    accepted_edges = []
    for row in accepted:
        i, j = int(row["idx_i"]), int(row["idx_j"])
        key = _edge_key(i, j)
        accepted_edges.append(
            {
                "edge": [i, j],
                "status": row.get("status"),
                "inliers": int(row.get("inliers", 0)),
                "point_artifact_status": "READY" if key in point_edges else "MISSING",
            }
        )

    missing = [row["edge"] for row in accepted_edges if row["point_artifact_status"] == "MISSING"]
    return {
        "status": "READY" if not missing else "INSUFFICIENT_EXISTING_ARTIFACTS",
        "rematch_allowed": False,
        "run_dir": str(run_dir),
        "scene_indices": [int(scene["index"]) for scene in dataset.get("scenes", [])],
        "reference_idx": registration.get("reference_index", spanning.get("reference_index")),
        "spanning_tree_edges": spanning.get("edges", []),
        "accepted_edges": accepted_edges,
        "missing_point_edges": missing,
    }


def write_adjustment_input_manifest(manifest: dict, output_dir: str | Path) -> Path:
    """Write the Task 0 audit without changing the source run artifacts."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "00_adjustment_input_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
