"""Diagnostic-only audit of the existing phase-validation chain.

This module never changes registration, phase-helper, radiometric, or mosaic
semantics.  It consumes saved two-edge artifacts and provides independent
counterfactual measurements for the exact validation inputs.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


AUDIT_EDGES = {
    (0, 6): "WORKING_CONTROL",
    (2, 5): "SUSPECT_PHASE_FALSE_ALARM",
}


def _edge_key(edge: str | tuple[int, int]) -> str:
    if isinstance(edge, str):
        i, j = (int(part) for part in edge.split("-"))
    else:
        i, j = int(edge[0]), int(edge[1])
    return f"{i}-{j}"


def _read_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _edge_rows(payload: Any) -> dict[str, dict]:
    if isinstance(payload, dict) and isinstance(payload.get("edges"), list):
        payload = payload["edges"]
    if isinstance(payload, dict):
        return {_edge_key(key): value for key, value in payload.items() if "-" in str(key)}
    if not isinstance(payload, list):
        return {}
    rows = {}
    for row in payload:
        edge = row.get("edge", row.get("edge_key"))
        if edge is not None:
            rows[_edge_key(edge)] = row
    return rows


def _load_tiles(path: Path) -> dict[str, list[dict]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    out: dict[str, list[dict]] = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = _edge_key(row["edge"])
            row["grid_n"] = int(row["grid_n"])
            row["tile_row"] = int(row["tile_row"])
            row["tile_col"] = int(row["tile_col"])
            row["phase_mag_px"] = float(row["phase_mag_px"])
            row["accepted"] = str(row.get("accepted", "")).strip().lower() in {"1", "true", "yes"}
            out.setdefault(key, []).append(row)
    return out


def load_phase_audit_baseline(
    edge_reliability_dir: str | Path,
    local_affine_dir: str | Path,
    selected_pair_dir: str | Path | None = None,
) -> dict:
    """Load the frozen two-edge phase evidence without rerunning registration."""
    edge_root = Path(edge_reliability_dir)
    local_root = Path(local_affine_dir)
    selected_root = Path(selected_pair_dir) if selected_pair_dir is not None else edge_root.parent / "nine_scene_selected_pair_consistency"
    required = (
        edge_root / "00_edge_reliability_baseline.json",
        edge_root / "07_direct_residual_summary.json",
        edge_root / "07_direct_residual_tiles.csv",
        local_root / "00_local_affine_baseline.json",
        local_root / "06_global_vs_local_phase_summary.json",
        selected_root / "06_canonical_world_edge_transforms.json",
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f"required phase audit artifact missing: {path}")

    reliability = _edge_rows(_read_json(required[0]))
    direct_summary = _edge_rows(_read_json(required[1]))
    local_summary = _edge_rows(_read_json(required[4]))
    canonical = _read_json(required[5]).get("edges", {})
    tiles = _load_tiles(required[2])
    edges: dict[str, dict] = {}
    for edge, role in AUDIT_EDGES.items():
        key = _edge_key(edge)
        if key not in reliability or key not in direct_summary:
            raise ValueError(f"missing baseline edge {key}")
        matrix_row = canonical.get(key, canonical.get(f"{edge[1]}-{edge[0]}"))
        if matrix_row is None or matrix_row.get("matrix") is None:
            raise ValueError(f"missing canonical world matrix for {key}")
        edges[key] = {
            "edge": [edge[0], edge[1]],
            "role": role,
            "reliability": reliability[key],
            "local_affine_phase": local_summary.get(key),
            "old_phase_median_px": float(direct_summary[key]["median_px"]),
            "old_phase_summary": direct_summary[key],
            "direct_tiles": tiles.get(key, []),
            "world_matrix": np.asarray(matrix_row["matrix"], dtype=float).tolist(),
            "pixel_matrix": None,
        }

    selected_json = selected_root / "02_new_pairwise_registration_results.json"
    if selected_json.is_file():
        for row in _read_json(selected_json).get("results", []):
            key = _edge_key((int(row["idx_i"]), int(row["idx_j"])))
            if key in edges:
                edges[key]["pixel_matrix"] = row.get("pixel_matrix")

    return {"edge_keys": [_edge_key(edge) for edge in AUDIT_EDGES], "edges": edges}


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.integer):
        return int(value)
    return value


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(value), indent=2, ensure_ascii=False), encoding="utf-8")


def sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
