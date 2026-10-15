"""Task14A generic adapter replay gate and downstream resume runner.

The first command is deliberately a five-scene gate.  It consumes only the
frozen Task13A/B artifacts and BAGRN rasters.  The 13-scene command is kept
behind that gate and is not allowed to create downstream artifacts when the
gate fails.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[1]
BASE5 = ROOT / "data/output/b9_five_scene_validation"
BASE13 = ROOT / "data/output/b9_13scene_task14"
ADAPTER5 = BASE13 / "adapter_validation/five_scene_replay"

import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_task13a1_source_side import _footprint_polygon, _load_coefficients, _load_saved_seam
from src.seam_local.adapter import run_multiscene_adapter


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return None
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items() if not isinstance(v, np.ndarray)}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    return value


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _array_diff(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    if a.shape != b.shape:
        return {"shape_equal": False, "max_abs": None, "finite_equal": False}
    finite = np.isfinite(a) & np.isfinite(b)
    same_nan = np.isnan(a) == np.isnan(b)
    if not np.any(finite):
        return {"shape_equal": True, "max_abs": 0.0, "finite_equal": bool(same_nan.all())}
    delta = np.abs(a[finite].astype(np.float64) - b[finite].astype(np.float64))
    return {"shape_equal": True, "max_abs": float(delta.max()),
            "finite_equal": bool(same_nan.all() and np.isfinite(delta).all())}


def _load_five_inputs(protocol: dict[str, Any]):
    paths = [ROOT / scene["path"] for scene in protocol["scenes"]]
    return paths, [(int(pair["scene_i"]), int(pair["scene_j"])) for pair in protocol["accepted_pairs"]]


def _load_five_footprints(paths: list[Path]):
    footprints = []
    for path in paths:
        with rasterio.open(path) as src:
            footprints.append(_footprint_polygon(src.read_masks(1) > 0, src.transform))
    return footprints


def _reference_pair_rows() -> dict[str, dict[str, str]]:
    return {row["pair_id"]: row for row in _read_csv(BASE5 / "multiscene_task13b/authoritative_pair_table.csv")}


def _compare_pair_artifacts(result, protocol: dict[str, Any]) -> dict[str, Any]:
    reference_rows = _reference_pair_rows()
    seam_checks: dict[str, Any] = {}
    coefficient_checks: dict[str, Any] = {}
    ownership_checks: dict[str, Any] = {}
    process_status_checks: dict[str, Any] = {}
    for pair_id, row in result.pair_results.items():
        process = row.process
        old_metrics = next(r for r in _read_csv(BASE5 / "seam_local_task13a/pair_metrics.csv") if r["pair_id"] == pair_id)
        process_status_checks[pair_id] = process.status == old_metrics["status"]
        origin = process.crop_origin
        shape = process.corrected_a.shape if process.corrected_a is not None else None
        pair_dir = BASE5 / "seam_local_task13a/pairs" / pair_id
        if origin is not None and shape is not None and process.initial_seam is not None and process.refined_seam is not None:
            with rasterio.open(ROOT / protocol["scenes"][0]["path"]) as src:
                transform = src.transform
            initial_ref = _load_saved_seam(pair_dir / "seam_initial.geojson", origin, transform, shape)
            refined_ref = _load_saved_seam(pair_dir / "seam_refined.geojson", origin, transform, shape)
            seam_checks[pair_id] = {
                "initial_equal": bool(np.array_equal(process.initial_seam.row_col_path, initial_ref.row_col_path)),
                "refined_equal": bool(np.array_equal(process.refined_seam.row_col_path, refined_ref.row_col_path)),
            }
            saved = _load_coefficients(pair_dir / "local_coefficients.csv")
            actual = tuple((s.index, s.center_arc, s.a_a, s.b_a, s.a_b, s.b_b) for s in process.local_segments)
            coefficient_checks[pair_id] = {
                "count_equal": len(actual) == len(saved),
                "max_abs": float(np.max(np.abs(np.asarray(actual, dtype=float) - np.asarray(saved, dtype=float)))) if len(actual) == len(saved) else None,
            }
        ref = reference_rows[pair_id]
        ownership_checks[pair_id] = {
            "expected": ref["ownership_status"],
            "actual": row.source_side_status,
            "equal": ref["ownership_status"] == row.source_side_status,
        }
    return {"process_status": process_status_checks, "seams": seam_checks,
            "coefficients": coefficient_checks, "ownership": ownership_checks}


def run_five_replay_gate() -> dict[str, Any]:
    started = time.perf_counter()
    protocol = _read_json(BASE5 / "seam_local_task13a/protocol.json")
    paths, pairs = _load_five_inputs(protocol)
    footprints = _load_five_footprints(paths)
    result = run_multiscene_adapter(paths, paths, pairs, output_dir=ADAPTER5, footprints=footprints)
    checks = _compare_pair_artifacts(result, protocol)

    old_labels_path = BASE5 / "multiscene_task13b1_tie_resolution/resolved_source_label_map.tif"
    old_methods_path = BASE5 / "multiscene_task13b1_tie_resolution/resolved_label_method_map.tif"
    if not old_methods_path.exists():
        old_methods_path = BASE5 / "multiscene_task13b/labels/label_method_map.tif"
    with rasterio.open(old_labels_path) as src:
        old_labels = src.read(1)
    label_diff = int(np.count_nonzero(old_labels != result.labels))
    reference_final = _read_json(BASE5 / "multiscene_task13b1_tie_resolution/continuation/final_metrics.json")
    v1_reference = BASE5 / "multiscene_task13b1_tie_resolution/continuation/v1_multiscene_label_blend.tif"
    v2_reference = BASE5 / "multiscene_task13b1_tie_resolution/continuation/v2_multiscene_local_blend.tif"
    with rasterio.open(v1_reference) as src:
        v1_ref = src.read(1)
    with rasterio.open(v2_reference) as src:
        v2_ref = src.read(1)
    v1_diff = _array_diff(result.v1_mosaic, v1_ref)
    v2_diff = _array_diff(result.v2_mosaic, v2_ref)
    support = int(np.count_nonzero(np.isfinite(result.v1_mosaic)))
    source_ok = all(v["equal"] for v in checks["ownership"].values())
    seams_ok = all(v["initial_equal"] and v["refined_equal"] for v in checks["seams"].values())
    coeff_ok = all(v["count_equal"] and (v["max_abs"] is not None and v["max_abs"] <= 1e-12) for v in checks["coefficients"].values())
    labels_ok = label_diff == 0 and result.diagnostics["unresolved_pixels"] == 0
    # Reuse the frozen Stage -1 streaming-equivalence tolerance exactly.
    v_ok = (v1_diff["max_abs"] <= 1e-3 and v2_diff["max_abs"] <= 1e-3)
    gate_pass = bool(source_ok and seams_ok and coeff_ok and labels_ok and v_ok and support == 22167910)
    report = {
        "task": "Task14A generic N-scene adapter five-scene replay gate",
        "status": "PASS" if gate_pass else "HARD_STOP_ADAPTER_FIVE_SCENE_REPLAY_MISMATCH",
        "scene_count": 5,
        "pair_count": len(result.pair_results),
        "checks": checks,
        "label_difference_pixels": label_diff,
        "adapter_diagnostics": _jsonable(result.diagnostics),
        "reference_unresolved": 0,
        "adapter_unresolved": int(result.diagnostics["unresolved_pixels"]),
        "reference_support": reference_final["support_pixels"],
        "adapter_support": support,
        "v1_difference": v1_diff,
        "v2_difference": v2_diff,
        "v1_v2_support_expected": 22167910,
        "elapsed_sec": time.perf_counter() - started,
        "no_13_scene_started": True,
    }
    ADAPTER5.mkdir(parents=True, exist_ok=True)
    (ADAPTER5 / "replay_gate.json").write_text(json.dumps(report, indent=2, default=_jsonable) + "\n", encoding="utf-8")
    lines = ["# Task14A five-scene adapter replay gate", "", f"Decision: **{report['status']}**.",
             "", f"Pair artifacts compared: {len(result.pair_results)}; seam paths exact: {seams_ok}; local coefficients exact: {coeff_ok}; ownership exact: {source_ok}.",
             f"Label difference pixels: {label_diff}; unresolved after Task13B.1: {report['adapter_unresolved']}; V1/V2 support: {support}.",
             f"V1 max absolute difference: {v1_diff['max_abs']}; V2 max absolute difference: {v2_diff['max_abs']}.",
             "", "13-scene execution is permitted only when this report status is PASS.", ""]
    (ADAPTER5 / "REPLAY_GATE_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    if not gate_pass:
        raise RuntimeError("HARD_STOP_ADAPTER_FIVE_SCENE_REPLAY_MISMATCH")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--five-replay-gate", action="store_true")
    args = parser.parse_args()
    if args.five_replay_gate:
        print(json.dumps(run_five_replay_gate(), indent=2, default=_jsonable))
        return
    parser.error("the 13-scene resume is locked until --five-replay-gate passes")


if __name__ == "__main__":
    main()
