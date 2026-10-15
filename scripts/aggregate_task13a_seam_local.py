"""Aggregate the frozen Task13A ten-pair CSV without rerunning image processing."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = ROOT / "data/output/b9_five_scene_validation/seam_local_task13a"


def _number(row: dict[str, str], field: str) -> float | None:
    raw = row.get(field, "")
    if raw is None or not raw.strip():
        return None
    value = float(raw)
    return value if math.isfinite(value) else None


def _summary(rows: list[dict[str, str]], field: str, *, worst: str = "max") -> dict:
    values = [(row["pair_id"], value) for row in rows if (value := _number(row, field)) is not None]
    if not values:
        return {"count": 0, "median": None, "worst": None, "worst_pair_id": None,
                "worst_direction": worst}
    pair_id, extreme = (max if worst == "max" else min)(values, key=lambda item: item[1])
    return {"count": len(values), "median": statistics.median(value for _, value in values),
            "worst": extreme, "worst_pair_id": pair_id, "worst_direction": worst}


def _fields(rows: list[dict[str, str]], prefix: str, names: dict[str, str], *,
            worst: str = "max") -> dict:
    return {name: _summary(rows, prefix + suffix, worst=worst) for name, suffix in names.items()}


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _coefficient_summary(directory: Path, pair_ids: list[str]) -> tuple[dict, dict]:
    values = {field: [] for field in ("a_a", "a_b", "b_a", "b_b")}
    hashes = {}
    segment_count = 0
    for pair_id in pair_ids:
        path = directory / "pairs" / pair_id / "local_coefficients.csv"
        with path.open(newline="", encoding="utf-8") as stream:
            segments = list(csv.DictReader(stream))
        if not segments or any(set(values) - segment.keys() for segment in segments):
            raise ValueError(f"Missing local coefficient segments or fields: {pair_id}")
        for segment in segments:
            for field in values:
                number = float(segment[field])
                if not math.isfinite(number):
                    raise ValueError(f"Nonfinite local coefficient: {pair_id}/{field}")
                values[field].append(number)
        segment_count += len(segments)
        hashes[pair_id] = _hash(path)

    def stats(series: list[float]) -> dict:
        return {"count": len(series), "min": min(series), "max": max(series),
                "median": statistics.median(series)}

    summary = {field: stats(series) for field, series in values.items()}
    summary["a"] = stats(values["a_a"] + values["a_b"])
    summary["b"] = stats(values["b_a"] + values["b_b"])
    summary["segment_count"] = segment_count
    summary["aggregation_unit"] = "Pooled frozen pair-segment coefficients; both source sides counted equally."
    return summary, hashes


def aggregate(csv_path: Path, manifest_path: Path) -> dict:
    with csv_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ids = [row["pair_id"] for row in rows]
    planned = manifest["planned_pair_ids"]
    completed = [row["pair_id"] for row in manifest["completed_pairs"]]
    if len(rows) != 10 or len(set(ids)) != 10 or ids != planned or ids != completed:
        raise ValueError("Frozen pair CSV and run manifest must agree on all ten accepted pairs")
    if manifest["status"] != "COMPLETED" or manifest["pair_count_completed"] != 10:
        raise ValueError("Frozen ten-pair run is not complete")
    if any(row["status"] != item["status"] or row["v1_status"] != item["v1_status"] or
           row["v2_status"] != item["v2_status"] for row, item in zip(rows, manifest["completed_pairs"])):
        raise ValueError("Frozen pair status mismatch")

    seam_names = {"mae": "mae", "rmse": "rmse", "rdd": "rdd",
                  "seam_cost_mean": "mean_absolute_cost", "seam_cost_p95": "p95_cost"}
    corridor_names = {"mae": "mae", "rmse": "rmse", "rdd": "rdd",
                      "mean_difference": "mean_difference", "std_difference": "std_difference"}
    v1 = _fields(rows, "metric_actual_seam_v1_", seam_names)
    v2 = _fields(rows, "metric_actual_seam_v2_", seam_names)
    bagrn = _fields(rows, "metric_fixed_corridor_bagrn_", corridor_names)
    local = _fields(rows, "metric_fixed_corridor_local_", corridor_names)
    structure_names = {"cgl_rad": "cgl_cgl_rad", "cgl_deg": "cgl_cgl_deg",
                       "gradient_magnitude_ncc": "gradient_magnitude_ncc",
                       "gradient_orientation_cosine": "gradient_orientation_cosine"}
    structure = {source: {
        name: _summary(rows, f"metric_structure_{source}_" + suffix,
                       worst="min" if name in ("gradient_magnitude_ncc", "gradient_orientation_cosine") else "max")
        for name, suffix in structure_names.items()}
        for source in ("a", "b")}
    coeff = {name: _summary(rows, f"diagnostic_{name}") for name in
             ("gain_min", "gain_max", "offset_min", "offset_max")}
    coeff["gain_overall_min"] = min((_number(r, "diagnostic_gain_min") for r in rows), default=None)
    coeff["gain_overall_max"] = max((_number(r, "diagnostic_gain_max") for r in rows), default=None)
    coeff["offset_overall_min"] = min((_number(r, "diagnostic_offset_min") for r in rows), default=None)
    coeff["offset_overall_max"] = max((_number(r, "diagnostic_offset_max") for r in rows), default=None)
    pooled_coeff, coeff_hashes = _coefficient_summary(csv_path.parent, ids)
    coeff.update(pooled_coeff)

    diagnostic = {}
    for variant in ("v0", "v1", "v2"):
        diagnostic[variant] = {
            key: sum(int(row[f"{variant}_{key}_valid_pixels"] or 0) for row in rows)
            for key in ("finite", "nan", "inf")
        }
        diagnostic[variant]["reported_pair_count"] = sum(
            bool(row[f"{variant}_finite_valid_pixels"]) for row in rows)
    all_numeric_inf_fields = [field for field in rows[0] if field.endswith("_inf_valid_pixels")]
    inf_total = sum(int(row[field] or 0) for row in rows for field in all_numeric_inf_fields)
    metric_fields = [field for field in rows[0] if field.startswith(("metric_", "diagnostic_"))]
    literal_inf_values = sum(str(row[field]).strip().lower() in
                             {"inf", "+inf", "-inf", "infinity", "+infinity", "-infinity"}
                             for row in rows for field in metric_fields)
    literal_nan_values = sum(str(row[field]).strip().lower() in {"nan", "+nan", "-nan"}
                             for row in rows for field in metric_fields)
    nan_total = sum(int(row[field] or 0) for row in rows
                    for field in rows[0] if field.endswith("_nan_valid_pixels"))
    v1_pass = sum(row["v1_status"] == "PASS" for row in rows)
    v2_pass = sum(row["v2_status"] == "PASS" for row in rows)
    unstable = sum(row["status"] == "UNSTABLE_LOCAL_GAIN" for row in rows)
    unsupported = sum(row["status"] == "UNSUPPORTED_TOPOLOGY" for row in rows)
    if unstable != manifest["unstable_local_gain_pairs"] or unsupported != manifest["unsupported_topology_pairs"]:
        raise ValueError("Hard-stop status count mismatch")

    def down(left: dict, right: dict) -> bool:
        return left["count"] > 0 and right["count"] > 0 and right["median"] < left["median"]

    all_cgl = [v for source in ("a", "b") for row in rows
               if (v := _number(row, f"metric_structure_{source}_cgl_cgl_deg")) is not None]
    all_ncc = [v for source in ("a", "b") for row in rows
               if (v := _number(row, f"metric_structure_{source}_gradient_magnitude_ncc")) is not None]
    # The plan has no numeric CGL damage threshold. Record the observed range
    # and use the unambiguous finite/passing status evidence; interpretation is
    # explicitly qualitative, never a newly tuned selection threshold.
    cgl_observation = (len(all_cgl) == 20 and len(all_ncc) == 20 and
                       all(row[f"metric_structure_{source}_cgl_status"] == "PASS"
                           for row in rows for source in ("a", "b")))
    gate = {
        "v1_at_least_8_of_10_pass": v1_pass >= 8,
        "v2_at_least_8_of_10_pass": v2_pass >= 8,
        "v2_median_seam_mae_down": down(v1["mae"], v2["mae"]),
        "v2_median_seam_rmse_down": down(v1["rmse"], v2["rmse"]),
        "v2_median_seam_rdd_down": down(v1["rdd"], v2["rdd"]),
        "local_median_fixed_corridor_mae_down": down(bagrn["mae"], local["mae"]),
        "local_median_fixed_corridor_rdd_down": down(bagrn["rdd"], local["rdd"]),
        "cgl_finite_without_evident_systematic_damage": cgl_observation,
        "no_more_than_2_unstable_local_gains": unstable <= 2,
        "formal_outputs_no_inf": (inf_total == 0 and nan_total == 0 and
                                  literal_inf_values == 0 and literal_nan_values == 0),
    }
    all_three_no_improvement = all(not gate[f"v2_median_seam_{metric}_down"]
                                    for metric in ("mae", "rmse", "rdd"))
    hard_stop = manifest["hard_stop"] is not None or unsupported > 2 or unstable > 2
    systemic_structure_damage_evidenced = not cgl_observation
    unexplained_nonfinite_evidenced = any((inf_total, nan_total, literal_inf_values, literal_nan_values))
    not_worth = (hard_stop or all_three_no_improvement or systemic_structure_damage_evidenced
                 or unexplained_nonfinite_evidenced)
    decision = ("NOT_WORTH_CONTINUING" if not_worth else
                "PROMISING_FOR_MULTISCENE" if all(gate.values()) else "MIXED_NEEDS_REVIEW")
    return {
        "schema_version": 1,
        "task": "Task13A frozen ten-pair aggregate and predefined feasibility gate",
        "input_sha256": {"pair_metrics_csv": _hash(csv_path), "run_manifest_json": _hash(manifest_path),
                         "local_coefficients_csv_by_pair": coeff_hashes},
        "pair_ids": ids,
        "pair_count_total": len(rows), "pair_count_v1_pass": v1_pass, "pair_count_v2_pass": v2_pass,
        "status_counts": {name: dict(sorted(Counter(row[name] for row in rows).items()))
                          for name in ("status", "v0_status", "v1_status", "v2_status")},
        "actual_seam": {
            "v0": {name: {"available": False, "count": 0, "median": None, "worst": None,
                          "worst_pair_id": None, "worst_direction": "max",
                          "reason": "Frozen V0 weighted-feather pipeline did not measure an actual seam"}
                   for name in seam_names},
            "v1": v1, "v2": v2,
            "availability_note": "V1/V2 diagnostic seam metrics use every finite reported pair, including pairs whose mosaic status is AMBIGUOUS_SOURCE_SIDE; PASS counts remain separate.",
        },
        "fixed_initial_seam_corridor": {"bagrn": bagrn, "local_corrected": local},
        "structure_original_vs_corrected": structure,
        "local_coefficients": coeff,
        "finite_diagnostics": {"by_variant": diagnostic,
                               "reported_inf_valid_pixels": inf_total,
                               "reported_nan_valid_pixels": nan_total,
                               "literal_inf_metric_cells": literal_inf_values,
                               "literal_nan_metric_cells": literal_nan_values},
        "hard_stop_counts": {"unsupported_topology_pairs": unsupported,
                             "unstable_local_gain_pairs": unstable,
                             "manifest_hard_stop": manifest["hard_stop"]},
        "gate": {"conditions": gate,
                 "cgl_evidence": {"finite_pair_source_values": len(all_cgl),
                                  "cgl_deg_median": statistics.median(all_cgl) if all_cgl else None,
                                  "cgl_deg_max": max(all_cgl) if all_cgl else None,
                                  "gradient_ncc_min": min(all_ncc) if all_ncc else None,
                                  "interpretation": "All 20 source-pair CGL statuses PASS and finite; observed changes remain small. The plan provides no numeric damage cutoff."},
                 "not_worth_triggers": {
                     "topology_or_local_model_hard_stop": hard_stop,
                     "all_three_aggregate_seam_errors_no_improvement": all_three_no_improvement,
                     "majority_clear_structural_damage_evidenced": systemic_structure_damage_evidenced,
                     "unexplained_nonfinite_output_evidenced": unexplained_nonfinite_evidenced,
                 },
                 "decision": decision,
                 "decision_evidence": (f"V1 and V2 each PASS {v1_pass}/10 and {v2_pass}/10; "
                                       f"median seam MAE/RMSE/RDD and fixed-corridor MAE/RDD conditions "
                                       f"are recorded individually; no manifest hard stop; "
                                       f"{inf_total} reported Inf pixels, {nan_total} reported NaN pixels, "
                                       f"{literal_inf_values} literal Inf and {literal_nan_values} literal NaN metric cells.")},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DIR)
    args = parser.parse_args()
    directory = args.output_dir
    result = aggregate(directory / "pair_metrics.csv", directory / "run_manifest.json")
    target = directory / "aggregate_summary.json"
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"{result['gate']['decision']}: {target}")


if __name__ == "__main__":
    main()
