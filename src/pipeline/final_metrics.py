"""Final metric collection and explicit scientific-quality decision gates."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _number(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def v0_semantic_gate(root: str | Path) -> dict[str, Any]:
    """Verify that Stage06's registered V0 is a formal weighted-feather output."""
    root = Path(root)
    summary = _json(root / "stages/10_mosaics/mosaic_summary.json")
    required = root / "stages/10_mosaics/v0_bagrn_weighted.tif"
    checks = {
        "mosaic_mode": summary.get("v0", "NOT_MEASURED"),
        "file_exists": required.is_file(),
        "weighted_feather_declared": "weighted" in str(summary.get("v0", "")).lower() or "feather" in str(summary.get("v0", "")).lower(),
    }
    checks["status"] = "PASS" if checks["file_exists"] and checks["weighted_feather_declared"] else "HARD_STOP_V0_SEMANTIC_MISMATCH"
    return checks


def five_scene_regression_gate(root: str | Path) -> dict[str, Any]:
    root = Path(root)
    labels = _json(root / "stages/08_multiscene_labeling/labeling_summary.json")
    stage11 = _json(root / "stages/11_metrics/metrics_summary.json")
    pair_path = root / "stages/07_pairwise_seam_local/pairwise_results.json"
    pairs = []
    try:
        pairs = json.loads(pair_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        pass
    counts = {}
    for row in pairs if isinstance(pairs, list) else []:
        status = row.get("status", "UNKNOWN")
        counts[status] = counts.get(status, 0) + 1
    checks = {
        "union_support": labels.get("union_valid_pixels") == 22167910,
        "source_side_9_pass": counts.get("PASS") == 9,
        "source_side_1_requires": counts.get("REQUIRES_MULTISCENE_LABELING") == 1,
        "unresolved_zero": labels.get("unresolved_pixels") == 0,
        "invalid_zero": labels.get("invalid_label_pixels") == 0,
        "two_scene_disagreement_zero": labels.get("two_scene_disagreement_pixels") == 0,
        "quality_gate": stage11.get("quality_gate") == "PASS",
        "v0_v1_v2_present": all((root / "stages/10_mosaics" / name).is_file() for name in ("v0_bagrn_weighted.tif", "v1_multiscene_label_blend.tif", "v2_local_corrected_multiscene.tif")),
    }
    return {"status": "PASS" if all(checks.values()) else "HARD_STOP_GENERIC_PIPELINE_FIVE_SCENE_REGRESSION", "checks": checks, "pair_status_counts": counts}


def _fresh_five_scene_status(root: Path) -> dict[str, Any]:
    """Classify the independent Task15 five-scene run, if it exists."""
    sibling = root.parent / "b9_5scene"
    if sibling.is_dir():
        gate = five_scene_regression_gate(sibling)
        return {
            "status": gate["status"],
            "evidence_type": "fresh_task15_independent_five_scene_run",
            "artifact_root": str(sibling),
            "gate": gate,
        }
    return {
        "status": "HISTORICAL_REFERENCE_PASS / NOT_REEXECUTED",
        "evidence_type": "historical_reference_only",
        "artifact_root": str(sibling),
    }


def _invariance_artifact(root: Path, name: str) -> dict[str, Any]:
    path = root / "11_metrics/invariance" / name
    value = _json(path)
    if not value:
        return {"status": "NOT_MEASURED", "artifact": str(path)}
    value.setdefault("artifact", str(path))
    return value


def compute_final_metrics(root: str | Path) -> dict[str, Any]:
    root = Path(root)
    labels = _json(root / "stages/08_multiscene_labeling/labeling_summary.json")
    metrics = _json(root / "stages/11_metrics/metrics_summary.json")
    scale = _json(root / "stages/12_report/scale_summary.json")
    correction_summary = _json(root / "stages/09_correction/correction_summary.json")
    scene_invariance = _invariance_artifact(root, "scene_order_invariance.json")
    correction_invariance = _invariance_artifact(root, "correction_order_invariance.json")
    boundary = metrics.get("boundary_metrics", {}) if isinstance(metrics.get("boundary_metrics"), dict) else {}
    structural = metrics.get("structural", [])
    ncc = [_number(row.get("gradient_magnitude_ncc")) for row in structural if isinstance(row, dict)]
    ncc = [value for value in ncc if value is not None]
    required_numeric = {
        "weighted_bagrn_mae": _number(boundary.get("bagrn_weighted_mae")),
        "weighted_v2_mae": _number(boundary.get("v2_weighted_mae")),
        "weighted_bagrn_rdd": _number(boundary.get("bagrn_weighted_rdd")),
        "weighted_v2_rdd": _number(boundary.get("v2_weighted_rdd")),
        "median_bagrn_mae": _number(boundary.get("median_bagrn_mae")),
        "median_v2_mae": _number(boundary.get("median_v2_mae")),
        "median_bagrn_rdd": _number(boundary.get("median_bagrn_rdd")),
        "median_v2_rdd": _number(boundary.get("median_v2_rdd")),
    }
    quality = {
        "unresolved_labels": labels.get("unresolved_pixels", "NOT_MEASURED"),
        "invalid_labels": labels.get("invalid_label_pixels", "NOT_MEASURED"),
        "two_scene_disagreement": labels.get("two_scene_disagreement_pixels", "NOT_MEASURED"),
        "union_support": labels.get("union_valid_pixels", scale.get("union_valid_pixels", "NOT_MEASURED")),
        "gradient_ncc_min": min(ncc) if ncc else "NOT_MEASURED",
        "weighted_boundary": required_numeric,
        "v0_semantic_gate": v0_semantic_gate(root),
        "pairwise_score_tie_pixels": labels.get("pairwise_score_tie_pixels", "NOT_MEASURED"),
        "clipped_interiority_fallback_pixels": labels.get("clipped_interiority_fallback_pixels", "NOT_MEASURED"),
        "resolved_by_unclipped_interiority": labels.get("resolved_by_unclipped_interiority", "NOT_MEASURED"),
        "resolved_by_raw_edt": labels.get("resolved_by_raw_edt", "NOT_MEASURED"),
        "cycle_pixels": labels.get("cycle_pixels", "NOT_MEASURED"),
        "cycle_fraction": labels.get("cycle_fraction", "NOT_MEASURED"),
        "resolved_preference_edge_count": labels.get("resolved_preference_edge_count", metrics.get("pairwise", {}).get("PASS", "NOT_MEASURED")),
        "unclipped_normalized_interiority_pixels": labels.get("unclipped_normalized_interiority_pixels", "NOT_MEASURED"),
        "raw_edt_pixels": labels.get("raw_edt_pixels", "NOT_MEASURED"),
        "label_pixels_per_scene": labels.get("label_pixels_per_scene", "NOT_MEASURED"),
        "label_method_counts": labels.get("label_method_counts", labels.get("label_method_pixels", "NOT_MEASURED")),
        "score_margin_min": labels.get("score_margin_min", "NOT_MEASURED"),
        "score_margin_median": labels.get("score_margin_median", "NOT_MEASURED"),
        "score_margin_p95": labels.get("score_margin_p95", "NOT_MEASURED"),
    }
    complete = all(value is not None and math.isfinite(float(value)) for value in required_numeric.values()) and bool(ncc) and all(math.isfinite(float(value)) for value in ncc)
    labels_clean = quality["unresolved_labels"] == 0 and quality["invalid_labels"] == 0 and quality["two_scene_disagreement"] == 0
    correction_order = correction_invariance.get("status") == "PASS" and correction_invariance.get("evidence_type", "").startswith("fresh_task15")
    scene_order = scene_invariance.get("status") == "PASS" and scene_invariance.get("evidence_type", "").startswith("fresh_task15")
    support_gate = quality["union_support"] == 62033096 and all((root / "stages/10_mosaics" / name).is_file() for name in ("v0_bagrn_weighted.tif", "v1_multiscene_label_blend.tif", "v2_local_corrected_multiscene.tif"))
    finite_structural = bool(structural) and all(bool(item.get("finite")) and _number(item.get("gradient_magnitude_ncc")) is not None and float(item.get("gradient_magnitude_ncc")) >= 0.99 for item in structural if isinstance(item, dict))
    no_invalid_outputs = labels_clean and metrics.get("status") == "SUCCESS" and all((root / "stages/09_correction" / f"corrected_scene_{i:03d}.tif").is_file() for i in range(int(scale.get("scene_count", 0))))
    core_metrics_match = (
        scale.get("union_valid_pixels") == 62033096
        and scale.get("multiscene_pixels") == 24731286
        and quality["cycle_pixels"] == 8577496
        and required_numeric["weighted_bagrn_mae"] == 105.95474750609073
        and required_numeric["weighted_v2_mae"] == 93.79802987358079
        and required_numeric["weighted_bagrn_rdd"] == 57.141824803056934
        and required_numeric["weighted_v2_rdd"] == 41.76302625936166
        and min(ncc) == 0.9986118416578263
    )
    quality["fresh_scene_order_invariance"] = scene_invariance
    quality["fresh_correction_order_invariance"] = correction_invariance
    quality["fresh_five_scene_regression"] = _fresh_five_scene_status(root)
    quality["support_gate"] = support_gate
    quality["finite_structural_gate"] = finite_structural
    quality["no_invalid_outputs_gate"] = no_invalid_outputs
    quality["all_core_metrics_match"] = core_metrics_match
    improvements = complete and all((
        required_numeric["weighted_v2_mae"] < required_numeric["weighted_bagrn_mae"],
        required_numeric["weighted_v2_rdd"] < required_numeric["weighted_bagrn_rdd"],
        required_numeric["median_v2_mae"] < required_numeric["median_bagrn_mae"],
        required_numeric["median_v2_rdd"] < required_numeric["median_bagrn_rdd"],
    ))
    gates_pass = all((
        complete, labels_clean, improvements, min(ncc) >= 0.99 if ncc else False,
        quality["v0_semantic_gate"]["status"] == "PASS", scene_order, correction_order,
        support_gate, finite_structural, no_invalid_outputs, core_metrics_match,
        quality["fresh_five_scene_regression"]["status"] == "PASS",
    ))
    quality["scientific_gate_status"] = "READY_FOR_NEXT_SCALE_STAGE" if gates_pass else ("MIXED_SCALE_REVIEW" if complete else "NOT_MEASURED")
    if scale.get("scene_count") == 5:
        quality["five_scene_regression"] = five_scene_regression_gate(root)
    return {"quality": quality, "legacy_stage11": metrics, "legacy_stage12": scale, "scene_order_invariance": scene_invariance, "correction_order_invariance": correction_invariance}

