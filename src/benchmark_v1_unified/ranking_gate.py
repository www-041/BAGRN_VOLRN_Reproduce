"""Fail-closed diagnostic and formal 13-scene eligibility gates."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from .freeze import verify_lock


def _finite_positive(value):
    try:
        return math.isfinite(float(value)) and float(value) > 0
    except (TypeError, ValueError):
        return False


def evaluate_regime_eligibility(evaluation, regime):
    """Report whether one scored regime is internally complete.

    This gate intentionally does not claim formal or spatial generalization.
    Small synthetic fixtures may pass this check while remaining ineligible for
    the frozen 13-scene ranking.
    """
    regime = str(regime).upper()
    reasons = []
    if regime not in {"A", "B1", "B2", "B3"}:
        return {"regime": regime, "eligible": False, "reasons": ["UNKNOWN_REGIME"]}
    if evaluation.get("status") != "EVALUATED":
        reasons.append("EVALUATION_NOT_COMPLETED")
    section = evaluation.get(regime)
    if not isinstance(section, dict):
        reasons.append("REGIME_NOT_REQUESTED")
        return {"regime": regime, "eligible": False, "reasons": reasons}
    if regime == "A":
        coverage = section.get("mosaic_coverage") or {}
        if section.get("status") != "EVALUABLE":
            reasons.append("A_STATUS_NOT_EVALUABLE")
        if coverage.get("holes") != 0 or coverage.get("extras") != 0 or coverage.get("coverage") != 1.0:
            reasons.append("A_COVERAGE_INCOMPLETE")
        if (section.get("single_source_structure") or {}).get("status") != "EVALUABLE":
            reasons.append("A_SINGLE_SOURCE_STRUCTURE_INCOMPLETE")
        if (section.get("seam") or {}).get("status") != "EVALUABLE":
            reasons.append("A_SEAM_NOT_EVALUABLE")
    elif regime == "B1":
        if section.get("status") != "EVALUABLE":
            reasons.append("B1_STATUS_NOT_EVALUABLE")
        rows = section.get("pair_rows") or []
        if not rows or len(rows) != section.get("expected_pair_rows"):
            reasons.append("B1_PAIR_POPULATION_INCOMPLETE")
        if any(row.get("status") not in {"EVALUABLE", "NO_VALID_OVERLAP"} for row in rows):
            reasons.append("B1_PAIR_OUTPUT_INCOMPLETE")
    elif regime == "B2":
        if section.get("status") != "EVALUABLE":
            reasons.append("B2_STATUS_NOT_EVALUABLE")
        rows = section.get("window_rows") or []
        expected = section.get("expected_window_rows")
        if not rows or len(rows) != expected or section.get("window_count") != expected:
            reasons.append("B2_WINDOW_POPULATION_INCOMPLETE")
        if any(row.get("status") != "EVALUABLE" or row.get("metrics") is None for row in rows):
            reasons.append("B2_WINDOW_OUTPUT_INCOMPLETE")
    elif regime == "B3" and section.get("status") != "EVALUABLE":
        reasons.append("B3_NOT_RUN_OR_INCOMPLETE")
    return {"regime": regime, "eligible": not reasons, "reasons": reasons}


def _frozen_populations(freeze_root):
    root = Path(freeze_root)
    frozen = json.loads((root / "frozen_manifest.json").read_text(encoding="utf-8"))
    with (root / "pair_overlap_inventory.csv").open("r", newline="", encoding="utf-8-sig") as stream:
        pairs = list(csv.DictReader(stream))
    with (root / "fixed_windows.csv").open("r", newline="", encoding="utf-8-sig") as stream:
        windows = list(csv.DictReader(stream))
    return frozen, pairs, windows


def evaluate_formal_13scene_ranking(submission, evaluation, freeze_root, *, provenance_audit=None,
                                   freeze_verification=None):
    """Evaluate the immutable full-study main-ranking conditions.

    Exact frozen inventory sizes are checked from files, never inferred from
    submitted result counts. No algorithm score can relax these gates.
    """
    reasons = []
    root = Path(freeze_root).resolve()
    lock_valid = bool(freeze_verification.get("valid")) if freeze_verification is not None else verify_lock(root).valid
    if not lock_valid:
        reasons.append("FREEZE_LOCK_INVALID")
    try:
        frozen, pairs, windows = _frozen_populations(root)
    except (OSError, ValueError, KeyError) as exc:
        return {"eligible": False, "status": "NOT_ELIGIBLE", "reasons": [f"FREEZE_INVENTORY_UNREADABLE:{exc}"]}

    scenes = set()
    for row in pairs:
        scenes.update((row["scene_i"], row["scene_j"]))
    overlap_pairs = sum(str(row.get("status", "")).upper() == "OVERLAP" for row in pairs)
    no_overlap_pairs = sum(str(row.get("status", "")).upper() == "NO_VALID_OVERLAP" for row in pairs)
    populations_ok = (len(scenes) == 13 and len(pairs) == 78 and overlap_pairs == 48
                      and no_overlap_pairs == 30 and len(windows) == 1533)
    if not populations_ok:
        reasons.append("FROZEN_POPULATION_NOT_13SCENE_PROTOCOL")
    if frozen.get("status") != "FROZEN" or frozen.get("protocol_id") != "DZ01V_UNIFIED_A_B_V1_20261013":
        reasons.append("FROZEN_MANIFEST_STATUS_OR_PROTOCOL_INVALID")

    if submission.get("method_kind") == "PAIR_ONLY":
        reasons.append("PAIR_ONLY_NOT_FORMAL_13SCENE")
    if submission.get("method_kind") != "FULL_MOSAIC":
        reasons.append("METHOD_KIND_NOT_FULL_MOSAIC")
    if submission.get("status") != "COMPLETED":
        reasons.append("METHOD_STATUS_NOT_COMPLETED")
    if submission.get("science_status") not in {"VALID", "STRICT_ADMM_CONVERGED", "NOT_APPLICABLE"}:
        reasons.append("SCIENCE_STATUS_NOT_FORMAL")
    if submission.get("input_freeze_sha256") != frozen.get("input_freeze_sha256"):
        reasons.append("INPUT_FREEZE_HASH_MISMATCH")
    if submission.get("geometry_freeze_sha256") != frozen.get("geometry_freeze_sha256"):
        reasons.append("GEOMETRY_FREEZE_HASH_MISMATCH")

    audit = evaluation.get("submission_audit") or {}
    if evaluation.get("status") != "EVALUATED" or audit.get("status") != "PASS" or audit.get("artifact_status") != "PASS":
        reasons.append("METHOD_ARTIFACT_AUDIT_NOT_PASS")
    regime_results = {key: evaluate_regime_eligibility(evaluation, key) for key in ("A", "B1", "B2")}
    if not all(item["eligible"] for item in regime_results.values()):
        reasons.append("MISSING_REQUIRED_REGIMES")

    a = evaluation.get("A") or {}
    coverage = a.get("mosaic_coverage") or {}
    if coverage.get("coverage") != 1.0 or coverage.get("holes") != 0 or coverage.get("extras") != 0:
        reasons.append("MOSAIC_COVERAGE_INCOMPLETE")
    b1 = evaluation.get("B1") or {}
    if (b1.get("expected_pair_rows") != 78 or b1.get("positive_overlap_pairs") != 48
            or b1.get("no_valid_overlap_pairs") != 30):
        reasons.append("B1_PAIR_POPULATION_MISMATCH")
    b2 = evaluation.get("B2") or {}
    if b2.get("expected_window_rows") != 1533 or b2.get("window_count") != 1533:
        reasons.append("B2_WINDOW_POPULATION_MISMATCH")

    runtime = evaluation.get("runtime") or {}
    resource_budget = runtime.get("resource_budget") or {}
    peak_rss = runtime.get("peak_rss_bytes")
    declared_rss_budget = resource_budget.get("max_rss_bytes")
    if (runtime.get("measurement_provenance_status") != "PASS"
            or not _finite_positive(runtime.get("wall_seconds"))
            or runtime.get("wall_measurement_source") in (None, "NOT_MEASURED")
            or not _finite_positive(runtime.get("peak_rss_bytes"))
            or int(runtime.get("rss_sample_count") or 0) <= 0
            or runtime.get("rss_measurement_source") in (None, "NOT_MEASURED")
            or runtime.get("rss_status") not in {"MEASURED", "SAMPLED"}
            or not _finite_positive(declared_rss_budget)
            or float(peak_rss or 0) > float(declared_rss_budget or 0)):
        reasons.append("RESOURCE_MEASUREMENT_INVALID")

    if submission.get("seam_type") != "HARD":
        reasons.append("HARD_SOURCE_PROVENANCE_REQUIRED")
    if not provenance_audit or provenance_audit.get("status") != "PASS":
        reasons.append("HARD_PROVENANCE_AUDIT_REQUIRED")
    status = "ELIGIBLE" if not reasons else "NOT_ELIGIBLE"
    return {"status": status, "eligible": not reasons, "reasons": list(dict.fromkeys(reasons)),
            "regime_eligibility": regime_results,
            "frozen_population": {"scene_count": len(scenes), "pair_count": len(pairs),
                                  "overlap_pair_count": overlap_pairs,
                                  "no_overlap_pair_count": no_overlap_pairs,
                                  "window_count": len(windows)}}


def apply_ranking_gate(submission, evaluation, freeze_root, *, provenance_audit=None,
                       freeze_verification=None):
    """Attach eligibility and map the legacy flag only to the formal gate."""
    result = dict(evaluation)
    regimes = ("A", "B1", "B2")
    result["regime_eligibility"] = {key: evaluate_regime_eligibility(evaluation, key) for key in regimes}
    formal = evaluate_formal_13scene_ranking(submission, evaluation, freeze_root,
                                             provenance_audit=provenance_audit,
                                             freeze_verification=freeze_verification)
    result["formal_13scene_ranking_eligibility"] = formal
    result["ranking_eligible"] = bool(formal["eligible"])
    return result
