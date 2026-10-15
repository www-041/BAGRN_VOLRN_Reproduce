from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("data/output/final_pipeline/b9_13scene")


def _read(name: str) -> dict:
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def test_task15_fresh_scene_order_artifact_is_exact_and_durable():
    artifact = _read("11_metrics/invariance/scene_order_invariance.json")
    assert artifact["status"] == "PASS"
    assert artifact["evidence_type"].startswith("fresh_task15")
    assert artifact["reverse_order_difference_pixels"] == 0
    assert artifact["fixed_permutation_difference_pixels"] == 0
    assert artifact["reference_label_support"] == 62033096
    assert artifact["reverse_label_support"] == 62033096
    assert artifact["permutation_label_support"] == 62033096


def test_task15_fresh_correction_order_artifact_is_exact_without_refit():
    artifact = _read("11_metrics/invariance/correction_order_invariance.json")
    assert artifact["status"] == "PASS"
    assert artifact["evidence_type"] == "fresh_task15_frozen_stage09_replay_no_refit"
    assert artifact["max_abs_difference"] == 0.0
    assert artifact["exact_equality"] is True
    assert artifact["inputs"]["refit_local_coefficients"] is False


def test_task15_report_consistency_closure_and_engineering_status():
    comparison = _read("12_report/reference_comparison.json")
    status = _read("pipeline_status.json")
    assert comparison["all_core_metrics_match"] is True
    assert all(comparison["report_consistency_checks"].values())
    assert status["pipeline_status"] == "SUCCESS"
    assert status["scientific_decision"] == "READY_FOR_NEXT_SCALE_STAGE"
    assert status["scene_order_invariance_status"] == "PASS"
    assert status["correction_order_invariance_status"] == "PASS"
    assert status["report_consistency_audit_status"] == "PASS"
