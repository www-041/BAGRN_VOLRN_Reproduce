from __future__ import annotations

import pytest

from scripts.run_task14a2_valid_footprint_fix import (
    classify_formal_pair_status,
    correction_order_equivalent,
    final_decision,
    preference_rows,
    _manifest,
    _detect_v0_semantic,
    _local_tile_seam_path,
    required_label_diagnostic_keys,
    scene_identity_difference,
    preference_eligible,
    validate_source_side_gate,
    validate_v0_semantics,
)


def test_task14a2_reads_the_frozen_stage06_manifest_once():
    manifest = _manifest()
    assert len(manifest["scenes"]) == 13


def test_adapter_mismatch_becomes_formal_pass_only_after_agreeing_resolver_replay():
    assert classify_formal_pair_status(
        "REQUIRES_MULTISCENE_LABELING",
        "EXCLUSIVE_CONTACT_RESOLVABLE",
        "EXCLUSIVE_CONTACT_RESOLVABLE",
        True,
        "STABLE",
    ) == "PASS"
    assert preference_eligible("PASS") is True


def test_true_geometry_and_unstable_statuses_are_not_promoted():
    assert classify_formal_pair_status(
        "REQUIRES_MULTISCENE_LABELING",
        "REQUIRES_MULTISCENE_LABELING",
        "REQUIRES_MULTISCENE_LABELING",
        False,
        "STABLE",
    ) == "REQUIRES_MULTISCENE_LABELING"
    assert classify_formal_pair_status(
        "UNSTABLE_LOCAL_GAIN", None, None, False, "UNSTABLE_LOCAL_GAIN"
    ) == "UNSTABLE_LOCAL_GAIN"
    assert classify_formal_pair_status(
        "NO_FINAL_SHARED_SUPPORT", None, None, False, "NO_FINAL_SHARED_SUPPORT"
    ) == "NO_FINAL_SHARED_SUPPORT"
    assert preference_eligible("REQUIRES_MULTISCENE_LABELING") is False
    assert preference_eligible("UNSTABLE_LOCAL_GAIN") is False


def test_source_side_gate_requires_exact_expected_counts():
    rows = [
        {
            "pair_id": f"mismatch_{i:02d}",
            "old_stage07_status": "REQUIRES_MULTISCENE_LABELING",
            "task14a1_dryrun_status": "PASS",
            "new_source_side_status": "PASS",
            "formal_pair_status": "PASS",
            "adapter_logic_mismatch": True,
        }
        for i in range(45)
    ]
    rows += [{
        "pair_id": "02_05", "old_stage07_status": "REQUIRES_MULTISCENE_LABELING",
        "task14a1_dryrun_status": "REQUIRES_MULTISCENE_LABELING",
        "new_source_side_status": "REQUIRES_MULTISCENE_LABELING",
        "formal_pair_status": "REQUIRES_MULTISCENE_LABELING",
        "adapter_logic_mismatch": False,
    }]
    rows += [{
        "pair_id": pair, "old_stage07_status": "UNSTABLE_LOCAL_GAIN",
        "task14a1_dryrun_status": "REQUIRES_MULTISCENE_LABELING",
        "new_source_side_status": "UNSTABLE_LOCAL_GAIN",
        "formal_pair_status": "UNSTABLE_LOCAL_GAIN",
        "adapter_logic_mismatch": False,
    } for pair in ("01_08", "08_11")]
    rows += [{
        "pair_id": f"no_{i:02d}", "old_stage07_status": "NO_FINAL_SHARED_SUPPORT",
        "task14a1_dryrun_status": "NO_FINAL_SHARED_SUPPORT",
        "new_source_side_status": "NO_FINAL_SHARED_SUPPORT",
        "formal_pair_status": "NO_FINAL_SHARED_SUPPORT",
        "adapter_logic_mismatch": False,
    } for i in range(12)]

    summary = validate_source_side_gate(rows)
    assert summary["adapter_logic_mismatch_pass_count"] == 45
    assert summary["true_geometric_requires_count"] == 1
    assert summary["unstable_count"] == 2
    assert summary["no_support_count"] == 12


def test_v0_semantic_gate_rejects_non_equivalent_metadata():
    expected = {
        "semantic": "weighted_feather",
        "crs": "EPSG:32650",
        "transform": [14.0, 0.0, 633346.0, 0.0, -14.0, 3359874.0],
        "shape": [12404, 7992],
        "dtype": "float32",
        "nodata": "nan",
        "union_support": 62033096,
    }
    assert validate_v0_semantics(expected, expected)["status"] == "PASS"
    bad = dict(expected, semantic="unknown")
    with pytest.raises(RuntimeError, match="HARD_STOP_V0_SEMANTIC_MISMATCH"):
        validate_v0_semantics(bad, expected)


def test_v0_semantic_detection_matches_frozen_stage06_writer():
    assert _detect_v0_semantic(
        {"radiometric_method": "BAGRN"},
        "_stream_blend(normalized_paths, distance_paths, mosaic_path)",
    ) == "weighted_feather"


def test_horizontal_tile_seam_path_applies_row_offset_once():
    path = _local_tile_seam_path("horizontal", [100, 102, 104], 90, 20, 3)
    assert path[:, 0].tolist() == [10, 12, 14]
    assert path[:, 1].tolist() == [0, 1, 2]


def test_only_formal_pass_rows_contribute_preference_edges():
    rows = [
        {"formal_pair_status": "PASS", "pair_id": "00_01"},
        {"formal_pair_status": "REQUIRES_MULTISCENE_LABELING", "pair_id": "02_05"},
        {"formal_pair_status": "UNSTABLE_LOCAL_GAIN", "pair_id": "01_08"},
        {"formal_pair_status": "NO_FINAL_SHARED_SUPPORT", "pair_id": "00_03"},
    ]
    assert [row["pair_id"] for row in preference_rows(rows)] == ["00_01"]


def test_label_diagnostics_contract_is_explicit():
    assert "cycle_pixels" in required_label_diagnostic_keys()
    assert "score_margin_p95" in required_label_diagnostic_keys()


def test_scene_identity_and_correction_order_gates_are_strict():
    original = [[0, 1, 2], [0, 1, 2]]
    reversed_labels = [[2, 1, 0], [2, 1, 0]]
    assert scene_identity_difference(original, reversed_labels, [2, 1, 0]) == 0
    assert correction_order_equivalent(0.0, 0.0) is True
    assert correction_order_equivalent(0.01, 0.01, rtol=1e-6, atol=1e-3) is False


def test_ready_decision_requires_scientific_radiometric_quality():
    gates = {name: True for name in (
        "source_side", "true_geometry_preserved", "labels", "invariance",
        "correction_order", "support", "structural", "no_invalid",
        "weighted_mae_improved", "weighted_rdd_improved", "median_mae_improved",
        "median_rdd_improved",
    )}
    assert final_decision(gates) == "READY_FOR_NEXT_SCALE_STAGE"
    gates["weighted_mae_improved"] = False
    assert final_decision(gates) == "MIXED_SCALE_REVIEW"
