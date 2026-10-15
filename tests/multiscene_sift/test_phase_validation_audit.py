import csv
import json

import numpy as np
import pytest

from src.multiscene_sift.phase_validation_audit import (
    AUDIT_EDGES,
    coordinate_invariance_check,
    load_phase_audit_baseline,
    phase_from_inputs,
    export_phase_inputs,
    reload_phase_inputs,
    phase_input_visual_stats,
    inject_translation_nonwrapping,
    run_phase_sign_convention,
    evaluate_shift_counterfactual,
    translation_metric_surface,
    mask_boundary_stress_test,
    representation_stability,
    real_crop_injection_recovery,
    phase_peak_diagnostics,
    compare_working_vs_suspect,
    classify_phase_validation_root_cause,
    build_phase_validation_conclusion,
    write_phase_validation_dashboard,
    write_phase_audit_artifacts,
    write_phase_input_visual_audit,
    trace_phase_validation_flow,
)
from scripts.audit_phase_validation import build_parser


def _write_minimal_baseline(root):
    edge_dir = root / "edge"
    local_dir = root / "local"
    selected_dir = root / "nine_scene_selected_pair_consistency"
    edge_dir.mkdir()
    local_dir.mkdir()
    selected_dir.mkdir()
    (edge_dir / "00_edge_reliability_baseline.json").write_text(json.dumps({
        "edges": [
            {"edge": [0, 6], "full_overlap_phase_residual_px": 0.58},
            {"edge": [2, 5], "full_overlap_phase_residual_px": 6.92},
        ]
    }), encoding="utf-8")
    (edge_dir / "07_direct_residual_summary.json").write_text(json.dumps({
        "0-6": {"median_px": 0.6324555320336759},
        "2-5": {"median_px": 6.715078575495168},
    }), encoding="utf-8")
    with (edge_dir / "07_direct_residual_tiles.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "edge", "grid_n", "tile_row", "tile_col", "phase_mag_px", "accepted"
        ])
        writer.writeheader()
        writer.writerow({"edge": "0-6", "grid_n": 4, "tile_row": 1, "tile_col": 1,
                         "phase_mag_px": 0.6324555320336759, "accepted": "True"})
        writer.writerow({"edge": "2-5", "grid_n": 4, "tile_row": 1, "tile_col": 1,
                         "phase_mag_px": 6.715078575495168, "accepted": "True"})
    (local_dir / "00_local_affine_baseline.json").write_text(json.dumps({"edges": {}}), encoding="utf-8")
    (local_dir / "06_global_vs_local_phase_summary.json").write_text(json.dumps({"0-6": {}, "2-5": {}}), encoding="utf-8")
    (selected_dir / "06_canonical_world_edge_transforms.json").write_text(json.dumps({
        "edges": {"0-6": {"matrix": np.eye(3).tolist()}, "2-5": {"matrix": np.eye(3).tolist()}}
    }), encoding="utf-8")
    (selected_dir / "02_new_pairwise_registration_results.json").write_text(json.dumps({
        "results": [
            {"idx_i": 0, "idx_j": 6, "pixel_matrix": np.eye(3).tolist()},
            {"idx_i": 2, "idx_j": 5, "pixel_matrix": np.eye(3).tolist()},
        ]
    }), encoding="utf-8")
    return edge_dir, local_dir, selected_dir


def test_audit_edges_are_exactly_working_control_and_suspect():
    assert AUDIT_EDGES == {
        (0, 6): "WORKING_CONTROL",
        (2, 5): "SUSPECT_PHASE_FALSE_ALARM",
    }


def test_baseline_loader_reads_old_phase_values_without_hardcoding(tmp_path):
    edge_dir, local_dir, selected_dir = _write_minimal_baseline(tmp_path)
    result = load_phase_audit_baseline(edge_dir, local_dir, selected_pair_dir=selected_dir)
    assert result["edge_keys"] == ["0-6", "2-5"]
    assert result["edges"]["0-6"]["old_phase_median_px"] == pytest.approx(0.6324555320336759)
    assert result["edges"]["2-5"]["old_phase_median_px"] == pytest.approx(6.715078575495168)
    assert result["edges"]["2-5"]["pixel_matrix"] == np.eye(3).tolist()


def test_baseline_loader_fails_when_required_artifact_is_missing(tmp_path):
    edge_dir, local_dir, selected_dir = _write_minimal_baseline(tmp_path)
    (edge_dir / "07_direct_residual_summary.json").unlink()
    with pytest.raises(FileNotFoundError, match="07_direct_residual_summary"):
        load_phase_audit_baseline(edge_dir, local_dir, selected_pair_dir=selected_dir)


def test_baseline_loader_retains_pixel_matrix_when_canonical_world_matrix_is_missing(tmp_path):
    edge_dir, local_dir, selected_dir = _write_minimal_baseline(tmp_path)
    canonical_path = selected_dir / "06_canonical_world_edge_transforms.json"
    canonical = json.loads(canonical_path.read_text(encoding="utf-8"))
    del canonical["edges"]["2-5"]
    canonical_path.write_text(json.dumps(canonical), encoding="utf-8")
    baseline = load_phase_audit_baseline(edge_dir, local_dir, selected_pair_dir=selected_dir)
    assert baseline["edges"]["2-5"]["world_matrix"] is None
    assert baseline["edges"]["2-5"]["pixel_matrix"] is not None


def test_phase_flow_trace_records_frame_crop_masks_and_phase():
    rng = np.random.default_rng(4)
    ref = rng.normal(size=(64, 64)).astype(np.float32)
    inputs = {
        "ref_crop": ref,
        "warped_target_crop": ref.copy(),
        "ref_valid_mask": np.ones_like(ref, dtype=bool),
        "target_valid_mask": np.ones_like(ref, dtype=bool),
        "joint_valid_mask": np.ones_like(ref, dtype=bool),
        "metadata": {
            "matrix_direction": "target -> reference",
            "ref_tgt_order": "reference, target",
            "crop_origin_row": 12,
            "crop_origin_col": 7,
            "pixel_center_semantics": "pixel centers",
        },
    }
    phase = phase_from_inputs(inputs)
    trace = trace_phase_validation_flow(inputs, phase)
    assert trace["matrix_direction"] == "target -> reference"
    assert trace["crop_origin"] == {"row": 12, "col": 7}
    assert trace["ref_array"]["shape"] == [64, 64]
    assert trace["joint_valid"]["count"] == 64 * 64
    assert trace["phase"]["magnitude_px"] == pytest.approx(0.0)
    assert trace["ref_tgt_order"] == "reference, target"


def test_coordinate_roundtrip_is_invariant_to_crop_origin():
    from rasterio.transform import Affine

    grid = Affine(2, 0, 1000, 0, -2, 2000)
    ref = Affine(2, 0, 900, 0, -2, 2100)
    tgt = Affine(2, 0, 1100, 0, -2, 2200)
    result = coordinate_invariance_check(grid, ref, tgt, (13, 17, 53, 67))
    assert result["status"] == "OK"
    assert result["ref_world_roundtrip_error"] < 1e-9
    assert result["tgt_world_roundtrip_error"] < 1e-9


def test_phase_flow_keeps_explicit_reference_target_order():
    ref = np.full((32, 32), 2.0, dtype=np.float32)
    tgt = np.full((32, 32), 7.0, dtype=np.float32)
    mask = np.ones_like(ref, dtype=bool)
    trace = trace_phase_validation_flow({
        "ref_crop": ref,
        "warped_target_crop": tgt,
        "ref_valid_mask": mask,
        "target_valid_mask": mask,
        "joint_valid_mask": mask,
        "metadata": {"ref_tgt_order": "reference, target"},
    }, phase={"status": "OK", "dx_px": 0.0, "dy_px": 0.0, "magnitude_px": 0.0})
    assert trace["ref_tgt_order"] == "reference, target"
    assert trace["ref_array"]["mean"] == pytest.approx(2.0)
    assert trace["target_array"]["mean"] == pytest.approx(7.0)


def test_forward_and_inverse_translation_directions_are_distinct():
    rng = np.random.default_rng(15)
    reference = rng.normal(size=(96, 96)).astype(np.float32)
    mask = np.ones_like(reference, dtype=bool)
    moving, moving_mask = inject_translation_nonwrapping(reference, mask, 4, -2)
    forward = evaluate_shift_counterfactual(reference, moving, moving_mask, 4, -2)
    inverse = evaluate_shift_counterfactual(reference, moving, moving_mask, -4, 2)
    assert inverse["gradient_ncc"] > forward["gradient_ncc"]
    assert inverse["raw_ncc"] > forward["raw_ncc"]


def test_export_reload_preserves_exact_phase_inputs_and_phase_result(tmp_path):
    rng = np.random.default_rng(5)
    ref = rng.normal(size=(64, 64)).astype(np.float32)
    inputs = {
        "ref_crop": ref,
        "warped_target_crop": ref.copy(),
        "ref_valid_mask": np.ones_like(ref, dtype=bool),
        "target_valid_mask": np.ones_like(ref, dtype=bool),
        "joint_valid_mask": np.ones_like(ref, dtype=bool),
        "metadata": {"edge": "0-6", "crop_origin_row": 2, "crop_origin_col": 3},
    }
    export_phase_inputs(inputs, tmp_path, "0-6")
    restored = reload_phase_inputs(tmp_path, "0-6")
    assert phase_from_inputs(inputs) == phase_from_inputs(restored)
    np.testing.assert_array_equal(inputs["ref_crop"], restored["ref_crop"])
    manifest = json.loads((tmp_path / "02_phase_input_manifest.json").read_text(encoding="utf-8"))
    assert manifest["0-6"]["shape"] == [64, 64]
    assert len(manifest["0-6"]["sha256"]) == 64


def test_visual_audit_stats_use_joint_mask_and_write_figure(tmp_path):
    rng = np.random.default_rng(6)
    ref = rng.normal(size=(64, 64)).astype(np.float32)
    inputs = {
        "ref_crop": ref,
        "warped_target_crop": ref.copy(),
        "ref_valid_mask": np.ones_like(ref, dtype=bool),
        "target_valid_mask": np.ones_like(ref, dtype=bool),
        "joint_valid_mask": np.ones_like(ref, dtype=bool),
        "metadata": {"edge": "2-5"},
    }
    stats = phase_input_visual_stats(inputs)
    assert stats["raw_ncc"] == pytest.approx(1.0)
    assert stats["gradient_ncc"] > 0.99
    assert stats["valid_fraction"] == pytest.approx(1.0)
    path = write_phase_input_visual_audit({"2-5": inputs}, tmp_path / "audit.png", tmp_path / "stats.json")
    assert path.is_file()
    assert (tmp_path / "stats.json").is_file()


@pytest.mark.parametrize("dx,dy", [(5, 0), (-5, 0), (0, 5), (0, -5), (4, -7)])
def test_nonwrapping_injection_has_explicit_inverse_phase_sign(dx, dy):
    rng = np.random.default_rng(7)
    image = rng.normal(size=(128, 128)).astype(np.float32)
    mask = np.ones_like(image, dtype=bool)
    shifted, shifted_mask = inject_translation_nonwrapping(image, mask, dx, dy)
    assert shifted.shape == image.shape
    assert shifted_mask.sum() < mask.sum()
    assert run_phase_sign_convention(image, mask, shifted, shifted_mask, dx, dy)["error_mag_px"] < 0.5


def test_counterfactual_reported_shift_improves_independent_metrics_on_known_shift():
    rng = np.random.default_rng(8)
    ref = rng.normal(size=(128, 128)).astype(np.float32)
    mask = np.ones_like(ref, dtype=bool)
    moving, moving_mask = inject_translation_nonwrapping(ref, mask, 5, -3)
    zero = evaluate_shift_counterfactual(ref, moving, moving_mask, 0, 0)
    corrected = evaluate_shift_counterfactual(ref, moving, moving_mask, -5, 3)
    assert corrected["gradient_ncc"] > zero["gradient_ncc"]
    assert corrected["raw_ncc"] > zero["raw_ncc"]
    assert corrected["masked_mae"] < zero["masked_mae"]


def test_translation_sweep_finds_known_correction_without_phase_correlation():
    rng = np.random.default_rng(9)
    clean = rng.normal(size=(128, 128)).astype(np.float32)
    ref = clean.copy()
    ref[:12, :] = np.nan
    mask = np.ones_like(clean, dtype=bool)
    moving, moving_mask = inject_translation_nonwrapping(clean, mask, 5, -3)
    result = translation_metric_surface(ref, moving, moving_mask, radius_px=6)
    assert (result["best_dx"], result["best_dy"]) == (-5, 3)
    assert result["best_score"] > result["score_at_zero"]
    assert len({row["valid_count"] for row in result["rows"]}) > 1


def test_mask_boundary_stress_test_is_deterministic_and_reports_validity():
    rng = np.random.default_rng(10)
    ref = rng.normal(size=(160, 160)).astype(np.float32)
    tgt = ref.copy()
    mask = np.ones_like(ref, dtype=bool)
    mask[:8, :] = False
    mask[-8:, :] = False
    first = mask_boundary_stress_test(ref, tgt, mask, mask)
    second = mask_boundary_stress_test(ref, tgt, mask, mask)
    assert first == second
    assert set(first) == {"V0_original", "V1_erode_16", "V2_erode_32", "V3_erode_64", "V4_central_valid_bbox"}
    assert first["V0_original"]["valid_fraction"] > first["V1_erode_16"]["valid_fraction"]


def test_representation_stability_covers_raw_standardized_gradient_sobel_and_edges():
    rng = np.random.default_rng(11)
    ref = rng.normal(size=(96, 96)).astype(np.float32)
    moving = ref * 2.0 + 17.0
    mask = np.ones_like(ref, dtype=bool)
    result = representation_stability(ref, moving, mask, include_eroded=True)
    names = {row["representation"] for row in result}
    assert {"R0_raw", "R1_standardized", "R2_gradient_magnitude", "R3_sobel_magnitude", "R4_edge_map"} <= names
    assert {row["mask_variant"] for row in result} == {"original", "erode_32"}
    assert all("phase" in row for row in result)


def test_real_crop_injection_recovery_recovers_zero_one_three_six_ten_pixels():
    rng = np.random.default_rng(12)
    ref = rng.normal(size=(192, 192)).astype(np.float32)
    mask = np.ones_like(ref, dtype=bool)
    inputs = {
        "ref_crop": ref,
        "warped_target_crop": ref.copy(),
        "ref_valid_mask": mask,
        "target_valid_mask": mask,
        "joint_valid_mask": mask,
    }
    rows = real_crop_injection_recovery(inputs, [(0, 0), (1, 0), (0, 1), (3, 2), (6, 0), (10, 5)])
    assert len(rows) == 6
    assert all(row["error_mag_px"] < 0.5 for row in rows)
    assert {row["injected_dx"] for row in rows} == {0, 1, 3, 6, 10}


def test_real_crop_injection_recovery_includes_exact_three_and_ten_pixel_axis_cases():
    rng = np.random.default_rng(16)
    ref = rng.normal(size=(192, 192)).astype(np.float32)
    mask = np.ones_like(ref, dtype=bool)
    inputs = {
        "ref_crop": ref,
        "warped_target_crop": ref.copy(),
        "ref_valid_mask": mask,
        "target_valid_mask": mask,
        "joint_valid_mask": mask,
    }
    rows = real_crop_injection_recovery(inputs, [(0, 0), (3, 0), (10, 0)])
    assert [row["injected_dx"] for row in rows] == [0, 3, 10]
    assert all(row["error_mag_px"] < 0.5 for row in rows)


def test_peak_diagnostics_reports_unavailable_without_existing_surface():
    assert phase_peak_diagnostics({})["status"] == "NOT_AVAILABLE_IN_CURRENT_HELPER"


def test_working_vs_suspect_comparison_rejects_reported_shift_when_zero_wins():
    result = compare_working_vs_suspect({
        "edge": "2-5", "old_phase_median_px": 6.7,
        "phase": {"magnitude_px": 6.7},
        "counterfactual": {
            "zero": {"gradient_ncc": 0.95, "raw_ncc": 0.96},
            "reported": {"gradient_ncc": 0.60, "raw_ncc": 0.65},
            "inverse": {"gradient_ncc": 0.55, "raw_ncc": 0.60},
        },
        "sweep": {"best_dx": 0, "best_dy": 0, "best_score": 0.95,
                   "score_at_zero": 0.95, "phase_candidate": {"gradient_ncc": 0.60}},
        "mask": {"V0_original": {"magnitude_px": 6.7}, "V1_erode_16": {"magnitude_px": 0.2}},
        "representations": [{"representation": "R0_raw", "phase": {"magnitude_px": 6.7}},
                             {"representation": "R2_gradient_magnitude", "phase": {"magnitude_px": 0.1}}],
        "injection": [{"injected_dx": 0, "injected_dy": 0, "error_mag_px": 6.7}],
    })
    assert result["answer"] == "NOT_SUPPORTED_PHASE_ARTIFACT"


@pytest.mark.parametrize("evidence, expected", [
    ({"frame_bug": True}, "FRAME_OR_DIRECTION_BUG_SUPPORTED"),
    ({"reported_improves": True, "sweep_near_reported": True, "mask_stable": True,
      "representation_agrees": True, "injection_correct": True}, "REAL_TRANSLATION_RESIDUAL_SUPPORTED"),
    ({"mask_boundary_collapse": True, "sweep_prefers_zero": True,
      "reported_improves": False}, "MASK_BOUNDARY_DOMINATED"),
    ({"representation_sensitive": True}, "RADIOMETRIC_REPRESENTATION_SENSITIVE"),
    ({"multipeak": True}, "TEXTURE_AMBIGUITY_OR_MULTIPEAK"),
    ({"zero_injection_bias": True}, "PHASE_HELPER_REAL_CROP_BIAS"),
    ({"reported_improves": False, "sweep_prefers_zero": True}, "PHASE_VALIDATION_ARTIFACT_SUPPORTED"),
    ({}, "MIXED_OR_UNDERDETERMINED"),
])
def test_root_cause_classifier_uses_explicit_evidence_states(evidence, expected):
    assert classify_phase_validation_root_cause(evidence) == expected


def test_final_conclusion_contains_can_and_cannot_conclude_limits():
    conclusion = build_phase_validation_conclusion({"edges": {}}, {
        "0-6": {"comparison": {"answer": "SUPPORTED_AS_REAL_TRANSLATION"}, "root_cause": "MIXED_OR_UNDERDETERMINED"},
        "2-5": {"comparison": {"answer": "NOT_SUPPORTED_PHASE_ARTIFACT"}, "root_cause": "PHASE_VALIDATION_ARTIFACT_SUPPORTED"},
    })
    assert "answers" in conclusion
    assert "can_conclude" in conclusion and conclusion["can_conclude"]
    cannot = " ".join(conclusion["cannot_conclude"])
    assert "do not directly modify production phase helper" in cannot
    assert "do not claim SIFT/Affine is always correct" in cannot


def test_phase_validation_dashboard_has_two_rows_and_five_diagnostic_columns(tmp_path):
    rng = np.random.default_rng(13)
    image = rng.normal(size=(64, 64)).astype(np.float32)
    mask = np.ones_like(image, dtype=bool)
    results = {}
    for edge, dx, dy in (("0-6", 1.0, -0.5), ("2-5", 6.0, 3.0)):
        results[edge] = {
            "inputs": {
                "ref_crop": image,
                "warped_target_crop": image.copy(),
                "joint_valid_mask": mask,
            },
            "phase": {"dx_px": dx, "dy_px": dy, "magnitude_px": float(np.hypot(dx, dy))},
            "sweep": {
                "rows": [
                    {"dx": -1, "dy": -1, "score": 0.2},
                    {"dx": 0, "dy": 0, "score": 0.9},
                    {"dx": 1, "dy": 1, "score": 0.4},
                ],
                "best_dx": 0,
                "best_dy": 0,
            },
            "mask": {
                "V0_original": {"magnitude_px": np.hypot(dx, dy)},
                "V1_erode_16": {"magnitude_px": np.hypot(dx, dy) * 0.8},
            },
            "injection": [
                {"injected_dx": 0, "injected_dy": 0, "error_mag_px": 0.0},
                {"injected_dx": 3, "injected_dy": 2, "error_mag_px": 0.1},
            ],
        }
    output = tmp_path / "13_phase_validation_audit_dashboard.png"
    written = write_phase_validation_dashboard(results, output)
    assert written == output
    assert output.exists()
    assert output.stat().st_size > 10_000


def test_phase_audit_cli_and_artifact_writer_follow_required_contract(tmp_path):
    args = build_parser().parse_args([
        "--input-root", "input",
        "--selected-pair-dir", "selected",
        "--edge-reliability-dir", "edges",
        "--local-affine-dir", "local",
        "--output-dir", "audit",
    ])
    assert args.band == "B14"
    assert args.output_dir == "audit"
    rng = np.random.default_rng(14)
    image = rng.normal(size=(64, 64)).astype(np.float32)
    mask = np.ones_like(image, dtype=bool)
    baseline = {
        "edge_keys": ["0-6", "2-5"],
        "edges": {
            edge: {"old_phase_median_px": 0.0, "role": role}
            for edge, role in (("0-6", "WORKING_CONTROL"), ("2-5", "SUSPECT_PHASE_FALSE_ALARM"))
        },
    }
    results = {}
    for edge in baseline["edge_keys"]:
        results[edge] = {
            "inputs": {
                "ref_crop": image,
                "warped_target_crop": image.copy(),
                "ref_valid_mask": mask,
                "target_valid_mask": mask,
                "joint_valid_mask": mask,
            },
            "baseline_reproduction_status": "EXACT",
            "flow": {"edge": edge},
            "phase": {"dx_px": 0.0, "dy_px": 0.0, "magnitude_px": 0.0},
            "sign_convention": {"status": "OK"},
            "counterfactual": {"zero": {"raw_ncc": 1.0, "gradient_ncc": 1.0}},
            "sweep": {"rows": [{"dx": 0, "dy": 0, "score": 1.0}], "best_dx": 0, "best_dy": 0},
            "mask": {"V0_original": {"magnitude_px": 0.0}},
            "representations": [{"representation": "R0_raw", "phase": {"magnitude_px": 0.0}}],
            "injection": [{"injected_dx": 0, "injected_dy": 0, "error_mag_px": 0.0}],
            "peak_diagnostics": {"status": "NOT_AVAILABLE_IN_CURRENT_HELPER"},
            "comparison": {"answer": "MIXED"},
            "root_cause": "MIXED_OR_UNDERDETERMINED",
        }
    manifest = write_phase_audit_artifacts(tmp_path / "audit", baseline, results)
    expected = {
        "00_phase_audit_baseline.json",
        "01_phase_dataflow_trace.json",
        "02_phase_input_manifest.json",
        "03_phase_input_visual_audit.png",
        "03_phase_input_visual_stats.json",
        "04_phase_sign_convention.json",
        "05_phase_shift_counterfactual.json",
        "05_phase_shift_counterfactual.png",
        "06_translation_sweep.csv",
        "06_translation_sweep_summary.json",
        "06_translation_sweep_surface.png",
        "07_mask_boundary_stress_test.csv",
        "07_mask_boundary_stress_test.json",
        "08_representation_stability.csv",
        "08_representation_stability.json",
        "09_real_crop_injection_recovery.csv",
        "09_real_crop_injection_recovery.json",
        "10_phase_peak_diagnostics.json",
        "11_working_vs_suspect_phase_audit.json",
        "11_working_vs_suspect_phase_audit.txt",
        "11_working_vs_suspect_phase_audit.png",
        "12_phase_validation_root_cause.json",
        "12_phase_validation_root_cause.txt",
        "13_phase_validation_audit_dashboard.png",
    }
    assert manifest["artifact_count"] == len(expected) + 2
    assert all((tmp_path / "audit" / name).exists() for name in expected)
    assert (tmp_path / "audit" / "02_phase_inputs" / "0_6_inputs.npz").exists()
    assert (tmp_path / "audit" / "02_phase_inputs" / "2_5_inputs.npz").exists()
