import numpy as np
import csv
import json

from src.multiscene_sift.local_affine_consistency_diagnostics import (
    LOCAL_AFFINE_EDGES,
    assign_points_to_overlap_regions,
    build_region_validation_crops,
    build_diagnostic_evidence,
    compare_global_vs_local_pixel_alignment,
    compare_local_to_global_at_points,
    compare_partition_stability,
    cross_validate_local_affine,
    classify_local_geometry_consistency,
    fit_region_local_affines,
    fit_affine_least_squares,
    load_local_affine_baseline,
    summarize_local_affine_variation,
    write_diagnostic_artifacts,
)


def test_fixed_edge_set_is_exactly_three_edges():
    assert LOCAL_AFFINE_EDGES == {
        (0, 6): "GOOD_REFERENCE",
        (2, 5): "HIGH_INLIER_FALSE_GOOD",
        (0, 5): "LOW_SUPPORT_CONTROL",
    }


def test_exact_affine_recovery_with_normalized_coordinates():
    src = np.array([[x, y] for y in range(4) for x in range(4)], dtype=float)
    expected = np.array([[1.02, 0.03, 8.0], [-0.02, 0.98, -4.0], [0, 0, 1]])
    dst = (expected[:2, :2] @ src.T).T + expected[:2, 2]
    result = fit_affine_least_squares(src, dst)
    assert result["status"] == "OK"
    np.testing.assert_allclose(result["matrix_3x3"], expected, atol=1e-10)


def test_region_assignment_uses_final_boundary_in_last_cell():
    xy = np.array([[0, 0], [5, 5], [10, 10], [9.9, 0.1]])
    labels = assign_points_to_overlap_regions(xy, (0, 0, 10, 10), 2)
    np.testing.assert_array_equal(labels, [0, 3, 3, 1])


def test_degenerate_and_insufficient_affine_inputs_are_rejected():
    assert fit_affine_least_squares(np.zeros((11, 2)), np.zeros((11, 2)))["status"] == "INSUFFICIENT_POINTS"
    src = np.column_stack([np.arange(12), np.arange(12)])
    dst = src.copy()
    assert fit_affine_least_squares(src, dst)["status"] == "DEGENERATE"


def test_baseline_loader_freezes_required_edges_and_context(tmp_path):
    edge_dir = tmp_path / "edge_reliability"
    edge_dir.mkdir()
    baseline = {
        "edges": [
            {"edge": [0, 6], "group": "GOOD_REFERENCE", "inliers": 291},
            {"edge": [2, 5], "group": "HIGH_INLIER_FALSE_GOOD", "inliers": 3010},
            {"edge": [0, 5], "group": "LOW_SUPPORT_CONTROL", "inliers": 23},
        ]
    }
    (edge_dir / "00_edge_reliability_baseline.json").write_text(
        json.dumps(baseline), encoding="utf-8"
    )
    for name in (
        "01_inlier_reproduction_summary.json",
        "02_inlier_spatial_metrics.json",
        "07_direct_residual_summary.json",
        "08_residual_vs_control_support_summary.json",
        "11_good_vs_false_good_comparison.json",
        "12_edge_diagnoses.json",
        "13_edge_reliability_conclusion.json",
    ):
        (edge_dir / name).write_text("{}", encoding="utf-8")
    with (edge_dir / "01_inlier_points.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["edge_i", "edge_j", "ref_x", "ref_y", "tgt_x", "tgt_y", "residual_px"],
        )
        writer.writeheader()
        for i, j in ((0, 6), (2, 5), (0, 5)):
            writer.writerow({"edge_i": i, "edge_j": j, "ref_x": 1, "ref_y": 2,
                             "tgt_x": 3, "tgt_y": 4, "residual_px": 0.5})

    result = load_local_affine_baseline(edge_dir)

    assert result["edge_keys"] == ["0-6", "2-5", "0-5"]
    for key in result["edge_keys"]:
        record = result["edges"][key]
        assert record["ref_xy"].shape == (1, 2)
        assert record["tgt_xy"].shape == (1, 2)
        assert "direct_overlap" in record
    assert result["missing_critical_artifacts"] == []


def _piecewise_points():
    src = np.array([[x, y] for y in np.linspace(4, 20, 12) for x in np.linspace(4, 96, 12)], dtype=float)
    dst = src.copy()
    dst[src[:, 0] >= 50] += np.array([8.0, 4.0])
    return src, dst


def test_region_local_affines_recover_piecewise_translation_without_new_ransac():
    src, dst = _piecewise_points()
    global_fit = fit_affine_least_squares(src, dst)
    models = fit_region_local_affines(
        ref_xy=dst,
        tgt_xy=src,
        global_matrix=global_fit["matrix_3x3"],
        overlap_bounds=(0, 0, 100, 50),
        grid_n=2,
    )
    fitted = [m for m in models if m["status"] == "OK"]
    assert len(fitted) == 2
    translations = sorted((m["translation_x"], m["translation_y"]) for m in fitted)
    np.testing.assert_allclose(translations, [(0.0, 0.0), (8.0, 4.0)], atol=1e-8)


def test_compare_local_global_uses_prediction_delta_at_region_points():
    global_matrix = np.eye(3)
    local_matrix = np.array([[1, 0, 8], [0, 1, 4], [0, 0, 1]], dtype=float)
    result = compare_local_to_global_at_points(local_matrix, global_matrix, np.array([[10.0, 20.0]]))
    assert result["center_delta_dx_px"] == 8.0
    assert result["center_delta_dy_px"] == 4.0
    assert result["center_delta_mag_px"] == np.hypot(8, 4)


def test_local_affine_variation_reports_ranges_and_spatial_trend():
    src, dst = _piecewise_points()
    global_fit = fit_affine_least_squares(src, dst)
    models = fit_region_local_affines(dst, src, global_fit["matrix_3x3"], (0, 0, 100, 50), 2)
    summary = summarize_local_affine_variation(models, 2)
    assert summary["n_regions_total"] == 4
    assert summary["n_regions_fittable"] == 2
    assert summary["translation_delta_range_px"][1] > 0
    assert "dx_r2" in summary and "dy_r2" in summary


def test_cross_validate_local_affine_is_deterministic_and_reports_holdout_metrics():
    src, dst = _piecewise_points()
    first = cross_validate_local_affine(src, dst, n_splits=5, seed=17)
    second = cross_validate_local_affine(src, dst, n_splits=5, seed=17)
    assert first["folds"] == second["folds"]
    assert first["summary"]["n_folds"] == 5
    assert {"global_affine_rmse", "local_affine_rmse", "improvement_px"} <= set(first["folds"][0])


def test_validation_crops_share_frame_shape_and_masks():
    yy, xx = np.mgrid[:64, :64]
    ref = (xx + 2 * yy).astype(np.float32)
    scene_ref = {"array": ref, "valid_mask": np.ones_like(ref, dtype=bool)}
    scene_tgt = {"array": ref.copy(), "valid_mask": np.ones_like(ref, dtype=bool)}
    crop = build_region_validation_crops(
        scene_ref, scene_tgt, np.eye(3), np.array([[1, 0, 2], [0, 1, 1], [0, 0, 1]], dtype=float),
        {"row": 0, "col": 0, "grid_n": 2, "bounds_px": (0, 0, 32, 32)},
    )
    assert crop["status"] == "OK"
    assert crop["reference"].shape == crop["global_warp"].shape == crop["local_warp"].shape
    assert crop["reference_valid"].shape == crop["local_valid"].shape
    assert crop["bounds_px"] == (0, 0, 32, 32)


def test_global_local_phase_alignment_reports_local_improvement_on_known_shift():
    rng = np.random.default_rng(123)
    ref = rng.normal(size=(64, 64)).astype(np.float32)
    global_warp = np.roll(ref, (3, 4), axis=(0, 1))
    local_warp = ref.copy()
    valid = np.ones_like(ref, dtype=bool)
    result = compare_global_vs_local_pixel_alignment(
        ref, global_warp, local_warp,
        {"reference_valid": valid, "global_valid": valid, "local_valid": valid},
    )
    assert result["status"] == "OK"
    assert result["local_phase_mag"] <= result["global_phase_mag"]
    assert result["phase_improvement_px"] >= 0


def test_partition_stability_classifies_matching_summary_contracts():
    global_summary = {
        "n_regions_fittable": 4, "median_center_delta_mag_px": 0.1,
        "p95_center_delta_mag_px": 0.2, "global_phase_median": 0.3,
        "median_phase_improvement_px": 0.1,
    }
    assert compare_partition_stability(global_summary, global_summary)["state"] == "STABLE_GLOBAL_CONSISTENCY"


def test_classifier_requires_independent_phase_evidence_for_local_geometry():
    evidence = {
        "edge": "2-5", "low_support": False,
        "partition_stability": "STABLE_LOCAL_VARIATION",
        "local_models_differ": True,
        "multiple_region_phase_improvement": True,
        "phase_validation_status": "OK",
    }
    assert classify_local_geometry_consistency(evidence) == "SPATIALLY_VARYING_LOCAL_GEOMETRY_SUPPORTED"


def test_classifier_marks_low_support_without_lowering_point_threshold():
    assert classify_local_geometry_consistency({"edge": "0-5", "low_support": True}) == "LOW_SUPPORT_EDGE"


def test_classifier_distinguishes_global_consistency_from_unexplained_residual():
    global_case = {
        "low_support": False, "partition_stability": "STABLE_GLOBAL_CONSISTENCY",
        "local_models_differ": False, "multiple_region_phase_improvement": False,
        "phase_validation_status": "OK",
    }
    unexplained = dict(global_case, local_models_differ=True)
    assert classify_local_geometry_consistency(global_case) == "GLOBAL_AFFINE_CONSISTENT"
    assert classify_local_geometry_consistency(unexplained) == "LOCAL_MODEL_DOES_NOT_EXPLAIN_RESIDUAL"


def test_build_evidence_preserves_edge_roles_and_cannot_conclude_limits():
    evidence = build_diagnostic_evidence(
        "2-5", "HIGH_INLIER_FALSE_GOOD", {"n_regions_fittable": 4},
        {"n_regions_fittable": 8}, {"state": "MIXED_OR_UNSTABLE"},
        {"summary": {"improvement_px": 0.0}}, {"n_regions_pixel_validated": 0},
    )
    assert evidence["edge_role"] == "HIGH_INLIER_FALSE_GOOD"
    assert "production algorithm" in " ".join(evidence["cannot_conclude"])


def _synthetic_artifact_result():
    src, dst = _piecewise_points()
    global_fit = fit_affine_least_squares(src, dst)
    models2 = fit_region_local_affines(dst, src, global_fit["matrix_3x3"], (0, 0, 100, 50), 2)
    models3 = fit_region_local_affines(dst, src, global_fit["matrix_3x3"], (0, 0, 100, 50), 3)
    variation2 = summarize_local_affine_variation(models2, 2)
    variation3 = summarize_local_affine_variation(models3, 3)
    cv = cross_validate_local_affine(src, dst, n_splits=5, seed=17)
    stability = compare_partition_stability(variation2, variation3)
    evidence = build_diagnostic_evidence(
        "2-5", "HIGH_INLIER_FALSE_GOOD", variation2, variation3, stability, cv, {},
    )
    return {"2-5": {
        "models": models2 + models3, "variation": {"2": variation2, "3": variation3},
        "cross_validation": cv, "phase": {"summary": {"n_regions_pixel_validated": 0}},
        "stability": stability, "evidence": evidence,
    }}


def test_diagnostic_artifacts_write_required_machine_readable_outputs(tmp_path):
    write_diagnostic_artifacts(tmp_path, _synthetic_artifact_result())
    for name in (
        "01_region_support.csv", "02_local_affine_models.csv", "02_local_affine_models.json",
        "03_local_vs_global_displacement.csv", "03_local_vs_global_summary.json",
        "04_local_affine_variation_summary.json", "05_local_affine_cross_validation.csv",
        "05_local_affine_cross_validation_summary.json", "06_global_vs_local_phase.csv",
        "06_global_vs_local_phase_summary.json", "09_partition_stability.json",
        "10_good_vs_false_good_local_geometry.json", "11_low_support_control.json",
        "12_local_affine_consistency_conclusion.json", "12_local_affine_consistency_conclusion.txt",
    ):
        assert (tmp_path / name).is_file(), name


def test_diagnostic_cli_accepts_required_artifact_roots():
    from scripts.diagnose_local_affine_consistency import build_parser

    parser = build_parser()
    args = parser.parse_args([
        "--input-root", "input", "--edge-reliability-dir", "edges",
        "--selected-pair-dir", "pairs", "--five-scene-run-dir", "run",
        "--output-dir", "out",
    ])
    assert args.output_dir.name == "out"
