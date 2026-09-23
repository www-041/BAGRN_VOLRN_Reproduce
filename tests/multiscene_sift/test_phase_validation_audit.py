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
    write_phase_input_visual_audit,
    trace_phase_validation_flow,
)


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
