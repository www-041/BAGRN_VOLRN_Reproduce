"""Diagnostic-only CLI for the frozen 0-6 and 2-5 phase-validation audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.multiscene_sift.dataset import discover_five_scenes
from src.multiscene_sift.frame_diagnostics import (
    match_view_frame_params,
    rebuild_pair_common_grids,
    world_transform_from_saved_pair,
)
from src.multiscene_sift.nine_scene_overlap_diagnostic import SCENE_NAMES_9
from src.multiscene_sift.phase_validation_audit import (
    AUDIT_EDGES,
    build_exact_phase_inputs,
    classify_phase_validation_root_cause,
    compare_working_vs_suspect,
    evaluate_shift_counterfactual,
    load_phase_audit_baseline,
    mask_boundary_stress_test,
    phase_from_inputs,
    phase_peak_diagnostics,
    real_crop_injection_recovery,
    representation_stability,
    run_phase_sign_convention,
    trace_phase_validation_flow,
    translation_metric_surface,
    write_json,
    write_phase_audit_artifacts,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--selected-pair-dir", required=True)
    parser.add_argument("--edge-reliability-dir", required=True)
    parser.add_argument("--local-affine-dir", required=True)
    parser.add_argument("--output-dir", default="data/output/phase_validation_audit")
    parser.add_argument("--band", default="B14")
    return parser


def _root_cause_evidence(result: dict) -> dict:
    comparison = result["comparison"]
    flow = result["flow"]
    injection = result["injection"]
    finite_errors = [
        row["error_mag_px"] for row in injection
        if row.get("error_mag_px") is not None and np.isfinite(row["error_mag_px"])
    ]
    coordinate = flow.get("coordinate_invariance", {})
    frame_bug = coordinate.get("status") not in {"OK", "NOT_PROVIDED"} or any(
        coordinate.get(key, 0.0) > 1e-6
        for key in ("ref_world_roundtrip_error", "tgt_world_roundtrip_error")
    )
    sweep = result["sweep"]
    return {
        "frame_bug": frame_bug,
        "reported_improves": comparison["reported_improves_independent_metrics"],
        "sweep_near_reported": comparison["sweep_near_reported"],
        "mask_stable": not comparison["mask_boundary_collapse"],
        "representation_agrees": not comparison["representation_sensitive"],
        "injection_correct": bool(finite_errors) and max(finite_errors) <= 1.0,
        "mask_boundary_collapse": comparison["mask_boundary_collapse"],
        "sweep_prefers_zero": sweep.get("best_dx") == 0 and sweep.get("best_dy") == 0,
        "zero_injection_bias": comparison["zero_injection_bias"],
        "representation_sensitive": comparison["representation_sensitive"],
    }


def _resolve_world_matrix(
    edge_baseline: dict,
    scenes,
    i: int,
    j: int,
    band: str,
    selected_pair_dir: str | Path,
) -> tuple[np.ndarray, str]:
    if edge_baseline.get("world_matrix") is not None:
        return np.asarray(edge_baseline["world_matrix"], dtype=float), "canonical_world_matrix"
    pixel_matrix = edge_baseline.get("pixel_matrix")
    if pixel_matrix is None:
        raise RuntimeError(f"missing world and pixel matrix for {i}-{j}")
    grids = rebuild_pair_common_grids(scenes, band, [(i, j)])
    from rasterio.transform import Affine

    grid = grids[f"pair_{i}_{j}"]
    common_transform = Affine(*grid["transform"])
    config_path = Path(selected_pair_dir) / "01_registration_config_snapshot.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    match_view = match_view_frame_params(tuple(grid["overlap_window"]), int(config.get("match_max_side", 1600)))
    pair_row = {"idx_i": i, "idx_j": j, "pixel_matrix": pixel_matrix}
    world_matrix = world_transform_from_saved_pair(
        i, j, [pair_row], common_transform, match_view, adjust_frame=True
    )
    return np.asarray(world_matrix, dtype=float), "selected_pair_pixel_matrix_with_existing_match_view_frame_normalization"


def run_audit(args: argparse.Namespace) -> dict:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline = load_phase_audit_baseline(
        args.edge_reliability_dir,
        args.local_affine_dir,
        selected_pair_dir=args.selected_pair_dir,
    )
    write_json(output_dir / "00_phase_audit_baseline.json", baseline)
    scenes, _ = discover_five_scenes(args.input_root, list(SCENE_NAMES_9), bands=(args.band,))

    results = {}
    for (i, j), role in AUDIT_EDGES.items():
        edge = f"{i}-{j}"
        baseline_edge = baseline["edges"][edge]
        world_matrix, world_matrix_source = _resolve_world_matrix(
            baseline_edge, scenes, i, j, args.band, args.selected_pair_dir
        )
        inputs = build_exact_phase_inputs(
            scenes[i], scenes[j], world_matrix,
            baseline_edge["direct_tiles"], edge, band=args.band,
        )
        inputs["metadata"]["world_matrix_source"] = world_matrix_source
        phase = phase_from_inputs(inputs)
        old_value = float(baseline_edge["old_phase_median_px"])
        selected_tile_value = float(inputs["metadata"]["selected_tile"]["phase_mag_px"])
        reproduced = phase.get("magnitude_px") is not None and abs(float(phase["magnitude_px"]) - selected_tile_value) <= 1e-6
        status = "EXACT" if reproduced else "BASELINE_REPRODUCTION_FAILED"
        if not reproduced:
            write_json(output_dir / "00_phase_audit_baseline.json", {
                **baseline,
                "status": status,
                "failed_edge": edge,
                "reproduced_phase": phase,
                "expected_selected_tile_phase_px": selected_tile_value,
                "expected_phase_median_px": old_value,
            })
            raise RuntimeError(f"{status}: {edge}")

        ref = np.asarray(inputs["ref_crop"])
        tgt = np.asarray(inputs["warped_target_crop"])
        ref_mask = np.asarray(inputs["ref_valid_mask"], dtype=bool)
        tgt_mask = np.asarray(inputs["target_valid_mask"], dtype=bool)
        reported_dx = int(round(phase["dx_px"]))
        reported_dy = int(round(phase["dy_px"]))
        counterfactual = {
            "zero": evaluate_shift_counterfactual(ref, tgt, tgt_mask, 0, 0),
            "reported": evaluate_shift_counterfactual(ref, tgt, tgt_mask, reported_dx, reported_dy),
            "inverse": evaluate_shift_counterfactual(ref, tgt, tgt_mask, -reported_dx, -reported_dy),
        }
        sweep = translation_metric_surface(
            ref, tgt, tgt_mask, radius_px=12, phase_shift=(phase["dx_px"], phase["dy_px"])
        )
        mask = mask_boundary_stress_test(ref, tgt, ref_mask, tgt_mask)
        representations = representation_stability(ref, tgt, inputs["joint_valid_mask"], include_eroded=True)
        injection = real_crop_injection_recovery(
            inputs,
            [
                (0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
                (3, 0), (-3, 0), (6, 0), (0, -6), (10, 0), (0, -10),
            ],
        )
        result = {
            "edge": edge,
            "role": role,
            "inputs": inputs,
            "baseline_reproduction_status": status,
            "flow": trace_phase_validation_flow(inputs, phase),
            "phase": phase,
            "sign_convention": run_phase_sign_convention(ref, ref_mask, tgt, tgt_mask, 4, -7),
            "counterfactual": counterfactual,
            "sweep": sweep,
            "mask": mask,
            "representations": representations,
            "injection": injection,
            "peak_diagnostics": phase_peak_diagnostics({}),
        }
        result["comparison"] = compare_working_vs_suspect(result)
        result["root_cause"] = classify_phase_validation_root_cause(_root_cause_evidence(result))
        results[edge] = result

    artifacts = write_phase_audit_artifacts(output_dir, baseline, results)
    return {"baseline": baseline, "results": results, "artifacts": artifacts}


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    run_audit(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
