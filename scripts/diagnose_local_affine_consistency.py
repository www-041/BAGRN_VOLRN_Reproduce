"""Run the fixed three-edge local affine consistency diagnosis.

The command consumes saved SIFT/RANSAC artifacts and performs only diagnostic
least-squares and pixel-validation probes.  It never writes pairwise results
or changes the production registration/mosaic pipeline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.multiscene_sift.local_affine_consistency_diagnostics import (
    LOCAL_AFFINE_EDGES,
    build_diagnostic_evidence,
    build_region_validation_crops,
    compare_global_vs_local_pixel_alignment,
    compare_partition_stability,
    cross_validate_local_affine,
    fit_region_local_affines,
    load_local_affine_baseline,
    summarize_local_affine_variation,
    write_diagnostic_artifacts,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--edge-reliability-dir", type=Path, required=True)
    parser.add_argument("--selected-pair-dir", type=Path, required=True)
    parser.add_argument("--five-scene-run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--band", default="B14")
    return parser


def _load_pair_matrices(selected_pair_dir: Path) -> dict[str, np.ndarray]:
    path = selected_pair_dir / "02_new_pairwise_registration_results.json"
    if not path.is_file():
        raise FileNotFoundError(f"pairwise registration results not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    matrices = {}
    for row in payload.get("results", []):
        if row.get("pixel_matrix") is not None:
            key = f"{int(row['idx_i'])}-{int(row['idx_j'])}"
            matrices[key] = np.asarray(row["pixel_matrix"], dtype=np.float64)
    return matrices


def _load_manifest(five_scene_run_dir: Path) -> dict:
    candidates = [
        five_scene_run_dir / "../nine_scene_overlap_diagnostic/01_nine_scene_manifest.json",
        five_scene_run_dir / "01_nine_scene_manifest.json",
    ]
    for path in candidates:
        path = path.resolve()
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    raise FileNotFoundError("nine-scene manifest not found beside the supplied run artifacts")


def _band_path(input_root: Path, manifest_row: dict, band: str) -> Path:
    saved = Path(manifest_row.get("b14_path", ""))
    if saved.is_file() and band == "B14":
        return saved
    name = manifest_row["scene_name"]
    scene_dir = input_root / name
    candidates = sorted(scene_dir.glob(f"*_{band}.TIF")) + sorted(scene_dir.glob(f"*_{band}.tif"))
    if not candidates:
        raise FileNotFoundError(f"{band} not found for scene {name} under {input_root}")
    return candidates[0]


def _load_pair_arrays(input_root: Path, manifest: dict, edge: str, band: str) -> tuple[dict, tuple[float, float, float, float]]:
    from src.registration_benchmark.common_grid import load_pair_to_common_grid

    i, j = (int(part) for part in edge.split("-"))
    rows = {int(row["scene_index"]): row for row in manifest["scenes"]}
    pair = load_pair_to_common_grid(
        str(_band_path(input_root, rows[i], band)),
        str(_band_path(input_root, rows[j], band)),
    )
    row_start, row_end, col_start, col_end = pair.overlap_window
    return {
        "array": pair.ref_raw, "valid_mask": pair.ref_valid,
        "target_array": pair.tgt_raw, "target_valid_mask": pair.tgt_valid,
    }, (float(col_start), float(row_start), float(col_end), float(row_end))


def _edge_bounds(record: dict, common_bounds) -> tuple[float, float, float, float]:
    ref = record["ref_xy"]
    if len(ref) == 0:
        return common_bounds
    # Saved inliers are already in the pair common-grid frame.  Keep the
    # geometric overlap bounds when available; otherwise use their envelope.
    if common_bounds is not None:
        return common_bounds
    return (float(np.min(ref[:, 0])), float(np.min(ref[:, 1])),
            float(np.max(ref[:, 0]) + 1), float(np.max(ref[:, 1]) + 1))


def _phase_summary(rows: list[dict]) -> dict:
    valid = [row for row in rows if row.get("status") == "OK"]
    if not valid:
        return {"n_regions_pixel_validated": 0, "global_phase_median": None,
                "local_phase_median": None, "median_phase_improvement_px": None,
                "fraction_regions_improved_gt_1px": 0.0,
                "fraction_regions_improved_gt_3px": 0.0}
    improvements = np.asarray([row["phase_improvement_px"] for row in valid], dtype=float)
    return {
        "n_regions_pixel_validated": len(valid),
        "global_phase_median": float(np.median([row["global_phase_mag"] for row in valid])),
        "local_phase_median": float(np.median([row["local_phase_mag"] for row in valid])),
        "median_phase_improvement_px": float(np.median(improvements)),
        "fraction_regions_improved_gt_1px": float(np.mean(improvements > 1.0)),
        "fraction_regions_improved_gt_3px": float(np.mean(improvements > 3.0)),
        "global_ncc_median": float(np.median([row["global_ncc"] for row in valid])),
        "local_ncc_median": float(np.median([row["local_ncc"] for row in valid])),
    }


def diagnose(args: argparse.Namespace) -> dict[str, dict]:
    baseline = load_local_affine_baseline(args.edge_reliability_dir)
    matrices = _load_pair_matrices(args.selected_pair_dir)
    manifest = _load_manifest(args.five_scene_run_dir)
    results = {}
    for edge, role in LOCAL_AFFINE_EDGES.items():
        key = f"{edge[0]}-{edge[1]}"
        record = baseline["edges"][key]
        matrix = matrices.get(key)
        if matrix is None:
            raise RuntimeError(f"missing saved global matrix for fixed edge {key}")
        record["global_matrix"] = matrix
        try:
            scene_pair, common_bounds = _load_pair_arrays(args.input_root, manifest, key, args.band)
            pixel_available = True
        except (FileNotFoundError, ValueError):
            scene_pair, common_bounds, pixel_available = None, None, False
        bounds = _edge_bounds(record, common_bounds)
        models = []
        phase_rows = []
        variations = {}
        for grid_n in (2, 3):
            grid_models = fit_region_local_affines(
                record["ref_xy"], record["tgt_xy"], matrix, bounds, grid_n
            )
            models.extend(grid_models)
            variations[str(grid_n)] = summarize_local_affine_variation(grid_models, grid_n)
            if pixel_available:
                for model in grid_models:
                    if model["status"] != "OK":
                        continue
                    region = {
                        "grid_n": grid_n, "row": model["region_row"], "col": model["region_col"],
                        "bounds_px": model["bounds_px"],
                    }
                    crop = build_region_validation_crops(
                        {"array": scene_pair["array"], "valid_mask": scene_pair["valid_mask"]},
                        {"array": scene_pair["target_array"], "valid_mask": scene_pair["target_valid_mask"]},
                        matrix, model["local_matrix"], region,
                    )
                    if crop["status"] == "OK":
                        phase = compare_global_vs_local_pixel_alignment(
                            crop["reference"], crop["global_warp"], crop["local_warp"], crop
                        )
                    else:
                        phase = {"status": crop["status"]}
                    phase_rows.append(dict(phase, grid_n=grid_n, region_id=model["region_id"]))
        phase_summary = _phase_summary(phase_rows)
        for variation in variations.values():
            variation.update(phase_summary)
        stability = compare_partition_stability(variations["2"], variations["3"])
        cv = cross_validate_local_affine(record["tgt_xy"], record["ref_xy"], n_splits=5, seed=20260923)
        evidence = build_diagnostic_evidence(key, role, variations["2"], variations["3"], stability, cv, phase_summary)
        results[key] = {
            "models": models, "variation": variations, "cross_validation": cv,
            "phase": {"rows": phase_rows, "summary": phase_summary},
            "stability": stability, "evidence": evidence,
        }
    write_diagnostic_artifacts(args.output_dir, results)
    return results


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    results = diagnose(args)
    for edge in ("0-6", "2-5", "0-5"):
        print(f"{edge}: {results[edge]['evidence']['diagnosis']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
