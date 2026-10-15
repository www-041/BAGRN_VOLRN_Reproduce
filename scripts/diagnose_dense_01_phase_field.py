"""CLI: dense multiscale local phase-field diagnosis of the 0-1 overlap.

Reads the scene-0 / scene-1 B14 bands and builds the metadata-only overlap
raster once; measures the local phase shift at 4×4 / 6×6 / 8×8 tile grids
under one shared quality config; fits spatial trends; classifies whether the
~40-45 px offset is a constant shift, a stable spatial gradient, or an
unstable/ambiguous measurement.  No SIFT/LoFTR/RANSAC/MST/BAGRN/VOLRN is run
or modified, and no direct/MST transform is applied before phase correlation.

Usage (PowerShell)::

    python -m scripts.diagnose_dense_01_phase_field `
        --run-dir "data/output/five_scene_sift_B14" `
        --geolocation-diagnostics-dir "data/output/five_scene_sift_B14/scene_geolocation_consistency" `
        --output-dir "data/output/five_scene_sift_B14/dense_01_phase_diagnostics" `
        --grids 4 6 8
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

from src.multiscene_sift import dense_phase_diagnostics as dpd
from src.multiscene_sift.frame_diagnostics import write_json
from src.multiscene_sift.loop_diagnostics import (
    load_run_config,
    load_scenes_for_run,
)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--geolocation-diagnostics-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--grids", nargs="+", type=int, default=[4, 6, 8])
    parser.add_argument("--min-joint-valid-fraction", type=float, default=0.60)
    parser.add_argument("--min-tile-size", type=int, default=128)
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir)
    geo_dir = Path(args.geolocation_diagnostics_dir)
    out_dir = Path(args.output_dir) if args.output_dir else (
        run_dir / "dense_01_phase_diagnostics"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    if not geo_dir.is_dir():
        print(f"ERROR: geolocation dir not found: {geo_dir}")
        return 1

    # --- 00 baseline ------------------------------------------------------
    baseline = dpd.load_dense_phase_baseline(geo_dir)
    write_json(out_dir / "00_dense_phase_baseline.json", baseline)

    config = load_run_config(run_dir)
    band = str(config.get("registration_band", "B14"))
    scenes = load_scenes_for_run(run_dir)

    # --- 01 overlap grid (metadata-only, shared by all scales) ------------
    overlap = dpd.build_metadata_only_overlap_01(scenes[0], scenes[1], band)
    write_json(out_dir / "01_overlap_grid.json", dpd.overlap_grid_metadata(overlap))

    quality_config = {
        **dpd.DEFAULT_QUALITY_CONFIG,
        "min_joint_valid_fraction": args.min_joint_valid_fraction,
        "min_tile_size_px": args.min_tile_size,
    }

    # --- 01/02 multiscale analysis -----------------------------------------
    result = dpd.run_multiscale_dense_phase(overlap, tuple(args.grids),
                                            quality_config)

    all_rows = []
    for name in sorted(result["fields"]):
        all_rows.extend(result["fields"][name])
    _write_csv(
        out_dir / "01_dense_tile_shifts.csv",
        ["grid_n", "tile_row", "tile_col", "row0", "row1", "col0", "col1",
         "center_pixel_x", "center_pixel_y", "center_world_x", "center_world_y",
         "valid_fraction_0", "valid_fraction_1", "joint_valid_fraction",
         "texture_0", "texture_1", "phase_dx_px", "phase_dy_px",
         "phase_mag_px", "phase_dx_m", "phase_dy_m", "phase_mag_m",
         "phase_confidence", "accepted", "reject_reason"],
        all_rows,
    )
    write_json(out_dir / "02_dense_shift_summary.json", {
        "grids": result["summaries"],
        "trends": result["trends"],
        "quality_config": quality_config,
    })

    # --- 03 vector map + 05 quality map -----------------------------------
    try:
        dpd.plot_dense_vector_map(
            result["fields"], overlap,
            out_dir / "03_dense_shift_vector_map.png",
        )
        dpd.plot_tile_quality_map(
            result["fields"], overlap,
            out_dir / "05_tile_quality_map.png",
        )
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING: figures skipped: {type(exc).__name__}: {exc}")

    # --- 04 multiscale stability ------------------------------------------
    stability = dpd.classify_multiscale_shift_behavior(
        result["summaries"], result["trends"]
    )
    write_json(out_dir / "04_multiscale_stability.json", stability)

    # --- 06 conclusion ------------------------------------------------------
    conclusion = dpd.build_dense_phase_conclusion(
        summaries=result["summaries"],
        trends=result["trends"],
        stability=stability,
        output_dir=str(out_dir),
    )
    write_json(out_dir / "06_dense_phase_conclusion.json", conclusion)
    dpd.write_dense_phase_conclusion_text(
        conclusion, out_dir / "06_dense_phase_conclusion.txt"
    )

    print("\n" + "=" * 66)
    print(f"Dense 0-1 phase diagnosis complete -> {out_dir}")
    for g in args.grids:
        s = result["summaries"].get(f"grid_{g}", {})
        print(
            f"  {g:>2}×{g}: accepted {s.get('n_phase_ok', '?')}/"
            f"{s.get('n_total', '?')}  "
            f"median=({s.get('median_dx_px', float('nan')):+.1f}, "
            f"{s.get('median_dy_px', float('nan')):+.1f}) px  "
            f"range=({s.get('dx_range_px', float('nan')):.1f}, "
            f"{s.get('dy_range_px', float('nan')):.1f}) px"
        )
    print(f"Multiscale state: {stability['state']}")
    print("=" * 66)
    return 0


if __name__ == "__main__":
    sys.exit(main())