#!/usr/bin/env python
"""CLI entry-point for the two-image flat-terrain registration benchmark.

Example
-------

.. code-block:: bash

    python scripts/run_two_image_registration_benchmark.py \\
        --ref data/input/REFERENCE.TIF \\
        --tgt data/input/TARGET.TIF \\
        --output-dir data/output/benchmark \\
        --band 1 \\
        --methods phase,sift \\
        --match-max-side 1600 \\
        --ransac-threshold 2.0
"""

from __future__ import annotations

import argparse
import logging
import sys


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Two-image flat-terrain registration benchmark",
    )
    parser.add_argument("--ref", required=True, help="Path to reference GeoTIFF")
    parser.add_argument("--tgt", required=True, help="Path to target GeoTIFF")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--band", type=int, default=1, help="Raster band (1-based)")
    parser.add_argument(
        "--methods",
        default="phase,sift",
        help="Comma-separated matcher names (default: phase,sift)",
    )
    parser.add_argument(
        "--match-max-side",
        type=int,
        default=1600,
        help="max_side for MatchView downscaling (default: 1600)",
    )
    parser.add_argument(
        "--ransac-threshold",
        type=float,
        default=2.0,
        help="RANSAC residual threshold in pixels (default: 2.0)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Torch device (default: auto)",
    )

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    from src.registration_benchmark.runner import run_two_image_benchmark

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    result = run_two_image_benchmark(
        ref_path=args.ref,
        tgt_path=args.tgt,
        output_dir=args.output_dir,
        band=args.band,
        methods=methods,
        match_max_side=args.match_max_side,
        ransac_threshold=args.ransac_threshold,
        device=args.device,
    )

    # Print summary table
    print("\n===== SUMMARY =====")
    for row in result["summary"]:
        print(f"  {row['method']:12s}  status={row['status']:22s}  "
              f"inliers={str(row.get('inliers','?')):>4s}  "
              f"rmse={_fmt(row.get('residual_rmse'))}")

    print(f"\nOutput: {result['output_dir']}")
    return 0


def _fmt(val) -> str:
    if val is None or (isinstance(val, float) and val != val):
        return "   N/A"
    return f"{val:6.3f}"


if __name__ == "__main__":
    sys.exit(main())