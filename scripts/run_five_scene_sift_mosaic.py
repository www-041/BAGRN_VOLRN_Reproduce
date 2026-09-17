"""CLI entry-point for five-scene SIFT + BAGRN + VOLRN multi-image mosaic.

Usage::

    python -m scripts.run_five_scene_sift_mosaic \\
        --input-root <path> \\
        --output-dir <path>
"""

from __future__ import annotations

import argparse
import logging
import sys

from src.multiscene_sift.runner import run_five_scene_mosaic

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)


def main():
    parser = argparse.ArgumentParser(
        description="Five-scene SIFT + BAGRN + VOLRN multi-image registration and mosaic"
    )
    # Registration
    parser.add_argument("--input-root", required=True,
                        help="Path to the flat/ input directory")
    parser.add_argument("--output-dir", required=True,
                        help="Output directory for all results")
    parser.add_argument("--registration-band", default="B14",
                        help="Band for SIFT registration (default: B14)")
    parser.add_argument("--bands", default="B14,B8,B5",
                        help="Comma-separated band list (default: B14,B8,B5)")
    parser.add_argument("--match-max-side", type=int, default=1600,
                        help="Max side for match-view (default: 1600)")
    parser.add_argument("--ransac-threshold", type=float, default=2.0,
                        help="RANSAC threshold in pixels (default: 2.0)")

    # Radiometric
    parser.add_argument("--block-size", type=int, default=200,
                        help="VOLRN block size (default: 200)")
    parser.add_argument("--lambda-param", type=float, default=0.1,
                        help="VOLRN lambda (default: 0.1)")
    parser.add_argument("--rho", type=float, default=1.0,
                        help="VOLRN rho (default: 1.0)")
    parser.add_argument("--max-iter", type=int, default=200,
                        help="VOLRN max iterations (default: 200)")
    parser.add_argument("--tol", type=float, default=1e-4,
                        help="VOLRN tolerance (default: 1e-4)")

    # Diagnostics
    parser.add_argument("--save-diagnostics", action="store_true",
                        help="Save diagnostic plots and detail mosaics")

    args = parser.parse_args()

    bands = tuple(b.strip() for b in args.bands.split(","))

    result = run_five_scene_mosaic(
        input_root=args.input_root,
        output_dir=args.output_dir,
        registration_band=args.registration_band,
        bands=bands,
        match_max_side=args.match_max_side,
        ransac_threshold=args.ransac_threshold,
        save_diagnostics=args.save_diagnostics,
        block_size=args.block_size,
        lambda_param=args.lambda_param,
        rho=args.rho,
        max_iter=args.max_iter,
        tol=args.tol,
    )

    if result.get("status") == "DISCONNECTED":
        print(f"\nSTOP: {result['error']}")
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()