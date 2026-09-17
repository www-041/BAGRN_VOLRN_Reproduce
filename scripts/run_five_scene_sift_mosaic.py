"""CLI entry-point for five-scene SIFT mosaic benchmark.

Usage::

    python -m scripts.run_five_scene_sift_mosaic \\
        --input-root <path> \\
        --output-dir <path> \\
        [--registration-band B14] \\
        [--bands B14,B8,B5] \\
        [--match-max-side 1600] \\
        [--ransac-threshold 2.0]
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
        description="Five-scene SIFT multi-image registration and mosaic"
    )
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

    args = parser.parse_args()

    bands = tuple(b.strip() for b in args.bands.split(","))

    result = run_five_scene_mosaic(
        input_root=args.input_root,
        output_dir=args.output_dir,
        registration_band=args.registration_band,
        bands=bands,
        match_max_side=args.match_max_side,
        ransac_threshold=args.ransac_threshold,
    )

    if result.get("status") == "DISCONNECTED":
        print(f"\nSTOP: {result['error']}")
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()