"""Run one fixed-geometry B9 radiometric experiment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.multiscene_sift.radiometric_runner import run_fixed_geometry_radiometric


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-config", required=True, type=Path)
    parser.add_argument("--global-run-dir", required=True, type=Path)
    parser.add_argument("--output-grid", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--method", required=True, choices=("RAW", "BAGRN", "BAGRN_VOLRN"))
    parser.add_argument("--geometry-run", default="sift_mst")
    parser.add_argument("--radiometric-control-idx", type=int, default=0)
    parser.add_argument("--block-size-pixels", type=int, default=200)
    parser.add_argument("--lambda-param", type=float, default=0.1)
    parser.add_argument("--rho", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=20)
    parser.add_argument("--tol", type=float, default=1e-4)
    args = parser.parse_args(argv)
    run_fixed_geometry_radiometric(
        args.source_config, args.global_run_dir, args.output_grid, args.output_dir,
        method=args.method, geometry_run=args.geometry_run,
        radiometric_control_idx=args.radiometric_control_idx,
        block_size_pixels=args.block_size_pixels, lambda_param=args.lambda_param,
        rho=args.rho, max_iter=args.max_iter, tol=args.tol,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
