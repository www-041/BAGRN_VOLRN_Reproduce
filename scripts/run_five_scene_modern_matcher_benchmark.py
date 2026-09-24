"""Run the matcher-only five-scene modern benchmark."""

from __future__ import annotations

import argparse
import logging

from src.multiscene_sift.modern_matcher_benchmark import (
    MODERN_MATCHERS,
    run_modern_matcher_benchmark,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark modern matchers on five scenes without BAGRN/VOLRN/mosaic."
    )
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--registration-band", default="B14")
    parser.add_argument("--matchers", nargs="+", default=list(MODERN_MATCHERS))
    parser.add_argument("--match-max-side", type=int, default=1600)
    parser.add_argument("--ransac-threshold", type=float, default=2.0)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    result = run_modern_matcher_benchmark(
        input_root=args.input_root,
        output_dir=args.output_dir,
        registration_band=args.registration_band,
        matchers=tuple(args.matchers),
        match_max_side=args.match_max_side,
        ransac_threshold=args.ransac_threshold,
        random_seed=args.random_seed,
        device=args.device,
    )
    print(f"status={result['status']}")
    for matcher, summary in result["methods"].items():
        print(f"{matcher}: {summary['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
