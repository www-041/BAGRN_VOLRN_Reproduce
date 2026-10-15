"""Run or verify the frozen-geometry BAGRN final-baseline replay."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.multiscene_sift.final_baseline_replay import (
    BaselineReplaySpec,
    _sha256,
    materialize_final_results,
    replay_baseline,
    verify_final_results,
)


def _verify(args) -> None:
    specs = (
        BaselineReplaySpec("efficient_loftr_translation_l2_bagrn", "efficient_loftr_translation_l2", args.frozen_config, args.main_global_run_dir, args.output_grid, args.replay_root, args.main_pairwise_summary),
        BaselineReplaySpec("sift_mst_bagrn", "sift_mst", args.frozen_config, args.traditional_global_run_dir, args.output_grid, args.replay_root, args.traditional_pairwise_summary),
    )
    verify_final_results(args.output_root, specs)
    print(f"status=PASS verified={args.output_root}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-config", type=Path, required=True)
    parser.add_argument("--output-grid", type=Path, required=True)
    parser.add_argument("--main-global-run-dir", type=Path, required=True)
    parser.add_argument("--traditional-global-run-dir", type=Path, required=True)
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--main-pairwise-summary", type=Path, default=None)
    parser.add_argument("--traditional-pairwise-summary", type=Path, default=None)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args(argv)
    if args.verify:
        _verify(args)
        return 0
    main_artifact = replay_baseline(BaselineReplaySpec(
        name="efficient_loftr_translation_l2_bagrn",
        geometry_run="efficient_loftr_translation_l2",
        source_config=args.frozen_config,
        global_run_dir=args.main_global_run_dir,
        output_grid=args.output_grid,
        replay_root=args.replay_root,
        pairwise_summary_csv=args.main_pairwise_summary,
    ))
    traditional_artifact = replay_baseline(BaselineReplaySpec(
        name="sift_mst_bagrn",
        geometry_run="sift_mst",
        source_config=args.frozen_config,
        global_run_dir=args.traditional_global_run_dir,
        output_grid=args.output_grid,
        replay_root=args.replay_root,
        pairwise_summary_csv=args.traditional_pairwise_summary,
    ))
    manifest = materialize_final_results(main_artifact, traditional_artifact, args.output_root)
    print(json.dumps({"status": "PASS", "manifest": manifest}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
