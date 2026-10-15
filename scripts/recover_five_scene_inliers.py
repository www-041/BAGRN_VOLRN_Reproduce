"""Replay frozen accepted five-scene SIFT pairs and export inlier artifacts."""

from __future__ import annotations

import argparse

from src.multiscene_sift.inlier_recovery import recover_accepted_pairs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--five-scene-run-dir", required=True)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    result = recover_accepted_pairs(
        args.five_scene_run_dir, args.input_root, args.output_dir
    )
    print(f"Replayed accepted pairs: {result['accepted_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
