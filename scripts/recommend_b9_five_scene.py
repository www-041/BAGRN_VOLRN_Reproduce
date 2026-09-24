"""Recommend a five-scene B9 validation topology from footprints only."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.multiscene_sift.b9_dataset import discover_b9_scenes
from src.multiscene_sift.b9_overlap import build_b9_overlap_graph
from src.multiscene_sift.b9_selection import (
    rank_five_scene_candidates,
    write_candidate_outputs,
)


DEFAULT_B9_ROOT = Path(
    r"D:\科研\地质一号\文献\BAGRN_VOLRN_Reproduce"
    r"\data\input\B9_reference_13scenes_full"
)
DEFAULT_OUTPUT_DIR = Path("data/output/b9_five_scene_validation/dataset_audit")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_B9_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)

    records = discover_b9_scenes(args.input_root)
    pairs = build_b9_overlap_graph(records)
    candidates = rank_five_scene_candidates(records, pairs)
    payload = write_candidate_outputs(records, candidates, args.output_dir)
    print(f"status={payload['status']}")
    if payload["recommended"]:
        print(f"recommended={payload['recommended']['scene_ids']}")
        print(f"recommended_score={payload['recommended']['score']:.6f}")
    print(f"top_candidates={len(payload['top_candidates'])}")
    print(f"output={args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
