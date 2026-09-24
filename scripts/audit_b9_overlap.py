"""Build B9 geographic overlap artifacts from metadata only."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.multiscene_sift.b9_dataset import discover_b9_scenes
from src.multiscene_sift.b9_overlap import (
    build_b9_overlap_graph,
    write_overlap_outputs,
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
    payload = write_overlap_outputs(records, pairs, args.output_dir)
    print(f"scenes={payload['n_scenes']}")
    print(f"pairs={payload['n_pairs']}")
    print(f"overlap_edges={payload['n_edges']}")
    print(f"connected_components={payload['connected_components']}")
    print(f"output={args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
