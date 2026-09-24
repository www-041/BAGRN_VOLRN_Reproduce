"""Freeze the manually confirmed B9 five-scene selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.multiscene_sift.b9_frozen_config import (
    build_frozen_config,
    write_frozen_config,
)


DEFAULT_INDICES = (2, 3, 5, 8, 10)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--overlap-graph", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--indices", nargs=5, type=int, default=DEFAULT_INDICES)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    overlap = json.loads(args.overlap_graph.read_text(encoding="utf-8"))
    records = manifest["scenes"]
    pairs = overlap["pairs"]
    config = build_frozen_config(records, pairs, args.indices)
    output = write_frozen_config(config, args.output)
    print(
        f"status=COMPLETE frozen_indices={config['selection']['manifest_indices']} "
        f"scenes={len(config['scenes'])} edges={len(config['graph_edges'])} "
        f"output={output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
