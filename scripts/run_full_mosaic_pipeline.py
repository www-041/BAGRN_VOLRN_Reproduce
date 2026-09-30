"""CLI for the Task15 generic end-to-end mosaic pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline.config import load_pipeline_config
from src.pipeline.runner import run_pipeline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the generic Task15 mosaic pipeline")
    parser.add_argument("--config", required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--fresh", action="store_true")
    group.add_argument("--resume", action="store_true")
    parser.add_argument("--from-stage", type=int, default=None)
    args = parser.parse_args(argv)
    config = load_pipeline_config(args.config)
    result = run_pipeline(config, mode="resume" if args.resume else "fresh", from_stage=args.from_stage)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
