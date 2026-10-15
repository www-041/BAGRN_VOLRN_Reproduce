"""Independent launcher for the six-scene mountain B9 pipeline run.

This wrapper is intentionally separate from the existing launchers. It only
selects the new dataset/configuration and delegates all scientific stages to
the unchanged Task15 pipeline implementation.
"""

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


CONFIG = ROOT / "configs/mountain_b9_6scene.yaml"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the unchanged Task15 pipeline on six mountain V-sensor B9 scenes")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--fresh", action="store_true", help="start a new six-scene run")
    group.add_argument("--resume", action="store_true", help="resume the configured output checkpoint")
    parser.add_argument("--from-stage", type=int, default=None, help="optional Task15 stage index")
    args = parser.parse_args(argv)

    config = load_pipeline_config(CONFIG)
    result = run_pipeline(config, mode="resume" if args.resume else "fresh", from_stage=args.from_stage)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
