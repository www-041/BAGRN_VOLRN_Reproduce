"""Run one matcher on the frozen B9 five-scene geometry configuration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.multiscene_sift.b9_runner import run_b9_registration


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--matcher",
        required=True,
        choices=("sift", "loftr", "efficient_loftr", "lightglue_disk"),
    )
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    result = run_b9_registration(
        config,
        args.output_dir,
        matcher=args.matcher,
        device=args.device,
    )
    print(f"status={result['status']} matcher={args.matcher} output={args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
