"""Execute Task16 without rerunning the frozen Task15 geometry/BAGRN stages."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.task16_volrn_comparison import run_task16


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task15-root", type=Path, default=ROOT / "data/output/final_pipeline/b9_13scene")
    parser.add_argument("--output-root", type=Path, default=ROOT / "data/output/b9_13scene_volrn_comparison")
    parser.add_argument("--protocol", type=Path, default=None)
    parser.add_argument("--resume", action="store_true", help="reuse a completed A baseline in a partially populated Task16 directory")
    args = parser.parse_args()
    summary = run_task16(args.task15_root, args.output_root, protocol_path=args.protocol, resume=args.resume)
    print(f"Task16 status: {summary['status']}")
    print(f"Output: {args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
