"""Run one frozen 1024x1024 LIRRN diagnostic crop at a time."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import traceback

# Allow the documented direct-script invocation from any working directory.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.benchmark_v1_lirrn import METHOD_ID, UPSTREAM_COMMIT
from src.benchmark_v1_lirrn.pair_runner import PROTOCOL_ID, run_pair


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--pair", choices=("00_01", "00_10"), required=True)
    parser.add_argument("--crop-size", type=int, default=1024)
    parser.add_argument("--pn", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261014)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.crop_size != 1024:
        parser.error("this diagnostic run is frozen to --crop-size 1024")

    command = " ".join(sys.argv)
    try:
        result = run_pair(args.freeze, args.pair, args.output_root,
                          seed=args.seed, p_n=args.pn)
    except Exception as exc:
        pair_dir = args.output_root / args.pair
        failure = {
            "protocol_id": PROTOCOL_ID,
            "method_id": METHOD_ID,
            "upstream_commit": UPSTREAM_COMMIT,
            "pair_id": args.pair,
            "status": "FAILED_RUN",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "command": command,
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        if pair_dir.is_dir():
            (pair_dir / "failure.json").write_text(
                json.dumps(failure, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            with (pair_dir / "execution.log").open("a", encoding="utf-8") as stream:
                stream.write(f"\nFAILED_RUN: {type(exc).__name__}: {exc}\n")
                stream.write(traceback.format_exc())
        print(json.dumps(failure, ensure_ascii=False), file=sys.stderr)
        traceback.print_exc()
        return 2
    print(json.dumps({k: v for k, v in result.items() if k not in {"metrics", "regression", "timing"}},
                     indent=2, ensure_ascii=False))
    print(json.dumps({"metrics": result["metrics"], "regression": result["regression"],
                      "timing": result["timing"]}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
