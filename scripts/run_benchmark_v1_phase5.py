"""Explicit one-pair/one-method command line for DZ01V Benchmark V1 Phase5."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from benchmark_v1_phase5.runner import METHODS, preflight, run_one


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--preflight", action="store_true")
    modes.add_argument("--synthetic", action="store_true")
    modes.add_argument("--smoke", action="store_true")
    modes.add_argument("--development", action="store_true")
    parser.add_argument("--pair", choices=("00_01", "02_03", "05_07", "08_10"))
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--output-root", type=Path)
    return parser


def validate_args(args) -> str | None:
    selected = [name for name in ("preflight", "synthetic", "smoke", "development")
                if getattr(args, name)]
    if len(selected) != 1:
        raise SystemExit("Choose exactly one of --preflight, --synthetic, --smoke, --development")
    mode = selected[0]
    if mode == "preflight":
        if any((args.pair, args.method, args.output_root)):
            raise SystemExit("--preflight does not accept --pair/--method/--output-root")
        return None
    missing = [name for name, value in (("--pair", args.pair),
                                        ("--method", args.method),
                                        ("--output-root", args.output_root)) if value is None]
    if missing:
        raise SystemExit(f"{mode} requires explicit " + ", ".join(missing))
    if mode == "smoke" and args.pair != "00_01":
        raise SystemExit("--smoke is restricted to pair 00_01")
    return mode


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        mode = validate_args(args)
        result = preflight(args.config) if mode is None else run_one(
            args.pair, args.method, args.config, args.output_root, mode)
    except SystemExit:
        raise
    except Exception as exc:
        print(json.dumps({"status": "RUNNER_ERROR",
                          "reason": f"{type(exc).__name__}: {exc}"}), file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    if mode is not None and result.get("execution_status") not in {"SUCCESS"}:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
