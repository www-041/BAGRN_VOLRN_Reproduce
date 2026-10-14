from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from benchmark_v1_radiometric_r15.report import build_aggregate_outputs  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Build the R1.5 four-pair candidate audit report and map.")
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)
    run_dir = ROOT / "data/output/benchmark_v1_radiometric_r1_5/auto_roi_candidates" / args.run_id
    report_dir = ROOT / "docs/benchmark/v1/radiometric_r1_5" / args.run_id
    try:
        output = build_aggregate_outputs(run_dir, report_dir)
    except Exception as exc:
        print(f"R1.5_REPORT_FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(f"R1.5_REPORT_READY output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
