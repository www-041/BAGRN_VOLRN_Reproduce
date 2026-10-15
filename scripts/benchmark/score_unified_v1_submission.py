from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.benchmark_v1_unified.reporting import write_json_atomic
from src.benchmark_v1_unified.submission_scoring import evaluate_submission


def main() -> int:
    parser = argparse.ArgumentParser(description="Score canonical-grid GeoTIFF artifacts against frozen Unified Benchmark V1.")
    parser.add_argument("--freeze", required=True, help="Path to freeze_20261013_v3")
    parser.add_argument("--method-submission", required=True, help="method_submission.json; relative artifact paths resolve beside this file")
    parser.add_argument("--regimes", nargs="+", choices=("A", "B1", "B2", "B3"), default=("A", "B1", "B2"))
    parser.add_argument("--output", required=True, help="Evaluation JSON output path")
    args = parser.parse_args()
    submission_path = Path(args.method_submission).resolve()
    submission = json.loads(submission_path.read_text(encoding="utf-8"))
    result = evaluate_submission(args.freeze, submission, submission_dir=submission_path.parent, regimes=args.regimes)
    output = write_json_atomic(args.output, result)
    print(json.dumps({"status": result.get("status"), "ranking_eligible": result.get("ranking_eligible", False), "report": str(output)}, ensure_ascii=False))
    return 0 if result.get("status") == "EVALUATED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
