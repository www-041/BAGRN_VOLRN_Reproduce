"""Phase 2.1 external submission scorer; does not modify frozen evaluator files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.benchmark_v1_spectralmatch_phase2.scoring import score_submission


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", default="data/output/benchmark_v1_unified/freeze_20261013_v3")
    parser.add_argument("--submission", required=True, type=Path)
    parser.add_argument("--regime", required=True, choices=["A", "B1", "B2", "B3"])
    parser.add_argument("--scope", choices=["FORMAL_FULL", "DIAGNOSTIC_CROP_ONLY"], default="FORMAL_FULL")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    submission = json.loads(args.submission.read_text(encoding="utf-8"))
    result = score_submission(args.freeze, submission, args.regime, expected_scope="FORMAL_FULL", scope=args.scope)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result.get("status") == "NOT_RUN_B3" or result.get("formal_ranking_eligible") else 2


if __name__ == "__main__":
    raise SystemExit(main())
