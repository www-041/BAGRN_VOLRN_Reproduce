"""Inspect four-pair DEVELOPMENT adapter readiness without running matchers."""

import argparse
import json
from pathlib import Path

from benchmark_phase4_freeze import load_frozen_trial
from benchmark_phase4_development import (environment_preflight, load_development_truth,
                                          prepare_trial_inputs, validate_manifest_destination)


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-env", action="store_true")
    parser.add_argument("--write-manifest", type=Path)
    args = parser.parse_args()
    snapshot = load_frozen_trial(ROOT)
    dev = load_development_truth(snapshot)
    report = {"scope": "FOUR_PAIR_DEVELOPMENT_ONLY", "development_counts": {k: len(v) for k, v in dev.items()},
              "environment": environment_preflight(), "inputs": prepare_trial_inputs(snapshot, ROOT)}
    if args.write_manifest:
        destination = validate_manifest_destination(args.write_manifest, ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"scope": report["scope"], "development_counts": report["development_counts"],
                      "environment": report["environment"], "matcher_executed": False}))


if __name__ == "__main__":
    main()
