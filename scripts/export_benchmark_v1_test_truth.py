"""Explicit gated export of independently reviewed four-pair TEST truth."""

import argparse
from pathlib import Path
import json

from benchmark_phase4_freeze import load_frozen_trial
from benchmark_phase4_review import load_review_events, review_state
from benchmark_phase4_export import export_formal_test


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true", help="Read-only review counts; writes no truth")
    parser.add_argument("--event-dir", type=Path, default=ROOT / "docs/benchmark/v1/phase4/second_reviews")
    parser.add_argument("--attestation", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.preflight:
        snapshot = load_frozen_trial(ROOT)
        events = [event for pair in snapshot.trial_pairs
                  for event in load_review_events(args.event_dir / f"{pair}.jsonl", snapshot)]
        state = review_state(snapshot, events)
        print(json.dumps({"frozen_test_candidates": len(state),
                          "pending": sum(s["human_outcome"] == "PENDING" for s in state.values()),
                          "human_approved_spatial_valid": sum(s["eligible"] for s in state.values()),
                          "formal_test_truth_written": False}))
        return
    if args.attestation is None or args.output_dir is None:
        parser.error("formal export requires --attestation and --output-dir")
    print(json.dumps(export_formal_test(ROOT, args.event_dir, args.attestation, args.output_dir)))


if __name__ == "__main__":
    main()
