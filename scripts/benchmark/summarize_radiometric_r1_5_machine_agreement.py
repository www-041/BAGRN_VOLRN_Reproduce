from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from benchmark_v1_radiometric_r15.inputs import ALLOWED_PAIRS  # noqa: E402
from benchmark_v1_radiometric_r15.machine_suggestions import (  # noqa: E402
    load_machine_suggestion_map,
    summarize_human_agreement,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Summarize machine-vs-human label agreement (not accuracy).")
    parser.add_argument("--machine-run-id", required=True)
    parser.add_argument("--human-review-dir", type=Path,
                        default=ROOT / "data/output/benchmark_v1_radiometric_r1_5/multiband_reviews")
    parser.add_argument("--output", type=Path, help="optional new JSON report path; existing paths are never overwritten")
    args = parser.parse_args(argv)
    machine_root = ROOT / "data/output/benchmark_v1_radiometric_r1_5/machine_label_suggestions" / args.machine_run_id
    machine_features = []
    human_rows = []
    try:
        for pair in sorted(ALLOWED_PAIRS):
            machine_path = machine_root / f"{pair}.machine_suggestions.geojson"
            if not machine_path.is_file():
                continue
            mapping = load_machine_suggestion_map(machine_path, expected_pair_id=pair)
            import json as _json
            collection = _json.loads(machine_path.read_text(encoding="utf-8"))
            machine_features.extend(collection["features"])
            snapshot = args.human_review_dir / f"{pair}_snapshot_v2.json"
            if snapshot.is_file():
                human_rows.extend(_json.loads(snapshot.read_text(encoding="utf-8")).get("records", []))
        result = summarize_human_agreement(machine_features, human_rows)
        result["machine_run_id"] = args.machine_run_id
        result["machine_candidate_count"] = len(machine_features)
        encoded = json.dumps(result, ensure_ascii=False, indent=2)
        if args.output:
            target = args.output if args.output.is_absolute() else ROOT / args.output
            if target.exists():
                raise FileExistsError(f"refusing to overwrite {target}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("x", encoding="utf-8") as stream:
                stream.write(encoded)
            print(f"AGREEMENT_REPORT_WRITTEN {target}")
        print(encoded)
        return 0
    except Exception as exc:
        print(f"R1.5_AGREEMENT_SUMMARY_FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
