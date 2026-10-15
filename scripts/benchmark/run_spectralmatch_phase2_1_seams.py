"""Apply the independent coverage-retaining assignment to a saved label grid."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.benchmark_v1_spectralmatch_phase2.coverage_seams import assign_complete_coverage, validate_assignment


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-npz", required=True, type=Path, help="keys: valid_i, valid_j, official_labels")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--priority", nargs=2, type=int, choices=(0, 1), default=(0, 1))
    args = parser.parse_args()
    with np.load(args.input_npz) as data:
        labels, audit = assign_complete_coverage(data["valid_i"], data["valid_j"], data["official_labels"], args.priority)
        validation = validate_assignment(labels, data["valid_i"], data["valid_j"], data["valid_i"] | data["valid_j"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "source_labels.npy", labels, allow_pickle=False)
    report = {"scope": "DIAGNOSTIC_CROP_ONLY", "variant": "coverage_retaining_source_assignment",
              "priority": list(args.priority), "assignment": audit, "validation": validation}
    (args.output_dir / "coverage_audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if validation["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
