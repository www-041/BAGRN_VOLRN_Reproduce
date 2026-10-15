from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.benchmark_v1_unified.reporting import write_json_atomic
from src.benchmark_v1_unified.provenance_audit import audit_mosaic_provenance
from src.benchmark_v1_unified.ranking_gate import apply_ranking_gate
from src.benchmark_v1_unified.portable_freeze_verification import (
    as_freeze_verification,
    verify_portable_lock,
)
from src.benchmark_v1_unified.submission_scoring import evaluate_submission
from src.benchmark_v1_unified.streaming_submission_scoring import (
    DISK_RESERVE_BYTES,
    RAM_BUDGET_BYTES,
    RAM_RESERVE_BYTES,
    evaluate_submission_streaming,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score canonical-grid GeoTIFF artifacts against frozen Unified Benchmark V1.")
    parser.add_argument("--freeze", required=True, help="Path to freeze_20261013_v3")
    parser.add_argument("--method-submission", required=True, help="method_submission.json; relative artifact paths resolve beside this file")
    parser.add_argument("--regimes", nargs="+", choices=("A", "B1", "B2", "B3"), default=("A", "B1", "B2"))
    parser.add_argument("--output", required=True, help="Evaluation JSON output path")
    parser.add_argument("--engine", choices=("streaming", "array"), default="streaming",
                        help="streaming is the bounded-memory default; array is the reference engine")
    parser.add_argument("--scratch-dir", help="Scratch directory; defaults beside the report")
    parser.add_argument("--ram-budget-mib", type=float, default=RAM_BUDGET_BYTES / 1024**2)
    parser.add_argument("--ram-reserve-mib", type=float, default=RAM_RESERVE_BYTES / 1024**2)
    parser.add_argument("--disk-reserve-gib", type=float, default=DISK_RESERVE_BYTES / 1024**3)
    parser.add_argument("--freeze-path-mapping", help="JSON mapping for original absolute lock paths on a different host")
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    submission_path = Path(args.method_submission).resolve()
    submission = json.loads(submission_path.read_text(encoding="utf-8"))
    output_path = Path(args.output).resolve()
    portable_check = None
    freeze_verifier = None
    if args.freeze_path_mapping:
        mapping_path = Path(args.freeze_path_mapping).resolve()
        mapping_document = json.loads(mapping_path.read_text(encoding="utf-8"))
        portable_check = verify_portable_lock(
            Path(args.freeze) / "freeze_lock.json",
            mapping_document.get("mappings", []),
            audit_path=output_path.with_suffix(output_path.suffix + ".freeze_audit.json"),
        )
        if not portable_check.get("valid"):
            missing = portable_check.get("counts_by_status", {}).get("MISSING_FILE", 0)
            status = "NOT_RUN_CLOUD_FREEZE_DATA_MISSING" if missing else portable_check.get("status", "FREEZE_INVALID")
            print(json.dumps({"status": status, "score_written": False,
                              "freeze_audit": str(output_path.with_suffix(output_path.suffix + ".freeze_audit.json"))},
                             ensure_ascii=False))
            return 2
        freeze_verifier = lambda _root: as_freeze_verification(portable_check)
    if args.engine == "streaming":
        scratch_dir = Path(args.scratch_dir).resolve() if args.scratch_dir else output_path.parent / ".unified_benchmark_scratch"
        scratch_dir.mkdir(parents=True, exist_ok=True)
        result = evaluate_submission_streaming(
            args.freeze,
            submission,
            submission_dir=submission_path.parent,
            regimes=args.regimes,
            scratch_dir=scratch_dir,
            ram_budget_bytes=int(args.ram_budget_mib * 1024**2),
            ram_reserve_bytes=int(args.ram_reserve_mib * 1024**2),
            disk_reserve_bytes=int(args.disk_reserve_gib * 1024**3),
            freeze_verifier=freeze_verifier,
        )
    else:
        result = evaluate_submission(args.freeze, submission, submission_dir=submission_path.parent,
                                     regimes=args.regimes, freeze_verifier=freeze_verifier)
    if result.get("status") == "RESOURCE_PREFLIGHT_FAILED":
        print(json.dumps({"status": result["status"], "report_written": False,
                          "resource_preflight": result.get("resource_preflight")}, ensure_ascii=False))
        return 2
    provenance = None
    if "A" in args.regimes and result.get("status") == "EVALUATED" and submission.get("seam_type") == "HARD":
        provenance = audit_mosaic_provenance(submission, args.freeze, submission_dir=submission_path.parent,
                                             freeze_verifier=freeze_verifier)
    result = apply_ranking_gate(submission, result, args.freeze, provenance_audit=provenance,
                                freeze_verification=portable_check)
    output = write_json_atomic(output_path, result)
    print(json.dumps({"status": result.get("status"), "ranking_eligible": result.get("ranking_eligible", False), "report": str(output)}, ensure_ascii=False))
    return 0 if result.get("status") == "EVALUATED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
