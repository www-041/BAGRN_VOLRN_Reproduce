"""Run the two frozen-support PIF sensitivity diagnostics as separate runs."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODES = ("flood_from_match_points", "entire")


def summarize_runs(run_reports: dict[str, dict]) -> dict:
    pairs = {}
    for mode, report in run_reports.items():
        pair = next((item for item in report.get("pairs", []) if item.get("pair_id") == "00_01"), {})
        smoke_path = pair.get("smoke_result")
        smoke = json.loads(Path(smoke_path).read_text(encoding="utf-8")) if smoke_path and Path(smoke_path).is_file() else {}
        diag = smoke.get("pif_diagnostic", {})
        pairs[mode] = {
            "run_status": pair.get("status", report.get("status")),
            "scope": "DIAGNOSTIC_CROP_ONLY",
            "pif_method": mode,
            "frozen_input_support_identity_sha256": diag.get("frozen_input_support_identity_sha256"),
            "evaluation_mask_sha256": diag.get("evaluation_mask_sha256"),
            "evaluation_mask_pixels": diag.get("evaluation_mask_pixels"),
            "frozen_scene_mask_sha256": diag.get("frozen_scene_mask_sha256"),
            "pif_mask": diag.get("pif_mask", "NOT_AVAILABLE"),
            "pif_sample_count": diag.get("pif_sample_count", "NOT_AVAILABLE"),
            "pif_spatial_distribution": diag.get("pif_spatial_distribution", "NOT_AVAILABLE"),
            "stage_metrics": smoke.get("radiometric_diagnostics"),
            "stage_seconds": smoke.get("per_stage_wall_seconds"),
            "global_adjustments_path": smoke.get("pif_diagnostic", {}).get("global_adjustments_path"),
        }
    identities = [pairs.get(mode, {}).get("frozen_input_support_identity_sha256") for mode in MODES]
    input_support_match = bool(identities[0] and identities[1] and identities[0] == identities[1])
    mask_hashes = [pairs.get(mode, {}).get("evaluation_mask_sha256") for mode in MODES]
    same_support = bool(mask_hashes[0] and mask_hashes[1] and mask_hashes[0] == mask_hashes[1])
    complete = all(pairs.get(mode, {}).get("run_status") == "COMPLETED_DIAGNOSTIC_ONLY" for mode in MODES)
    return {
        "status": "PIF_DIAGNOSTIC_COMPLETE" if complete and same_support else "PIF_DIAGNOSTIC_INCOMPLETE",
        "pair_id": "00_01",
        "scope": "DIAGNOSTIC_CROP_ONLY",
        "evaluation_mask_changed": False if same_support else "NOT_VERIFIED",
        "same_frozen_input_support_identity": input_support_match,
        "modes": pairs,
        "ranking_eligible": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", default="data/output/benchmark_v1_unified/freeze_20261013_v3")
    parser.add_argument("--environment-prefix", required=True)
    parser.add_argument("--output-root", default="data/output/benchmark_v1_unified/spectralmatch_phase2_1_20261013/pif_sensitivity")
    args = parser.parse_args()
    root = Path(args.output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    reports = {}
    for mode in MODES:
        mode_root = root / mode
        command = [
            sys.executable, str(ROOT / "scripts/benchmark/run_spectralmatch_phase2.py"),
            "--freeze", str(Path(args.freeze).resolve()),
            "--output-root", str(mode_root),
            "--environment-prefix", str(Path(args.environment_prefix).resolve()),
            "--pairs", "00_01", "--crop-only", "--pif-method", mode,
        ]
        completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
        report_path = mode_root / "phase2_run_status.json"
        reports[mode] = json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else {
            "status": "RUNNER_FAILED", "return_code": completed.returncode,
            "stdout": completed.stdout, "stderr": completed.stderr,
        }
    summary = summarize_runs(reports)
    (root / "pif_sensitivity_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary["status"] == "PIF_DIAGNOSTIC_COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
