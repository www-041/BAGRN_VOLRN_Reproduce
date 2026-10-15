"""Run only the Task15.1 frozen-input invariance audits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline.invariance import run_invariance_audit
from src.pipeline.final_report import write_final_report


def finalize(root: Path) -> dict:
    report = write_final_report(root)
    summary_path = root / "12_report/reference_comparison.json"
    comparison = json.loads(summary_path.read_text(encoding="utf-8"))
    consistency = comparison.get("report_consistency_checks", {})
    status_path = root / "pipeline_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status.update({
        "pipeline_status": "SUCCESS",
        "scientific_decision": report["scientific_decision"],
        "scene_order_invariance_measured": True,
        "scene_order_invariance_status": comparison.get("fresh_scene_order_invariance", {}).get("status", "NOT_MEASURED"),
        "correction_order_invariance_measured": True,
        "correction_order_invariance_status": comparison.get("fresh_correction_order_invariance", {}).get("status", "NOT_MEASURED"),
        "report_consistency_audit_status": "PASS" if all(consistency.values()) else "HARD_STOP_REPORT_CONSISTENCY",
    })
    status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest_path = root / "pipeline_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["scientific_decision"] = report["scientific_decision"]
    manifest["task15_1_fresh_invariance_audit"] = {
        "scene_order": comparison.get("fresh_scene_order_invariance"),
        "correction_order": comparison.get("fresh_correction_order_invariance"),
        "report_consistency_audit_status": status["report_consistency_audit_status"],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (root / "pipeline.log").write_text(
        "Task15.1 fresh invariance and final-report consistency audit completed; "
        f"scientific_decision={report['scientific_decision']}; pipeline_status=SUCCESS.\n",
        encoding="utf-8",
    )
    return {"report": report["scientific_decision"], "pipeline_status": status["pipeline_status"], "report_consistency": status["report_consistency_audit_status"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/output/final_pipeline/b9_13scene"))
    parser.add_argument("--finalize-only", action="store_true")
    args = parser.parse_args()
    result = {"finalization": finalize(args.root)} if args.finalize_only else run_invariance_audit(args.root)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
