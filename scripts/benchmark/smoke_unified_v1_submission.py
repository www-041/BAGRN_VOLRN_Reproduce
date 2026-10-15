from __future__ import annotations

import argparse
import importlib.util
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.benchmark_v1_unified.reporting import write_json_atomic
from src.benchmark_v1_unified.submission_scoring import evaluate_submission


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the deterministic synthetic 13-scene GeoTIFF scorer smoke test.")
    parser.add_argument("--output", default="docs/benchmark/v1/unified_v1/evaluator_smoke_report.json")
    args = parser.parse_args()
    fixture_path = ROOT / "tests" / "benchmark_v1_unified" / "test_submission_scoring.py"
    spec = importlib.util.spec_from_file_location("_submission_scoring_fixture", fixture_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load checked-in synthetic fixture: {fixture_path}")
    fixture_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture_module)
    with tempfile.TemporaryDirectory(prefix="unified-v1-artifact-smoke-") as temp:
        freeze, submission, _ = fixture_module.make_thirteen_scene_fixture(Path(temp))
        result = evaluate_submission(freeze, submission, submission_dir=Path(temp) / "method", regimes=("A", "B1", "B2", "B3"))
        report = {
            "smoke_kind": "SYNTHETIC_GEOTIFF_ONLY",
            "real_algorithm_run": False,
            "synthetic_scene_count": result.get("submission_audit", {}).get("scene_count"),
            "synthetic_pair_rows": result.get("B1", {}).get("expected_pair_rows"),
            "synthetic_positive_overlap_pairs": result.get("B1", {}).get("positive_overlap_pairs"),
            "synthetic_no_overlap_pairs": result.get("B1", {}).get("no_valid_overlap_pairs"),
            "synthetic_frozen_windows": result.get("B2", {}).get("expected_window_rows"),
            "A_status": result.get("A", {}).get("status"),
            "A_coverage": result.get("A", {}).get("mosaic_coverage"),
            "A_seam_status": result.get("A", {}).get("seam", {}).get("status"),
            "B1_status": result.get("B1", {}).get("status"),
            "B1_metrics": sorted(result.get("B1", {}).get("metric_summaries", {})),
            "B2_status": result.get("B2", {}).get("status"),
            "B2_window_count": result.get("B2", {}).get("window_count"),
            "B_scene_structure_rows": len(result.get("scene_structure", [])),
            "B2_scene_window_structure_rows": len(result.get("B2", {}).get("structure_by_scene_window", [])),
            "runtime_provenance": result.get("runtime"),
            "B3_status": result.get("B3", {}).get("status"),
            "ranking_eligible": result.get("ranking_eligible"),
            "assertions": {
                "all_13_scenes": result.get("submission_audit", {}).get("scene_count") == 13,
                "all_fixture_pairs_retained": result.get("B1", {}).get("expected_pair_rows") == 78,
                "all_fixture_windows_scored": result.get("B2", {}).get("window_count") == result.get("B2", {}).get("expected_window_rows"),
                "B3_explicitly_not_run": result.get("B3", {}).get("status") == "NOT_RUN_SYNTHETIC_METHOD_OUTPUT_REQUIRED",
                "not_a_real_method_result": True,
            },
        }
        report["status"] = "PASS" if all(report["assertions"].values()) else "FAIL"
        dest = write_json_atomic(args.output, report)
    print(f"{report['status']}: {dest}")
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
