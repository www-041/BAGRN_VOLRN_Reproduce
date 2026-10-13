from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.benchmark_v1_spectralmatch_phase2.freeze_audit import audit_frozen_inputs
from src.benchmark_v1_spectralmatch_phase2.adapter import build_pair_submission
from src.benchmark_v1_spectralmatch_phase2 import UPSTREAM_COMMIT, UPSTREAM_VERSION
from src.benchmark_v1_unified.resource import run_instrumented_method


def validate_pair_selection(pair_ids):
    pairs = list(pair_ids)
    allowed = {"00_01", "00_10"}
    if len(set(pairs)) != len(pairs):
        raise ValueError("duplicate pair in selection")
    unsupported = sorted(set(pairs) - allowed)
    if unsupported:
        raise ValueError(f"unsupported pair selection: {unsupported}")
    if not pairs:
        raise ValueError("at least one predeclared pair is required")
    return pairs


def _schema_check(submission, schema_path):
    schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))
    errors = []
    for key in schema.get("required", []):
        if key not in submission:
            errors.append(f"MISSING_REQUIRED:{key}")
    for key, spec in schema.get("properties", {}).items():
        if key not in submission:
            continue
        value = submission[key]
        if "const" in spec and value != spec["const"]:
            errors.append(f"CONST_MISMATCH:{key}")
        if "enum" in spec and value not in spec["enum"]:
            errors.append(f"ENUM_MISMATCH:{key}")
        if spec.get("type") == "string" and not isinstance(value, str):
            errors.append(f"TYPE_MISMATCH:{key}")
        if spec.get("pattern") and isinstance(value, str):
            import re
            if not re.fullmatch(spec["pattern"], value):
                errors.append(f"PATTERN_MISMATCH:{key}")
    return {"status": "PASS" if not errors else "FAIL", "errors": errors}


def _write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def run(args):
    freeze_root = Path(args.freeze).resolve()
    output_root = Path(args.output_root).resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing to reuse non-empty output root: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    audit = audit_frozen_inputs(freeze_root)
    _write_json(output_root / "freeze_audit.json", audit)
    if audit["status"] != "PASS":
        raise RuntimeError(f"freeze audit rejected run: {audit['errors']}")
    schema_path = ROOT / "docs/benchmark/v1/unified_v1/method_submission_schema.json"
    freeze_manifest = json.loads((freeze_root / "frozen_manifest.json").read_text(encoding="utf-8"))
    results = []

    for pair_id in validate_pair_selection(args.pairs):
        pair_dir = output_root / pair_id
        telemetry_dir = pair_dir / "telemetry"
        worker_out = pair_dir / "worker_output"
        worker_out.mkdir(parents=True, exist_ok=True)
        environment = Path(args.environment_prefix).resolve()
        python_exe = environment / "python.exe"
        if not python_exe.is_file():
            raise FileNotFoundError(f"isolated environment interpreter not found: {python_exe}")
        base_command = [str(python_exe), "-u", str(ROOT / "scripts/benchmark/spectralmatch_phase2_worker.py"),
                        "--pair", pair_id, "--freeze", str(freeze_root), "--output-root", str(worker_out)]
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        env["PATH"] = os.pathsep.join(str(environment / part) for part in ("", "Library/bin", "Scripts", "Library/usr/bin")) + os.pathsep + env.get("PATH", "")
        env.update({"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                    "GDAL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"})
        env["GDAL_DATA"] = str(environment / "Library/share/gdal")
        env["PROJ_DATA"] = str(environment / "Library/share/proj")
        env["PROJ_LIB"] = str(environment / "Library/share/proj")
        attempts = []
        if args.crop_only:
            crop_command = base_command + ["--crop-smoke"]
            crop_measured = run_instrumented_method(crop_command, ROOT, env, args.crop_time_limit_s,
                                                    args.rss_limit_bytes, [], telemetry_dir / "crop_only", sample_interval_s=1.0)
            attempts.append({"regime": "DETERMINISTIC_1024_CROP_SMOKE", "instrumentation": crop_measured})
            measured = crop_measured
            run_suffix = "crop_smoke"
        else:
            measured = run_instrumented_method(base_command, ROOT, env, args.time_limit_s,
                                               args.rss_limit_bytes, [], telemetry_dir / "full_pair", sample_interval_s=1.0)
            attempts.append({"regime": "FULL_PAIR_SMOKE", "instrumentation": measured})
            run_suffix = "full_pair"
            if measured["status"] in ("RESOURCE_LIMIT", "TIMEOUT"):
                crop_command = base_command + ["--crop-smoke"]
                crop_measured = run_instrumented_method(crop_command, ROOT, env, args.crop_time_limit_s,
                                                        args.rss_limit_bytes, [], telemetry_dir / "crop_smoke", sample_interval_s=1.0)
                attempts.append({"regime": "DETERMINISTIC_1024_CROP_SMOKE", "instrumentation": crop_measured})
                measured = crop_measured
                run_suffix = "crop_smoke"
        result_dir = worker_out / pair_id / run_suffix
        result_file = result_dir / "smoke_result.json"
        failure_file = result_dir / "worker_failure.json"
        pair_result = {"pair_id": pair_id, "attempts": attempts, "instrumentation": measured,
                       "upstream_version": UPSTREAM_VERSION, "upstream_commit": UPSTREAM_COMMIT}
        if result_file.is_file() and measured["status"] == "COMPLETED":
            smoke = json.loads(result_file.read_text(encoding="utf-8"))
            pair_meta = audit["requested_pairs"][pair_id]
            corrected = smoke["stage_outputs"]["scene_outputs"]
            submission = build_pair_submission(
                pair_id=pair_id, scene_i=pair_meta["scene_i"], scene_j=pair_meta["scene_j"],
                frozen=freeze_manifest,
                corrected_scene_paths={pair_meta["scene_i"]: corrected[0], pair_meta["scene_j"]: corrected[1]},
                output_mosaic_path=smoke["stage_outputs"]["mosaic"],
                output_mosaic_mask_path=str(result_dir / "output_masks/mosaic_mask.tif"),
                run_id=f"spectralmatch_phase2_{pair_id}_20261013",
                outputs={"smoke_result": str(result_file), "diagnostic_only": True,
                         "runtime": measured, "environment_prefix": args.environment_prefix},
            )
            submission_path = worker_out / pair_id / "method_submission.json"
            _write_json(submission_path, submission)
            schema_audit = _schema_check(submission, schema_path)
            checker = subprocess.run(
                [sys.executable, str(ROOT / "scripts/benchmark/check_unified_submission.py"),
                 "--freeze", str(freeze_root), "--method-submission", str(submission_path)],
                cwd=ROOT, capture_output=True, text=True, check=False,
            )
            checker_result = None
            try:
                checker_result = json.loads(checker.stdout.strip().splitlines()[-1])
            except (json.JSONDecodeError, IndexError):
                checker_result = {"status": "CHECKER_ERROR", "stdout": checker.stdout, "stderr": checker.stderr}
            pair_result.update({"status": smoke["status"], "smoke_result": str(result_file),
                                "submission_path": str(submission_path), "schema_audit": schema_audit,
                                "actual_frozen_checker": checker_result, "checker_return_code": checker.returncode,
                                "checker_interpretation": "STRUCTURAL_SCHEMA_AND_FREEZE_VALIDATION_ONLY; does not make a crop smoke eligible for formal ranking",
                                "benchmark_eligibility": "DIAGNOSTIC_ONLY_NOT_FORMAL_RANKING",
                                "smoke": smoke})
        elif failure_file.is_file():
            pair_result.update(json.loads(failure_file.read_text(encoding="utf-8")))
        else:
            pair_result["status"] = measured["status"]
            pair_result["failure_class"] = "RESOURCE_LIMIT_OR_TIMEOUT" if measured["status"] in ("RESOURCE_LIMIT", "TIMEOUT") else "WORKER_DID_NOT_COMPLETE"
        _write_json(pair_dir / "pair_run_status.json", pair_result)
        results.append(pair_result)

    final = {"status": "COMPLETE_DIAGNOSTIC_SMOKES" if all(x.get("status") == "COMPLETED_DIAGNOSTIC_ONLY" for x in results)
             else "INCOMPLETE_DIAGNOSTIC_SMOKES", "protocol_id": freeze_manifest["protocol_id"],
             "freeze": str(freeze_root), "freeze_audit": str(output_root / "freeze_audit.json"),
             "upstream_version": UPSTREAM_VERSION, "upstream_commit": UPSTREAM_COMMIT,
             "resource_limits": {"rss_limit_bytes": args.rss_limit_bytes, "time_limit_s": args.time_limit_s},
             "formal_13_scene_run": "NOT_AUTHORIZED_AND_NOT_RUN", "pairs": results}
    _write_json(output_root / "phase2_run_status.json", final)
    return final


def main():
    parser = argparse.ArgumentParser(description="Run two isolated Spectralmatch compatibility smokes.")
    parser.add_argument("--freeze", default="data/output/benchmark_v1_unified/freeze_20261013_v3")
    parser.add_argument("--output-root", default="data/output/benchmark_v1_unified/spectralmatch_phase2_20261013")
    parser.add_argument("--environment-prefix", required=True)
    parser.add_argument("--rss-limit-bytes", type=int, default=2_147_483_648)
    parser.add_argument("--time-limit-s", type=int, default=7200)
    parser.add_argument("--crop-time-limit-s", type=int, default=900)
    parser.add_argument("--pairs", nargs="+", default=["00_01", "00_10"])
    parser.add_argument("--crop-only", action="store_true", help="skip already-attempted full runs; run the fixed deterministic crop only")
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({"status": report["status"], "report": str(Path(args.output_root).resolve() / "phase2_run_status.json")}, ensure_ascii=False))
    return 0 if report["status"] == "COMPLETE_DIAGNOSTIC_SMOKES" else 2


if __name__ == "__main__":
    raise SystemExit(main())
