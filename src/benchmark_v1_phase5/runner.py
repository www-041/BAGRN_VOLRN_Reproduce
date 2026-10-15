"""Single-pair Phase5 orchestration with isolated worker and frozen provenance."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any

from .evaluation import (load_approved_development_truth, score_development,
                         score_georef_only)
from .inputs import prepare_pair
from .protocol import (PairManifest, Phase5Config, build_pair_manifest,
                      load_config, verify_manifest_bytes)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_BASE = PROJECT_ROOT / "data/output/benchmark_v1_phase5"
METHODS = ("SIFT", "KARIOS", "AROSICS_GLOBAL", "LoFTR",
           "EfficientLoFTR", "LightGlue+DISK")
MODES = ("synthetic", "smoke", "development")
RESULT_STATUSES = {"SUCCESS", "ENV_UNAVAILABLE", "WEIGHTS_UNAVAILABLE",
                   "NO_VALID_OVERLAP", "UNSUPPORTED_GRID", "PREPROCESSING_FAIL",
                   "ALGORITHM_FAIL", "NOT_ATTEMPTED"}


def _json_default(value: Any):
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _write_json_atomic(path: Path, value: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                         suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, indent=2, sort_keys=True, default=_json_default)
            stream.write("\n")
        os.replace(temporary, path)
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def _load_snapshot_manifest(pair_id: str):
    scripts = PROJECT_ROOT / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    from benchmark_phase4_freeze import load_frozen_trial, verify_b9_identity
    snapshot = load_frozen_trial(PROJECT_ROOT)
    verify_b9_identity(snapshot)
    return snapshot, build_pair_manifest(snapshot, pair_id)


def _manifest_dict(manifest: PairManifest) -> dict:
    return {"pair_id": manifest.pair_id, "scene_i": manifest.scene_i,
            "scene_j": manifest.scene_j, "path_i": str(manifest.path_i),
            "path_j": str(manifest.path_j), "size_i": manifest.size_i,
            "size_j": manifest.size_j, "sha256_i": manifest.sha256_i,
            "sha256_j": manifest.sha256_j}


def _validate_worker_result(result: dict, method: str, pair_id: str) -> None:
    required = {"method", "pair_id", "status", "delta_row_col", "direction", "frame",
                "translation_semantics", "elapsed_s", "reason", "provenance"}
    if not isinstance(result, dict) or set(result) != required:
        raise ValueError("Worker result schema mismatch")
    if result["method"] != method or result["pair_id"] != pair_id:
        raise ValueError("Worker result method/pair identity mismatch")
    if result["status"] not in RESULT_STATUSES:
        raise ValueError("Worker result has unknown execution status")
    try:
        elapsed = float(result["elapsed_s"])
    except (TypeError, ValueError) as exc:
        raise ValueError("Worker elapsed time is invalid") from exc
    if not (elapsed >= 0 and math.isfinite(elapsed)):
        raise ValueError("Worker elapsed time is nonfinite or negative")
    if not isinstance(result["provenance"], dict):
        raise ValueError("Worker provenance must be an object")
    if result["status"] == "SUCCESS":
        if (result["direction"] != "scene_i_to_scene_j" or result["frame"] != "native"
                or result["translation_semantics"] != "full_native_i_to_j"):
            raise ValueError("Worker success result is not normalized to native i-to-j")
        delta = result["delta_row_col"]
        if not isinstance(delta, (list, tuple)) or len(delta) != 2:
            raise ValueError("Worker success result requires a two-vector")
        try:
            finite_delta = all(math.isfinite(float(value)) for value in delta)
        except (TypeError, ValueError):
            finite_delta = False
        if not finite_delta:
            raise ValueError("Worker translation must be finite")
    elif result["delta_row_col"] is not None:
        raise ValueError("Non-success worker result cannot carry a translation")


def _config_lock(config: Phase5Config, config_path: Path,
                 pair_id: str, method: str, mode: str) -> dict:
    return {"protocol": "DZ01V_BENCHMARK_V1_PHASE5",
            "config_path": str(Path(config_path).resolve()),
            "config_sha256": config.config_sha256,
            "run_selection": {"pair_id": pair_id, "method": method, "mode": mode},
            "frozen_values": {
                "pairs": list(config.pair_ids), "methods": list(config.method_ids),
                "crs": config.crs, "native_pixel_m": config.native_pixel_m,
                "stretch_percentiles": list(config.stretch_percentiles),
                "max_view_side": config.max_view_side,
                "translation": {"seed": config.seed,
                                "hypotheses": config.ransac_hypotheses,
                                "inlier_radius_native_px": config.ransac_radius_native_px,
                                "min_matches": config.min_matches,
                                "min_inliers": config.min_inliers,
                                "min_inlier_fraction": config.min_inlier_fraction,
                                "coverage_grid": list(config.coverage_grid),
                                "min_coverage_cells": config.min_coverage_cells,
                                "min_coverage_rows": config.min_coverage_rows,
                                "min_coverage_columns": config.min_coverage_columns},
                "external_python": dict(config.external_python),
                "method_settings": {name: dict(settings)
                                    for name, settings in config.method_settings},
                "resource_budget": {"wall_seconds": config.smoke_seconds,
                                    "rss_gib": config.smoke_rss_gib}},
            "test_points_in_lock": False}


def _process_rss_bytes(pid: int) -> int | None:
    import psutil
    try:
        root = psutil.Process(pid)
        processes = [root, *root.children(recursive=True)]
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return None
    total = 0
    for process in processes:
        try:
            total += process.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return None
    return total


def _stop_process(process) -> None:
    psutil_module = None
    try:
        import psutil as psutil_module
        root = psutil_module.Process(process.pid)
        children = root.children(recursive=True)
        for child in children:
            try:
                child.terminate()
            except (psutil_module.NoSuchProcess, psutil_module.AccessDenied):
                pass
    except Exception:
        children = []
    try:
        process.terminate()
        process.wait(timeout=5)
    except Exception:
        try:
            process.kill()
            process.wait(timeout=5)
        except Exception:
            pass
    if children and psutil_module is not None:
        try:
            _, alive = psutil_module.wait_procs(children, timeout=2)
            for child in alive:
                try:
                    child.kill()
                except (psutil_module.NoSuchProcess, psutil_module.AccessDenied):
                    pass
        except Exception:
            pass


def _run_process_with_limits(command: list[str], output_dir: Path,
                             timeout_s: float, rss_ceiling_bytes: int) -> dict:
    stdout_path, stderr_path = output_dir / "worker_stdout.log", output_dir / "worker_stderr.log"
    started = time.monotonic()
    peak_rss = 0
    rss_sampled = False
    reason = None
    monitor_unavailable = False
    try:
        import psutil  # noqa: F401
    except ImportError:
        return {"status": "ENV_UNAVAILABLE", "reason": "RESOURCE_MONITOR_PSUTIL_MISSING",
                "wall_seconds": 0.0, "peak_rss_bytes": 0}
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        try:
            process = subprocess.Popen(command, cwd=PROJECT_ROOT,
                                       stdout=stdout, stderr=stderr)
        except OSError as exc:
            return {"status": "ENV_UNAVAILABLE", "reason": f"WORKER_START: {exc}",
                    "wall_seconds": time.monotonic() - started, "peak_rss_bytes": 0}
        while process.poll() is None:
            elapsed = time.monotonic() - started
            if elapsed >= timeout_s:
                reason = "RESOURCE_LIMIT: wall timeout"
                _stop_process(process)
                break
            current_rss = _process_rss_bytes(process.pid)
            if current_rss is None:
                monitor_unavailable = True
                reason = "RESOURCE_MONITOR_UNAVAILABLE: RSS sample could not be read"
                _stop_process(process)
                break
            rss_sampled = True
            peak_rss = max(peak_rss, current_rss)
            if current_rss > rss_ceiling_bytes:
                reason = "RESOURCE_LIMIT: host RSS ceiling exceeded"
                _stop_process(process)
                break
            time.sleep(min(0.1, max(0.01, timeout_s - elapsed)))
        returncode = process.poll()
        if returncode is None:
            try:
                returncode = process.wait(timeout=1)
            except Exception:
                returncode = -1
    if reason:
        status = "ENV_UNAVAILABLE" if monitor_unavailable else "ALGORITHM_FAIL"
    elif returncode == 0 and not rss_sampled:
        status = "ENV_UNAVAILABLE"
        reason = "RESOURCE_MONITOR_UNAVAILABLE: worker exited before an RSS sample"
    elif returncode == 0:
        status = "SUCCESS"
    else:
        status = "ALGORITHM_FAIL"
        reason = f"WORKER_EXIT_{returncode}"
    return {"status": status, "reason": reason, "worker_exit_code": returncode,
            "wall_seconds": time.monotonic() - started,
            "peak_rss_bytes": peak_rss if rss_sampled else None,
            "rss_sample_count": int(rss_sampled),
            "rss_measurement": "parent psutil sum of worker and descendants; 100ms polling"
            if rss_sampled else "UNKNOWN: no successful live process sample"}


def _run_worker(command: list[str], output_dir: Path,
                timeout_s: float, rss_gib: float) -> dict:
    return _run_process_with_limits(command, output_dir, timeout_s,
                                    int(rss_gib * 1024**3))


def _output_path(path: Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    allowed = Path(OUTPUT_BASE).resolve()
    if not resolved.is_relative_to(allowed) or resolved == allowed:
        raise ValueError(f"Phase5 outputs must be a new child of {allowed}")
    if resolved.exists():
        raise FileExistsError(f"Refusing to overwrite Phase5 run: {resolved}")
    return resolved


def _validate_request(pair_id: str, method: str, mode: str) -> None:
    from .protocol import PAIRS
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    if pair_id not in PAIRS:
        raise ValueError("Pair must be one of the frozen four Phase5 pairs")
    if method not in METHODS:
        raise ValueError("Method must be one of the six frozen Matcher IDs")
    if mode == "smoke" and pair_id != "00_01":
        raise ValueError("Real Phase5 smoke is restricted to pair 00_01")


def _score_after_persist(result: dict, snapshot, manifest: PairManifest,
                         config: Phase5Config, output_dir: Path) -> dict:
    """Load truth only after method_result.json exists on disk."""
    if not (output_dir / "method_result.json").is_file():
        raise RuntimeError("Method result must be persisted before DEVELOPMENT truth loading")
    development = load_approved_development_truth(snapshot)[manifest.pair_id]
    scored = score_development(result, development)
    _write_json_atomic(output_dir / "development_metrics.json", scored)
    pair = prepare_pair(manifest, config)
    georef = score_georef_only(pair, development)
    _write_json_atomic(output_dir / "georef_only_metrics.json", georef)
    return scored


def run_one(pair_id: str, method: str, config_path: Path,
            output_root: Path, mode: str) -> dict:
    """Run one explicitly selected pair/method, never a batch."""
    _validate_request(pair_id, method, mode)
    config_path = Path(config_path).expanduser().resolve()
    config = load_config(config_path)
    out = _output_path(output_root)
    snapshot = manifest = None
    if mode != "synthetic":
        snapshot, manifest = _load_snapshot_manifest(pair_id)
        verify_manifest_bytes(manifest)
    out.mkdir(parents=True)
    _write_json_atomic(out / "protocol_lock.json",
                       _config_lock(config, config_path, pair_id, method, mode))
    if mode == "synthetic":
        input_provenance = {"source": "generated_synthetic_coordinates",
                            "pair_id": pair_id, "test_points_included": False}
    else:
        input_provenance = {"source": "Phase4 frozen B9 identity",
                            "pair_manifest": _manifest_dict(manifest),
                            "phase3_assignment_sha256": snapshot.assignment_sha256,
                            "phase3_audit_sha256": snapshot.audit["output_sha256"].get(
                                "split_leakage_audit.json"),
                            "test_points_included": False}
    _write_json_atomic(out / "input_provenance.json", input_provenance)
    request = {"mode": mode, "pair_id": pair_id, "method": method,
               "config_path": str(config_path), "config_sha256": config.config_sha256,
               "output_dir": str(out),
               "pair_manifest": _manifest_dict(manifest) if manifest else None,
               "truth_keys": []}
    expected_keys = {"mode", "pair_id", "method", "config_path", "config_sha256",
                     "output_dir", "pair_manifest", "truth_keys"}
    if set(request) != expected_keys:
        raise AssertionError("Unexpected worker descriptor schema")
    _write_json_atomic(out / "worker_request.json", request)
    worker = PROJECT_ROOT / "scripts/run_benchmark_v1_phase5_worker.py"
    command = [sys.executable, str(worker), "--request", str(out / "worker_request.json")]
    budget = _run_worker(command, out, config.smoke_seconds, config.smoke_rss_gib)
    result_path = out / "method_result.json"
    if not result_path.is_file():
        status = budget["status"] if budget["status"] != "SUCCESS" else "ALGORITHM_FAIL"
        result = {"method": method, "pair_id": pair_id, "status": status,
                  "delta_row_col": None, "direction": "scene_i_to_scene_j",
                  "frame": "native", "translation_semantics": "full_native_i_to_j",
                  "elapsed_s": budget["wall_seconds"],
                  "reason": budget.get("reason") or "WORKER_RESULT_MISSING",
                  "provenance": {"worker_invoked": True}}
        _write_json_atomic(result_path, result)
    else:
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
            _validate_worker_result(result, method, pair_id)
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
            _write_json_atomic(out / "worker_result_validation_error.json",
                               {"reason": f"{type(exc).__name__}: {exc}"})
            result = {"method": method, "pair_id": pair_id,
                      "status": "ALGORITHM_FAIL", "delta_row_col": None,
                      "direction": "scene_i_to_scene_j", "frame": "native",
                      "translation_semantics": "full_native_i_to_j",
                      "elapsed_s": budget["wall_seconds"],
                      "reason": "INVALID_WORKER_RESULT",
                      "provenance": {"worker_result_rejected": True}}
            _write_json_atomic(result_path, result)
    if budget["status"] != "SUCCESS" and result["status"] == "SUCCESS":
        _write_json_atomic(out / "worker_result_before_resource_gate.json", result)
        result = {**result, "status": budget["status"], "delta_row_col": None,
                  "reason": budget.get("reason") or "RESOURCE_GATE_FAILED",
                  "provenance": {**result["provenance"],
                                 "worker_result_rejected_by_resource_gate": True}}
        _write_json_atomic(result_path, result)
    _write_json_atomic(out / "resource_usage.json", budget)
    evaluation = None
    if mode in {"smoke", "development"} and snapshot is not None:
        evaluation = _score_after_persist(result, snapshot, manifest, config, out)
    run_status = {"mode": mode, "pair_id": pair_id, "method": method,
                  "execution_status": result["status"],
                  "evaluation_status": evaluation["evaluation_status"] if evaluation else "NOT_RUN",
                  "config_sha256": config.config_sha256, "worker_status": budget["status"]}
    _write_json_atomic(out / "run_status.json", run_status)
    return {**run_status, "result_path": str(result_path), "output_dir": str(out)}


def preflight(config_path: Path) -> dict:
    """Read-only config, frozen input hash and dependency check; no inference."""
    scripts = PROJECT_ROOT / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    from benchmark_phase4_development import environment_preflight

    config_path = Path(config_path).expanduser().resolve()
    config = load_config(config_path)
    snapshot, _ = _load_snapshot_manifest("00_01")
    dependencies = environment_preflight()
    external = {}
    for method, executable in config.external_python:
        external[method] = {"configured": bool(executable),
                            "path_exists": bool(executable and Path(executable).is_file()),
                            "end_to_end": "NOT_VERIFIED_END_TO_END"}
    return {"status": "PREFLIGHT_COMPLETE", "config_sha256": config.config_sha256,
            "frozen_b9_inputs_verified": len(snapshot.audit["b9_verified_inputs"]),
            "phase3_assignment_sha256": snapshot.assignment_sha256,
            "dependencies": dependencies, "external_environments": external,
            "model_inference_performed": False}
