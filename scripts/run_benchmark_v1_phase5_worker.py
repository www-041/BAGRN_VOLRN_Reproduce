"""Truth-free one-pair matcher subprocess for the Phase5 parent controller."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import traceback
import warnings

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_BASE = ROOT / "data/output/benchmark_v1_phase5"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark_v1_phase5.adapters import run_local_matcher
from benchmark_v1_phase5.coordinates import NativeMatches, UnsupportedGridError
from benchmark_v1_phase5.external import run_external
from benchmark_v1_phase5.inputs import prepare_pair
from benchmark_v1_phase5.protocol import PairManifest, load_config
from benchmark_v1_phase5.translation import estimate_translation

REQUEST_KEYS = {"mode", "pair_id", "method", "config_path", "config_sha256",
                "output_dir", "pair_manifest", "truth_keys"}
MANIFEST_KEYS = {"pair_id", "scene_i", "scene_j", "path_i", "path_j",
                 "size_i", "size_j", "sha256_i", "sha256_j"}


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def _validate_output_target(request: dict) -> Path:
    output = Path(request["output_dir"]).expanduser().resolve()
    allowed = Path(OUTPUT_BASE).resolve()
    if not output.is_relative_to(allowed) or output == allowed:
        raise ValueError("Worker output must be a child of the Phase5 output directory")
    if not output.is_dir():
        raise ValueError("Worker output directory must already exist")
    result_path = output / "method_result.json"
    if result_path.exists():
        raise FileExistsError("Worker refuses to overwrite method_result.json")
    for required in ("protocol_lock.json", "worker_request.json"):
        if not (output / required).is_file():
            raise ValueError(f"Worker output is missing parent-runner file: {required}")
    return output


def _verify_frozen_b9_manifest(request: dict) -> None:
    """Verify B9 identity from Phase3 audit only; do not read candidate truth."""
    from benchmark_phase4_freeze import FROZEN_PHASE3_AUDIT_SHA256, PAIRS, sha256_file

    manifest = request["pair_manifest"]
    pair_id = request["pair_id"]
    if pair_id not in PAIRS or manifest.get("pair_id") != pair_id:
        raise ValueError("Worker pair identity does not match frozen request")
    audit_path = ROOT / "docs/benchmark/v1/phase3/split_leakage_audit.json"
    if sha256_file(audit_path) != FROZEN_PHASE3_AUDIT_SHA256:
        raise ValueError("Frozen Phase3 audit SHA256 mismatch")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("freeze_scope") != "TRIAL4_ONLY" or audit.get("leakage_pass") is not True:
        raise ValueError("Phase3 audit is not a passing four-pair freeze")
    i, j = map(int, pair_id.split("_"))
    by_index = {int(row["scene_index"]): row for row in audit["b9_verified_inputs"]}
    if i not in by_index or j not in by_index:
        raise ValueError("Pair scenes are absent from frozen B9 inventory")
    expected = (by_index[i], by_index[j])
    for side, record in zip(("i", "j"), expected):
        if (manifest[f"scene_{side}"] != record["scene_id"]
                or Path(manifest[f"path_{side}"]).resolve() != Path(record["path"]).resolve()
                or int(manifest[f"size_{side}"]) != int(record["size_bytes"])
                or manifest[f"sha256_{side}"] != record["sha256"]):
            raise ValueError(f"Worker B9 scene_{side} manifest is not frozen")
        path = Path(record["path"])
        if not path.is_file() or path.stat().st_size != int(record["size_bytes"]):
            raise ValueError(f"Frozen B9 scene_{side} is missing or has changed size")
        if sha256_file(path) != record["sha256"]:
            raise ValueError(f"Frozen B9 scene_{side} SHA256 mismatch")


def _synthetic_matches(method: str) -> NativeMatches:
    """Known-shift fixture only; never dispatches a real Matcher."""
    rows, cols = np.mgrid[80:1521:6j, 80:1521:8j]
    xy = np.column_stack((cols.ravel(), rows.ravel()))
    i_rc = np.column_stack((rows.ravel(), cols.ravel())).astype(float)
    j_rc = i_rc + np.array([2.25, -3.5])
    j_rc[:4] += np.array([30.0, -25.0])
    return NativeMatches(i_rc, j_rc, xy, xy.copy(), np.ones(len(xy)),
                         method, (1600, 1600))


def _gpu_stats(method: str, *, finish: bool = False, torch_module=None, state=None):
    deep = {"LoFTR", "EfficientLoFTR", "LightGlue+DISK"}
    if method not in deep:
        return None, {"status": "NOT_APPLICABLE", "reason": "non-deep adapter"}
    if not finish:
        try:
            import torch
            if not torch.cuda.is_available():
                return torch, {"status": "UNKNOWN", "reason": "CUDA unavailable"}
            torch.cuda.reset_peak_memory_stats()
            return torch, {"status": "MEASURING"}
        except Exception as exc:
            return None, {"status": "UNKNOWN", "reason": f"{type(exc).__name__}: {exc}"}
    if not state or state.get("status") != "MEASURING" or torch_module is None:
        return torch_module, state or {"status": "UNKNOWN", "reason": "measurement unavailable"}
    try:
        return torch_module, {"status": "MEASURED",
                              "peak_allocated_bytes": int(torch_module.cuda.max_memory_allocated())}
    except Exception as exc:
        return torch_module, {"status": "UNKNOWN", "reason": f"{type(exc).__name__}: {exc}"}


def _result(request: dict, status: str, delta=None, elapsed=0.0,
            reason=None, provenance=None) -> dict:
    return {"method": request["method"], "pair_id": request["pair_id"],
            "status": status, "delta_row_col": delta,
            "direction": "scene_i_to_scene_j", "frame": "native",
            "translation_semantics": "full_native_i_to_j",
            "elapsed_s": float(elapsed), "reason": reason,
            "provenance": provenance or {}}


def _execute_real(request: dict, config) -> dict:
    raw = request.get("pair_manifest")
    if not isinstance(raw, dict) or set(raw) != MANIFEST_KEYS:
        raise ValueError("Real mode requires B9-only pair manifest")
    manifest = PairManifest(raw["pair_id"], raw["scene_i"], raw["scene_j"],
                            Path(raw["path_i"]), Path(raw["path_j"]),
                            int(raw["size_i"]), int(raw["size_j"]),
                            str(raw["sha256_i"]), str(raw["sha256_j"]))
    started = time.perf_counter()
    try:
        pair = prepare_pair(manifest, config)
    except UnsupportedGridError as exc:
        return _result(request, "UNSUPPORTED_GRID", elapsed=time.perf_counter() - started,
                       reason=str(exc), provenance={"matcher_invoked": False})
    except ValueError as exc:
        status = "UNSUPPORTED_GRID" if "Unsupported CRS/grid" in str(exc) else "PREPROCESSING_FAIL"
        return _result(request, status, elapsed=time.perf_counter() - started,
                       reason=str(exc), provenance={"matcher_invoked": False})
    except Exception as exc:
        return _result(request, "PREPROCESSING_FAIL", elapsed=time.perf_counter() - started,
                       reason=f"{type(exc).__name__}: {exc}",
                       provenance={"matcher_invoked": False,
                                   "traceback": traceback.format_exc()[-3000:]})
    preprocessing_s = time.perf_counter() - started
    if pair.status != "READY":
        return _result(request, pair.status, elapsed=preprocessing_s, reason=pair.reason,
                       provenance={"native_joint_valid_count": pair.native_joint_valid_count,
                                   "view_joint_valid_count": pair.view_joint_valid_count,
                                   "matcher_invoked": False})

    method = request["method"]
    matcher_started = time.perf_counter()
    torch_module, gpu_state = _gpu_stats(method)
    if method in {"KARIOS", "AROSICS_GLOBAL"}:
        external_output = Path(request["output_dir"]) / "external_worker"
        external = run_external(method, pair, config,
                                dict(config.external_python).get(method), external_output)
        matcher_s = time.perf_counter() - matcher_started
        if external.status != "SUCCESS":
            return _result(request, external.status, elapsed=preprocessing_s + matcher_s,
                           reason=external.reason,
                           provenance={**(external.provenance or {}),
                                       "preprocessing_s": preprocessing_s,
                                       "matcher_s": matcher_s, "gpu_memory": gpu_state})
        if method == "AROSICS_GLOBAL":
            direct = external.direct_translation
            return _result(request, "SUCCESS", list(direct.delta_row_col),
                           preprocessing_s + matcher_s,
                           provenance={**(external.provenance or {}),
                                       "solver": "native AROSICS global COREG",
                                       "raw_shift_xy": list(direct.raw_shift_xy),
                                       "shift_convention": direct.shift_convention,
                                       "sign_verified": direct.sign_verified,
                                       "preprocessing_s": preprocessing_s,
                                       "matcher_s": matcher_s, "gpu_memory": gpu_state})
        matches = external.matches
        adapter_provenance = external.provenance or {}
        adapter_timings = {"external_wall_s": matcher_s}
    else:
        adapter = run_local_matcher(method, pair, config)
        matcher_s = time.perf_counter() - matcher_started
        if adapter.status != "SUCCESS" or adapter.matches is None:
            return _result(request, adapter.status, elapsed=preprocessing_s + matcher_s,
                           reason=adapter.reason,
                           provenance={**adapter.provenance,
                                       "preprocessing_s": preprocessing_s,
                                       "matcher_s": matcher_s, "gpu_memory": gpu_state})
        matches = adapter.matches
        adapter_provenance = adapter.provenance
        adapter_timings = {"matcher_runtime_s": adapter.matcher_runtime_sec,
                           "feature_runtime_s": adapter.feature_runtime_sec,
                           "model_init_runtime_s": adapter.model_init_runtime_sec}

    fit_started = time.perf_counter()
    fit = estimate_translation(matches, config)
    fit_s = time.perf_counter() - fit_started
    _, gpu_state = _gpu_stats(method, finish=True, torch_module=torch_module, state=gpu_state)
    provenance = {
        "preprocessing_s": preprocessing_s, "matcher_s": matcher_s, "fit_s": fit_s,
        "match_count": fit.match_count, "inlier_count": fit.inlier_count,
        "inlier_fraction": fit.inlier_fraction,
        "coverage_cells": fit.coverage_cells, "coverage_rows": fit.coverage_rows,
        "coverage_columns": fit.coverage_columns, "ransac_seed": fit.seed,
        "hypotheses_tested": fit.hypotheses_tested,
        "median_residual_px": fit.median_residual_px,
        "adapter_provenance": adapter_provenance,
        "adapter_timings": adapter_timings, "gpu_memory": gpu_state,
    }
    return _result(request, fit.status,
                   list(fit.delta_row_col) if fit.delta_row_col is not None else None,
                   preprocessing_s + matcher_s + fit_s, fit.failure_code, provenance)


def execute_request(request: dict) -> dict:
    if not isinstance(request, dict) or set(request) != REQUEST_KEYS:
        raise ValueError("Unexpected worker request schema")
    if request["truth_keys"] != []:
        raise ValueError("Checkpoint/truth data is forbidden in matcher worker")
    config = load_config(Path(request["config_path"]))
    if config.config_sha256 != request["config_sha256"]:
        raise ValueError("Worker config SHA256 differs from locked config")
    if request["pair_id"] not in config.pair_ids or request["method"] not in config.method_ids:
        raise ValueError("Worker request outside frozen protocol")
    if request["mode"] not in {"synthetic", "smoke", "development"}:
        raise ValueError("Unknown worker mode")
    output = _validate_output_target(request)
    lock = json.loads((output / "protocol_lock.json").read_text(encoding="utf-8"))
    if lock.get("config_sha256") != config.config_sha256:
        raise ValueError("Worker config does not match persisted protocol lock")
    if lock.get("run_selection") != {"pair_id": request["pair_id"],
                                     "method": request["method"],
                                     "mode": request["mode"]}:
        raise ValueError("Worker request does not match persisted run selection")
    if request["mode"] != "synthetic":
        _verify_frozen_b9_manifest(request)
    elif request["pair_manifest"] is not None:
        raise ValueError("Synthetic worker must not receive a B9 manifest")
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        if request["mode"] == "synthetic":
            started = time.perf_counter()
            fit = estimate_translation(_synthetic_matches(request["method"]), config)
            result = _result(request, fit.status,
                             list(fit.delta_row_col) if fit.delta_row_col is not None else None,
                             time.perf_counter() - started, fit.failure_code,
                             {"synthetic_only": True, "match_count": fit.match_count,
                              "inlier_count": fit.inlier_count,
                              "coverage_cells": fit.coverage_cells,
                              "ransac_seed": fit.seed,
                              "hypotheses_tested": fit.hypotheses_tested})
        else:
            result = _execute_real(request, config)
    result["provenance"]["warnings"] = [str(item.message) for item in captured]
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True, type=Path)
    args = parser.parse_args(argv)
    request = json.loads(args.request.read_text(encoding="utf-8"))
    try:
        result = execute_request(request)
        output = _validate_output_target(request) / "method_result.json"
    except Exception as exc:
        try:
            output = _validate_output_target(request) / "method_result.json"
        except Exception:
            print(f"Refusing unsafe Phase5 worker request: {type(exc).__name__}: {exc}",
                  file=sys.stderr)
            return 2
        result = {"method": request.get("method", "UNKNOWN"),
                  "pair_id": request.get("pair_id", "UNKNOWN"),
                  "status": "ALGORITHM_FAIL", "delta_row_col": None,
                  "direction": "scene_i_to_scene_j", "frame": "native",
                  "translation_semantics": "full_native_i_to_j", "elapsed_s": 0.0,
                  "reason": f"{type(exc).__name__}: {exc}",
                  "provenance": {"traceback": traceback.format_exc()[-3000:]}}
    _atomic_json(output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
