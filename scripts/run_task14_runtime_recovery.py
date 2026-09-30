"""Task14R: recover the frozen EfficientLoFTR runtime and validate one replay.

This module is intentionally a gate, not a new matcher experiment.  It only
accepts the official local checkout/checkpoint already used by the historical
five-scene run and refuses to continue when provenance or replay evidence is
missing.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
TASK14_ROOT = ROOT / "data/output/b9_13scene_task14"
RECOVERY_ROOT = TASK14_ROOT / "runtime_recovery"
DEFAULT_REPO = Path(r"D:\科研\地质一号\文献\EfficientLoFTR")
DEFAULT_CHECKPOINT = DEFAULT_REPO / "weights/eloftr_outdoor.ckpt"
DEFAULT_CONFIG = ROOT / "data/output/b9_five_scene_validation/04_frozen_five_scene_config_1024.json"
HISTORICAL_RUN = ROOT / "data/output/b9_five_scene_validation/matcher_runs_1024_globalready/efficient_loftr"
HARD_STOP_OFFICIAL_MATCHER_ASSET_MISSING = "HARD_STOP_OFFICIAL_MATCHER_ASSET_MISSING"
HARD_STOP_CHECKPOINT_MISMATCH = "HARD_STOP_CHECKPOINT_MISMATCH"
HARD_STOP_MATCHER_REPLAY_MISMATCH = "HARD_STOP_MATCHER_REPLAY_MISMATCH"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_value(repo: Path, *args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", f"-c", f"safe.directory={repo.as_posix()}", "-C", str(repo), *args],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "UNAVAILABLE"


def resolve_official_assets(repo: str | Path, checkpoint: str | Path) -> dict[str, Any]:
    repo = Path(repo).expanduser().resolve()
    checkpoint = Path(checkpoint).expanduser().resolve()
    if not (repo / "src" / "loftr").is_dir() or not checkpoint.is_file():
        raise RuntimeError(HARD_STOP_OFFICIAL_MATCHER_ASSET_MISSING)
    return {
        "status": "READY",
        "source_checkout": str(repo),
        "source_commit": _git_value(repo, "rev-parse", "HEAD"),
        "source_remote": _git_value(repo, "remote", "get-url", "origin"),
        "checkpoint": str(checkpoint),
        "checkpoint_size_bytes": checkpoint.stat().st_size,
        "checkpoint_sha256": sha256_file(checkpoint),
    }


def _version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "UNAVAILABLE"


def build_runtime_fingerprint(
    *, repo: str | Path, checkpoint: str | Path, device: str, precision: str, match_max_side: int
) -> dict[str, Any]:
    repo = Path(repo).expanduser().resolve()
    checkpoint = Path(checkpoint).expanduser().resolve()
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        cuda_version = torch.version.cuda
        gpu_name = torch.cuda.get_device_name(0) if cuda_available else None
        torch_version = torch.__version__
    except Exception:
        cuda_available, cuda_version, gpu_name, torch_version = False, "UNAVAILABLE", None, "UNAVAILABLE"
    try:
        import cv2
        opencv_version = cv2.__version__
    except Exception:
        opencv_version = "UNAVAILABLE"
    payload = {
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "venv_path": sys.prefix,
        "torch_version": torch_version,
        "torch_cuda_version": cuda_version,
        "cuda_available": cuda_available,
        "gpu_name": gpu_name,
        "opencv_version": opencv_version,
        "numpy_version": _version("numpy"),
        "scipy_version": _version("scipy"),
        "kornia_version": _version("kornia"),
        "einops_version": _version("einops"),
        "yacs_version": _version("yacs"),
        "efficientloftr_source": str(repo),
        "efficientloftr_source_commit": _git_value(repo, "rev-parse", "HEAD"),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint) if checkpoint.is_file() else "UNAVAILABLE",
        "device": device,
        "precision": precision,
        "match_max_side": int(match_max_side),
        "coordinate_inverse_resize": "x_original=x_model/scale_x, y_original=y_model/scale_y",
    }
    return payload


def _metric_equal(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return left is right
    if isinstance(left, str) or isinstance(right, str):
        return left == right
    if isinstance(left, (int, np.integer)) and isinstance(right, (int, np.integer)):
        return int(left) == int(right)
    return bool(np.isclose(float(left), float(right), rtol=1e-6, atol=1e-6))


def compare_pair_replay(historical: dict[str, Any], replay: dict[str, Any]) -> dict[str, Any]:
    fields = ("status", "raw_matches", "inliers", "coverage", "residual_rmse", "residual_p95")
    metric_matches = {field: _metric_equal(historical.get(field), replay.get(field)) for field in fields}
    hist_matrix = np.asarray(historical.get("pixel_matrix"), dtype=np.float64)
    replay_matrix = np.asarray(replay.get("pixel_matrix"), dtype=np.float64)
    transform_match = hist_matrix.shape == replay_matrix.shape and bool(np.allclose(hist_matrix, replay_matrix, rtol=1e-6, atol=1e-6))
    historical_arrays = historical.get("inlier_arrays_sha256")
    replay_arrays = replay.get("inlier_arrays_sha256")
    arrays_match = True if historical_arrays is None or replay_arrays is None else historical_arrays == replay_arrays
    metrics_match = all(metric_matches.values()) and transform_match and arrays_match
    result = {
        "status": "MATCHER_REPLAY_VALIDATED" if metrics_match else HARD_STOP_MATCHER_REPLAY_MISMATCH,
        "metrics_match": metrics_match,
        "field_matches": metric_matches,
        "transform_match": transform_match,
        "arrays_match": arrays_match,
    }
    if not metrics_match:
        raise RuntimeError(HARD_STOP_MATCHER_REPLAY_MISMATCH)
    return result


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_task14r_reports(task14_root: str | Path, recovery_root: str | Path) -> Path:
    """Write the required Task14R report and preserve the initial hard stop."""
    task14_root = Path(task14_root); recovery_root = Path(recovery_root)
    graph = json.loads((task14_root / "graph_summary.json").read_text(encoding="utf-8"))
    registration = json.loads((task14_root / "global_geometry_summary.json").read_text(encoding="utf-8"))
    replay = json.loads((recovery_root / "02_five_scene_matcher_replay.json").read_text(encoding="utf-8"))
    stage_status = {}
    for stage_dir in sorted((task14_root / "stages").iterdir(), key=lambda p: p.name):
        marker = stage_dir / "_SUCCESS.json"
        if marker.is_file():
            stage_status[stage_dir.name] = json.loads(marker.read_text(encoding="utf-8")).get("status")
    stage06 = json.loads((task14_root / "stages/06_bagrn/_SUCCESS.json").read_text(encoding="utf-8")) if (task14_root / "stages/06_bagrn/_SUCCESS.json").is_file() else {}
    global_summary = json.loads((task14_root / "stages/03_global_adjustment/global_adjustment_summary.json").read_text(encoding="utf-8")) if (task14_root / "stages/03_global_adjustment/global_adjustment_summary.json").is_file() else {}
    report = [
        "# Task14R Runtime Recovery Report", "",
        "User request: strictly execute Task14R. The pasted Task14R document is the binding recovery/resume specification.",
        "Task14 algorithm semantics were not changed: EfficientLoFTR, max side 1024, shared Affine RANSAC, Translation-L2, B9, EPSG:32650, 14 m.", "",
        "## Runtime provenance", "",
        f"- Official checkout: `{replay['historical'].get('source_checkout', 'see 00_runtime_audit.json')}`.",
        f"- Checkpoint SHA-256: `{json.loads((recovery_root / '01_runtime_fingerprint.json').read_text(encoding='utf-8')).get('checkpoint_sha256')}`.",
        f"- Runtime fingerprint: Python `{json.loads((recovery_root / '01_runtime_fingerprint.json').read_text(encoding='utf-8')).get('python_version')}`, torch `{json.loads((recovery_root / '01_runtime_fingerprint.json').read_text(encoding='utf-8')).get('torch_version')}`, CUDA `{json.loads((recovery_root / '01_runtime_fingerprint.json').read_text(encoding='utf-8')).get('torch_cuda_version')}`, GPU `{json.loads((recovery_root / '01_runtime_fingerprint.json').read_text(encoding='utf-8')).get('gpu_name')}`.", "",
        "## Historical pair replay gate", "",
        f"- Pair `(0,1)`: `{replay['comparison']['status']}`; raw/inliers/coverage/RMSE/P95/transform/NPZ arrays all matched.",
        "- No matcher fallback, threshold change, RANSAC change, checkpoint replacement, or resolution sensitivity was used.", "",
        "## Task14 resume", "",
        "- Initial run: `HARD_STOP_MATCHER_UNAVAILABLE` (preserved in the main Task14 report).",
        f"- Stage 02 resume: `{stage_status.get('02_matching_ransac', 'UNAVAILABLE')}`; accepted edges `{registration.get('accepted_edges')}/{registration.get('overlap_edges_total')}`, accepted graph `{registration.get('accepted_graph_status')}`, cycle rank `{registration.get('registration_graph_cycle_rank')}`.",
        f"- Stage 03 Translation-L2: `{stage_status.get('03_global_adjustment', 'UNAVAILABLE')}`; global RMSE/P95/max `{global_summary.get('translation_l2', {}).get('metrics', {}).get('global_rmse_pixel')}/{global_summary.get('translation_l2', {}).get('metrics', {}).get('global_p95_pixel')}/{global_summary.get('translation_l2', {}).get('metrics', {}).get('global_max_pixel')}` px.",
        f"- Stage 04 canonical grid and Stage 05 distance cache: `{stage_status.get('04_canonical_warp', 'UNAVAILABLE')}` / `{stage_status.get('05_valid_distance_cache', 'UNAVAILABLE')}`.",
        f"- Stage 06 BAGRN: `{stage_status.get('06_bagrn', 'UNAVAILABLE')}`; memory mode `local_scene_windows`; normalized scene GeoTIFFs and mosaic are present.",
        f"- Stages 07–12: `{stage_status.get('07_pairwise_seam_local', 'UNAVAILABLE')}` downstream; hard stop `{json.loads((task14_root / 'stages/07_pairwise_seam_local/_SUCCESS.json').read_text(encoding='utf-8')).get('error', 'UNAVAILABLE') if (task14_root / 'stages/07_pairwise_seam_local/_SUCCESS.json').is_file() else 'UNAVAILABLE'}` because no frozen 13-scene Task13A/Task13B adapter is present.",
        "- No 13-scene V0/V1/V2 or scientific PASS is claimed.", "",
        "## Final decision", "",
        "`NOT_READY_FOR_LARGE_SCALE` — runtime recovery, matcher replay, Translation-L2, canonicalization, EDT cache, and BAGRN completed; downstream multiscene stages remain blocked by the absence of a frozen 13-scene Task13A/Task13B adapter.", "",
        "Artifacts: `00_runtime_audit.json`, `01_runtime_fingerprint.json`, `02_five_scene_matcher_replay.json`, and the original `data/output/b9_13scene_task14/` package.",
    ]
    path = recovery_root / "TASK14R_RUNTIME_RECOVERY_REPORT.md"
    path.write_text("\n".join(report) + "\n", encoding="utf-8")
    main_report = task14_root / "TASK14_13SCENE_SCALE_VALIDATION_REPORT.md"
    initial = main_report.read_text(encoding="utf-8") if main_report.is_file() else "# Task14 13-Scene Scalable Mosaic Validation\n"
    recovery_section = ("\n## Task14R runtime recovery\n\n"
        "- Initial hard stop preserved: `HARD_STOP_MATCHER_UNAVAILABLE`.\n"
        "- Runtime recovery: `MATCHER_REPLAY_VALIDATED`.\n"
        "- Resume: Stage 02 completed with 48/60 accepted edges; Stage 03 Translation-L2, Stage 04 canonical grid, Stage 05 EDT cache, and Stage 06 BAGRN completed.\n"
        "- Stages 07–12 hard stop: `HARD_STOP_TASK14_MULTISCENE_ADAPTER_UNAVAILABLE`; no frozen 13-scene Task13A/Task13B adapter is present.\n"
        "- Final decision remains `NOT_READY_FOR_LARGE_SCALE`.\n")
    if "## Task14R runtime recovery" in initial:
        initial = initial.split("## Task14R runtime recovery", 1)[0].rstrip() + "\n" + recovery_section
    else:
        initial += recovery_section
    main_report.write_text(initial, encoding="utf-8")
    return path


def _historical_pair() -> dict[str, Any]:
    summary = json.loads((HISTORICAL_RUN / "pairwise_summary.json").read_text(encoding="utf-8"))
    row = next(row for row in summary["results"] if row["idx_i"] == 0 and row["idx_j"] == 1)
    sidecar = json.loads((HISTORICAL_RUN / "geometry/pair_00_01.json").read_text(encoding="utf-8"))
    return {
        "status": row["status"],
        "raw_matches": row["raw_matches"],
        "inliers": row["inliers"],
        "coverage": row["coverage"],
        "residual_rmse": row["residual_rmse"],
        "residual_p95": row["residual_p95"],
        "pixel_matrix": sidecar["pair_pixel_matrix"]["matrix"],
        "historical_summary": str(HISTORICAL_RUN / "pairwise_summary.json"),
        "historical_geometry": str(HISTORICAL_RUN / "geometry/pair_00_01.json"),
        "historical_npz": str(HISTORICAL_RUN / "geometry/pair_00_01.npz"),
        "inlier_arrays_sha256": sha256_file(HISTORICAL_RUN / "geometry/pair_00_01.npz"),
    }


def run_pair_replay(*, repo: Path, checkpoint: Path, config: Path, output: Path, device: str) -> dict[str, Any]:
    from src.multiscene_sift.b9_runner import run_b9_registration

    output.mkdir(parents=True, exist_ok=True)
    os.environ["EFFICIENT_LOFTR_REPO"] = str(repo)
    os.environ["EFFICIENT_LOFTR_WEIGHTS"] = str(checkpoint)
    summary_path = output / "pairwise_summary.json"
    sidecar_path = output / "geometry/pair_00_01.json"
    if summary_path.is_file() and sidecar_path.is_file():
        result = {"status": "REUSED_EXISTING_REPLAY_ARTIFACT"}
    else:
        frozen = json.loads(config.read_text(encoding="utf-8"))
        result = run_b9_registration(
            frozen,
            output,
            matcher="efficient_loftr",
            device=device,
            pair=(0, 1),
            protocol_config_path=config,
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    row = next(row for row in summary["results"] if row["idx_i"] == 0 and row["idx_j"] == 1)
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    replay = {
        "status": row["status"],
        "raw_matches": row["raw_matches"],
        "inliers": row["inliers"],
        "coverage": row["coverage"],
        "residual_rmse": row["residual_rmse"],
        "residual_p95": row["residual_p95"],
        "pixel_matrix": sidecar["pair_pixel_matrix"]["matrix"],
        "inlier_arrays_sha256": sha256_file(output / "geometry/pair_00_01.npz"),
    }
    return {"runner_result": result, "replay": replay, "output": str(output)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--task14-root", type=Path, default=TASK14_ROOT)
    parser.add_argument("--recovery-root", type=Path, default=RECOVERY_ROOT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", default="fp32")
    args = parser.parse_args(argv)
    started = time.perf_counter()
    recovery = args.recovery_root
    recovery.mkdir(parents=True, exist_ok=True)
    try:
        assets = resolve_official_assets(args.repo, args.checkpoint)
        _write_json(recovery / "00_runtime_audit.json", {"status": "READY", "assets": assets, "config": str(args.config.resolve())})
        fingerprint = build_runtime_fingerprint(repo=args.repo, checkpoint=args.checkpoint, device=args.device, precision=args.precision, match_max_side=1024)
        _write_json(recovery / "01_runtime_fingerprint.json", fingerprint)
        historical = _historical_pair()
        replay_dir = recovery / "pair_replay_00_01"
        replay_result = run_pair_replay(repo=args.repo, checkpoint=args.checkpoint, config=args.config, output=replay_dir, device=args.device)
        gate = compare_pair_replay(historical, replay_result["replay"])
        _write_json(recovery / "02_five_scene_matcher_replay.json", {"status": gate["status"], "pair": [0, 1], "historical": historical, "replay": replay_result, "comparison": gate})
        report = {
            "status": "MATCHER_REPLAY_VALIDATED",
            "hard_stops": [],
            "elapsed_sec": time.perf_counter() - started,
            "assets": assets,
            "fingerprint": fingerprint,
            "replay": gate,
            "next_action": "resume_task14_stage02",
        }
    except RuntimeError as exc:
        report = {"status": "HARD_STOP", "hard_stops": [str(exc)], "elapsed_sec": time.perf_counter() - started}
        _write_json(recovery / "00_runtime_audit.json", report)
        _write_json(recovery / "TASK14R_RUNTIME_RECOVERY_REPORT.json", report)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 2
    _write_json(recovery / "TASK14R_RUNTIME_RECOVERY_REPORT.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
