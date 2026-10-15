import json
from pathlib import Path

import pytest

from scripts.run_task14_runtime_recovery import (
    HARD_STOP_OFFICIAL_MATCHER_ASSET_MISSING,
    HARD_STOP_MATCHER_REPLAY_MISMATCH,
    build_runtime_fingerprint,
    compare_pair_replay,
    resolve_official_assets,
)
from scripts.run_task14_13scene_scale import load_task14_prefix, validate_task14_prefix


def test_resolve_official_assets_requires_both_checkout_and_checkpoint(tmp_path: Path):
    repo = tmp_path / "EfficientLoFTR"
    (repo / "src" / "loftr").mkdir(parents=True)
    weights = repo / "weights" / "eloftr_outdoor.ckpt"
    weights.parent.mkdir()
    weights.write_bytes(b"checkpoint")
    resolved = resolve_official_assets(repo, weights)
    assert resolved["status"] == "READY"
    assert resolved["source_checkout"] == str(repo.resolve())
    assert resolved["checkpoint"] == str(weights.resolve())

    with pytest.raises(RuntimeError, match=HARD_STOP_OFFICIAL_MATCHER_ASSET_MISSING):
        resolve_official_assets(repo, repo / "missing.ckpt")


def test_compare_pair_replay_requires_frozen_metrics_and_transform(tmp_path: Path):
    historical = {
        "status": "OK",
        "raw_matches": 10,
        "inliers": 8,
        "coverage": 0.5,
        "residual_rmse": 0.25,
        "residual_p95": 0.5,
        "pixel_matrix": [[1.0, 0.0, 1.0], [0.0, 1.0, 2.0], [0.0, 0.0, 1.0]],
    }
    replay = dict(historical)
    result = compare_pair_replay(historical, replay)
    assert result["status"] == "MATCHER_REPLAY_VALIDATED"
    assert result["metrics_match"] is True
    assert result["arrays_match"] is True

    replay["inliers"] = 7
    with pytest.raises(RuntimeError, match=HARD_STOP_MATCHER_REPLAY_MISMATCH):
        compare_pair_replay(historical, replay)


def test_runtime_fingerprint_contains_required_environment_fields(tmp_path: Path):
    payload = build_runtime_fingerprint(
        repo=tmp_path / "EfficientLoFTR",
        checkpoint=tmp_path / "eloftr_outdoor.ckpt",
        device="cuda",
        precision="fp32",
        match_max_side=1024,
    )
    for field in (
        "python_version", "python_executable", "venv_path", "torch_version",
        "torch_cuda_version", "cuda_available", "gpu_name", "opencv_version",
        "numpy_version", "scipy_version", "efficientloftr_source_commit",
        "checkpoint_sha256", "device", "precision", "match_max_side",
    ):
        assert field in payload


def test_task14_resume_prefix_reads_existing_stage_artifacts_without_discovery(tmp_path: Path):
    root = tmp_path / "task14"
    root.mkdir()
    (root / "protocol.json").write_text(json.dumps({"scene_count": 1, "band": "B9", "match_max_side": 1024}), encoding="utf-8")
    (root / "scene_manifest.csv").write_text(
        "scene_index,scene_id,b9_path,mtl_path,crs,pixel_size_x_m,pixel_size_y_m,width,height,left,bottom,right,top,nodata\n"
        "0,S0,/tmp/s0.tif,/tmp/s0.mtl,EPSG:32650,14,14,10,10,0,0,140,140,0\n",
        encoding="utf-8",
    )
    (root / "overlap_edges.csv").write_text(
        "idx_i,idx_j,scene_i,scene_j,intersection_area,has_overlap\n",
        encoding="utf-8",
    )
    (root / "graph_summary.json").write_text(json.dumps({"scene_count": 1}), encoding="utf-8")
    prefix = load_task14_prefix(root)
    assert prefix["protocol"]["scene_count"] == 1
    assert prefix["records"][0]["scene_id"] == "S0"
    assert prefix["pairs"] == []


def test_task14_resume_prefix_requires_stage_minus1_00_01_success(tmp_path: Path):
    root = tmp_path / "task14"
    for stage in ("-1_streaming_equivalence", "00_preflight", "01_spatial_index_graph"):
        (root / "stages" / stage).mkdir(parents=True)
        (root / "stages" / stage / "_SUCCESS.json").write_text(json.dumps({"status": "SUCCESS", "input_hashes": {}, "output_hashes": {}}), encoding="utf-8")
    assert validate_task14_prefix(root) is True
    (root / "stages" / "01_spatial_index_graph" / "_SUCCESS.json").write_text(json.dumps({"status": "HARD_STOP", "input_hashes": {}, "output_hashes": {}}), encoding="utf-8")
    with pytest.raises(RuntimeError):
        validate_task14_prefix(root)
