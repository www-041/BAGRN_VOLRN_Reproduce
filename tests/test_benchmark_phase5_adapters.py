"""Mocked contract tests for the four in-repository Phase5 matchers."""

from dataclasses import replace
import hashlib
from pathlib import Path
import sys

from affine import Affine
import numpy as np
import pytest
import rasterio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from benchmark_v1_phase5.inputs import prepare_pair
from benchmark_v1_phase5.protocol import PairManifest, load_config
from benchmark_v1_phase5.adapters import run_local_matcher
from registration_benchmark.models import MatchSet


def _pair(tmp_path):
    paths = []
    for name, transform in (("i", Affine(14, 0, 100, 0, -14, 300)),
                            ("j", Affine(14, 0, 128, 0, -14, 286))):
        path = tmp_path / f"{name}.tif"
        with rasterio.open(path, "w", driver="GTiff", height=64, width=64,
                           count=1, dtype="float32", crs="EPSG:32650",
                           transform=transform, nodata=0) as dst:
            dst.write(np.random.default_rng(3).random((64, 64), dtype=np.float32), 1)
            dst.write_mask(np.ones((64, 64), dtype="uint8") * 255)
        paths.append(path)
    a, b = paths
    manifest = PairManifest("00_01", "i", "j", a, b, a.stat().st_size, b.stat().st_size,
                            hashlib.sha256(a.read_bytes()).hexdigest(),
                            hashlib.sha256(b.read_bytes()).hexdigest())
    config = replace(load_config(ROOT / "configs/benchmark_v1_phase5.yaml"), max_view_side=64)
    return prepare_pair(manifest, config)


def _matchset():
    xy = np.array([[10.0, 10.0], [20.0, 20.0]])
    return MatchSet("sift", xy, xy + 1, np.array([.8, .7]), .25,
                    {"coordinate_frame": "pair_common_grid",
                     "confidence_semantics": "method_internal_only",
                     "runtime_breakdown": {"feature_runtime_sec": .1,
                                           "matcher_runtime_sec": .15,
                                           "total_runtime_sec": .25}})


def _efficient_artifact(tmp_path):
    repo = tmp_path / "eloftr_repo"
    repo.mkdir()
    weights = tmp_path / "eloftr.ckpt"
    weights.write_bytes(b"synthetic frozen checkpoint")
    return {"repo_dir": repo, "weights_path": weights,
            "sha256": hashlib.sha256(weights.read_bytes()).hexdigest()}


@pytest.mark.parametrize("method,expected", [
    ("SIFT", {"nfeatures": 8000, "ratio_threshold": .75}),
    ("LoFTR", {"device": "auto", "confidence_threshold": .2, "max_matches": 5000}),
    ("EfficientLoFTR", {"device": "auto", "confidence_threshold": 0.0,
                         "model_type": "full", "precision": "fp32"}),
    ("LightGlue+DISK", {"device": "auto", "max_num_keypoints": 2048,
                        "disk_resize": 1024, "min_confidence": 0.0}),
])
def test_dispatch_uses_frozen_args_shared_view_and_native_frame(
        tmp_path, monkeypatch, method, expected):
    import benchmark_v1_phase5.adapters as adapters
    pair = _pair(tmp_path)
    seen = {}

    def fake(view, **kwargs):
        seen["view"] = view
        seen["kwargs"] = kwargs
        return _matchset()

    name = {"SIFT": "match_sift", "LoFTR": "match_loftr",
            "EfficientLoFTR": "match_efficient_loftr",
            "LightGlue+DISK": "match_lightglue_disk"}[method]
    monkeypatch.setattr(adapters, name, fake)
    artifact = _efficient_artifact(tmp_path) if method == "EfficientLoFTR" else None
    if artifact:
        expected.update(repo_dir=artifact["repo_dir"], weights_path=artifact["weights_path"])
    result = run_local_matcher(method, pair,
                               load_config(ROOT / "configs/benchmark_v1_phase5.yaml"),
                               efficient_artifact=artifact)
    assert result.status == "SUCCESS"
    assert result.matches.i_rc.shape == (2, 2)
    assert seen["kwargs"] == expected
    assert seen["view"].origin_x == 0 and seen["view"].origin_y == 0
    assert seen["view"].scale_x == 1 and seen["view"].scale_y == 1
    assert result.matches.i_view_xy[0].tolist() == [10.0, 10.0]
    assert result.matches.j_view_xy[0].tolist() == [11.0, 11.0]
    if method == "EfficientLoFTR":
        assert seen["kwargs"]["repo_dir"] == artifact["repo_dir"]
        assert seen["kwargs"]["weights_path"] == artifact["weights_path"]
        assert result.provenance["weights_sha256"] == artifact["sha256"]
    # Independently calculate expected native row/col from the overlap world
    # coordinates and the two different affine origins.
    overlap = pair.view.provenance.overlap_affine
    world_i = overlap * (10.5, 10.5)
    world_j = overlap * (11.5, 11.5)
    expected_i_xy = (~pair.native_affine_i) * world_i
    expected_j_xy = (~pair.native_affine_j) * world_j
    expected_i = np.array([expected_i_xy[1] - .5, expected_i_xy[0] - .5])
    expected_j = np.array([expected_j_xy[1] - .5, expected_j_xy[0] - .5])
    np.testing.assert_allclose(result.matches.i_rc[0], expected_i)
    np.testing.assert_allclose(result.matches.j_rc[0], expected_j)


@pytest.mark.parametrize("method,message,status", [
    ("LoFTR", "LoFTR (kornia) is not available", "ENV_UNAVAILABLE"),
    ("LightGlue+DISK", "LIGHTGLUE_DISK_UNAVAILABLE: missing torch", "ENV_UNAVAILABLE"),
    ("LightGlue+DISK", "LIGHTGLUE_DISK_UNAVAILABLE: CUDA out of memory", "ALGORITHM_FAIL"),
    ("EfficientLoFTR", "EFFICIENT_LOFTR_UNAVAILABLE: overlap view must be at least 32x32",
     "ALGORITHM_FAIL"),
])
def test_dependency_and_weight_unavailability_are_classified(
        tmp_path, monkeypatch, method, message, status):
    import benchmark_v1_phase5.adapters as adapters
    name = {"LoFTR": "match_loftr", "EfficientLoFTR": "match_efficient_loftr",
            "LightGlue+DISK": "match_lightglue_disk"}[method]
    def unavailable(*args, **kwargs):
        raise RuntimeError(message)
    monkeypatch.setattr(adapters, name, unavailable)
    artifact = _efficient_artifact(tmp_path) if method == "EfficientLoFTR" else None
    result = run_local_matcher(method, _pair(tmp_path),
                               load_config(ROOT / "configs/benchmark_v1_phase5.yaml"),
                               efficient_artifact=artifact)
    assert result.status == status
    assert message in result.reason
    assert result.matches is None


def test_pair_without_joint_overlap_is_not_reported_as_matcher_failure(tmp_path):
    pair = _pair(tmp_path)
    pair = replace(pair, status="NO_VALID_OVERLAP", view=None)
    result = run_local_matcher("SIFT", pair,
                               load_config(ROOT / "configs/benchmark_v1_phase5.yaml"))
    assert result.status == "NO_VALID_OVERLAP"
    assert result.matches is None


def test_efficient_weights_require_path_and_matching_sha256(tmp_path, monkeypatch):
    import benchmark_v1_phase5.adapters as adapters
    pair = _pair(tmp_path)
    config = load_config(ROOT / "configs/benchmark_v1_phase5.yaml")
    called = []
    monkeypatch.setattr(adapters, "match_efficient_loftr",
                        lambda *a, **k: called.append(True))
    missing = run_local_matcher("EfficientLoFTR", pair, config)
    assert missing.status == "WEIGHTS_UNAVAILABLE"
    assert missing.reason == "MISSING_FROZEN_WEIGHTS_MANIFEST"
    artifact = _efficient_artifact(tmp_path)
    bad_hash = {**artifact, "sha256": "0" * 64}
    mismatch = run_local_matcher("EfficientLoFTR", pair, config,
                                 efficient_artifact=bad_hash)
    assert mismatch.status == "WEIGHTS_UNAVAILABLE"
    assert mismatch.reason == "CHECKPOINT_SHA256_MISMATCH"
    assert called == []


@pytest.mark.parametrize("field,value,status,reason", [
    ("repo_dir", None, "ENV_UNAVAILABLE", "INVALID_MODEL_REPOSITORY_PATH"),
    ("weights_path", None, "WEIGHTS_UNAVAILABLE", "INVALID_CHECKPOINT_PATH"),
])
def test_malformed_efficient_artifact_returns_classified_status(
        tmp_path, field, value, status, reason):
    pair = _pair(tmp_path)
    artifact = _efficient_artifact(tmp_path)
    artifact[field] = value
    result = run_local_matcher("EfficientLoFTR", pair,
                               load_config(ROOT / "configs/benchmark_v1_phase5.yaml"),
                               efficient_artifact=artifact)
    assert result.status == status
    assert reason in result.reason


def test_efficient_artifact_path_resolution_and_hash_read_errors_are_classified(
        tmp_path, monkeypatch):
    import benchmark_v1_phase5.adapters as adapters
    pair = _pair(tmp_path)
    config = load_config(ROOT / "configs/benchmark_v1_phase5.yaml")
    artifact = _efficient_artifact(tmp_path)
    original_resolve = Path.resolve

    def loop_on_repo(path, *args, **kwargs):
        if path == Path(artifact["repo_dir"]):
            raise RuntimeError("symlink loop")
        return original_resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", loop_on_repo)
    result = run_local_matcher("EfficientLoFTR", pair, config,
                               efficient_artifact=artifact)
    assert result.status == "ENV_UNAVAILABLE"
    assert "symlink loop" in result.reason
    monkeypatch.setattr(Path, "resolve", original_resolve)

    weights = Path(artifact["weights_path"])
    original_open = Path.open

    def permission_error(path, *args, **kwargs):
        if path == weights:
            raise PermissionError("synthetic access denied")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", permission_error)
    result = run_local_matcher("EfficientLoFTR", pair, config,
                               efficient_artifact=artifact)
    assert result.status == "WEIGHTS_UNAVAILABLE"
    assert "CHECKPOINT_UNREADABLE" in result.reason


def test_corrupt_checkpoint_error_is_algorithm_failure(tmp_path, monkeypatch):
    import benchmark_v1_phase5.adapters as adapters
    monkeypatch.setattr(adapters, "match_efficient_loftr",
                        lambda *a, **k: (_ for _ in ()).throw(
                            RuntimeError("EFFICIENT_LOFTR_UNAVAILABLE: checkpoint tensor shape invalid")))
    result = run_local_matcher("EfficientLoFTR", _pair(tmp_path),
                               load_config(ROOT / "configs/benchmark_v1_phase5.yaml"),
                               efficient_artifact=_efficient_artifact(tmp_path))
    assert result.status == "ALGORITHM_FAIL"


def test_different_scene_masks_are_checked_at_their_own_points(tmp_path, monkeypatch):
    import benchmark_v1_phase5.adapters as adapters
    pair = _pair(tmp_path)
    pair.view.mask_i[11, 11] = False
    pair.view.mask_j[10, 10] = False
    pair.view.joint_mask[10, 10] = False
    pair.view.joint_mask[11, 11] = False
    monkeypatch.setattr(adapters, "match_sift", lambda *a, **k: _matchset())
    result = run_local_matcher("SIFT", pair,
                               load_config(ROOT / "configs/benchmark_v1_phase5.yaml"))
    assert result.status == "SUCCESS"
    assert len(result.matches.i_rc) == 2


def test_unknown_method_and_matcher_exception_are_explicit(tmp_path, monkeypatch):
    import benchmark_v1_phase5.adapters as adapters
    config = load_config(ROOT / "configs/benchmark_v1_phase5.yaml")
    pair = _pair(tmp_path)
    with pytest.raises(ValueError, match="method"):
        run_local_matcher("KARIOS", pair, config)
    monkeypatch.setattr(adapters, "match_sift", lambda *a, **k: (_ for _ in ()).throw(
        RuntimeError("synthetic matcher crash")))
    result = run_local_matcher("SIFT", pair, config)
    assert result.status == "ALGORITHM_FAIL"
    assert "synthetic matcher crash" in result.reason
