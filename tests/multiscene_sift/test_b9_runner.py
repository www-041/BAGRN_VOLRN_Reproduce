"""Small-fixture tests for the frozen B9 registration runner."""

from __future__ import annotations

import json

import numpy as np
import pytest
from rasterio.transform import Affine

from src.multiscene_sift.b9_frozen_config import build_frozen_config
from src.multiscene_sift.models import PairwiseRegistration


def _record(index: int) -> dict:
    return {
        "scene_id": f"scene_{index}",
        "scene_dir": f"D:/B9/scene_{index}",
        "b9_path": f"D:/B9/scene_{index}/scene_{index}_B9.TIF",
        "mtl_path": f"D:/B9/scene_{index}/scene_{index}_MTL.txt",
        "crs": "EPSG:32650",
        "pixel_size_x_m": 14.0,
        "pixel_size_y_m": 14.0,
        "width": 100,
        "height": 100,
        "left": float(index * 50),
        "bottom": 0.0,
        "right": float(index * 50 + 100),
        "top": 100.0,
    }


def _pair(i: int, j: int) -> dict:
    return {
        "idx_i": i,
        "idx_j": j,
        "scene_i": f"scene_{i}",
        "scene_j": f"scene_{j}",
        "intersection_area": 100.0,
        "overlap_area_i_ratio": 0.5,
        "overlap_area_j_ratio": 0.5,
        "symmetric_overlap_ratio": 0.5,
        "has_overlap": True,
    }


def _config():
    records = [_record(index) for index in range(5)]
    pairs = [_pair(i, j) for i in range(5) for j in range(i + 1, 5)]
    return build_frozen_config(records, pairs, [0, 1, 2, 3, 4])


def _result(i: int, j: int, status: str = "OK") -> PairwiseRegistration:
    return PairwiseRegistration(
        idx_i=i,
        idx_j=j,
        status=status,
        raw_matches=30,
        inliers=25 if status == "OK" else 0,
        inlier_ratio=0.8 if status == "OK" else 0.0,
        coverage=0.5 if status == "OK" else 0.0,
        residual_median=0.4 if status == "OK" else float("nan"),
        residual_rmse=0.5 if status == "OK" else float("nan"),
        residual_p95=0.9 if status == "OK" else float("nan"),
        pair_pixel_matrix=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        pair_common_transform=None,
        runtime_sec=1.5,
        matcher="sift",
        matcher_runtime_sec=1.0,
        geometry_runtime_sec=0.5,
    )


def test_b9_runner_delegates_to_shared_pairwise_api_and_writes_runtime(tmp_path):
    from src.multiscene_sift import b9_runner

    calls = []

    def fake_run_all_pairs(scenes, edges, out_dir, **kwargs):
        calls.append((scenes, edges, out_dir, kwargs))
        return [_result(i, j) for i in range(5) for j in range(i + 1, 5)]

    output = tmp_path / "sift"
    b9_runner.run_b9_registration(
        _config(), output, matcher="sift", pair_runner=fake_run_all_pairs
    )

    assert len(calls) == 1
    scenes, edges, _, kwargs = calls[0]
    assert len(scenes) == 5
    assert len(edges) == 10
    assert kwargs["band"] == "B9"
    assert kwargs["ransac_threshold"] == 2.0
    assert kwargs["random_seed"] == 0
    assert json.loads((output / "accepted_graph.json").read_text())["status"] == "CONNECTED"
    runtime = json.loads((output / "runtime.json").read_text())
    assert runtime["matcher_runtime_sec"] == 10.0
    assert runtime["geometry_runtime_sec"] == 5.0
    assert (output / "pairwise_summary.csv").exists()
    assert (output / "pairwise_summary.json").exists()


def test_b9_runner_records_disconnected_network_without_lowering_thresholds(tmp_path):
    from src.multiscene_sift import b9_runner

    def fake_run_all_pairs(scenes, edges, out_dir, **kwargs):
        return [_result(0, 1)]

    output = tmp_path / "lightglue_disk"
    b9_runner.run_b9_registration(
        _config(), output, matcher="lightglue_disk", pair_runner=fake_run_all_pairs
    )

    graph = json.loads((output / "accepted_graph.json").read_text())
    assert graph["status"] == "NETWORK_DISCONNECTED"
    assert graph["thresholds_unchanged"] is True


def test_b9_runner_reports_efficient_loftr_unavailable_without_fallback(tmp_path, monkeypatch):
    from src.multiscene_sift import b9_runner

    called = False

    def fake_run_all_pairs(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("unavailable EfficientLoFTR must not fallback")

    monkeypatch.setattr(b9_runner, "is_efficient_loftr_available", lambda: False)
    output = tmp_path / "efficient_loftr"
    b9_runner.run_b9_registration(
        _config(), output, matcher="efficient_loftr", pair_runner=fake_run_all_pairs
    )

    summary = json.loads((output / "pairwise_summary.json").read_text())
    assert not called
    assert {row["status"] for row in summary["results"]} == {
        "EFFICIENT_LOFTR_UNAVAILABLE"
    }


def test_b9_runner_pair_selector_limits_execution_to_requested_candidate(tmp_path):
    from src.multiscene_sift import b9_runner

    calls = []

    def fake_run_all_pairs(scenes, edges, out_dir, **kwargs):
        calls.append(edges)
        return [_result(edges[0].idx_i, edges[0].idx_j)]

    b9_runner.run_b9_registration(
        _config(),
        tmp_path / "pair",
        matcher="sift",
        pair=(0, 1),
        pair_runner=fake_run_all_pairs,
    )

    assert len(calls) == 1
    assert [(edge.idx_i, edge.idx_j) for edge in calls[0]] == [(0, 1)]


def test_b9_runner_pair_selector_rejects_non_candidate_pair(tmp_path):
    from src.multiscene_sift import b9_runner

    with pytest.raises(ValueError, match="not a frozen candidate edge"):
        b9_runner.run_b9_registration(
            _config(),
            tmp_path / "invalid-pair",
            matcher="sift",
            pair=(0, 99),
            pair_runner=lambda *args, **kwargs: [],
        )


def test_b9_runner_records_protocol_config_path(tmp_path):
    from src.multiscene_sift import b9_runner

    output = tmp_path / "formal"
    b9_runner.run_b9_registration(
        _config(),
        output,
        matcher="sift",
        protocol_config_path="data/output/b9_five_scene_validation/04_frozen_five_scene_config_1024.json",
        pair_runner=lambda scenes, edges, out_dir, **kwargs: [
            _result(edge.idx_i, edge.idx_j) for edge in edges
        ],
    )

    run_config = json.loads((output / "run_config.json").read_text())
    assert run_config["protocol_config_path"].endswith("04_frozen_five_scene_config_1024.json")


def test_b9_runner_persists_geometry_before_pairwise_summary(tmp_path):
    from src.multiscene_sift import b9_runner
    from src.multiscene_sift.geometry_artifacts import load_pair_geometry_bundle

    config = _config()
    config["registration"]["match_max_side"] = 1024
    config_path = tmp_path / "04_frozen_five_scene_config_1024.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    result = _result(0, 1)
    result.pair_common_transform = Affine(14.0, 0.0, 100.0, 0.0, -14.0, 200.0)
    result.inlier_ref_xy = np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float64)
    result.inlier_tgt_xy = np.array([[1.5, 3.0], [3.5, 5.0]], dtype=np.float64)

    def fake_run_all_pairs(scenes, edges, out_dir, **kwargs):
        return [result]

    output = tmp_path / "globalready"
    b9_runner.run_b9_registration(
        config,
        output,
        matcher="sift",
        pair=(0, 1),
        protocol_config_path=config_path,
        pair_runner=fake_run_all_pairs,
    )

    index = json.loads((output / "geometry_index.json").read_text(encoding="utf-8"))
    assert index["match_max_side"] == 1024
    assert index["accepted"][0]["pair"] == [0, 1]
    assert (output / "pairwise_summary.json").exists()
    bundle = load_pair_geometry_bundle(output / "geometry" / "pair_00_01.json")
    np.testing.assert_array_equal(bundle["inlier_ref_xy"], result.inlier_ref_xy)


def test_b9_runner_indexes_rejected_geometry_and_failed_pair_without_fake_npz(tmp_path):
    from src.multiscene_sift import b9_runner

    config = _config()
    config["registration"]["match_max_side"] = 1024
    rejected = _result(0, 1, status="TOO_FEW_INLIERS")
    rejected.pair_common_transform = Affine(14.0, 0.0, 100.0, 0.0, -14.0, 200.0)
    rejected.inlier_ref_xy = np.empty((0, 2), dtype=np.float64)
    rejected.inlier_tgt_xy = np.empty((0, 2), dtype=np.float64)
    failed = _result(0, 1, status="FAILED")

    output = tmp_path / "rejected"
    b9_runner.run_b9_registration(
        config,
        output,
        matcher="sift",
        pair=(0, 1),
        pair_runner=lambda *args, **kwargs: [rejected],
    )
    index = json.loads((output / "geometry_index.json").read_text(encoding="utf-8"))
    assert index["rejected"][0]["accepted"] is False
    assert index["rejected"][0]["point_count"] == 0
    assert len(list((output / "geometry").glob("*.npz"))) == 1

    failed_output = tmp_path / "failed"
    b9_runner.run_b9_registration(
        config,
        failed_output,
        matcher="sift",
        pair=(0, 1),
        pair_runner=lambda *args, **kwargs: [failed],
    )
    failed_index = json.loads((failed_output / "geometry_index.json").read_text(encoding="utf-8"))
    assert failed_index["failed"][0]["reason"] == "GEOMETRY_UNAVAILABLE"
    assert not list((failed_output / "geometry").glob("*.npz"))
