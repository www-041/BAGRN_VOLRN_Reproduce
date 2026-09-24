"""Task 5 tests for the matcher-only five-scene benchmark."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.multiscene_sift.models import OverlapEdge, PairwiseRegistration, Scene
from src.multiscene_sift.modern_matcher_benchmark import (
    MODERN_MATCHERS,
    _resolve_scene_names,
    run_modern_matcher_benchmark,
)


def _scene(index: int) -> Scene:
    return Scene(
        index=index,
        name=f"scene_{index}",
        directory="synthetic",
        band_paths={"B14": "synthetic.tif"},
        crs=None,
        transforms={"B14": None},
        shapes={"B14": (10, 10)},
        nodata={"B14": 0.0},
        bounds={"B14": (0.0, 0.0, 10.0, 10.0)},
    )


def _failed_pair(i: int, j: int, matcher: str) -> PairwiseRegistration:
    return PairwiseRegistration(
        idx_i=i,
        idx_j=j,
        status="FAILED",
        raw_matches=0,
        inliers=0,
        inlier_ratio=0.0,
        coverage=0.0,
        residual_median=float("nan"),
        residual_rmse=float("nan"),
        residual_p95=float("nan"),
        pair_pixel_matrix=np.eye(3).tolist(),
        pair_common_transform=None,
        runtime_sec=0.1,
        matcher=matcher,
        matcher_runtime_sec=0.1,
        geometry_runtime_sec=0.0,
    )


def test_benchmark_auto_discovers_exactly_five_scene_directories(tmp_path):
    root = tmp_path / "flat"
    for name in ("scene_3", "scene_1", "scene_5", "scene_2", "scene_4"):
        (root / name).mkdir(parents=True)

    assert _resolve_scene_names(str(root), None) == [
        "scene_1", "scene_2", "scene_3", "scene_4", "scene_5"
    ]


def test_modern_matchers_are_frozen_and_use_one_geometry_config():
    assert MODERN_MATCHERS == ("sift", "loftr", "efficient_loftr", "lightglue_disk")

    from src.registration_benchmark.geometry import (
        MAX_RANSAC_TRIALS,
        MIN_INLIERS,
        MIN_INLIER_RATIO,
        RANSAC_RESIDUAL_THRESHOLD,
    )
    from src.multiscene_sift.pairwise import (
        MIN_INLIERS as PAIR_MIN_INLIERS,
        MIN_INLIER_RATIO as PAIR_MIN_INLIER_RATIO,
        RANSAC_MAX_TRIALS,
    )

    assert RANSAC_MAX_TRIALS == MAX_RANSAC_TRIALS
    assert PAIR_MIN_INLIERS == MIN_INLIERS
    assert PAIR_MIN_INLIER_RATIO == MIN_INLIER_RATIO
    assert RANSAC_RESIDUAL_THRESHOLD == 2.0


def test_disconnected_graph_is_recorded_without_radiometric_or_mosaic_steps(
    monkeypatch, tmp_path
):
    scenes = [_scene(i) for i in range(3)]
    edges = [OverlapEdge(0, 1, 1.0, 0.5, 0.5), OverlapEdge(1, 2, 1.0, 0.5, 0.5)]
    calls = []

    monkeypatch.setattr(
        "src.multiscene_sift.modern_matcher_benchmark.discover_five_scenes",
        lambda *args, **kwargs: (scenes, {"scenes": []}),
    )
    monkeypatch.setattr(
        "src.multiscene_sift.modern_matcher_benchmark.build_geographic_overlap_graph",
        lambda *args, **kwargs: edges,
    )

    def fake_run_all_pairs(scenes_arg, edges_arg, out_dir, **kwargs):
        calls.append(kwargs)
        return [_failed_pair(edge.idx_i, edge.idx_j, kwargs["matcher"]) for edge in edges_arg]

    monkeypatch.setattr(
        "src.multiscene_sift.modern_matcher_benchmark.run_all_pairs",
        fake_run_all_pairs,
    )

    result = run_modern_matcher_benchmark(
        input_root="synthetic",
        output_dir=str(tmp_path / "modern"),
        scene_names=[scene.name for scene in scenes],
        matchers=("sift",),
    )

    assert result["status"] == "COMPLETE"
    assert result["methods"]["sift"]["status"] == "DISCONNECTED"
    assert calls[0]["ransac_threshold"] == 2.0
    assert (tmp_path / "modern" / "sift" / "network_summary.json").exists()
    payload = json.loads((tmp_path / "modern" / "sift" / "network_summary.json").read_text())
    assert payload["graph_connected"] is False
    assert not list((tmp_path / "modern").rglob("*BAGRN*"))
    assert not list((tmp_path / "modern").rglob("*VOLRN*"))
    assert not list((tmp_path / "modern").rglob("*.tif"))


def test_benchmark_writes_comparison_table_for_each_requested_method(monkeypatch, tmp_path):
    scenes = [_scene(i) for i in range(2)]
    edges = [OverlapEdge(0, 1, 1.0, 0.5, 0.5)]

    monkeypatch.setattr(
        "src.multiscene_sift.modern_matcher_benchmark.discover_five_scenes",
        lambda *args, **kwargs: (scenes, {"scenes": []}),
    )
    monkeypatch.setattr(
        "src.multiscene_sift.modern_matcher_benchmark.build_geographic_overlap_graph",
        lambda *args, **kwargs: edges,
    )
    monkeypatch.setattr(
        "src.multiscene_sift.modern_matcher_benchmark.run_all_pairs",
        lambda scenes_arg, edges_arg, out_dir, **kwargs: [
            _failed_pair(0, 1, kwargs["matcher"])
        ],
    )

    result = run_modern_matcher_benchmark(
        input_root="synthetic",
        output_dir=str(tmp_path / "modern"),
        scene_names=[scene.name for scene in scenes],
        matchers=("sift", "efficient_loftr"),
    )

    assert set(result["methods"]) == {"sift", "efficient_loftr"}
    comparison = json.loads((tmp_path / "modern" / "comparison.json").read_text())
    assert set(comparison["methods"]) == {"sift", "efficient_loftr"}
    assert (tmp_path / "modern" / "comparison.csv").exists()
    assert (tmp_path / "modern" / "comparison.png").exists()
