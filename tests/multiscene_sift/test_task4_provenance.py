"""Task4 regression tests for canonical identity and metric provenance."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from src.multiscene_sift.global_comparison import _world_metrics
from src.multiscene_sift.global_geometric_adjustment import (
    load_frozen_edge_observations,
    summarize_network_residuals,
)
from src.multiscene_sift.global_ready_validation import _replay_report


def test_replay_uses_canonical_pair_identity_without_inverting_coordinates(tmp_path):
    old_dir = tmp_path / "old"
    old_dir.mkdir()
    (old_dir / "pairwise_summary.json").write_text(
        json.dumps(
            {
                "results": [
                    {
                        "idx_i": 0,
                        "idx_j": 1,
                        "status": "OK",
                        "raw_matches": 5,
                        "inliers": 4,
                        "residual_rmse": 0.5,
                        "residual_p95": 0.8,
                        "coverage": 0.6,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (old_dir / "accepted_graph.json").write_text(
        json.dumps({"accepted_edges": [{"idx_i": 0, "idx_j": 1}]}),
        encoding="utf-8",
    )

    reversed_result = SimpleNamespace(
        idx_i=1,
        idx_j=0,
        status="OK",
        raw_matches=5,
        inliers=4,
        residual_rmse=0.5,
        residual_p95=0.8,
        coverage=0.6,
    )
    report = _replay_report(
        tmp_path / "new",
        old_dir,
        [reversed_result],
        {"accepted_edges": [{"idx_i": 1, "idx_j": 0}]},
    )

    assert report["replay"] == "REPLAY_CONSISTENT"
    assert report["metric_rows"][0]["pair"] == [0, 1]


def test_world_metric_weights_only_rows_with_the_same_valid_metric(tmp_path):
    payload = {
        "results": [
            {"n_points": 2, "global_rmse_world_m": 3.0, "global_p95_world_m": 4.0},
            {"n_points": 100, "global_rmse_world_m": None, "global_p95_world_m": None},
            {"n_points": 4, "global_rmse_world_m": 5.0, "global_p95_world_m": 6.0},
        ]
    }
    path = tmp_path / "global_edge_consistency.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    metrics = _world_metrics(tmp_path)

    assert metrics["global_rmse_world_m"] == pytest.approx(
        np.sqrt((2.0 * 3.0**2 + 4.0 * 5.0**2) / 6.0)
    )
    assert metrics["mean_edge_p95_world_m"] == pytest.approx(5.0)


def test_metric_frame_is_supplied_by_the_global_method():
    class Records:
        def to_dict(self, orient):
            assert orient == "records"
            return [
            {
                "edge_i": 0,
                "edge_j": 1,
                "residual_px": 1.0,
                "dx": 1.0,
                "dy": 0.0,
                "is_tree_edge": True,
            }
            ]

    rows = Records()

    summary = summarize_network_residuals(
        rows, metric_frame="translation_l2_global_frame"
    )

    assert summary["metric_frame"] == "translation_l2_global_frame"


def test_legacy_edge_observation_loader_accepts_explicit_band(tmp_path):
    manifest = {
        "scenes": [
            {
                "index": 0,
                "bands": {
                    "B9": {
                        "bounds": [0.0, 0.0, 10.0, 10.0],
                        "resolution": [30.0, 30.0],
                    }
                },
            },
            {
                "index": 1,
                "bands": {
                    "B9": {
                        "bounds": [5.0, 0.0, 15.0, 10.0],
                        "resolution": [30.0, 30.0],
                    }
                },
            },
        ]
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    tree_path = tmp_path / "tree.json"
    tree_path.write_text(json.dumps({"edges": []}), encoding="utf-8")
    csv_path = tmp_path / "inliers.csv"
    csv_path.write_text(
        "edge_i,edge_j,x_i,y_i,x_j,y_j\n0,1,1,2,3,4\n", encoding="utf-8"
    )

    observations = load_frozen_edge_observations(
        csv_path, tree_path, manifest_path, registration_band="B9"
    )

    assert observations[(0, 1)]["pair_common_transform"][0, 0] == pytest.approx(30.0)
