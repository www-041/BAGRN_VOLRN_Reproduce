"""Hard-fail contracts for global graph and transform composition."""

from __future__ import annotations

import numpy as np
import pytest
from rasterio.transform import from_origin

from src.multiscene_sift.global_registration import (
    build_accepted_graph,
    compose_global_transforms,
)
from src.multiscene_sift.models import PairwiseRegistration


def _pair(i: int, j: int) -> PairwiseRegistration:
    return PairwiseRegistration(
        idx_i=i,
        idx_j=j,
        status="OK",
        raw_matches=30,
        inliers=25,
        inlier_ratio=0.8,
        coverage=0.5,
        residual_median=0.1,
        residual_rmse=0.2,
        residual_p95=0.3,
        pair_pixel_matrix=np.eye(3).tolist(),
        pair_common_transform=from_origin(0.0, 100.0, 1.0, 1.0),
        runtime_sec=0.0,
    )


def test_explicit_scene_count_keeps_isolated_scene_in_connectivity_check():
    with pytest.raises(RuntimeError, match="DISCONNECTED"):
        build_accepted_graph([_pair(0, 1)], matcher_name="synthetic", n_scenes=3)


def test_missing_pair_for_required_tree_edge_is_hard_failure():
    with pytest.raises(RuntimeError, match="No pair registration"):
        compose_global_transforms(
            scenes=[object(), object()],
            accepted=[],
            tree_edges=[{"parent": 0, "child": 1, "depth": 1}],
            ref_idx=0,
        )


def test_scene_without_composed_transform_is_hard_failure():
    with pytest.raises(RuntimeError, match="no global transform"):
        compose_global_transforms(
            scenes=[object(), object(), object()],
            accepted=[_pair(0, 1)],
            tree_edges=[{"parent": 0, "child": 1, "depth": 1}],
            ref_idx=0,
        )
