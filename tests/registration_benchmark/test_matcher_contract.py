"""Task 2 tests for the common matcher-to-geometry contract."""

from __future__ import annotations

import numpy as np
import pytest

from src.registration_benchmark.models import (
    MATCHER_INFO,
    MatchSet,
    MatchView,
    PAIR_COMMON_GRID_FRAME,
    make_matchset_from_view,
)


def _view() -> MatchView:
    return MatchView(
        ref=np.zeros((40, 50), dtype=np.float32),
        tgt=np.zeros((40, 50), dtype=np.float32),
        ref_valid=np.ones((40, 50), dtype=bool),
        tgt_valid=np.ones((40, 50), dtype=bool),
        origin_x=120.0,
        origin_y=80.0,
        scale_x=0.5,
        scale_y=0.25,
    )


def test_resize_and_crop_are_undone_into_pair_common_grid():
    view = _view()
    ref_view = np.array([[10.0, 20.0], [0.0, 0.0]])
    tgt_view = np.array([[12.0, 16.0], [1.0, 4.0]])

    matches = make_matchset_from_view(
        method="synthetic",
        view=view,
        ref_xy_view=ref_view,
        tgt_xy_view=tgt_view,
        confidence=np.array([0.8, 0.7]),
        runtime_sec=1.25,
    )

    np.testing.assert_allclose(matches.ref_xy, [[140.0, 160.0], [120.0, 80.0]])
    np.testing.assert_allclose(matches.tgt_xy, [[144.0, 144.0], [122.0, 96.0]])
    assert matches.metadata["coordinate_frame"] == PAIR_COMMON_GRID_FRAME


def test_geometry_validation_rejects_match_view_coordinates():
    matches = MatchSet(
        method="bad",
        ref_xy=np.zeros((3, 2)),
        tgt_xy=np.zeros((3, 2)),
        confidence=np.ones(3),
        runtime_sec=0.1,
        metadata={"coordinate_frame": "match_view"},
    )

    with pytest.raises(ValueError, match="pair_common_grid"):
        matches.validate_for_geometry()


def test_runtime_breakdown_and_confidence_semantics_are_explicit():
    matches = make_matchset_from_view(
        method="synthetic",
        view=_view(),
        ref_xy_view=np.zeros((2, 2)),
        tgt_xy_view=np.ones((2, 2)),
        confidence=np.ones(2),
        runtime_sec=1.25,
        runtime_breakdown={
            "feature_runtime_sec": 0.75,
            "matcher_runtime_sec": 0.50,
        },
    )

    matches.validate_for_geometry()
    assert matches.metadata["confidence_semantics"] == "method_internal_only"
    assert matches.metadata["runtime_breakdown"] == {
        "feature_runtime_sec": 0.75,
        "matcher_runtime_sec": 0.50,
        "total_runtime_sec": 1.25,
    }


def test_capability_descriptions_are_method_specific_and_non_comparable():
    for method in ("sift", "loftr", "efficient_loftr", "lightglue"):
        info = MATCHER_INFO[method]
        assert info["name"] == method
        assert info["confidence_semantics"] == "method_internal_only"
        assert "family" in info
        assert "detector_free" in info


def test_shared_geometry_contract_has_one_ransac_configuration():
    from src.registration_benchmark.geometry import (
        MAX_RANSAC_TRIALS,
        MIN_INLIERS,
        MIN_INLIER_RATIO,
        RANSAC_RANDOM_SEED,
        RANSAC_RESIDUAL_THRESHOLD,
    )

    assert RANSAC_RESIDUAL_THRESHOLD == 2.0
    assert MAX_RANSAC_TRIALS == 5000
    assert RANSAC_RANDOM_SEED == 0
    assert MIN_INLIERS == 20
    assert MIN_INLIER_RATIO == 0.30
