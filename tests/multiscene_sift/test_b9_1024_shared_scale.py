"""Synthetic regression tests for the shared 1024 matcher input protocol."""

from __future__ import annotations

from unittest import mock

import numpy as np

from src.multiscene_sift import pairwise
from src.registration_benchmark.common_grid import build_match_view
from src.registration_benchmark.models import CommonGridPair, MatchSet


def _synthetic_pair(height: int = 1400, width: int = 2200) -> CommonGridPair:
    ref = np.linspace(0.0, 1.0, height * width, dtype=np.float32).reshape(height, width)
    tgt = np.flip(ref, axis=1).copy()
    valid = np.ones((height, width), dtype=bool)
    return CommonGridPair(
        ref_raw=ref,
        tgt_raw=tgt,
        ref_valid=valid,
        tgt_valid=valid.copy(),
        transform=None,
        crs=None,
        overlap_window=(0, height, 0, width),
    )


def _fake_matchset(method: str) -> MatchSet:
    return MatchSet(
        method=method,
        ref_xy=np.zeros((2, 2), dtype=np.float64),
        tgt_xy=np.ones((2, 2), dtype=np.float64),
        confidence=np.ones(2, dtype=np.float64),
        runtime_sec=0.01,
        metadata={
            "coordinate_frame": "pair_common_grid",
            "confidence_semantics": "method_internal_only",
        },
    )


def test_all_matchers_receive_the_same_1024_shared_view():
    view = build_match_view(_synthetic_pair(), max_side=1024)
    seen = {}

    def record(name):
        def _match(received_view, **kwargs):
            seen[name] = (id(received_view), received_view.ref.shape, received_view.tgt.shape)
            return _fake_matchset(name)

        return _match

    with mock.patch.object(pairwise, "match_sift", side_effect=record("sift")), \
         mock.patch.object(pairwise, "match_efficient_loftr", side_effect=record("efficient_loftr")), \
         mock.patch.object(pairwise, "match_lightglue_disk", side_effect=record("lightglue_disk")), \
         mock.patch("src.registration_benchmark.matchers.loftr.match_loftr", side_effect=record("loftr")):
        for matcher in ("sift", "loftr", "efficient_loftr", "lightglue_disk"):
            assert pairwise._run_matcher(view, matcher, device="cpu").method == matcher

    assert set(seen) == {"sift", "loftr", "efficient_loftr", "lightglue_disk"}
    assert {item[0] for item in seen.values()} == {id(view)}
    assert {item[1] for item in seen.values()} == {view.ref.shape}
    assert max(view.ref.shape) <= 1024


def test_shared_1024_view_preserves_resize_crop_inverse_mapping():
    pair = _synthetic_pair(height=1400, width=2200)
    pair.overlap_window = (113, 1313, 227, 2027)
    view = build_match_view(pair, max_side=1024)

    view_xy = np.array([[view.ref.shape[1] / 2.0, view.ref.shape[0] / 2.0]])
    common_xy = view.to_common_grid(view_xy)

    expected = np.array([
        [227.0 + view.ref.shape[1] / view.scale_x / 2.0,
         113.0 + view.ref.shape[0] / view.scale_y / 2.0]
    ])
    np.testing.assert_allclose(common_xy, expected)
    assert max(view.ref.shape) <= 1024
    assert view.scale_x == view.scale_y
