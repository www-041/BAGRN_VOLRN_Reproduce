import numpy as np

from src.registration_benchmark.models import (
    MatchView,
    filter_matches_by_valid_mask,
)


def _view() -> MatchView:
    ref_valid = np.ones((6, 7), dtype=bool)
    tgt_valid = np.ones((6, 7), dtype=bool)
    ref_valid[2, 2] = False
    tgt_valid[3, 3] = False
    image = np.zeros((6, 7), dtype=np.float32)
    return MatchView(
        ref=image,
        tgt=image.copy(),
        ref_valid=ref_valid,
        tgt_valid=tgt_valid,
        origin_x=0.0,
        origin_y=0.0,
        scale_x=1.0,
        scale_y=1.0,
    )


def test_shared_filter_removes_invalid_and_out_of_bounds_matches():
    ref_xy = np.array([[1.0, 1.0], [2.0, 2.0], [4.0, 4.0], [-1.0, 1.0]])
    tgt_xy = np.array([[1.0, 1.0], [1.0, 1.0], [3.0, 3.0], [1.0, 1.0]])
    confidence = np.array([0.1, 0.2, 0.3, 0.4])

    ref, tgt, conf, keep = filter_matches_by_valid_mask(
        ref_xy, tgt_xy, confidence, _view()
    )

    np.testing.assert_array_equal(keep, [True, False, False, False])
    np.testing.assert_allclose(ref, [[1.0, 1.0]])
    np.testing.assert_allclose(tgt, [[1.0, 1.0]])
    np.testing.assert_allclose(conf, [0.1])


def test_shared_filter_preserves_all_valid_coordinates_and_confidence():
    ref_xy = np.array([[1.25, 1.75], [4.5, 4.5]])
    tgt_xy = np.array([[1.25, 1.75], [4.5, 4.5]])
    confidence = np.array([0.2, 0.8])

    ref, tgt, conf, keep = filter_matches_by_valid_mask(
        ref_xy, tgt_xy, confidence, _view()
    )

    np.testing.assert_array_equal(keep, [True, True])
    np.testing.assert_allclose(ref, ref_xy)
    np.testing.assert_allclose(tgt, tgt_xy)
    np.testing.assert_allclose(conf, confidence)


def test_four_frozen_adapters_import_the_same_shared_filter():
    from src.registration_benchmark.matchers import efficient_loftr, lightglue_disk, loftr, sift

    for module in (sift, loftr, efficient_loftr, lightglue_disk):
        assert module.filter_matches_by_valid_mask is filter_matches_by_valid_mask
