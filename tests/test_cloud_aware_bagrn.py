import numpy as np


def test_bagrn_excludes_cloud_from_overlap_stats_but_transforms_cloud_pixel():
    from src.bagrn import bagrn_normalize

    ref = np.array([[[10.0, 20.0], [10.0, 20.0]]])
    tar = np.array([[[30.0, 50.0], [30.0, 1000.0]]])
    overlaps = [{
        'idx_i': 0, 'idx_j': 1,
        'window_i': (0, 2, 0, 2),
        'window_j': (0, 2, 0, 2),
        'pixel_count': 4,
    }]
    cloud_masks = [
        np.zeros((2, 2), dtype=bool),
        np.array([[False, False], [False, True]]),
    ]

    result, _, _ = bagrn_normalize(
        [ref, tar], [None, None], overlaps, control_idx=0,
        cloud_masks=cloud_masks,
    )

    # Clear target pixels are matched using only clear overlap statistics.
    assert 10.0 < result[1][0, 0, 0] < 13.0
    assert 21.0 < result[1][0, 0, 1] < 23.5
    # Cloud is excluded from estimation, but the final linear transform still applies to it.
    assert np.isfinite(result[1][0, 1, 1])
    assert result[1][0, 1, 1] > 100.0
