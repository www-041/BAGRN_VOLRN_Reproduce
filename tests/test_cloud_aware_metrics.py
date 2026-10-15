import numpy as np


def test_metrics_ignore_cloud_outlier():
    from src.metrics import compute_all

    a = np.array([[[10.0, 10.0], [10.0, 1000.0]]])
    b = np.array([[[10.0, 10.0], [10.0, 20.0]]])
    overlaps = [{
        'idx_i': 0, 'idx_j': 1,
        'window_i': (0, 2, 0, 2),
        'window_j': (0, 2, 0, 2),
        'pixel_count': 4,
    }]
    cloud_masks = [
        np.array([[False, False], [False, True]]),
        np.array([[False, False], [False, True]]),
    ]
    metrics = compute_all(
        [a, b], [a, b], [None, None], overlaps, bands=[0],
        cloud_masks=cloud_masks,
    )
    assert metrics['adm'] == 0.0
    assert metrics['adsd'] == 0.0
    assert metrics['cd'] == 0.0
    assert metrics['gl'] == 0.0
