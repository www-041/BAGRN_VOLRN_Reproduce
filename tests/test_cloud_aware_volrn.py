import numpy as np
from rasterio.transform import Affine


def test_image_blocking_excludes_cloud_from_block_statistics():
    from src.volrn import image_blocking

    arr0 = np.ones((1, 4, 4), dtype=float)
    arr1 = np.ones((1, 4, 4), dtype=float)
    arr0[0, 0, 0] = 1000.0
    tf = Affine(1, 0, 0, 0, -1, 4)
    b = (0.0, 0.0, 4.0, 4.0)
    cloud_masks = [np.zeros((4, 4), bool), np.zeros((4, 4), bool)]
    cloud_masks[0][0, 0] = True

    blocks, _ = image_blocking(
        [arr0, arr1], [tf, tf], [b, b], [None, None], 4, [0],
        cloud_masks=cloud_masks,
    )
    block0 = next(x for x in blocks if x.image_idx == 0)
    assert abs(block0.mu[0] - 1.0) < 1e-12
    assert abs(block0.sigma[0]) < 1e-12


def test_volrn_keeps_cloud_pixel_finite_in_output():
    from src.volrn import volrn_normalize

    arr0 = np.ones((1, 6, 6), dtype=float) * 10
    arr1 = np.ones((1, 6, 6), dtype=float) * 12
    arr0[0, 1, 1] = 1000.0
    tf = Affine(1, 0, 0, 0, -1, 6)
    b = (0.0, 0.0, 6.0, 6.0)
    cloud_masks = [np.zeros((6, 6), bool), np.zeros((6, 6), bool)]
    cloud_masks[0][1, 1] = True

    out, _ = volrn_normalize(
        [arr0, arr1], [tf, tf], [b, b], [None, None],
        block_size_pixels=3, lambda_param=0.1, rho=1.0,
        max_iter=20, tol=1e-4, cloud_masks=cloud_masks,
    )
    assert np.isfinite(out[0][0, 1, 1])
