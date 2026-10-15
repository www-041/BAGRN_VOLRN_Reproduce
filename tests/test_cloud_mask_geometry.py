import numpy as np
from rasterio.transform import Affine


def test_reproject_mask_to_grid_uses_nearest_and_exact_target_shape():
    from src.multiscene_sift.band_geometry import reproject_mask_to_grid

    mask = np.zeros((4, 4), dtype=bool)
    mask[1:3, 1:3] = True
    src_tf = Affine(1, 0, 0, 0, -1, 4)
    dst_tf = Affine(1, 0, 0, 0, -1, 4)

    out = reproject_mask_to_grid(
        mask,
        src_transform=src_tf,
        src_crs='EPSG:32649',
        dst_transform=dst_tf,
        dst_crs='EPSG:32649',
        dst_shape=(4, 4),
    )
    assert out.dtype == np.bool_
    assert out.shape == (4, 4)
    np.testing.assert_array_equal(out, mask)
