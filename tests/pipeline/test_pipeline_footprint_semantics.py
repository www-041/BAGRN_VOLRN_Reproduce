import numpy as np
from affine import Affine
from rasterio.features import geometry_mask
from shapely.geometry import box

from src.seam_local.footprint import footprint_polygon_from_valid_mask


def test_pipeline_footprint_preserves_holes_and_disconnected_components():
    mask = np.zeros((8, 12), dtype=bool)
    mask[1:7, 1:6] = True
    mask[3:5, 3:5] = False
    mask[0:2, 9:12] = True
    footprint = footprint_polygon_from_valid_mask(mask, Affine(2, 0, 100, 0, -2, 200))

    assert footprint.geom_type == "MultiPolygon"
    assert footprint.area == 2 * 2 * (30 - 4) + 2 * 2 * 6
    assert not footprint.contains(box(106, 190, 110, 194).centroid)


def test_pipeline_uses_the_same_helper_as_task13a1_compatibility_wrapper():
    from scripts.run_task13a1_source_side import _footprint_polygon

    mask = np.array([[1, 1, 0, 0], [1, 0, 0, 1], [0, 0, 1, 1]], dtype=bool)
    transform = Affine(3, 0, 10, 0, -3, 20)
    old = _footprint_polygon(mask, transform)
    new = footprint_polygon_from_valid_mask(mask, transform)
    assert old.equals(new)


def test_all_false_mask_is_a_hard_input_error():
    import pytest

    with pytest.raises(ValueError, match="valid footprint"):
        footprint_polygon_from_valid_mask(np.zeros((2, 2), dtype=bool), Affine.identity())
