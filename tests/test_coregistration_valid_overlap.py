import numpy as np
import pytest


def test_map_pixel_center_between_grids_identity_round_trip():
    from rasterio.transform import from_origin
    from src.coregistration import map_pixel_center_between_grids

    transform = from_origin(100.0, 200.0, 14.0, 14.0)
    mapped = map_pixel_center_between_grids(12.25, 8.5, transform, transform)

    assert mapped == pytest.approx((12.25, 8.5))


def test_map_pixel_center_between_grids_translated_origins():
    from rasterio.transform import from_origin
    from src.coregistration import map_pixel_center_between_grids

    ref = from_origin(100.0, 200.0, 14.0, 14.0)
    target = from_origin(128.0, 172.0, 14.0, 14.0)

    mapped = map_pixel_center_between_grids(10.0, 20.0, ref, target)

    assert mapped == pytest.approx((8.0, 18.0))


def test_map_pixel_center_between_grids_preserves_subpixel_offset():
    from rasterio.transform import from_origin
    from src.coregistration import map_pixel_center_between_grids

    ref = from_origin(0.0, 100.0, 14.0, 14.0)
    target = from_origin(7.0, 93.0, 14.0, 14.0)

    mapped = map_pixel_center_between_grids(10.25, 20.75, ref, target)

    assert mapped == pytest.approx((9.75, 20.25))


def test_map_pixel_center_between_grids_round_trip_between_two_14m_grids():
    from rasterio.transform import from_origin
    from src.coregistration import map_pixel_center_between_grids

    ref = from_origin(500000.0, 4200000.0, 14.0, 14.0)
    target = from_origin(500123.0, 4199877.0, 14.0, 14.0)

    target_xy = map_pixel_center_between_grids(321.125, 654.875, ref, target)
    round_trip = map_pixel_center_between_grids(*target_xy, target, ref)

    assert round_trip == pytest.approx((321.125, 654.875), abs=1e-9)


def test_vector_pixel_center_mapping_matches_scalar_for_many_points():
    from rasterio.transform import from_origin
    from src.coregistration import (
        map_pixel_center_between_grids,
        map_pixel_centers_between_grids,
    )

    ref = from_origin(500000.0, 4200000.0, 14.0, 14.0)
    target = from_origin(500123.0, 4199877.0, 14.0, 14.0)
    points = np.array([[0.0, 0.0], [1.25, 2.5], [321.125, 654.875]])

    vector_result = map_pixel_centers_between_grids(points, ref, target)
    scalar_result = np.array([
        map_pixel_center_between_grids(x, y, ref, target)
        for x, y in points
    ])

    assert vector_result.shape == (3, 2)
    assert np.allclose(vector_result, scalar_result, atol=1e-10)


def test_vector_pixel_center_mapping_round_trip():
    from rasterio.transform import from_origin
    from src.coregistration import map_pixel_centers_between_grids

    ref = from_origin(100.0, 200.0, 14.0, 14.0)
    target = from_origin(128.0, 172.0, 14.0, 14.0)
    points = np.array([[0.0, 0.0], [10.25, 20.75], [99.5, 42.125]])

    mapped = map_pixel_centers_between_grids(points, ref, target)
    round_trip = map_pixel_centers_between_grids(mapped, target, ref)

    assert np.allclose(round_trip, points, atol=1e-10)


def test_common_valid_mask_excludes_diagonal_nodata_regions():
    from src.coregistration import build_common_valid_mask

    ref = np.ones((4, 4), dtype=float)
    tgt = np.ones((4, 4), dtype=float)
    tgt[0, 0] = -9999.0
    tgt[1, 1] = np.nan

    mask = build_common_valid_mask(ref, tgt, -9999.0, -9999.0)

    assert mask.dtype == bool
    assert mask.sum() == 14
    assert not mask[0, 0]
    assert not mask[1, 1]


def test_overlap_bbox_does_not_imply_usable_overlap():
    from rasterio.transform import from_origin
    from src.multiband_pipeline import MultibandPipeline
    from src.experiment_config import ExperimentConfig

    ref = np.ones((1, 32, 32), dtype=float)
    tgt = np.ones((1, 32, 32), dtype=float)
    tgt[:, :24, :] = -9999.0
    pipeline = MultibandPipeline(ExperimentConfig(
        scenes=[], selected_bands=["B14"], registration_band="B14"
    ))
    scene_data = {
        "arrays": [ref, tgt],
        "transforms": [from_origin(0, 32, 1, 1), from_origin(0, 32, 1, 1)],
        "bounds": [(0, 0, 32, 32), (0, 0, 32, 32)],
        "nodata_values": [-9999.0, -9999.0],
    }

    overlaps = pipeline.detect_overlaps(scene_data)

    assert len(overlaps) == 1
    assert overlaps[0]["bbox_overlap_pixels"] == 32 * 32
    assert overlaps[0]["common_valid_pixels"] == 8 * 32
    assert overlaps[0]["common_valid_ratio"] == 0.25


def test_valid_zero_pixels_are_not_treated_as_nodata():
    from src.coregistration import build_common_valid_mask

    ref = np.array([[0.0, 1.0], [2.0, 3.0]])
    tgt = np.array([[0.0, 4.0], [5.0, 6.0]])

    mask = build_common_valid_mask(ref, tgt, None, None)

    assert mask.all()
