import numpy as np


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
