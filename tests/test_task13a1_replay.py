"""Frozen Task13A.1 replay contracts on small, hand-checkable arrays."""

import numpy as np

from src.seam_local.blend import blend_across_seam
from src.seam_local.seam import SeamResult
from src.seam_local.source_side import SourceSideResult


def _vertical_seam(shape: tuple[int, int], column: int) -> SeamResult:
    path = np.column_stack((np.arange(shape[0]), np.full(shape[0], column)))
    return SeamResult("vertical", path, 0.0, 0.0, 0.0, "OK")


def test_explicit_assignment_blends_b_on_low_side_without_mutating_seam():
    shape = (5, 9)
    a = np.full(shape, 10.0)
    b = np.full(shape, 30.0)
    valid = np.ones(shape, dtype=bool)
    seam = _vertical_seam(shape, 4)
    saved_path = seam.row_col_path.copy()
    assignment = SourceSideResult("CENTROID_RESOLVABLE", "B", "A", "FOOTPRINT_CENTROID_PROJECTION", 1.0, {})

    default = blend_across_seam(a, b, seam, valid, valid, blend_half_width=1)
    result = blend_across_seam(a, b, seam, valid, valid, blend_half_width=1, source_side=assignment)

    assert default.status == "AMBIGUOUS_SOURCE_SIDE"
    assert result.status == "OK"
    assert result.source_side == "B_LOW_A_HIGH"
    assert result.image[0, 0] == 30.0
    assert result.image[0, 8] == 10.0
    assert np.isfinite(result.image[result.valid_mask]).all()
    assert np.array_equal(seam.row_col_path, saved_path)


def test_saved_coefficients_replay_local_taper_without_refitting():
    from scripts.run_task13a1_source_side import replay_saved_local_correction

    shape = (5, 9)
    a = np.full(shape, 10.0)
    b = np.full(shape, 30.0)
    valid = np.ones(shape, dtype=bool)
    seam = _vertical_seam(shape, 4)
    saved_path = seam.row_col_path.copy()
    coefficients = ((0, 0.0, 2.0, 1.0, 0.5, -5.0),)

    corrected_a, corrected_b = replay_saved_local_correction(
        a, b, valid, valid, seam, coefficients, half_width=2,
    )

    assert corrected_a[0, 4] == 21.0
    assert corrected_b[0, 4] == 10.0
    assert corrected_a[0, 2] == 10.0
    assert corrected_b[0, 6] == 30.0
    assert np.array_equal(seam.row_col_path, saved_path)
    assert coefficients == ((0, 0.0, 2.0, 1.0, 0.5, -5.0),)


def test_registered_footprint_is_polygon_from_valid_pixel_bounds():
    from affine import Affine
    from scripts.run_task13a1_source_side import _footprint_polygon

    valid = np.zeros((5, 7), dtype=bool)
    valid[1:3, 2:5] = True
    footprint = _footprint_polygon(valid, Affine(14, 0, 100, 0, -14, 200))

    assert footprint.geom_type == "Polygon"
    assert footprint.bounds == (128.0, 158.0, 170.0, 186.0)
    assert (footprint.centroid.x, footprint.centroid.y) == (149.0, 172.0)
