"""Synthetic checks for seam-side inference and cosine blending."""

import numpy as np

from src.seam_local.blend import blend_across_seam
from src.seam_local.seam import SeamResult


def _seam(shape: tuple[int, int], orientation: str, coordinate: int) -> SeamResult:
    length = shape[0] if orientation == "vertical" else shape[1]
    lines = np.arange(length)
    path = np.column_stack((lines, np.full(length, coordinate))) if orientation == "vertical" else np.column_stack((np.full(length, coordinate), lines))
    return SeamResult(orientation, path, 0.0, 0.0, 0.0, "OK")


def test_vertical_seam_uses_geometry_and_exact_sources_outside_band():
    shape = (6, 20)
    a, b = np.full(shape, 10.0), np.full(shape, 30.0)
    valid_a = np.broadcast_to(np.arange(shape[1]) < 15, shape).copy()
    valid_b = np.broadcast_to(np.arange(shape[1]) > 4, shape).copy()
    result = blend_across_seam(a, b, _seam(shape, "vertical", 10), valid_a, valid_b, blend_half_width=2)
    assert result.status == "OK"
    assert result.source_side == "A_LOW_B_HIGH"
    assert np.array_equal(result.image[:, 7], a[:, 7])
    assert np.array_equal(result.image[:, 13], b[:, 13])
    assert np.all(result.image[:, 10] == 20.0)
    assert np.allclose(result.image[:, 9], 10.0 + 20.0 * (0.5 - np.sqrt(2) / 4))
    assert np.allclose(result.image[:, 11], 10.0 + 20.0 * (0.5 + np.sqrt(2) / 4))
    assert result.blend_pixels == 18


def test_horizontal_seam_can_reverse_source_order():
    shape = (20, 6)
    a, b = np.full(shape, 10.0), np.full(shape, 30.0)
    valid_a = np.broadcast_to((np.arange(shape[0]) > 4)[:, None], shape).copy()
    valid_b = np.broadcast_to((np.arange(shape[0]) < 15)[:, None], shape).copy()
    result = blend_across_seam(a, b, _seam(shape, "horizontal", 10), valid_a, valid_b, blend_half_width=2)
    assert result.status == "OK"
    assert result.source_side == "B_LOW_A_HIGH"
    assert np.all(result.image[7] == 30.0)
    assert np.all(result.image[13] == 10.0)
    assert np.all(result.image[10] == 20.0)


def test_single_source_valid_region_is_preserved_even_within_band():
    shape = (6, 20)
    a, b = np.full(shape, 10.0), np.full(shape, 30.0)
    valid_a = np.broadcast_to(np.arange(shape[1]) < 15, shape).copy()
    valid_b = np.broadcast_to(np.arange(shape[1]) > 4, shape).copy()
    valid_b[0, 10] = False
    valid_a[1, 10] = False
    result = blend_across_seam(a, b, _seam(shape, "vertical", 10), valid_a, valid_b, blend_half_width=2)
    assert result.status == "OK"
    assert result.image[0, 10] == 10.0
    assert result.image[1, 10] == 30.0
    assert result.image[0, 19] == 30.0


def test_missing_unique_exclusive_geometry_returns_explicit_status():
    shape = (6, 20)
    a, b = np.ones(shape), np.ones(shape) * 2
    both = np.ones(shape, dtype=bool)
    result = blend_across_seam(a, b, _seam(shape, "vertical", 10), both, both, blend_half_width=2)
    assert result.status == "AMBIGUOUS_SOURCE_SIDE"
    assert result.image is None


def test_finite_inputs_produce_finite_valid_output():
    shape = (6, 20)
    a, b = np.full(shape, 1e200), np.full(shape, -1e200)
    valid_a = np.broadcast_to(np.arange(shape[1]) < 15, shape).copy()
    valid_b = np.broadcast_to(np.arange(shape[1]) > 4, shape).copy()
    result = blend_across_seam(a, b, _seam(shape, "vertical", 10), valid_a, valid_b, blend_half_width=2)
    assert result.status == "OK"
    assert np.isfinite(result.image[result.valid_mask]).all()
