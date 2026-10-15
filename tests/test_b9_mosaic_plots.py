import importlib
import warnings

import numpy as np


plots = importlib.import_module("scripts.plot_b9_mosaic_audit")


def test_checkerboard_alternates_tiles_without_changing_input_shape():
    first = np.full((4, 4), 10, dtype=np.uint8)
    second = np.full((4, 4), 20, dtype=np.uint8)

    result = plots.make_checkerboard(first, second, tiles=2)

    assert result.shape == first.shape
    assert set(np.unique(result)) == {10, 20}
    assert result[0, 0] == result[0, 1] == result[1, 0]
    assert result[0, 0] != result[0, 2]
    assert result[0, 0] != result[2, 0]


def test_gradient_overlay_is_rgb_and_finite():
    first = np.arange(16, dtype=np.float32).reshape(4, 4)
    second = np.flipud(first)

    result = plots.gradient_overlay(first, second)

    assert result.shape == (4, 4, 3)
    assert result.dtype == np.uint8
    assert np.isfinite(result).all()


def test_display_stretch_converts_nonfinite_nodata_to_zero():
    image = np.array([[1.0, np.nan], [3.0, 4.0]], dtype=np.float32)
    mask = np.array([[True, False], [True, True]])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = plots._stretch(image, mask)

    assert result.dtype == np.uint8
    assert result[0, 1] == 0
    assert not caught
