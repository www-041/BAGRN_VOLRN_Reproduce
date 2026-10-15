import numpy as np

from src.multiscene_sift.mosaic_protocol import (
    RUN_KEYS,
    build_canonical_output_grid,
    same_grid,
    transformed_scene_bounds,
)


def _record(left=0.0, top=100.0, shape=(10, 20)):
    return {
        "shape": list(shape),
        "left": left,
        "top": top,
        "transform": [1.0, 0.0, left, 0.0, -1.0, top],
    }


def _identity_transform_sets(scene_count=5):
    return {
        key: {index: np.eye(3) for index in range(scene_count)}
        for key in RUN_KEYS
    }


def test_transformed_scene_bounds_apply_positive_pixel_translation():
    record = _record()
    translated = np.array(
        [[1.0, 0.0, 2.0], [0.0, 1.0, 3.0], [0.0, 0.0, 1.0]]
    )

    assert transformed_scene_bounds(record, translated) == (2.0, 93.0, 22.0, 103.0)


def test_canonical_grid_covers_all_transformed_scene_bounds():
    records = [_record(left=index * 20.0) for index in range(5)]
    transforms = _identity_transform_sets()
    grid = build_canonical_output_grid(records, transforms, 1.0, "EPSG:32650")

    assert grid["bounds"] == [0.0, 90.0, 100.0, 100.0]
    assert grid["width"] == 100
    assert grid["height"] == 10
    assert grid["transform"] == [1.0, 0.0, 0.0, 0.0, -1.0, 100.0]


def test_canonical_grid_identity_has_explicit_pixel_size():
    grid = build_canonical_output_grid(
        [_record(left=index * 20.0) for index in range(5)],
        _identity_transform_sets(),
        14.0,
        "EPSG:32650",
    )

    assert grid["pixel_size"] == 14.0
    assert grid["grid_identity"]["pixel_size"] == 14.0


def test_all_eight_runs_share_the_same_grid_identity():
    records = [_record(left=index * 20.0) for index in range(5)]
    grids = [
        build_canonical_output_grid(records, _identity_transform_sets(), 14.0, "EPSG:32650")
        for _ in RUN_KEYS
    ]

    assert all(same_grid(grids[0], grid) for grid in grids[1:])
