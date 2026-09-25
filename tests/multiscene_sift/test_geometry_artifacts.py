"""Tests for persisted point-level Global-ready pair geometry bundles."""

from __future__ import annotations

import json

import numpy as np
import pytest
from rasterio.transform import Affine

from src.multiscene_sift.geometry_artifacts import (
    load_pair_geometry_bundle,
    save_pair_geometry_bundle,
    validate_pair_geometry_bundle,
)


def _save(tmp_path, **overrides):
    values = {
        "matcher": "sift",
        "idx_i": 0,
        "idx_j": 1,
        "scene_i": "scene_0",
        "scene_j": "scene_1",
        "accepted": True,
        "status": "OK",
        "inlier_ref_xy": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "inlier_tgt_xy": np.array([[0.5, 2.0], [2.5, 4.0]], dtype=np.float32),
        "pair_pixel_matrix": np.eye(3, dtype=np.float64),
        "pair_common_transform": Affine(14.0, 0.0, 100.0, 0.0, -14.0, 200.0),
        "crs": "EPSG:32650",
        "pixel_size_x_m": 14.0,
        "pixel_size_y_m": 14.0,
        "coordinate_frame": "pair_common_grid",
        "transform_direction": "target_to_reference",
        "config_path": "04_frozen_five_scene_config_1024.json",
        "config_sha256": "abc123",
        "match_max_side": 1024,
    }
    values.update(overrides)
    return save_pair_geometry_bundle(tmp_path / "geometry", **values)


def test_pair_geometry_bundle_round_trip_preserves_arrays_and_metadata(tmp_path):
    sidecar = _save(tmp_path)

    assert validate_pair_geometry_bundle(sidecar)["valid"] is True
    bundle = load_pair_geometry_bundle(sidecar)

    np.testing.assert_array_equal(
        bundle["inlier_ref_xy"], np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    )
    np.testing.assert_array_equal(
        bundle["inlier_tgt_xy"], np.array([[0.5, 2.0], [2.5, 4.0]], dtype=np.float32)
    )
    assert bundle["inlier_ref_xy"].dtype == np.dtype("float32")
    assert bundle["metadata"]["coordinate_frame"] == "pair_common_grid"
    assert bundle["metadata"]["transform_direction"] == "target_to_reference"
    assert bundle["metadata"]["match_max_side"] == 1024
    assert bundle["pair_common_transform"].a == pytest.approx(14.0)
    assert bundle["pair_common_transform"].f == pytest.approx(200.0)


def test_pair_geometry_bundle_rejects_mismatched_point_lengths(tmp_path):
    with pytest.raises(ValueError, match="same number of rows"):
        _save(tmp_path, inlier_tgt_xy=np.zeros((1, 2), dtype=np.float32))


def test_pair_geometry_bundle_rejects_nonfinite_points(tmp_path):
    with pytest.raises(ValueError, match="finite"):
        _save(tmp_path, inlier_ref_xy=np.array([[np.nan, 1.0]], dtype=np.float32))


def test_pair_geometry_bundle_rejects_wrong_frame_on_validation(tmp_path):
    with pytest.raises(ValueError, match="pair_common_grid"):
        _save(tmp_path, coordinate_frame="match_view")


def test_pair_geometry_bundle_rejects_missing_npz(tmp_path):
    sidecar = _save(tmp_path)
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    npz = sidecar.parent / payload["npz_path"]
    npz.unlink()

    with pytest.raises(FileNotFoundError, match="NPZ"):
        load_pair_geometry_bundle(sidecar)


def test_pair_geometry_bundle_does_not_overwrite_existing_files(tmp_path):
    sidecar = _save(tmp_path)

    with pytest.raises(FileExistsError):
        _save(tmp_path)
