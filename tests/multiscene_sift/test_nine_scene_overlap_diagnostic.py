"""Synthetic tests for the nine-scene overlap network diagnostic.

Uses constructed footprint boxes — no real raster and no registration.
"""

from __future__ import annotations

import numpy as np
import pytest
from rasterio.coords import BoundingBox

from src.multiscene_sift import nine_scene_overlap_diagnostic as nsd
from src.multiscene_sift.models import Scene


def _scene(idx, name, cx, cy, size=0.30):
    left, right = cx - size / 2, cx + size / 2
    bottom, top = cy - size / 2, cy + size / 2
    return Scene(
        index=idx, name=name, directory="", band_paths={"B14": ""},
        crs="EPSG:32650",
        transforms={"B14": None},
        shapes={"B14": (100, 100)},
        nodata={"B14": 0.0},
        bounds={"B14": BoundingBox(left=left, bottom=bottom,
                                   right=right, top=top)},
    )


def _nine_scenes():
    names = list(nsd.SCENE_NAMES_9)
    return [
        _scene(0, names[0], 113.40, 36.60),
        _scene(1, names[1], 113.65, 36.50),
        _scene(2, names[2], 113.00, 36.40),
        _scene(3, names[3], 112.60, 36.00),
        _scene(4, names[4], 113.62, 36.72),
        _scene(5, names[5], 112.80, 36.80),
        _scene(6, names[6], 113.45, 36.75),
        _scene(7, names[7], 113.80, 36.20),
        _scene(8, names[8], 113.75, 36.35),
    ]


def test_nine_scenes_produce_thirty_six_pairs():
    scenes = _nine_scenes()
    overlaps = nsd.all_pair_overlaps(scenes)
    assert len(overlaps) == 36
    assert all(r["idx_i"] < r["idx_j"] for r in overlaps)


def test_duplicate_scene_name_rejected():
    names = list(nsd.SCENE_NAMES_9)
    assert nsd.validate_scene_names(names) is True
    assert nsd.validate_scene_names(names + [names[4]]) is False
    assert nsd.validate_scene_names(names[:-1]) is False


def test_bounds_intersection_correct():
    a = _scene(0, nsd.SCENE_NAMES_9[0], 113.4, 36.6, size=0.2)
    b = _scene(1, nsd.SCENE_NAMES_9[1], 113.48, 36.6, size=0.2)
    overlaps = nsd.all_pair_overlaps([a, b])
    r = overlaps[0]
    # boxes: a[113.30,113.50] b[113.38,113.58]; intersection width 0.08°
    # (approx in deg^2 units; we only check geometric consistency)
    assert r["intersection_exists"] is True
    assert r["intersection_width_m"] > 0.0
    assert r["intersection_area_m2"] > 0.0
    assert r["overlap_ratio_i"] == pytest.approx(
        r["intersection_area_m2"] / r["scene_i_area_m2"])


def test_no_overlap_is_zero():
    a = _scene(0, nsd.SCENE_NAMES_9[0], 113.4, 36.6)
    b = _scene(1, nsd.SCENE_NAMES_9[1], 114.5, 37.5)
    r = nsd.all_pair_overlaps([a, b])[0]
    assert r["intersection_exists"] is False
    assert r["intersection_area_m2"] == 0.0
    assert r["overlap_ratio_i"] == 0.0


def test_overlap_matrix_symmetric():
    scenes = _nine_scenes()
    overlaps = nsd.all_pair_overlaps(scenes)
    ratio, exists = nsd.overlap_matrices(overlaps, len(scenes))
    np.testing.assert_allclose(ratio, ratio.T, atol=1e-12)
    np.testing.assert_array_equal(exists, exists.T)


def test_overlap_matrix_diagonal():
    scenes = _nine_scenes()
    overlaps = nsd.all_pair_overlaps(scenes)
    ratio, exists = nsd.overlap_matrices(overlaps, len(scenes))
    np.testing.assert_allclose(np.diag(ratio), np.ones(9))
    np.testing.assert_array_equal(np.diag(exists), np.ones(9, dtype=int))


def test_neighbor_ranking():
    scenes = [
        _scene(0, nsd.SCENE_NAMES_9[0], 113.4, 36.6),
        _scene(1, nsd.SCENE_NAMES_9[1], 113.48, 36.60, size=0.2),
        _scene(4, nsd.SCENE_NAMES_9[4], 113.44, 36.64, size=0.2),
    ]
    overlaps = nsd.all_pair_overlaps(scenes)
    neigh = nsd.neighbors_of(scenes, overlaps, 0)
    assert len(neigh) == 2
    assert neigh[0]["overlap_ratio"] >= neigh[1]["overlap_ratio"]


def test_triangle_enumeration_includes_0_4_1():
    scenes = _nine_scenes()
    overlaps = nsd.all_pair_overlaps(scenes)
    triangles = nsd.enumerate_triangles(overlaps)
    assert (0, 1, 4) in triangles


def test_triangle_no_duplicates():
    scenes = _nine_scenes()
    overlaps = nsd.all_pair_overlaps(scenes)
    triangles = nsd.enumerate_triangles(overlaps)
    assert len(triangles) == len(set(triangles))


def test_triangle_priority_abc():
    scenes = _nine_scenes()
    overlaps = nsd.all_pair_overlaps(scenes)
    triangles = nsd.enumerate_triangles(overlaps)
    by_pair = {
        (r["idx_i"], r["idx_j"]): r["overlap_ratio_i"]
        for r in overlaps if r["intersection_exists"]
    }
    seen = set()
    for tri in triangles:
        pri, _, _ = nsd.triangle_priority(tri, by_pair, overlaps)
        if 0 in tri and 1 in tri and any(x in nsd.NEW_INDICES for x in tri):
            assert pri == "A"
        elif (0 in tri or 1 in tri) and any(x in nsd.NEW_INDICES for x in tri):
            assert pri == "B"
        else:
            assert pri == "C"
        seen.add(pri)
    # this synthetic geometry should produce all three classes
    assert seen == {"A", "B", "C"}


def test_recommended_pairs_prefer_scene0_new():
    scenes = _nine_scenes()
    overlaps = nsd.all_pair_overlaps(scenes)
    triangles = nsd.enumerate_triangles(overlaps)
    recs = nsd.select_recommended_pairs(scenes, overlaps, triangles, max_n=10)
    assert len(recs) <= 10
    assert len(recs) >= 3
    # scene0-new-neighbour pairs score highest
    assert any(r["probes_scene0"] and any(
        int(x) in nsd.NEW_INDICES for x in r["pair"].split("-")
    ) for r in recs[:3])