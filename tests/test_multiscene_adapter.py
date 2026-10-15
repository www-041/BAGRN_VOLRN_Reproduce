from __future__ import annotations

import numpy as np
from shapely.geometry import box

from src.seam_local.adapter import (
    LABEL_UNRESOLVED,
    aggregate_labels_with_ties,
    blend_labeled_scenes,
    normalize_overlap_pairs,
    run_multiscene_adapter,
)


def _scene_stack(n: int = 5, shape: tuple[int, int] = (32, 48)):
    rows, cols = np.indices(shape)
    scenes = []
    masks = []
    for i in range(n):
        left = i * 4
        right = min(shape[1], left + 32)
        mask = np.zeros(shape, dtype=bool)
        mask[4:28, left:right] = True
        masks.append(mask)
        scenes.append((rows * 0.1 + cols + i).astype(np.float64))
    return np.asarray(scenes), np.asarray(masks)


def test_normalize_pairs_preserves_spatial_edges_and_registration_flag():
    rows = normalize_overlap_pairs(
        [{"scene_i": 0, "scene_j": 2, "shared_valid_pixel_count": 4,
          "registration_edge_accepted": False}, (1, 2)]
    )
    assert [(r.scene_i, r.scene_j) for r in rows] == [(0, 2), (1, 2)]
    assert rows[0].registration_edge_accepted is False
    assert rows[0].shared_valid_pixel_count == 4


def test_generic_adapter_supports_n2_n3_n5_and_missing_pair_ownership():
    for n in (2, 3, 5):
        scenes, masks = _scene_stack(n)
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        result = run_multiscene_adapter(
            scenes, masks, pairs, footprints=[box(i * 4, 4, i * 4 + 32, 28) for i in range(n)]
        )
        assert result.scene_count == n
        assert set(result.pair_results) == {f"{i:02d}_{j:02d}" for i, j in pairs}
        assert result.labels.shape == masks.shape[1:]
        assert result.diagnostics["union_valid_pixels"] > 0
    # An unresolved ownership pair is retained and delegated to global labels.
    scenes, masks = _scene_stack(3)
    result = run_multiscene_adapter(scenes, masks, [(0, 1), (1, 2)])
    assert result.pair_results["00_01"].source_side_status == "REQUIRES_MULTISCENE_LABELING"


def test_containment_and_invalid_candidate_do_not_use_scene_index_tiebreak():
    scenes, masks = _scene_stack(2)
    footprints = [box(0, 0, 48, 32), box(8, 8, 16, 16)]
    result = run_multiscene_adapter(scenes, masks, [(0, 1)], footprints=footprints)
    assert result.pair_results["00_01"].source_side_status == "REQUIRES_MULTISCENE_LABELING"

    labels, methods, diag = aggregate_labels_with_ties(
        masks=np.ones((2, 1, 3), dtype=bool),
        pairwise_fields={},
        raw_edt=np.array([[[1.2, 1.2, 1.2]], [[1.4, 1.4, 0.5]]]),
    )
    assert labels[0, 0] == 1
    assert labels[0, 1] == 1
    assert labels[0, 2] == 0
    assert diag["resolved_by_raw_edt"] == 2
    invalid = np.array([[[True]], [[False]]])
    labels, _, _ = aggregate_labels_with_ties(invalid, {}, np.array([[[1.0]], [[99.0]]]))
    assert labels[0, 0] == 0


def test_label_and_blend_are_order_invariant_and_tile_boundary_safe():
    scenes, masks = _scene_stack(5, (33, 49))
    pairs = [(i, j) for i in range(5) for j in range(i + 1, 5)]
    a = run_multiscene_adapter(scenes, masks, pairs)
    permutation = [3, 0, 4, 1, 2]
    b = run_multiscene_adapter(scenes[permutation], masks[permutation],
                               [(i, j) for i in range(5) for j in range(i + 1, 5)])
    remapped = np.full_like(b.labels, LABEL_UNRESOLVED)
    for new, old in enumerate(permutation):
        remapped[b.labels == new] = old
    assert np.array_equal(a.labels, remapped)
    one_tile = blend_labeled_scenes(scenes, masks, a.labels, tile_size=16, halo=4)
    two_tiles = blend_labeled_scenes(scenes, masks, a.labels, tile_size=17, halo=4)
    assert np.allclose(one_tile, two_tiles, equal_nan=True, atol=1e-6)
