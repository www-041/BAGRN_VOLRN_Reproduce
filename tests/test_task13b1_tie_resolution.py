import numpy as np

from src.seam_local.multiscene_label import LABEL_UNRESOLVED
from src.seam_local.tie_resolution import resolve_unresolved_geometry


def _run(unclipped, raw, old=None, masks=None, p95=None):
    u = np.asarray(unclipped, dtype=float)[:, None, None]
    r = np.asarray(raw, dtype=float)[:, None, None]
    if masks is None:
        masks = np.ones_like(u, dtype=bool)
    if old is None:
        old = np.array([[LABEL_UNRESOLVED]], dtype=np.int16)
    return resolve_unresolved_geometry(old, masks, np.zeros_like(u), r, np.ones(u.shape[0]) if p95 is None else np.asarray(p95))


def test_unclipped_normalized_breaks_clipped_one_tie():
    result = _run([1.1, 1.4], [110, 140])
    assert result.labels[0, 0] == 1
    assert result.resolved_by_unclipped == 1


def test_three_candidate_unclipped_winner():
    result = _run([1.2, 1.2, 1.5], [120, 120, 150])
    assert result.labels[0, 0] == 2


def test_raw_edt_breaks_normalized_tie():
    result = _run([1.2, 1.2], [300, 420], p95=[250, 350])
    assert result.labels[0, 0] == 1
    assert result.resolved_by_raw_edt == 1


def test_exact_geometry_tie_remains_unresolved():
    result = _run([1.2, 1.2], [420, 420])
    assert result.labels[0, 0] == LABEL_UNRESOLVED
    assert result.still_unresolved == 1


def test_old_resolved_pixels_are_immutable():
    old = np.array([[2, LABEL_UNRESOLVED]], dtype=np.int16)
    masks = np.ones((3, 1, 2), bool)
    result = resolve_unresolved_geometry(old, masks, np.zeros_like(masks, float), np.ones_like(masks, float), np.ones(3))
    assert result.labels[0, 0] == 2
    assert result.changed_old_resolved_pixels == 0


def test_invalid_candidate_cannot_win():
    masks = np.array([[[True]], [[False]]])
    result = resolve_unresolved_geometry(np.array([[-1]], np.int16), masks, np.zeros_like(masks, float), np.array([[[1.0]], [[999.0]]]), np.ones(2))
    assert result.labels[0, 0] == 0


def test_scene_order_permutation_maps_back_identically():
    old = np.array([[-1]], np.int16)
    masks = np.ones((3, 1, 1), bool)
    raw = np.array([[[100.0]], [[200.0]], [[150.0]]])
    p95 = np.ones(3)
    base = resolve_unresolved_geometry(old, masks, np.zeros_like(raw), raw, p95).labels[0, 0]
    perm = [2, 0, 1]
    result = resolve_unresolved_geometry(old, masks[perm], np.zeros_like(raw)[perm], raw[perm], p95[perm])
    assert perm[int(result.labels[0, 0])] == base


def test_no_radiometric_inputs_are_required():
    result = _run([1.01, 1.02], [101, 102])
    assert result.labels[0, 0] == 1
