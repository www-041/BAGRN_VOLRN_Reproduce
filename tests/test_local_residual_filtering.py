import numpy as np


def _controls(dx, dy=None, confidence=None):
    dx = np.asarray(dx, dtype=float)
    dy = np.zeros_like(dx) if dy is None else np.asarray(dy, dtype=float)
    n = len(dx)
    if confidence is None:
        confidence = np.full(n, 0.9)
    points = np.column_stack([np.arange(n) % 5, np.arange(n) // 5])
    return {
        "points_xy": points.astype(float),
        "residual_dx": dx,
        "residual_dy": dy,
        "confidence": np.asarray(confidence, dtype=float),
    }


def test_smooth_large_local_residual_is_not_rejected_by_fixed_three_pixel_rule():
    from src.coregistration import filter_local_residual_controls

    controls = _controls([-5.0, -6.0, -7.0, -8.0, -7.0, -6.0, -5.0, -6.0])
    result = filter_local_residual_controls(
        controls, confidence_threshold=0.6, mad_scale=3.0,
        hard_max_shift=8.0, neighbor_radius=3.0,
    )

    assert result["n_valid"] == 8
    assert all(reason is None for reason in result["reject_reasons"])


def test_low_confidence_extreme_phase_peak_is_rejected():
    from src.coregistration import filter_local_residual_controls

    controls = _controls(
        [-5.0, -6.0, -7.0, 120.0], confidence=[0.9, 0.9, 0.9, 0.1]
    )
    result = filter_local_residual_controls(
        controls, confidence_threshold=0.6, mad_scale=3.0,
        hard_max_shift=8.0, neighbor_radius=3.0,
    )

    assert result["n_valid"] == 3
    assert result["reject_reasons"][-1] == "low_confidence"


def test_local_search_bound_is_distinct_from_final_component_cap(monkeypatch):
    from src import coregistration

    seen = []

    def fake_collect(*args, **kwargs):
        seen.append(kwargs["max_global_shift"])
        return [], {"total": 0}

    monkeypatch.setattr(coregistration, "collect_block_matches", fake_collect)
    coregistration.rematch_pair_on_registered(
        np.ones((64, 64)), np.ones((64, 64)),
        None, None, None, None, max_residual_shift=3.0,
        block_size=16, confidence_threshold=0.6,
        local_search_max_shift=12.0,
    )

    assert seen
    assert all(value == 12.0 for value in seen)
