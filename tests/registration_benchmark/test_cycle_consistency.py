from __future__ import annotations

import numpy as np

from src.multiscene_sift.global_geometry_audit import compute_cycle_diagnostics


def _tx(dx: float) -> np.ndarray:
    return np.array([[1.0, 0.0, dx], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])


def test_perfect_pair_transform_loop_has_zero_closure() -> None:
    # Sidecar pair matrices map j -> i. All identity matrices form a perfect loop.
    bundles = {(0, 1): {"pair_pixel_matrix": np.eye(3)},
               (1, 2): {"pair_pixel_matrix": np.eye(3)},
               (0, 2): {"pair_pixel_matrix": np.eye(3)}}
    row = compute_cycle_diagnostics(bundles, [(0, 1), (1, 2), (0, 2)], 3)[0]
    assert row["matrix_identity_error"] == 0.0
    assert row["translation_closure_px"] == 0.0


def test_pair_transform_loop_reports_three_pixel_translation_closure() -> None:
    bundles = {(0, 1): {"pair_pixel_matrix": np.eye(3)},
               (1, 2): {"pair_pixel_matrix": np.eye(3)},
               # The stored 0-2 matrix maps scene 2 to scene 0.
               (0, 2): {"pair_pixel_matrix": _tx(3.0)}}
    row = compute_cycle_diagnostics(bundles, [(0, 1), (1, 2), (0, 2)], 3)[0]
    assert row["translation_closure_px"] == 3.0
    assert row["linear_part_error"] == 0.0
