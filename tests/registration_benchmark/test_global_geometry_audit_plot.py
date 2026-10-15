from __future__ import annotations

import numpy as np

from scripts.plot_b9_global_geometry_audit import plot_residual_cdf


def test_residual_cdf_smoke_writes_nonempty_png(tmp_path) -> None:
    path = plot_residual_cdf({"MST": np.array([0.0, 1.0]), "Translation-L2": np.array([0.2, 0.8])}, tmp_path / "cdf.png")
    assert path.exists()
    assert path.stat().st_size > 0
