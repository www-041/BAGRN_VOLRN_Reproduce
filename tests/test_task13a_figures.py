"""Selection and display reconstruction checks for Task13A figures."""

import numpy as np
import subprocess
import sys
from pathlib import Path

from scripts.build_task13a_figures import choose_examples, reconstruct_local_window


def test_direct_script_check_only_runs_without_regenerating_figures():
    script = Path(__file__).resolve().parents[1] / "scripts/build_task13a_figures.py"
    result = subprocess.run([sys.executable, str(script), "--check-only"],
                            cwd=script.parent.parent, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "verified three figures" in result.stdout


def test_choose_examples_uses_finite_pass_v2_mae_and_pair_id_median_tie_break():
    rows = [
        {"pair_id": "01_02", "status": "PASS", "metric_actual_seam_v2_mae": "4"},
        {"pair_id": "00_01", "status": "PASS", "metric_actual_seam_v2_mae": "3"},
        {"pair_id": "00_02", "status": "PASS", "metric_actual_seam_v2_mae": "1"},
        {"pair_id": "02_04", "status": "PASS", "metric_actual_seam_v2_mae": "2"},
        {"pair_id": "01_03", "status": "PASS", "metric_actual_seam_v2_mae": "nan"},
        {"pair_id": "00_04", "status": "AMBIGUOUS_SOURCE_SIDE", "metric_actual_seam_v2_mae": "100"},
    ]
    assert choose_examples(rows) == {"best": ("00_02", 1.0),
                                     "median": ("00_01", 3.0),
                                     "worst": ("01_02", 4.0)}


def test_reconstruct_local_window_has_cosine_center_and_zero_boundary():
    a = np.full((1, 5), 10.0)
    b = np.full((1, 5), 20.0)
    path = np.array([[0, 2]])
    rows = [{"center_arc": "0", "a_a": "2", "b_a": "0",
             "a_b": "1", "b_b": "0"}]
    ca, cb = reconstruct_local_window(a, b, np.ones_like(a, bool),
                                      np.ones_like(b, bool), path, rows,
                                      (0, 0), "vertical", half_width=2)
    np.testing.assert_allclose(ca, [[10, 15, 20, 15, 10]])
    np.testing.assert_allclose(cb, b)
