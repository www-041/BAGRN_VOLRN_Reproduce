import numpy as np
import pytest

from src.benchmark_v1_lsstaa.core import solve_joint
from src.benchmark_v1_lsstaa.diagnostics import direct_joint_solution


def test_direct_and_normal_use_identical_observation_equations():
    x = np.linspace(50, 300, 150)
    a = (x - 7) / 1.1
    b = (x + 11) / .9
    edges = [(0, 1, x, a), (1, 2, a, b), (0, 2, x, b)]
    normal = solve_joint(3, 0, edges)
    direct = direct_joint_solution(3, 0, edges)
    assert direct["rank"] == 4
    assert direct["gains"] == pytest.approx(normal.gains, abs=1e-9)
    assert direct["offsets"] == pytest.approx(normal.offsets, abs=1e-8)
    assert direct["residual_rmse"] < 1e-9
