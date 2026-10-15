"""Independent direct least-squares check of the pinned normal equations."""
from __future__ import annotations

import numpy as np


def direct_joint_solution(scene_count: int, control: int, edges: list) -> dict:
    others = [scene for scene in range(scene_count) if scene != control]
    positions = {scene: 2 * k for k, scene in enumerate(others)}
    rows, observations = [], []
    edge_counts = []
    for i, j, x, y in edges:
        x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
        if len(x) != len(y) or not len(x):
            raise ValueError("EMPTY_EDGE_TIES")
        design = np.zeros((len(x), 2 * len(others)), dtype=np.float64)
        if i != control:
            design[:, positions[i]] = x
            design[:, positions[i] + 1] = 1
        if j != control:
            design[:, positions[j]] = -y
            design[:, positions[j] + 1] = -1
        rows.append(design)
        observations.append(y - x)
        edge_counts.append({"scene_i": i, "scene_j": j, "subset_count": len(x),
                            "x_min": float(x.min()), "x_max": float(x.max()),
                            "y_min": float(y.min()), "y_max": float(y.max())})
    if not rows:
        raise ValueError("NO_OBSERVATIONS")
    design = np.vstack(rows)
    observation = np.concatenate(observations)
    delta, _, rank, singular = np.linalg.lstsq(design, observation, rcond=None)
    if rank != design.shape[1]:
        raise ValueError(f"RANK_DEFICIENT:{rank}/{design.shape[1]}")
    gains = np.ones(scene_count)
    offsets = np.zeros(scene_count)
    for scene, pos in positions.items():
        gains[scene] += delta[pos]
        offsets[scene] = delta[pos + 1]
    residual = design @ delta - observation
    return {"gains": gains, "offsets": offsets, "rank": int(rank),
            "design_condition": float(singular[0] / singular[-1]),
            "residual_rmse": float(np.sqrt(np.mean(residual ** 2))),
            "max_abs_residual": float(np.max(np.abs(residual))),
            "tie_count": len(observation), "edge_counts": edge_counts}
