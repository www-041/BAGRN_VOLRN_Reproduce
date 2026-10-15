"""Selected-pair registration and cycle-consistency diagnostics (9 scenes).

Runs exactly the 8 fixed new direct SIFT pairs (same config source as the
original five-scene baseline), validates each direct edge independently,
normalises eligible edges into the already-validated canonical world frame,
and evaluates cycl background closures:

    C0 baseline = 0-1-4   (reused, no re-registration)
    C1           = 0-4-6
    C2           = 0-2-5
    C3           = 0-5-6
    C4           = 1-7-8

All closure math happens in the canonical world frame — pair-local /
match-view matrices are never multiplied directly.  No full 9-scene MST,
radiometric normalisation, or mosaicking is run.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import numpy as np

from src.multiscene_sift.models import Scene

logger = logging.getLogger(__name__)

# Fixed plan specs (Task 1)
SELECTED_NEW_PAIRS = [
    (0, 6),
    (4, 6),
    (0, 5),
    (2, 5),
    (5, 6),
    (1, 8),
    (1, 7),
    (7, 8),
]

TARGET_CYCLES: dict[str, tuple[int, int, int]] = {
    "C0_baseline": (0, 1, 4),
    "C1_scene0_ref4": (0, 4, 6),
    "C2_scene0_ref2": (0, 2, 5),
    "C3_scene0_new": (0, 5, 6),
    "C4_scene1_new": (1, 7, 8),
}

REUSED_OLD_PAIRS = [(0, 4), (0, 2), (0, 1), (1, 4)]

CYCLE_CLASS_THRESHOLDS = {
    "closed_px": 2.0,
    "small_residual_px": 5.0,
    "translation_std_max_px": 1.5,
    "linear_rotation_max_deg": 0.1,
    "linear_scale_max_diff": 1e-3,
}

REQUIRED_BASELINE_CONFIG_FIELDS = [
    "matcher", "registration_band", "match_max_side",
    "ransac_threshold", "random_seed",
]


def load_baseline_registration_config(five_scene_run_dir: str | Path) -> dict:
    """Freeze the configuration fields the new pairs must share.

    Reads the original five-scene run's ``run_config.json`` and merges the
    fixed SIFT constants used by the production runner.  Never guesses a
    missing key.
    """
    from src.multiscene_sift.loop_diagnostics import load_run_config
    from src.multiscene_sift.pairwise import (
        SIFT_NFEATURES, LOWE_RATIO, RANSAC_MAX_TRIALS,
        MIN_INLIERS, MIN_INLIER_RATIO,
    )

    config = load_run_config(five_scene_run_dir)
    missing = [k for k in REQUIRED_BASELINE_CONFIG_FIELDS if k not in config]
    if missing:
        raise ValueError(
            f"baseline run config is missing required fields {missing}; "
            "refusing to guess defaults"
        )
    return {
        "matcher": str(config["matcher"]).lower(),
        "registration_band": str(config["registration_band"]),
        "match_max_side": int(config["match_max_side"]),
        "ransac_threshold": float(config["ransac_threshold"]),
        "random_seed": int(config["random_seed"]),
        "source_run_dir": str(five_scene_run_dir),
        "fixed_sift_constants": {
            "nfeatures": SIFT_NFEATURES,
            "lowe_ratio": LOWE_RATIO,
            "ransac_model": "Affine",
            "ransac_max_trials": RANSAC_MAX_TRIALS,
            "minimum_inliers": MIN_INLIERS,
            "minimum_inlier_ratio": MIN_INLIER_RATIO,
        },
        "no_loftr_fallback": True,
        "no_threshold_relaxation": True,
    }


def run_selected_pair_registration(
    scene_i: Scene, scene_j: Scene, config: dict
):
    """Run one selected pair with the frozen baseline configuration.

    Delegates to the production ``register_pair`` (SIFT + RANSAC Affine).
    """
    from src.multiscene_sift.pairwise import register_pair

    assert config["matcher"] == "sift", "this round is SIFT-only by design"
    return register_pair(
        scene_i, scene_j,
        band=str(config["registration_band"]),
        match_max_side=int(config["match_max_side"]),
        ransac_threshold=float(config["ransac_threshold"]),
        random_seed=int(config["random_seed"]),
        matcher="sift",
    )


def registration_as_row(reg) -> dict:
    """Turn a PairwiseRegistration into the summary-row shape we store."""
    return {
        "idx_i": reg.idx_i,
        "idx_j": reg.idx_j,
        "status": reg.status,
        "raw_matches": reg.raw_matches,
        "inliers": reg.inliers,
        "inlier_ratio": reg.inlier_ratio,
        "coverage": reg.coverage,
        "residual_rmse": reg.residual_rmse,
        "residual_p95": reg.residual_p95,
        "residual_median": reg.residual_median,
        "pixel_matrix": reg.pair_pixel_matrix,
        "matcher": reg.matcher,
        "runtime_s": round(float(reg.runtime_sec), 4),
        "has_common_transform": bool(reg.pair_common_transform is not None),
    }


def direct_edge_world_transform(
    pair_rows: list[dict],
    i: int,
    j: int,
    common_transform,
    match_view: dict,
) -> np.ndarray:
    """Canonical world transform ``T_world_i_from_j`` (scene j -> i).

    Reuses the already-validated frame-normalising conversion
    (match-view -> common -> world, conjugated consistently).
    """
    from src.multiscene_sift.frame_diagnostics import world_transform_from_saved_pair

    return world_transform_from_saved_pair(
        i, j, pair_rows, common_transform, match_view, adjust_frame=True
    )


def cycle_edge_eligibility(
    cycle: tuple[int, int, int],
    eligible_edges: set,
) -> tuple[bool, list]:
    """Whether every edge of a cycle is present (undirected) in *eligible*."""
    pairs = [(cycle[0], cycle[1]), (cycle[1], cycle[2]), (cycle[0], cycle[2])]
    missing = [
        p for p in pairs
        if p not in eligible_edges and (p[1], p[0]) not in eligible_edges
    ]
    return (len(missing) == 0, missing)


def compose_cycle_world(
    cycle: tuple[int, int, int],
    world_edges: dict[tuple[int, int], np.ndarray],
) -> np.ndarray:
    """Three-edge closure matrix for traversal ``a -> b -> c -> a``.

    ``world_edges[(p, q)] = T_world_p_from_q`` (maps scene *q* into scene *p*
    world frame).  Missing / reversed directions are handled transparently.
    Raises ``KeyError`` if an edge is absent.
    """
    a, b, c = cycle

    def _t(from_scene: int, to_scene: int) -> np.ndarray:
        # maps *to_scene* -> *from_scene* (returns the stored direction or its inverse)
        if (from_scene, to_scene) in world_edges:
            return world_edges[(from_scene, to_scene)]
        if (to_scene, from_scene) in world_edges:
            return np.linalg.inv(world_edges[(to_scene, from_scene)])
        raise KeyError(f"no stored edge between {from_scene} and {to_scene}")

    return _t(a, c) @ _t(c, b) @ _t(b, a)


def sample_cycle_world_points(
    scenes: list[Scene],
    cycle: tuple[int, int, int],
    band: str = "B14",
    n_per_axis: int = 3,
) -> np.ndarray:
    """Deterministic world points inside the cycle nodes' common overlap.

    Falls back to the overlap of the cycle's first two nodes when a triple
    intersection is empty or trivial.
    """
    nodes = list(cycle)

    def _overlap(a: int, b: int) -> tuple[float, float, float, float] | None:
        ba = scenes[a].bounds[band]
        bb = scenes[b].bounds[band]
        left = max(ba.left, bb.left)
        bottom = max(ba.bottom, bb.bottom)
        right = min(ba.right, bb.right)
        top = min(ba.top, bb.top)
        if right <= left or top <= bottom:
            return None
        return left, bottom, right, top

    region = _overlap(nodes[0], nodes[1])
    for k in range(1, len(nodes)):
        r = _overlap(nodes[0], nodes[k])
        if r is None:
            continue
        if region is None:
            region = r
        else:
            region = (max(region[0], r[0]), max(region[1], r[1]),
                      min(region[2], r[2]), min(region[3], r[3]))
    if region is None or (region[2] - region[0]) < 1.0 or \
            (region[3] - region[1]) < 1.0:
        return np.empty((0, 2))
    left, bottom, right, top = region
    fracs = [(k + 0.5) / n_per_axis for k in range(n_per_axis)]
    pts = []
    for fy in fracs:
        for fx in fracs:
            pts.append((left + fx * (right - left),
                        bottom + fy * (top - bottom)))
    return np.asarray(pts, dtype=np.float64)


def evaluate_cycle_on_points(
    closure: np.ndarray,
    sample_world_points: np.ndarray,
    pixel_size_m: float = 14.0,
) -> dict:
    """Displacement induced by the closure matrix on world sample points."""
    pts = np.asarray(sample_world_points, dtype=np.float64)
    if len(pts) == 0:
        return {"n_points": 0}
    h = np.hstack([pts, np.ones((len(pts), 1))])
    moved = (closure @ h.T).T[:, :2]
    d = (moved - pts) / pixel_size_m
    mag = np.linalg.norm(d, axis=1)
    return {
        "n_points": int(len(pts)),
        "median_px": float(np.median(mag)),
        "rmse_px": float(np.sqrt(np.mean(mag**2))),
        "p95_px": float(np.percentile(mag, 95)),
        "max_px": float(np.max(mag)),
        "std_px": float(np.std(mag)),
        "dx_mean_px": float(np.mean(d[:, 0])),
        "dy_mean_px": float(np.mean(d[:, 1])),
    }


def linear_components(closure: np.ndarray) -> dict:
    """Rotation / scale / shear of a closure matrix (for classification)."""
    a, b = closure[0, 0], closure[0, 1]
    c, d = closure[1, 0], closure[1, 1]
    sx = float(math.hypot(a, c))
    sy = float(math.hypot(b, d))
    rot = math.degrees(math.atan2(c, a))
    rot2 = math.degrees(math.atan2(-b, d))
    rot = (rot + rot2) / 2.0
    shear = math.degrees(math.atan2(-b * a - d * c, a * d - b * c))
    return {"rotation_deg": rot, "scale_x": sx, "scale_y": sy,
            "shear_deg": shear}


def classify_cycle(closure: np.ndarray, stats: dict) -> str:
    """Descriptive cycle state (thresholds exposed, not hidden)."""
    th = CYCLE_CLASS_THRESHOLDS
    med = float(stats.get("median_px", float("nan")))
    if not np.isfinite(med):
        return "UNAVAILABLE"
    if med <= th["closed_px"]:
        return "CLOSED"
    if med <= th["small_residual_px"]:
        return "SMALL_RESIDUAL"
    lin = linear_components(closure)
    translational = (
        float(stats.get("std_px", float("inf"))) <= th["translation_std_max_px"]
        and abs(lin["rotation_deg"]) <= th["linear_rotation_max_deg"]
        and abs(lin["scale_x"] - 1.0) <= th["linear_scale_max_diff"]
        and abs(lin["scale_y"] - 1.0) <= th["linear_scale_max_diff"]
    )
    return "LARGE_SYSTEMATIC_TRANSLATION" if translational \
        else "NON_TRANSLATIONAL_INCONSISTENCY"


def neighbour_new_pairs_count(
    pair_results: list[dict], anchor: int
) -> tuple[int, int]:
    """(ok, total) new pairs touching *anchor*."""
    ok = tot = 0
    for r in pair_results:
        if anchor not in (r["idx_i"], r["idx_j"]):
            continue
        tot += 1
        if r["status"] == "OK":
            ok += 1
    return ok, tot


def _is_available(status) -> bool:
    return status is not None and not (
        isinstance(status, str) and status.startswith("UNAVAILABLE")
    )


def classify_network_inconsistency(evidence: dict) -> str:
    """Network-level state from per-cycle + edge evidence (Task 12)."""
    c0 = evidence.get("c0_status")
    scene0_cycles = [evidence.get(k) for k in ("c1_status", "c2_status", "c3_status")]
    c4 = evidence.get("c4_status")
    available_s0 = [s for s in scene0_cycles if _is_available(s)]
    closed_s0 = [s for s in available_s0 if s in ("CLOSED", "SMALL_RESIDUAL")]
    n_new_ok = evidence.get("new_pair_ok", 0)
    n_new_tried = evidence.get("new_pair_tried", 8)
    if n_new_ok < 4 or len(available_s0) < 2:
        return "NEW_EDGES_INSUFFICIENT"

    c4_ok = _is_available(c4) and c4 in ("CLOSED", "SMALL_RESIDUAL")
    c0_bad = c0 in ("LARGE_SYSTEMATIC_TRANSLATION",
                    "NON_TRANSLATIONAL_INCONSISTENCY", "SMALL_RESIDUAL") \
        and c0 not in ("CLOSED",)

    if c0_bad and len(closed_s0) >= 2 and c4_ok:
        return "PAIR_0_1_OVERLAP_SPECIFIC_SUPPORTED"

    large_s0 = [s for s in available_s0
                if s in ("LARGE_SYSTEMATIC_TRANSLATION",
                         "NON_TRANSLATIONAL_INCONSISTENCY")]
    if len(large_s0) >= 2 and c4_ok:
        return "SCENE0_LEVEL_INCONSISTENCY_SUSPECT"

    if c4 in ("LARGE_SYSTEMATIC_TRANSLATION", "NON_TRANSLATIONAL_INCONSISTENCY") \
            and len(closed_s0) >= 1:
        return "MIXED_OR_UNDERDETERMINED"

    if c4 in ("LARGE_SYSTEMATIC_TRANSLATION", "NON_TRANSLATIONAL_INCONSISTENCY") \
            and len(closed_s0) < 1:
        return "SCENE1_LEVEL_INCONSISTENCY_SUSPECT"

    return "MIXED_OR_UNDERDETERMINED"


def build_scene_evidence(
    cycle_states: dict[str, tuple[str, dict | None]],
    pair_results: list[dict],
) -> dict:
    """Assemble the scene0 / scene1 evidence table (Task 11)."""
    evidence: dict[str, Any] = {
        "scene0_cycles": {
            k: cycle_states.get(k, ("UNAVAILABLE", None))[0]
            for k in ("C1_scene0_ref4", "C2_scene0_ref2", "C3_scene0_new")
        },
        "scene1_cycles": {
            "C4_scene1_new": cycle_states.get("C4_scene1_new",
                                              ("UNAVAILABLE", None))[0]
        },
        "baseline_contrast": {
            "C0_baseline": cycle_states.get("C0_baseline", ("UNAVAILABLE", None))[0]
        },
        "scene0_new_edges": {
            f"{r['idx_i']}-{r['idx_j']}": r["status"]
            for r in pair_results
            if (r["idx_i"], r["idx_j"]) in ((0, 5), (0, 6), (5, 0), (6, 0))
        },
        "scene1_new_edges": {
            f"{r['idx_i']}-{r['idx_j']}": r["status"]
            for r in pair_results
            if (r["idx_i"], r["idx_j"]) in ((1, 7), (1, 8), (7, 1), (8, 1))
        },
    }
    return evidence