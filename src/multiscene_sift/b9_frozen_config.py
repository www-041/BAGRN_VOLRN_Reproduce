"""Build the user-confirmed, immutable B9 five-scene configuration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from src.multiscene_sift.pairwise import (
    LOWE_RATIO,
    MIN_INLIER_RATIO,
    MIN_INLIERS,
    RANSAC_MAX_TRIALS,
    SIFT_NFEATURES,
)
from src.registration_benchmark.geometry import (
    MAX_RANSAC_TRIALS,
    MIN_INLIER_RATIO as GEOMETRY_MIN_INLIER_RATIO,
    MIN_INLIERS as GEOMETRY_MIN_INLIERS,
    RANSAC_RANDOM_SEED,
    RANSAC_RESIDUAL_THRESHOLD,
)


MANUAL_SELECTION_MODE = "manual_confirmation"
MATCHERS = ["sift", "loftr", "efficient_loftr", "lightglue_disk"]
GLOBAL_METHODS = ["mst", "equal_l2_translation"]


def build_frozen_config(
    records: list[dict],
    pairs: list[dict],
    manifest_indices: Iterable[int],
) -> dict:
    """Create a B9 config from exact manifest indices, without re-ranking.

    The indices are intentionally validated against the supplied manifest. An
    invalid manual selection is an error; it is never silently substituted by
    a ranked or nearby scene.
    """
    selected_indices = [int(index) for index in manifest_indices]
    if len(selected_indices) != 5 or len(set(selected_indices)) != 5:
        raise ValueError("exactly five unique manifest indices are required")
    if any(index < 0 or index >= len(records) for index in selected_indices):
        raise ValueError(
            f"manifest indices are outside the discovered manifest: {selected_indices}"
        )

    selected = set(selected_indices)
    selected_records = []
    for manifest_index in selected_indices:
        scene = dict(records[manifest_index])
        scene["manifest_index"] = manifest_index
        selected_records.append(scene)

    selected_edges = []
    for pair in pairs:
        if not pair.get("has_overlap"):
            continue
        if pair["idx_i"] not in selected or pair["idx_j"] not in selected:
            continue
        edge = dict(pair)
        edge["manifest_idx_i"] = int(edge.pop("idx_i"))
        edge["manifest_idx_j"] = int(edge.pop("idx_j"))
        selected_edges.append(edge)

    if not selected_edges:
        raise ValueError("manual five-scene selection has no geographic overlap edges")

    return {
        "schema_version": 1,
        "selection": {
            "manifest_indices": selected_indices,
            "selection_mode": MANUAL_SELECTION_MODE,
            "source": "user-confirmed Rank 1",
            "frozen": True,
            "auto_rescore": False,
            "auto_replace": False,
        },
        "scenes": selected_records,
        "graph_edges": selected_edges,
        "band": "B9",
        "pixel_size_m": 14.0,
        "crs": _common_value(selected_records, "crs"),
        "registration": {
            "match_max_side": 1600,
            "random_seed": RANSAC_RANDOM_SEED,
            "coordinate_frame": "pair_common_grid",
        },
        "ransac": {
            "model": "AffineTransform",
            "residual_threshold_px": RANSAC_RESIDUAL_THRESHOLD,
            "max_trials": MAX_RANSAC_TRIALS,
            "random_seed": RANSAC_RANDOM_SEED,
            "minimum_inliers": GEOMETRY_MIN_INLIERS,
            "minimum_inlier_ratio": GEOMETRY_MIN_INLIER_RATIO,
        },
        "pair_acceptance_thresholds": {
            "residual_threshold_px": RANSAC_RESIDUAL_THRESHOLD,
            "max_trials": RANSAC_MAX_TRIALS,
            "minimum_inliers": MIN_INLIERS,
            "minimum_inlier_ratio": MIN_INLIER_RATIO,
            "coverage_rule": (
                "spatial_coverage_ratio(inlier_points, overlap_window); "
                "descriptive metric, no additional acceptance threshold"
            ),
        },
        "sift": {
            "nfeatures": SIFT_NFEATURES,
            "lowe_ratio": LOWE_RATIO,
        },
        "matchers": MATCHERS,
        "confidence_semantics": "method_internal_only",
        "global_methods": GLOBAL_METHODS,
        "reference_selection_policy": (
            "existing select_reference_scene: degree, sum(Q), index tie-break"
        ),
        "mosaic_method": "weighted_feather",
        "mosaic_status": "deferred_until_geometry_validation",
    }


def write_frozen_config(config: dict, output_path: str | Path) -> Path:
    """Write a frozen config artifact without changing its selected indices."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return path


def _common_value(records: list[dict], key: str):
    values = {record.get(key) for record in records}
    if len(values) != 1:
        raise ValueError(f"selected scenes do not share one {key}: {sorted(values)}")
    return next(iter(values))
