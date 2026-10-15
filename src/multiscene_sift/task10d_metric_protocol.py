"""Frozen, independently defined Task10D radiometric metric protocol."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


EXPECTED_PROTOCOL: dict[str, Any] = {
    "schema_version": 1,
    "task": "Task10D",
    "paper_metrics": {"cd_status": "UNVERIFIED", "gl_status": "UNVERIFIED"},
    "global_metrics": {
        "mamd": {"definition": "weighted_shared_valid_abs_mean_difference", "direction": "lower"},
        "msdd": {"definition": "weighted_shared_valid_abs_std_difference", "direction": "lower"},
    },
    "rdd": {
        "definition": "weighted_shared_valid_one_dimensional_wasserstein_1",
        "implementation": "scipy.stats.wasserstein_distance",
        "histogram_official": False,
        "direction": "lower",
    },
    "local_grid": {
        "tile_size_pixels": 256,
        "anchor": "canonical_mosaic_grid_origin",
        "min_shared_valid_pixels": 4096,
        "support_shared_across_methods": True,
        "direction": "lower",
    },
    "seam_zone": {
        "definition_source": "Task9 build_seam_zone_mask",
        "min_normalized_weight": 0.25,
        "weight_rule": "pairwise normalized feather weights",
        "direction": "lower",
    },
    "cgl": {
        "name": "Circular Gradient Orientation Loss",
        "gradient_operator": "scipy.ndimage.sobel",
        "angle_definition": "theta=atan2(Gy,Gx)",
        "angle_difference": "abs(atan2(sin(delta), cos(delta)))",
        "stencil": "fully_valid_3x3",
        "gradient_threshold": 1e-12,
        "unit": "radians",
        "report_degrees": True,
        "direction": "lower",
    },
    "shared_valid_fairness": True,
    "no_composite_score": True,
}


def load_task10d_metric_protocol(path: str | Path) -> dict[str, Any]:
    """Load and strictly validate the on-disk Task10D protocol."""

    protocol_path = Path(path)
    with protocol_path.open("r", encoding="utf-8") as handle:
        protocol = json.load(handle)
    validate_task10d_metric_protocol(protocol)
    return protocol


def validate_task10d_metric_protocol(protocol: Mapping[str, Any]) -> None:
    """Reject any runtime definition that differs from the frozen protocol."""

    _validate_exact(protocol, EXPECTED_PROTOCOL, path="protocol")


def _validate_exact(actual: Any, expected: Any, *, path: str) -> None:
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping):
            raise ValueError(f"{path} must be an object")
        missing = [key for key in expected if key not in actual]
        if missing:
            raise ValueError(f"{path} missing required keys: {missing}")
        for key, expected_value in expected.items():
            _validate_exact(actual[key], expected_value, path=f"{path}.{key}")
        return
    if actual != expected:
        raise ValueError(f"{path} must equal frozen value {expected!r}; got {actual!r}")
