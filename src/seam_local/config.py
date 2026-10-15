"""Runtime parameters shared by Task15 seam/local/label/blend stages.

The dataclass is deliberately small so pure array functions can accept one
validated object rather than silently relying on duplicated numeric literals.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SeamLocalRuntimeConfig:
    intensity_weight: float = 0.5
    gradient_weight: float = 0.5
    normalization: str = "p95"
    coarse_factor: int = 4
    refine_half_width: int = 64
    corridor_half_width: int = 128
    segment_length: int = 256
    min_valid_pixels: int = 4096
    percentile_low: float = 1.0
    percentile_high: float = 99.0
    stability_gain_min: float = 0.5
    stability_gain_max: float = 2.0
    preference_distance_scale: float = 64.0
    tie_tolerance: float = 1e-6
    blend_method: str = "multilabel_cosine"
    blend_half_width: int = 64

    def __post_init__(self) -> None:
        if self.coarse_factor < 1 or self.refine_half_width < 0:
            raise ValueError("invalid seam search parameters")
        if self.corridor_half_width < 1 or self.segment_length < 1 or self.min_valid_pixels < 1:
            raise ValueError("invalid local correction parameters")
        if not (0.0 <= self.percentile_low < self.percentile_high <= 100.0):
            raise ValueError("invalid percentile range")
        if self.stability_gain_min <= 0 or self.stability_gain_max < self.stability_gain_min:
            raise ValueError("invalid gain stability range")
        if self.preference_distance_scale <= 0 or self.tie_tolerance < 0:
            raise ValueError("invalid labeling parameters")
        if self.blend_half_width < 1:
            raise ValueError("blend_half_width must be positive")


__all__ = ["SeamLocalRuntimeConfig"]
