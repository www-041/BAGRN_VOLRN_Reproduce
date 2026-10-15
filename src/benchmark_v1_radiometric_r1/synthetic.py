from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SyntheticCase:
    case_id: str
    clean_target: np.ndarray
    observed: np.ndarray
    valid_mask: np.ndarray
    change_mask: np.ndarray
    change_amplitude: np.ndarray
    saturation_mask: np.ndarray
    perturbation: dict[str, np.ndarray | float | int]


def _valid_array(source: np.ndarray, valid_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    source = np.asarray(source, dtype=np.float64)
    if source.ndim not in (2, 3):
        raise ValueError("source must be 2D or bands-first 3D")
    if source.ndim == 2:
        if np.shape(valid_mask) != source.shape:
            raise ValueError("valid mask shape must match the source")
        valid = np.asarray(valid_mask, dtype=bool) & np.isfinite(source)
    else:
        if source.shape[1:] != np.shape(valid_mask) and source.shape != np.shape(valid_mask):
            raise ValueError("valid mask must match spatial shape or full bands-first source")
        if np.shape(valid_mask) == source.shape:
            valid = np.asarray(valid_mask, dtype=bool) & np.isfinite(source)
        else:
            valid = np.broadcast_to(np.asarray(valid_mask, dtype=bool), source.shape).copy() & np.isfinite(source)
    return source, valid


def _spatial_field(shape: tuple[int, int], config: dict[str, Any], prefix: str, base_default: float) -> np.ndarray:
    h, w = shape
    x = np.linspace(-0.5, 0.5, w, dtype=np.float64)[None, :]
    y = np.linspace(-0.5, 0.5, h, dtype=np.float64)[:, None]
    base = float(config.get(f"{prefix}_base", base_default))
    return base + float(config.get(f"{prefix}_gradient_x", 0.0)) * x + float(config.get(f"{prefix}_gradient_y", 0.0)) * y


def generate_case(source, valid_mask, config: dict[str, Any], seed: int) -> SyntheticCase:
    """Apply a known synthetic perturbation; no correction method is fitted or run."""
    data, valid = _valid_array(np.asarray(source), np.asarray(valid_mask))
    cfg = dict(config)
    case_id = str(cfg.get("case", "")).upper()
    if case_id not in {"S1", "S2", "S3", "S4"}:
        raise ValueError(f"unknown synthetic case: {case_id}")
    rng = np.random.default_rng(int(seed))
    change_mask = np.zeros(data.shape[-2:], dtype=bool)
    change_amplitude = np.zeros(data.shape, dtype=np.float64)
    clean_target = data.copy()
    perturbation: dict[str, np.ndarray | float | int] = {"seed": int(seed)}

    if case_id == "S1":
        gain, offset = float(cfg.get("gain", 1.0)), float(cfg.get("offset", 0.0))
        observed = clean_target * gain + offset
        perturbation.update(gain=gain, offset=offset)
    elif case_id == "S2":
        if data.ndim == 2:
            spatial_shape = data.shape
        else:
            spatial_shape = data.shape[-2:]
        gain_field = _spatial_field(spatial_shape, cfg, "gain", 1.0)
        offset_field = _spatial_field(spatial_shape, cfg, "offset", 0.0)
        if data.ndim == 3:
            gain_field, offset_field = gain_field[None], offset_field[None]
        observed = clean_target * gain_field + offset_field
        perturbation.update(gain_field=gain_field, offset_field=offset_field)
    elif case_id == "S3":
        if data.ndim != 3:
            raise ValueError("S3 requires bands-first 3D source")
        gains = np.asarray(cfg.get("gains", []), dtype=np.float64)
        offsets = np.asarray(cfg.get("offsets", []), dtype=np.float64)
        if len(gains) != data.shape[0] or len(offsets) != data.shape[0]:
            raise ValueError("S3 gain/offset parameter count must equal the number of bands")
        observed = clean_target * gains[:, None, None] + offsets[:, None, None]
        perturbation.update(gain_per_band=gains, offset_per_band=offsets)
    else:
        if "change_mask" in cfg:
            supplied_change = np.asarray(cfg["change_mask"], dtype=bool)
            if supplied_change.shape != data.shape[-2:]:
                raise ValueError("S4 change mask must match source spatial shape")
            change_mask = supplied_change.copy()
        else:
            region = cfg.get("change_region", [0, 0, 0, 0])
            if len(region) != 4:
                raise ValueError("S4 change_region must be [row_start,row_stop,col_start,col_stop]")
            r0, r1, c0, c1 = (int(v) for v in region)
            change_mask[max(0, r0):min(change_mask.shape[0], r1), max(0, c0):min(change_mask.shape[1], c1)] = True
        amplitude = float(cfg.get("change_amplitude", 0.0))
        change_amplitude = np.broadcast_to(change_mask, data.shape).astype(np.float64).copy() * amplitude
        changed_target = clean_target + change_amplitude
        gain, offset = float(cfg.get("gain", 1.0)), float(cfg.get("offset", 0.0))
        observed = changed_target * gain + offset
        perturbation.update(gain=gain, offset=offset, change_amplitude=amplitude)

    noise_std = float(cfg.get("noise_std", 0.0))
    if noise_std < 0:
        raise ValueError("noise_std cannot be negative")
    if noise_std:
        observed = observed + rng.normal(0.0, noise_std, size=data.shape)
    low, high = cfg.get("clip_min"), cfg.get("clip_max")
    unclipped = observed.copy()
    saturation = np.zeros(data.shape, dtype=bool)
    if low is not None:
        saturation |= unclipped < float(low)
    if high is not None:
        saturation |= unclipped > float(high)
    if low is not None or high is not None:
        observed = np.clip(observed, low if low is not None else -np.inf, high if high is not None else np.inf)
    observed = np.where(valid, observed, 0.0)
    clean_target = np.where(valid, clean_target, 0.0)
    change_amplitude = np.where(valid, change_amplitude, 0.0)
    saturation &= valid
    return SyntheticCase(case_id, clean_target, observed, valid, change_mask, change_amplitude, saturation, perturbation)
