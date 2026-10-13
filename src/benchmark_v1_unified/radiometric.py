"""Pure, method-neutral radiometric consistency and coverage metrics."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import numpy as np
from scipy.stats import wasserstein_distance


@dataclass(frozen=True)
class PairDirectMetrics:
    status: str
    n: int
    mae: float | None
    rmse: float | None
    bias: float | None
    p95ae: float | None


@dataclass(frozen=True)
class PairDistributionMetrics:
    status: str
    n: int
    mean_a: float | None
    mean_b: float | None
    std_a: float | None
    std_b: float | None
    mamd: float | None
    msdd: float | None
    rdd: float | None


@dataclass(frozen=True)
class CoverageAudit:
    input_pixels: int
    output_pixels: int
    missing_pixels: int
    coverage: float
    status: str
    primary_eligible: bool


def _arrays(x, y, support):
    a, b, m = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64), np.asarray(support, dtype=bool)
    if a.shape != b.shape or a.shape != m.shape:
        raise ValueError("x, y and frozen input support must have identical shapes")
    if a.ndim < 1:
        raise ValueError("metric inputs must have at least one dimension")
    if np.any(m & (~np.isfinite(a) | ~np.isfinite(b))):
        raise ValueError("nonfinite value on declared valid input support")
    return a[m], b[m]


def pair_support(mask_i, mask_j) -> np.ndarray:
    a, b = np.asarray(mask_i, dtype=bool), np.asarray(mask_j, dtype=bool)
    if a.shape != b.shape:
        raise ValueError("input masks must have identical shapes")
    return a & b


def compute_pair_direct_metrics(x, y, support) -> PairDirectMetrics:
    a, b = _arrays(x, y, support)
    n = int(a.size)
    if not n:
        return PairDirectMetrics("NO_VALID_OVERLAP", 0, None, None, None, None)
    delta = a - b
    abs_delta = np.abs(delta)
    return PairDirectMetrics("EVALUABLE", n, float(abs_delta.mean()), float(np.sqrt(np.mean(delta**2))), float(delta.mean()), float(np.quantile(abs_delta, .95, method="linear")))


def compute_pair_distribution_metrics(x, y, support, *, exact_w1: bool = True) -> PairDistributionMetrics:
    if not exact_w1:
        raise ValueError("V1 requires exact empirical Wasserstein-1")
    a, b = _arrays(x, y, support)
    n = int(a.size)
    if not n:
        return PairDistributionMetrics("NO_VALID_OVERLAP", 0, None, None, None, None, None, None, None)
    ma, mb = float(a.mean()), float(b.mean())
    sa, sb = float(a.std(ddof=0)), float(b.std(ddof=0))
    return PairDistributionMetrics("EVALUABLE", n, ma, mb, sa, sb, abs(ma-mb), abs(sa-sb), float(wasserstein_distance(a, b)))


def audit_pair_output_support(input_pair_mask, output_values, output_valid) -> CoverageAudit:
    source = np.asarray(input_pair_mask, dtype=bool)
    values = np.asarray(output_values)
    valid = np.asarray(output_valid, dtype=bool)
    if source.shape != values.shape or source.shape != valid.shape:
        raise ValueError("input support, output values and explicit output mask must match")
    n = int(source.sum())
    out = int(np.count_nonzero(source & valid & np.isfinite(values)))
    if n == 0:
        return CoverageAudit(0, 0, 0, 0.0, "NO_VALID_OVERLAP", False)
    coverage = out / n
    status = "FULL_SUPPORT" if out == n else ("OUTPUT_MISSING" if out == 0 else "PARTIAL_OUTPUT")
    return CoverageAudit(n, out, n-out, coverage, status, coverage == 1.0)


def cross_band_summary(per_band_values: dict[str, float], *, units_known: bool, scales=None) -> dict:
    if not units_known:
        return {"status": "NOT_EVALUABLE_UNKNOWN_UNITS", "value": None, "per_band": dict(per_band_values)}
    if scales is None or set(scales) != set(per_band_values) or any(not np.isfinite(scales[k]) or scales[k] <= 0 for k in per_band_values):
        return {"status": "NOT_EVALUABLE_NO_FROZEN_SCALE", "value": None, "per_band": dict(per_band_values)}
    vals = [float(per_band_values[k]) / float(scales[k]) for k in sorted(per_band_values)]
    return {"status": "EVALUABLE", "value": float(np.mean(vals)), "per_band": dict(per_band_values), "scale_policy": dict(scales)}


def metric_dict(value):
    return asdict(value)
