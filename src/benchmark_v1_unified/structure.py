"""Input-frozen structural diagnostics; these are not paper GL."""
from __future__ import annotations
from dataclasses import asdict, dataclass
import numpy as np
from scipy.ndimage import convolve, binary_erosion

KX = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=np.float64) / 8.0
KY = KX.T


@dataclass(frozen=True)
class StructureStats:
    status: str
    gradient_centers: int
    edge_centers: int
    input_edge_threshold: float | None
    direction_loss: float | None
    relative_gradient_mae: float | None


@dataclass(frozen=True)
class MetricValue:
    value: float | None
    unit: str
    status: str
    n_input: int
    n_output: int
    diagnostics: dict


def _gradient(image):
    return convolve(np.asarray(image, dtype=np.float64), KX, mode="constant", cval=0), convolve(np.asarray(image, dtype=np.float64), KY, mode="constant", cval=0)


def compute_sobel_structure(original, corrected, valid) -> StructureStats:
    x, y, mask = np.asarray(original, float), np.asarray(corrected, float), np.asarray(valid, bool)
    if x.ndim != 2 or x.shape != y.shape or x.shape != mask.shape:
        raise ValueError("structure inputs must be same-shaped 2D arrays")
    if np.any(mask & (~np.isfinite(x) | ~np.isfinite(y))):
        raise ValueError("nonfinite value on valid structure support")
    eligible = binary_erosion(mask, structure=np.ones((3,3), bool), border_value=0)
    gx, gy = _gradient(np.where(mask, x, 0.0)); hx, hy = _gradient(np.where(mask, y, 0.0))
    mag_x, mag_y = np.hypot(gx, gy), np.hypot(hx, hy)
    n = int(eligible.sum())
    if not n:
        return StructureStats("NO_VALID_GRADIENT_CENTERS", 0, 0, None, None, None)
    tau = max(1e-12, float(np.median(mag_x[eligible])))
    edges = eligible & (mag_x > tau)
    ne = int(edges.sum())
    if not ne:
        return StructureStats("NOT_EVALUABLE_NO_EDGES", n, 0, tau, None, None)
    denom = mag_x[edges] * mag_y[edges]
    cosine = np.zeros(ne, dtype=np.float64)
    nonzero = denom > 1e-12
    cosine[nonzero] = (gx[edges][nonzero]*hx[edges][nonzero] + gy[edges][nonzero]*hy[edges][nonzero]) / denom[nonzero]
    cosine = np.clip(cosine, -1.0, 1.0)
    total = float(mag_x[eligible].sum())
    relative = None if total == 0 else float(np.abs(mag_y[eligible]-mag_x[eligible]).sum() / total)
    status = "NOT_EVALUABLE_FLAT_INPUT" if total == 0 else "EVALUABLE"
    return StructureStats(status, n, ne, tau, float(np.mean(1-cosine)), relative)


def compute_range_retention(original, corrected, valid) -> MetricValue:
    x, y, mask = np.asarray(original, float), np.asarray(corrected, float), np.asarray(valid, bool)
    if x.shape != y.shape or x.shape != mask.shape:
        raise ValueError("range inputs must have identical shapes")
    if np.any(mask & (~np.isfinite(x) | ~np.isfinite(y))):
        raise ValueError("nonfinite valid range sample")
    a, b = x[mask], y[mask]
    if not a.size:
        return MetricValue(None, "ratio", "NO_VALID_SUPPORT", 0, 0, {})
    input_range = float(np.quantile(a,.95,method="linear")-np.quantile(a,.05,method="linear"))
    if input_range == 0:
        return MetricValue(None,"ratio","NOT_EVALUABLE_FLAT_INPUT",int(a.size),int(b.size),{"input_range":input_range})
    output_range = float(np.quantile(b,.95,method="linear")-np.quantile(b,.05,method="linear"))
    return MetricValue(output_range/input_range,"ratio","EVALUABLE",int(a.size),int(b.size),{"input_range":input_range,"output_range":output_range})
