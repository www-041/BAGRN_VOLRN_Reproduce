"""Unified RANSAC + Affine geometry fitting.

All four matchers feed into the same :func:`fit_affine_ransac` so that
the comparison is purely about tie-point quality, not geometry choice.
"""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass

import numpy as np
from skimage.measure import ransac
from skimage.transform import AffineTransform, warp

from src.registration_benchmark.models import MatchSet

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Status constants
# ---------------------------------------------------------------------------

STATUS_OK = "OK"
STATUS_TOO_FEW_INLIERS = "TOO_FEW_INLIERS"
STATUS_LOW_INLIER_RATIO = "LOW_INLIER_RATIO"
STATUS_INVALID_GEOMETRY = "INVALID_GEOMETRY"

RANSAC_RESIDUAL_THRESHOLD = 2.0
MAX_RANSAC_TRIALS = 5000
RANSAC_RANDOM_SEED = 0
MIN_INLIERS = 20
MIN_INLIER_RATIO = 0.30


# ---------------------------------------------------------------------------
# GeometryResult
# ---------------------------------------------------------------------------


@dataclass
class GeometryResult:
    """Output of :func:`fit_affine_ransac`.

    Attributes:
        model: Fitted ``AffineTransform`` (maps ``tgt_xy → ref_xy``).
        status: One of the ``STATUS_*`` constants.
        inlier_mask: Boolean array over input matches.
        n_raw: Total matches fed into RANSAC.
        n_inlier: Number of RANSAC inliers.
        inlier_ratio: ``n_inlier / n_raw``.
        residual_median: Median Euclidean residual (pixels).
        residual_rmse: Root-mean-square Euclidean residual (pixels).
        residual_p90: 90th percentile residual.
        residual_p95: 95th percentile residual.
        residual_max: Maximum residual among inliers.
        affine_scale_x: Decomposed X scale.
        affine_scale_y: Decomposed Y scale.
        affine_rotation_deg: Decomposed rotation in degrees.
        affine_shear_deg: Decomposed shear in degrees.
        residual_threshold: RANSAC threshold used.
    """

    model: AffineTransform | None
    status: str

    inlier_mask: np.ndarray

    n_raw: int
    n_inlier: int
    inlier_ratio: float

    residual_median: float
    residual_rmse: float
    residual_p90: float
    residual_p95: float
    residual_max: float

    affine_scale_x: float
    affine_scale_y: float
    affine_rotation_deg: float
    affine_shear_deg: float

    residual_threshold: float


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fit_affine_ransac(
    matches: MatchSet,
    residual_threshold: float = RANSAC_RESIDUAL_THRESHOLD,
    max_trials: int = MAX_RANSAC_TRIALS,
    random_seed: int = RANSAC_RANDOM_SEED,
) -> GeometryResult:
    """Fit a 2-D affine transform to tie-points via RANSAC.

    The fitted model maps **tgt_xy → ref_xy**::

        ref_xy ≈ model(tgt_xy)

    Args:
        matches: A validated :class:`MatchSet`.
        residual_threshold: Inlier threshold in pixels (Euclidean distance).
        max_trials: Maximum RANSAC iterations.
        random_seed: Seed passed to scikit-image RANSAC for reproducibility.

    Returns:
        A :class:`GeometryResult`, with ``status`` indicating whether the fit
        met the minimum quality thresholds.
    """
    src = matches.tgt_xy
    dst = matches.ref_xy

    n_raw = len(src)

    # Edge case: too few points for RANSAC (need ≥ 3 for affine)
    if n_raw < 3:
        logger.warning("Geometry: only %d raw matches (need ≥ 3)", n_raw)
        return GeometryResult(
            model=None,
            status=STATUS_TOO_FEW_INLIERS,
            inlier_mask=np.zeros(n_raw, dtype=bool),
            n_raw=n_raw,
            n_inlier=0,
            inlier_ratio=0.0,
            residual_median=float("nan"),
            residual_rmse=float("nan"),
            residual_p90=float("nan"),
            residual_p95=float("nan"),
            residual_max=float("nan"),
            affine_scale_x=float("nan"),
            affine_scale_y=float("nan"),
            affine_rotation_deg=float("nan"),
            affine_shear_deg=float("nan"),
            residual_threshold=residual_threshold,
        )

    # --- RANSAC ------------------------------------------------------------------
    model, inliers = ransac(
        (src, dst),
        AffineTransform,
        min_samples=3,
        residual_threshold=residual_threshold,
        max_trials=max_trials,
        **(
            {"rng": random_seed}
            if "rng" in inspect.signature(ransac).parameters
            else {"random_state": random_seed}
        ),
    )

    inlier_mask = inliers.astype(bool)
    n_inlier = int(inlier_mask.sum())
    inlier_ratio = n_inlier / n_raw if n_raw > 0 else 0.0

    # --- Residual statistics ----------------------------------------------------
    if n_inlier > 0:
        predicted = model(src[inlier_mask])
        residuals = np.linalg.norm(predicted - dst[inlier_mask], axis=1)
        resid_median = float(np.median(residuals))
        resid_rmse = float(np.sqrt(np.mean(residuals ** 2)))
        resid_p90 = float(np.percentile(residuals, 90))
        resid_p95 = float(np.percentile(residuals, 95))
        resid_max = float(np.max(residuals))
    else:
        resid_median = resid_rmse = resid_p90 = resid_p95 = resid_max = float("nan")

    # --- Affine decomposition ---------------------------------------------------
    scale_x, scale_y, rot_deg, shear_deg = _decompose_affine(model.params)

    # --- Status checks ----------------------------------------------------------
    status = STATUS_OK

    if n_inlier < MIN_INLIERS:
        status = STATUS_TOO_FEW_INLIERS
    elif inlier_ratio < MIN_INLIER_RATIO:
        status = STATUS_LOW_INLIER_RATIO
    elif not _is_geometry_valid(scale_x, scale_y, rot_deg, shear_deg):
        status = STATUS_INVALID_GEOMETRY

    if status != STATUS_OK:
        logger.warning("Geometry: status=%s (inliers=%d, ratio=%.3f, "
                       "scale=(%.3f,%.3f), rot=%.2f°, shear=%.2f°)",
                       status, n_inlier, inlier_ratio,
                       scale_x, scale_y, rot_deg, shear_deg)

    return GeometryResult(
        model=model,
        status=status,
        inlier_mask=inlier_mask,
        n_raw=n_raw,
        n_inlier=n_inlier,
        inlier_ratio=inlier_ratio,
        residual_median=resid_median,
        residual_rmse=resid_rmse,
        residual_p90=resid_p90,
        residual_p95=resid_p95,
        residual_max=resid_max,
        affine_scale_x=scale_x,
        affine_scale_y=scale_y,
        affine_rotation_deg=rot_deg,
        affine_shear_deg=shear_deg,
        residual_threshold=residual_threshold,
    )


def warp_target_common_grid(
    target: np.ndarray,
    valid_mask: np.ndarray,
    model: AffineTransform,
) -> tuple[np.ndarray, np.ndarray]:
    """Warp the target image onto the reference grid using an affine model.

    Performs a **single** warp of the original-DN target image.  The valid
    mask is warped with nearest-neighbour to avoid blending artefacts.

    Args:
        target: Target image in common-grid pixel space (2-D, original DN).
        valid_mask: Boolean valid-pixel mask for *target*.
        model: ``AffineTransform`` mapping ``tgt_xy → ref_xy``.

    Returns:
        ``(warped, warped_valid)`` — both 2-D arrays with the same shape
        as *target*.  Invalid regions are set to ``np.nan`` in the warped
        array.
    """
    target_float = target.astype(np.float64)
    target_float[~valid_mask] = np.nan

    # Warp image data (bilinear)
    warped = warp(
        target_float,
        model.inverse,
        output_shape=target.shape,
        order=1,
        mode="constant",
        cval=np.nan,
        preserve_range=True,
    )

    # Warp valid mask (nearest-neighbour)
    mask_uint8 = valid_mask.astype(np.uint8)
    warped_mask = warp(
        mask_uint8,
        model.inverse,
        output_shape=target.shape,
        order=0,
        mode="constant",
        cval=0,
        preserve_range=True,
    ).astype(bool)

    return warped, warped_mask


# ---------------------------------------------------------------------------
# Affine decomposition helpers
# ---------------------------------------------------------------------------


def _decompose_affine(
    matrix: np.ndarray,
) -> tuple[float, float, float, float]:
    """Decompose a 3×3 affine matrix into scale, rotation, shear.

    Returns:
        ``(scale_x, scale_y, rotation_deg, shear_deg)``
    """
    # Extract 2×2 linear part
    a, b = matrix[0, 0], matrix[0, 1]
    c, d = matrix[1, 0], matrix[1, 1]

    scale_x = float(np.sqrt(a * a + c * c))
    scale_y = float(np.sqrt(b * b + d * d))

    # Rotation (average of the two estimates)
    rot1 = np.arctan2(c, a)
    rot2 = np.arctan2(-b, d)
    rot = np.arctan2(np.sin(rot1) + np.sin(rot2),
                     np.cos(rot1) + np.cos(rot2))
    rot_deg = float(np.degrees(rot))

    # Shear from the residual
    shear = float(np.arctan2(-b * a - d * c, a * d - b * c))
    shear_deg = float(np.degrees(shear))

    return scale_x, scale_y, rot_deg, shear_deg


def _is_geometry_valid(
    scale_x: float,
    scale_y: float,
    rot_deg: float,
    shear_deg: float,
) -> bool:
    """Check whether affine parameters are within plausible range."""
    if not (0.95 <= scale_x <= 1.05):
        return False
    if not (0.95 <= scale_y <= 1.05):
        return False
    if abs(rot_deg) > 3.0:
        return False
    if abs(shear_deg) > 3.0:
        return False
    return True
