"""Two-image registration benchmark runner.

Orchestrates the full pipeline: load → match → RANSAC → metrics → diagnostics.
"""

from __future__ import annotations

import csv
import json
import logging
import time
from pathlib import Path

import numpy as np

from src.registration_benchmark.common_grid import (
    build_match_view,
    load_pair_to_common_grid,
)
from src.registration_benchmark.diagnostics import MethodDiagnostics
from src.registration_benchmark.geometry import (
    STATUS_OK,
    fit_affine_ransac,
    warp_target_common_grid,
)
from src.registration_benchmark.matchers.phase import match_phase
from src.registration_benchmark.matchers.sift import match_sift
from src.registration_benchmark.models import MatchSet

logger = logging.getLogger(__name__)

# --- Optional matchers (graceful degradation) --------------------------------
try:
    from src.registration_benchmark.matchers.lightglue import (
        is_lightglue_available,
        match_lightglue,
    )
except ImportError:
    is_lightglue_available = lambda: False  # noqa: E731
    match_lightglue = None

# ---------------------------------------------------------------------------
# Matcher registry
# ---------------------------------------------------------------------------

MATCHERS: dict[str, callable] = {
    "phase": match_phase,
    "sift": match_sift,
}

if is_lightglue_available():
    MATCHERS["lightglue"] = match_lightglue

try:
    from src.registration_benchmark.matchers.loftr import is_loftr_available, match_loftr
except ImportError:
    is_loftr_available = lambda: False  # noqa: E731
    match_loftr = None

if is_loftr_available():
    MATCHERS["loftr"] = match_loftr


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_two_image_benchmark(
    ref_path: str,
    tgt_path: str,
    output_dir: str,
    band: int = 1,
    methods: list[str] | None = None,
    match_max_side: int = 1600,
    ransac_threshold: float = 2.0,
    device: str = "auto",
) -> dict:
    """Run the full two-image registration benchmark.

    Args:
        ref_path: Path to the reference GeoTIFF.
        tgt_path: Path to the target GeoTIFF.
        output_dir: Directory for all outputs.
        band: Raster band index (1-based).
        methods: List of matcher names.  Default: ``["phase", "sift"]``.
        match_max_side: ``max_side`` for :func:`build_match_view`.
        ransac_threshold: RANSAC residual threshold in pixels.
        device: ``"auto"``, ``"cpu"``, or ``"cuda"``.

    Returns:
        Dict with keys ``"summary"`` (list of per-method result dicts) and
        ``"output_dir"``.
    """
    t_total = time.perf_counter()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if methods is None:
        methods = ["phase", "sift"]

    # Validate methods
    unknown = set(methods) - set(MATCHERS.keys())
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}. Available: {list(MATCHERS.keys())}")

    # --- Save run config -------------------------------------------------------
    config = {
        "ref_path": str(ref_path),
        "tgt_path": str(tgt_path),
        "output_dir": str(output_dir),
        "band": band,
        "methods": methods,
        "match_max_side": match_max_side,
        "ransac_threshold": ransac_threshold,
        "device": device,
    }
    with open(out / "run_config.json", "w") as f:
        json.dump(sanitize_json(config), f, indent=2, allow_nan=False)

    # --- Common grid preparation -----------------------------------------------
    logger.info("=== Common grid preparation ===")
    pair = load_pair_to_common_grid(ref_path, tgt_path, band=band)

    # --- Match view ------------------------------------------------------------
    view = build_match_view(pair, max_side=match_max_side)

    # --- Per-method pipeline ---------------------------------------------------
    summary = []

    for method in methods:
        logger.info("=== Method: %s ===", method)
        try:
            result = _run_one_method(
                method, pair, view, out, ransac_threshold, device
            )
        except Exception as exc:
            logger.exception("Method %s FAILED: %s", method, exc)
            result = {
                "method": method,
                "status": "FAILED",
                "error": str(exc),
                "raw_matches": None,
                "inliers": None,
                "inlier_ratio": None,
                "coverage": None,
                "residual_median": None,
                "residual_rmse": None,
                "residual_p90": None,
                "residual_p95": None,
                "residual_max": None,
                "gradient_ncc_before": None,
                "gradient_ncc_after": None,
                "verification_dx": None,
                "verification_dy": None,
                "verification_magnitude": None,
                "verification_confidence": None,
                "verification_status": None,
                "feature_runtime_sec": None,
                "matcher_runtime_sec": None,
                "geometry_runtime_sec": None,
                "match_runtime_sec": None,
                "total_runtime_sec": None,
            }
        summary.append(result)

    # --- Write summary files ---------------------------------------------------
    _write_summary_csv(out / "summary.csv", summary)
    with open(out / "summary.json", "w") as f:
        json.dump(sanitize_json(summary), f, indent=2, allow_nan=False)

    elapsed = time.perf_counter() - t_total
    logger.info("=== Benchmark complete (%.1f s) ===", elapsed)

    return {"summary": summary, "output_dir": str(output_dir)}


# ---------------------------------------------------------------------------
# Internal: single-method pipeline
# ---------------------------------------------------------------------------


def _run_matcher(method: str, matcher_fn: callable, view, device: str):
    """Invoke a matcher, forwarding ``device`` only to learned methods."""
    if method in {"lightglue", "loftr"}:
        return matcher_fn(view, device=device)
    return matcher_fn(view)


def _run_one_method(
    method: str,
    pair,
    view,
    out: Path,
    ransac_threshold: float,
    device: str,
) -> dict:
    t0 = time.perf_counter()

    # 1. Match ---------------------------------------------------------------
    matcher_fn = MATCHERS[method]
    matches = _run_matcher(method, matcher_fn, view, device)
    matches.validate_for_geometry()
    logger.info("  raw matches: %d", len(matches.ref_xy))

    # Phase block screening diagnostics
    screening = matches.metadata.get("screening")
    if screening:
        logger.info(
            "  Phase screening: total=%d low_valid=%d low_texture=%d "
            "low_conf=%d large_shift=%d accepted=%d",
            screening.get("total", 0),
            screening.get("low_valid", 0),
            screening.get("low_texture", 0),
            screening.get("low_conf", 0),
            screening.get("large_shift", 0),
            screening.get("accepted", 0),
        )

    # 2. RANSAC Affine -------------------------------------------------------
    geometry_t0 = time.perf_counter()
    geom = fit_affine_ransac(matches, residual_threshold=ransac_threshold)
    geometry_runtime = time.perf_counter() - geometry_t0
    logger.info("  geometry: status=%s, inliers=%d/%d",
                geom.status, geom.n_inlier, geom.n_raw)

    # 3. Warp (only if geometry is valid) ---------------------------------
    warp_applied = False
    if geom.status == STATUS_OK and geom.model is not None:
        registered, registered_valid = warp_target_common_grid(
            pair.tgt_raw, pair.tgt_valid, geom.model
        )
        warp_applied = True
    else:
        registered = pair.tgt_raw.copy()
        registered_valid = pair.tgt_valid.copy()

    # 4. Diagnostics ---------------------------------------------------------
    diag = MethodDiagnostics(str(out), method, pair, matches, geom)
    diag.run_all(registered, registered_valid, warp_applied=warp_applied)

    elapsed = time.perf_counter() - t0

    # 5. Read back metrics.json ---------------------------------------------
    metrics_path = out / method / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)
    else:
        metrics = {}

    runtime_breakdown = matches.metadata.get("runtime_breakdown", {})

    return {
        "method": method,
        "status": geom.status,
        "raw_matches": int(geom.n_raw),
        "inliers": int(geom.n_inlier),
        "inlier_ratio": float(geom.inlier_ratio),
        "coverage": metrics.get("coverage"),
        "residual_median": geom.residual_median,
        "residual_rmse": geom.residual_rmse,
        "residual_p90": geom.residual_p90,
        "residual_p95": geom.residual_p95,
        "residual_max": geom.residual_max,
        "gradient_ncc_before": metrics.get("gradient_ncc_before"),
        "gradient_ncc_after": metrics.get("gradient_ncc_after"),
        "verification_dx": metrics.get("verification_dx"),
        "verification_dy": metrics.get("verification_dy"),
        "verification_magnitude": metrics.get("verification_magnitude"),
        "verification_confidence": metrics.get("verification_confidence"),
        "verification_status": metrics.get("verification_status"),
        "feature_runtime_sec": runtime_breakdown.get("feature_runtime_sec"),
        "matcher_runtime_sec": runtime_breakdown.get("matcher_runtime_sec"),
        "geometry_runtime_sec": geometry_runtime,
        "match_runtime_sec": matches.runtime_sec,
        "total_runtime_sec": elapsed,
        "error": metrics.get("error"),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_summary_csv(path: Path, summary: list[dict]):
    """Write summary.csv with uniform columns."""
    if not summary:
        return
    fieldnames = [
        "method", "status", "raw_matches", "inliers", "inlier_ratio",
        "coverage", "residual_median", "residual_rmse", "residual_p90",
        "residual_p95", "residual_max", "gradient_ncc_before",
        "gradient_ncc_after", "verification_magnitude",
        "verification_confidence", "verification_status",
        "feature_runtime_sec", "matcher_runtime_sec", "geometry_runtime_sec",
        "match_runtime_sec", "total_runtime_sec",
        "error",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(summary)


def sanitize_json(obj):
    """Recursively sanitise a Python object for strict JSON output.

    Rules:
    * ``NaN``, ``+Inf``, ``-Inf`` → ``None`` (serialised as ``null``)
    * ``np.integer`` → ``int``
    * ``np.floating`` → ``float`` or ``None``
    * ``dict`` / ``list`` → recurse
    * other → ``str(obj)``
    """
    if isinstance(obj, dict):
        return {k: sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_json(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return sanitize_json(obj.tolist())
    if isinstance(obj, (np.floating,)):
        val = float(obj)
        return None if val != val or val in (float("inf"), float("-inf")) else val
    if isinstance(obj, float):
        return None if obj != obj or obj in (float("inf"), float("-inf")) else obj
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj
