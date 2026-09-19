"""Pairwise matcher + shared RANSAC registration for multi-scene mosaicking.

Only tie-point generation differs between ``sift`` and ``loftr``.  Both
methods feed the same common-grid coordinates into the same affine RANSAC,
quality checks, coverage metric, graph construction, and global registration.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np

from src.registration_benchmark.common_grid import (
    load_pair_to_common_grid,
    build_match_view,
)
from src.registration_benchmark.matchers.sift import match_sift
from src.registration_benchmark.geometry import fit_affine_ransac, STATUS_OK
from src.registration_benchmark.metrics import spatial_coverage_ratio

from src.multiscene_sift.models import OverlapEdge, PairwiseRegistration, Scene

logger = logging.getLogger(__name__)

SUPPORTED_MATCHERS = ("sift", "loftr")

# Fixed registration parameters (quality thresholds, not tunable via CLI)
SIFT_NFEATURES = 8000
LOWE_RATIO = 0.75
RANSAC_MAX_TRIALS = 5000
MIN_INLIERS = 20
MIN_INLIER_RATIO = 0.30


def _normalize_matcher_name(matcher: str) -> str:
    name = str(matcher).strip().lower()
    if name not in SUPPORTED_MATCHERS:
        raise ValueError(
            f"Unknown matcher {matcher!r}; expected one of {SUPPORTED_MATCHERS}"
        )
    return name


def _run_matcher(view, matcher: str):
    """Run exactly one matcher and return a unified MatchSet."""
    matcher = _normalize_matcher_name(matcher)
    if matcher == "sift":
        return match_sift(
            view,
            nfeatures=SIFT_NFEATURES,
            ratio_threshold=LOWE_RATIO,
        )

    # Lazy import keeps the SIFT-only path usable when torch/kornia is absent.
    from src.registration_benchmark.matchers.loftr import match_loftr

    return match_loftr(view)


def register_pair(
    scene_i: Scene,
    scene_j: Scene,
    band: str = "B14",
    match_max_side: int = 1600,
    ransac_threshold: float = 2.0,
    random_seed: int = 0,
    matcher: str = "sift",
) -> PairwiseRegistration:
    """Register scene_j onto scene_i using the requested matcher + shared RANSAC."""
    matcher = _normalize_matcher_name(matcher)
    t0 = time.perf_counter()

    ref_path = scene_i.band_paths[band]
    tgt_path = scene_j.band_paths[band]

    # 1. Load into common grid
    pair = load_pair_to_common_grid(ref_path, tgt_path, band=1)
    common_transform = pair.transform

    # Guard: zero overlap
    row_start, row_end, col_start, col_end = pair.overlap_window
    if row_start >= row_end or col_start >= col_end:
        logger.warning(
            "Pair (%d, %d): zero overlap in common grid", scene_i.index, scene_j.index
        )
        return PairwiseRegistration(
            idx_i=scene_i.index,
            idx_j=scene_j.index,
            status="NO_OVERLAP",
            raw_matches=0,
            inliers=0,
            inlier_ratio=0.0,
            coverage=0.0,
            residual_median=float("nan"),
            residual_rmse=float("nan"),
            residual_p95=float("nan"),
            pair_pixel_matrix=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            pair_common_transform=common_transform,
            runtime_sec=time.perf_counter() - t0,
            matcher=matcher,
            matcher_runtime_sec=0.0,
            geometry_runtime_sec=0.0,
        )

    # 2. Build one shared match view for either matcher
    view = build_match_view(pair, max_side=match_max_side)

    # 3. Tie-point matching -- the only matcher-specific branch
    matches = _run_matcher(view, matcher)

    # 4. Shared RANSAC Affine
    t_geom = time.perf_counter()
    geom = fit_affine_ransac(
        matches,
        residual_threshold=ransac_threshold,
        max_trials=RANSAC_MAX_TRIALS,
        random_seed=random_seed,
    )
    geometry_runtime = time.perf_counter() - t_geom

    # Quality checks
    if geom.status != STATUS_OK:
        return _make_result(
            scene_i.index,
            scene_j.index,
            geom.status,
            geom,
            matches,
            common_transform,
            0.0,
            time.perf_counter() - t0,
            matcher,
            geometry_runtime,
        )

    if geom.n_inlier < MIN_INLIERS or geom.inlier_ratio < MIN_INLIER_RATIO:
        return _make_result(
            scene_i.index,
            scene_j.index,
            "TOO_FEW_INLIERS",
            geom,
            matches,
            common_transform,
            0.0,
            time.perf_counter() - t0,
            matcher,
            geometry_runtime,
        )

    # Coverage
    inlier_pts = (
        matches.ref_xy[geom.inlier_mask]
        if geom.inlier_mask is not None and geom.inlier_mask.any()
        else np.empty((0, 2))
    )
    coverage = spatial_coverage_ratio(inlier_pts, pair.overlap_window)

    return _make_result(
        scene_i.index,
        scene_j.index,
        STATUS_OK,
        geom,
        matches,
        common_transform,
        coverage,
        time.perf_counter() - t0,
        matcher,
        geometry_runtime,
    )


def _make_result(
    idx_i: int,
    idx_j: int,
    status: str,
    geom,
    matches,
    common_transform,
    coverage: float,
    runtime: float,
    matcher: str,
    geometry_runtime: float,
) -> PairwiseRegistration:
    """Build a PairwiseRegistration from geometry result."""
    inlier_ref = np.empty((0, 2))
    inlier_tgt = np.empty((0, 2))

    if status == STATUS_OK and geom.model is not None:
        pixel_mat = geom.model.params.tolist()
        raw_matches = int(geom.n_raw)
        inliers = int(geom.n_inlier)
        inlier_ratio = float(geom.inlier_ratio)
        res_median = (
            float(geom.residual_median)
            if not np.isnan(geom.residual_median)
            else None
        )
        res_rmse = (
            float(geom.residual_rmse)
            if not np.isnan(geom.residual_rmse)
            else None
        )
        res_p95 = (
            float(geom.residual_p95)
            if not np.isnan(geom.residual_p95)
            else None
        )
        if geom.inlier_mask is not None and geom.inlier_mask.any():
            inlier_ref = np.asarray(
                matches.ref_xy[geom.inlier_mask], dtype=np.float64
            )
            inlier_tgt = np.asarray(
                matches.tgt_xy[geom.inlier_mask], dtype=np.float64
            )
    else:
        pixel_mat = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        raw_matches = int(geom.n_raw) if geom.n_raw else 0
        inliers = int(geom.n_inlier) if geom.n_inlier else 0
        inlier_ratio = (
            float(geom.inlier_ratio) if geom.inlier_ratio is not None else 0.0
        )
        res_median = None
        res_rmse = None
        res_p95 = None

    return PairwiseRegistration(
        idx_i=idx_i,
        idx_j=idx_j,
        status=status,
        raw_matches=raw_matches,
        inliers=inliers,
        inlier_ratio=inlier_ratio,
        coverage=float(coverage),
        residual_median=(
            float(res_median) if res_median is not None else float("nan")
        ),
        residual_rmse=(float(res_rmse) if res_rmse is not None else float("nan")),
        residual_p95=(float(res_p95) if res_p95 is not None else float("nan")),
        pair_pixel_matrix=pixel_mat,
        pair_common_transform=common_transform,
        runtime_sec=runtime,
        matcher=matcher,
        matcher_runtime_sec=float(getattr(matches, "runtime_sec", 0.0)),
        geometry_runtime_sec=float(geometry_runtime),
        inlier_ref_xy=inlier_ref,
        inlier_tgt_xy=inlier_tgt,
    )


def run_all_pairs(
    scenes: list[Scene],
    edges: list[OverlapEdge],
    out_dir: str | Path,
    band: str = "B14",
    match_max_side: int = 1600,
    ransac_threshold: float = 2.0,
    random_seed: int = 0,
    matcher: str = "sift",
) -> list[PairwiseRegistration]:
    """Run the selected matcher + shared RANSAC for all geographic edges."""
    matcher = _normalize_matcher_name(matcher)
    out = Path(out_dir)
    (out / "pairwise").mkdir(parents=True, exist_ok=True)

    results: list[PairwiseRegistration] = []

    for edge in edges:
        pair_start = time.perf_counter()
        logger.info(
            "Registering scene %d -> %d with %s (%s -> %s)",
            edge.idx_j,
            edge.idx_i,
            matcher.upper(),
            scenes[edge.idx_j].name,
            scenes[edge.idx_i].name,
        )
        try:
            reg = register_pair(
                scenes[edge.idx_i],
                scenes[edge.idx_j],
                band=band,
                match_max_side=match_max_side,
                ransac_threshold=ransac_threshold,
                random_seed=random_seed,
                matcher=matcher,
            )
        except Exception as exc:
            logger.exception(
                "Pair (%d, %d) failed with exception: %s",
                edge.idx_i,
                edge.idx_j,
                exc,
            )
            reg = PairwiseRegistration(
                idx_i=edge.idx_i,
                idx_j=edge.idx_j,
                status="FAILED",
                raw_matches=0,
                inliers=0,
                inlier_ratio=0.0,
                coverage=0.0,
                residual_median=float("nan"),
                residual_rmse=float("nan"),
                residual_p95=float("nan"),
                pair_pixel_matrix=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                pair_common_transform=None,
                runtime_sec=time.perf_counter() - pair_start,
                matcher=matcher,
                matcher_runtime_sec=0.0,
                geometry_runtime_sec=0.0,
            )
        results.append(reg)

    save_pairwise_summary(results, out)
    return results


def save_pairwise_summary(
    results: list[PairwiseRegistration],
    out_dir: Path,
) -> None:
    """Write pairwise_summary.csv and pairwise_summary.json."""
    csv_path = out_dir / "pairwise_summary.csv"
    with open(csv_path, "w", newline="") as f:
        f.write(
            "idx_i,idx_j,status,raw_matches,inliers,inlier_ratio,"
            "coverage,residual_median,residual_rmse,residual_p95,"
            "runtime_sec,matcher,matcher_runtime_sec,geometry_runtime_sec\n"
        )
        for r in results:
            f.write(
                f"{r.idx_i},{r.idx_j},{r.status},{r.raw_matches},"
                f"{r.inliers},{r.inlier_ratio:.4f},{r.coverage:.4f},"
                f"{r.residual_median},{r.residual_rmse},{r.residual_p95},"
                f"{r.runtime_sec:.6f},{r.matcher},"
                f"{r.matcher_runtime_sec:.6f},{r.geometry_runtime_sec:.6f}\n"
            )

    json_path = out_dir / "pairwise_summary.json"
    data = {
        "results": [
            {
                "idx_i": r.idx_i,
                "idx_j": r.idx_j,
                "matcher": r.matcher,
                "status": r.status,
                "raw_matches": r.raw_matches,
                "inliers": r.inliers,
                "inlier_ratio": r.inlier_ratio,
                "coverage": r.coverage,
                "residual_median": r.residual_median,
                "residual_rmse": r.residual_rmse,
                "residual_p95": r.residual_p95,
                "pixel_matrix": r.pair_pixel_matrix,
                "matcher_runtime_sec": r.matcher_runtime_sec,
                "geometry_runtime_sec": r.geometry_runtime_sec,
                "runtime_sec": r.runtime_sec,
            }
            for r in results
        ],
        "n_pairs": len(results),
    }
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)

    logger.info("Pairwise summary saved: %d pairs", len(results))
