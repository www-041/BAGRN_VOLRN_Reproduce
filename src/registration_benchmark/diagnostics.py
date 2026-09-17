"""Diagnostic visualisation and output for the registration benchmark.

Each method gets a sub-directory containing:

* ``matches_raw.png`` / ``matches_inliers.png`` — side-by-side match plots
* ``coverage.png`` — spatial coverage grid
* ``checkerboard_before.png`` / ``checkerboard_after.png``
* ``registered_target.tif`` — warped in original DN
* ``mosaic_source_selection.tif``
* ``matches.csv`` / ``metrics.json``
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend

import matplotlib.pyplot as plt
import numpy as np
from rasterio import Affine

from src.mosaic import create_mosaic
from src.registration_benchmark.geometry import GeometryResult
from src.registration_benchmark.metrics import (
    gradient_ncc,
    phase_verification,
    spatial_coverage_ratio,
)
from src.registration_benchmark.models import CommonGridPair, MatchSet

logger = logging.getLogger(__name__)

MAX_DRAW_MATCHES = 300


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class MethodDiagnostics:
    """Produce all diagnostic artefacts for one method run."""

    def __init__(
        self,
        output_dir: str,
        method: str,
        pair: CommonGridPair,
        matches: MatchSet,
        geom: GeometryResult,
    ):
        self.out = Path(output_dir) / method
        self.out.mkdir(parents=True, exist_ok=True)
        self.method = method
        self.pair = pair
        self.matches = matches
        self.geom = geom

    def run_all(self, registered_tgt: np.ndarray, registered_valid: np.ndarray,
                warp_applied: bool = True):
        """Generate all diagnostics.

        When *warp_applied* is ``False`` (failed geometry), skip outputs
        that imply a successful registration.
        """
        self._save_matches_csv()
        self._save_metrics_json(registered_tgt, registered_valid, warp_applied)
        self._draw_raw_matches()
        self._draw_inlier_matches()
        self._draw_coverage()
        self._draw_checkerboard("before", self.pair.ref_raw, self.pair.tgt_raw,
                                self.pair.ref_valid, self.pair.tgt_valid)
        if warp_applied:
            self._draw_checkerboard("after", self.pair.ref_raw, registered_tgt,
                                    self.pair.ref_valid, registered_valid)
            self._save_registered_geotiff(registered_tgt, registered_valid)
            self._save_mosaic(registered_tgt, registered_valid)
        else:
            logger.info("  [%s] skipping warp/mosaic (geometry failed)", self.method)

    # -----------------------------------------------------------------------
    # CSV
    # -----------------------------------------------------------------------

    def _save_matches_csv(self):
        path = self.out / "matches.csv"
        n = len(self.matches.ref_xy)
        rows = []
        inlier = self.geom.inlier_mask if self.geom.inlier_mask is not None else np.zeros(n, dtype=bool)
        for i in range(n):
            rows.append({
                "idx": i,
                "ref_x": self.matches.ref_xy[i, 0],
                "ref_y": self.matches.ref_xy[i, 1],
                "tgt_x": self.matches.tgt_xy[i, 0],
                "tgt_y": self.matches.tgt_xy[i, 1],
                "confidence": self.matches.confidence[i],
                "inlier": int(inlier[i]),
            })
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys()) if rows else csv.writer(f)
            if rows:
                w.writeheader()
                w.writerows(rows)
        logger.info("  [%s] matches.csv  (%d rows)", self.method, n)

    # -----------------------------------------------------------------------
    # JSON metrics
    # -----------------------------------------------------------------------

    def _save_metrics_json(self, registered_tgt, registered_valid, warp_applied=True):
        row_start, row_end, col_start, col_end = self.pair.overlap_window
        ref_crop = self.pair.ref_raw[row_start:row_end, col_start:col_end]
        tgt_crop_before = self.pair.tgt_raw[row_start:row_end, col_start:col_end]
        tgt_crop_after = registered_tgt[row_start:row_end, col_start:col_end]
        valid_ref_crop = self.pair.ref_valid[row_start:row_end, col_start:col_end]
        valid_tgt_crop_before = self.pair.tgt_valid[row_start:row_end, col_start:col_end]
        valid_tgt_crop_after = registered_valid[row_start:row_end, col_start:col_end]

        joint_before = valid_ref_crop & valid_tgt_crop_before
        joint_after = valid_ref_crop & valid_tgt_crop_after

        ncc_before = gradient_ncc(ref_crop, tgt_crop_before, joint_before)
        ncc_after = gradient_ncc(ref_crop, tgt_crop_after, joint_after)

        # Phase verification on entire overlap
        verify = phase_verification(
            self.pair.ref_raw, registered_tgt,
            self.pair.ref_valid, registered_valid,
        )

        # Coverage (on inliers only)
        inlier_pts = self.matches.ref_xy[self.geom.inlier_mask] if self.geom.inlier_mask is not None and self.geom.inlier_mask.any() else np.empty((0, 2))
        coverage = spatial_coverage_ratio(inlier_pts, self.pair.overlap_window)

        metrics = {
            "method": self.method,
            "raw_matches": int(self.geom.n_raw),
            "inliers": int(self.geom.n_inlier),
            "inlier_ratio": float(self.geom.inlier_ratio),
            "coverage": float(coverage),
            "residual_median": self.geom.residual_median,
            "residual_rmse": self.geom.residual_rmse,
            "residual_p90": self.geom.residual_p90,
            "residual_p95": self.geom.residual_p95,
            "residual_max": self.geom.residual_max,
            "gradient_ncc_before": ncc_before,
            "gradient_ncc_after": ncc_after,
            "verification_dx": verify["dx"],
            "verification_dy": verify["dy"],
            "verification_magnitude": verify["magnitude"],
            "verification_confidence": verify["confidence"],
            "match_runtime_sec": self.matches.runtime_sec,
            "warp_applied": warp_applied,
            "status": self.geom.status,
        }
        with open(self.out / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2, default=_json_default)
        logger.info("  [%s] metrics.json", self.method)

    # -----------------------------------------------------------------------
    # Figure: raw matches
    # -----------------------------------------------------------------------

    def _draw_raw_matches(self):
        """Draw up to MAX_DRAW_MATCHES raw matches, side by side."""
        n = len(self.matches.ref_xy)
        if n == 0:
            logger.info("  [%s] matches_raw.png  (skip: 0 matches)", self.method)
            return

        # Sort by confidence descending, take top N
        idx = np.argsort(self.matches.confidence)[::-1][:MAX_DRAW_MATCHES]
        ref_pts = self.matches.ref_xy[idx]
        tgt_pts = self.matches.tgt_xy[idx]

        _draw_match_figure(
            self.pair.ref_raw,
            self.pair.tgt_raw,
            ref_pts,
            tgt_pts,
            self.out / "matches_raw.png",
            title=f"{self.method} — raw matches (top {len(idx)}/{n})",
        )

    # -----------------------------------------------------------------------
    # Figure: inlier matches
    # -----------------------------------------------------------------------

    def _draw_inlier_matches(self):
        if self.geom.inlier_mask is None or not self.geom.inlier_mask.any():
            logger.info("  [%s] matches_inliers.png  (skip: 0 inliers)", self.method)
            return

        ref_in = self.matches.ref_xy[self.geom.inlier_mask]
        tgt_in = self.matches.tgt_xy[self.geom.inlier_mask]

        # Cap at MAX_DRAW_MATCHES
        if len(ref_in) > MAX_DRAW_MATCHES:
            idx = np.random.default_rng(42).choice(len(ref_in), MAX_DRAW_MATCHES, replace=False)
            ref_in = ref_in[idx]
            tgt_in = tgt_in[idx]

        _draw_match_figure(
            self.pair.ref_raw,
            self.pair.tgt_raw,
            ref_in,
            tgt_in,
            self.out / "matches_inliers.png",
            title=f"{self.method} — inliers ({len(ref_in)}/{self.geom.n_inlier})",
        )

    # -----------------------------------------------------------------------
    # Figure: coverage
    # -----------------------------------------------------------------------

    def _draw_coverage(self):
        """Draw overlap region with 8×8 grid and inlier locations."""
        row_start, row_end, col_start, col_end = self.pair.overlap_window
        height = row_end - row_start
        width = col_end - col_start

        if height <= 0 or width <= 0:
            logger.info("  [%s] coverage.png  (skip: empty overlap)", self.method)
            return

        # Background: target overlap
        bg = self.pair.tgt_raw[row_start:row_end, col_start:col_end].copy()
        bg = np.clip(bg, np.nanpercentile(bg, 2), np.nanpercentile(bg, 98))
        bg = (bg - np.nanmin(bg)) / max(np.nanmax(bg) - np.nanmin(bg), 1e-8)

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(bg, cmap="gray", extent=[col_start, col_end, row_end, row_start])

        # 8×8 grid
        for i in range(9):
            ax.axhline(row_start + i * height / 8, color="cyan", linewidth=0.5, alpha=0.6)
            ax.axvline(col_start + i * width / 8, color="cyan", linewidth=0.5, alpha=0.6)

        # Inlier points
        if self.geom.inlier_mask is not None and self.geom.inlier_mask.any():
            pts = self.matches.ref_xy[self.geom.inlier_mask]
            ax.scatter(pts[:, 0], pts[:, 1], c="red", s=8, alpha=0.7, marker="o")

        ax.set_title(f"{self.method} — coverage ({self.geom.n_inlier} inliers)")
        fig.tight_layout()
        fig.savefig(self.out / "coverage.png", dpi=150)
        plt.close(fig)
        logger.info("  [%s] coverage.png", self.method)

    # -----------------------------------------------------------------------
    # Figure: checkerboard
    # -----------------------------------------------------------------------

    def _draw_checkerboard(self, tag, ref, tgt, valid_ref, valid_tgt, tile=64):
        row_start, row_end, col_start, col_end = self.pair.overlap_window
        ref_c = ref[row_start:row_end, col_start:col_end]
        tgt_c = tgt[row_start:row_end, col_start:col_end]
        valid_ref_c = valid_ref[row_start:row_end, col_start:col_end]
        valid_tgt_c = valid_tgt[row_start:row_end, col_start:col_end]
        joint = valid_ref_c & valid_tgt_c & np.isfinite(ref_c) & np.isfinite(tgt_c)

        h, w = ref_c.shape
        if h < tile or w < tile:
            logger.info("  [%s] checkerboard_%s.png  (skip: too small)", self.method, tag)
            return

        # Create checkerboard: alternate tiles from ref and tgt
        cb = np.full((h, w), np.nan, dtype=np.float64)
        for r in range(0, h, tile):
            for c in range(0, w, tile):
                r2 = min(r + tile, h)
                c2 = min(c + tile, w)
                if ((r // tile) + (c // tile)) % 2 == 0:
                    cb[r:r2, c:c2] = ref_c[r:r2, c:c2]
                else:
                    cb[r:r2, c:c2] = tgt_c[r:r2, c:c2]

        # Percentile clip for display
        v = cb[joint]
        if len(v) < 10:
            logger.info("  [%s] checkerboard_%s.png  (skip: too few valid)", self.method, tag)
            return
        p2, p98 = np.nanpercentile(v, [2, 98])
        display = np.clip(cb, p2, p98)

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(display, cmap="gray")
        ax.set_title(f"{self.method} — checkerboard {tag}")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(self.out / f"checkerboard_{tag}.png", dpi=150)
        plt.close(fig)
        logger.info("  [%s] checkerboard_%s.png", self.method, tag)

    # -----------------------------------------------------------------------
    # registered_target.tif
    # -----------------------------------------------------------------------

    def _save_registered_geotiff(self, registered_tgt, registered_valid):
        from src.io_utils import write_geotiff

        path = str(self.out / "registered_target.tif")
        arr = registered_tgt.copy()
        arr[~registered_valid] = np.nan
        write_geotiff(
            path, arr,
            self.pair.transform, str(self.pair.crs),
            nodata=np.nan, dtype="float64",
        )
        logger.info("  [%s] registered_target.tif", self.method)

    # -----------------------------------------------------------------------
    # mosaic_source_selection.tif
    # -----------------------------------------------------------------------

    def _save_mosaic(self, registered_tgt, registered_valid):
        """Create a source-selection mosaic of ref + registered target."""
        # Prepare arrays as (bands, rows, cols) for create_mosaic
        ref_arr = self.pair.ref_raw.copy()
        tgt_arr = registered_tgt.copy()

        # Apply valid masks as NaN
        ref_arr[~self.pair.ref_valid] = np.nan
        tgt_arr[~registered_valid] = np.nan

        path = str(self.out / "mosaic_source_selection.tif")
        try:
            create_mosaic(
                arrays=[
                    ref_arr[np.newaxis, :, :],
                    tgt_arr[np.newaxis, :, :],
                ],
                transforms=[self.pair.transform, self.pair.transform],
                crs=str(self.pair.crs),
                nodata_values=[np.nan, np.nan],
                output_path=path,
                mode="source_selection",
            )
        except Exception as e:
            logger.warning("  [%s] mosaic creation failed: %s", self.method, e)
            return
        logger.info("  [%s] mosaic_source_selection.tif", self.method)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _draw_match_figure(ref_img, tgt_img, ref_pts, tgt_pts, path, title):
    """Side-by-side match figure: Reference | Target."""
    # Percentile clip each image for display
    def _clip_for_display(arr):
        v = arr[np.isfinite(arr)]
        if len(v) < 10:
            return np.zeros_like(arr)
        p2, p98 = np.nanpercentile(v, [2, 98])
        clipped = np.clip(arr, p2, p98)
        return (clipped - p2) / max(p98 - p2, 1e-8)

    ref_d = _clip_for_display(ref_img)
    tgt_d = _clip_for_display(tgt_img)

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(16, 8))
    ax_l.imshow(ref_d, cmap="gray")
    ax_l.scatter(ref_pts[:, 0], ref_pts[:, 1], c="lime", s=3, alpha=0.7)
    ax_l.set_title("Reference")
    ax_l.axis("off")

    ax_r.imshow(tgt_d, cmap="gray")
    ax_r.scatter(tgt_pts[:, 0], tgt_pts[:, 1], c="lime", s=3, alpha=0.7)
    ax_r.set_title("Target")
    ax_r.axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _json_default(obj):
    """Handle non-serialisable types in JSON."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        if np.isnan(obj):
            return None
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)