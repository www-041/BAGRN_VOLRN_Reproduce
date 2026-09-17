"""Tests for :mod:`src.registration_benchmark.diagnostics`."""

import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import Affine
from skimage.transform import AffineTransform

from src.registration_benchmark.diagnostics import MethodDiagnostics
from src.registration_benchmark.geometry import GeometryResult, STATUS_OK
from src.registration_benchmark.models import CommonGridPair, MatchSet


def _make_fake_pair() -> CommonGridPair:
    """Build a minimal CommonGridPair for diagnostic testing."""
    rng = np.random.default_rng(42)
    ref = rng.uniform(50, 200, size=(128, 128)).astype(np.float64)
    tgt = rng.uniform(50, 200, size=(128, 128)).astype(np.float64)
    return CommonGridPair(
        ref_raw=ref,
        tgt_raw=tgt,
        ref_valid=np.ones((128, 128), dtype=bool),
        tgt_valid=np.ones((128, 128), dtype=bool),
        transform=Affine(1.0, 0.0, 0.0, 0.0, -1.0, 128.0),
        crs=rasterio.crs.CRS.from_string("EPSG:4326"),
        overlap_window=(10, 118, 10, 118),
    )


def _make_fake_matches(n: int = 50) -> MatchSet:
    rng = np.random.default_rng(99)
    return MatchSet(
        method="test",
        ref_xy=rng.uniform(10, 118, size=(n, 2)),
        tgt_xy=rng.uniform(10, 118, size=(n, 2)),
        confidence=rng.uniform(0.3, 1.0, size=n),
        runtime_sec=0.5,
    )


def _make_fake_geom(matches: MatchSet) -> GeometryResult:
    n = len(matches.ref_xy)
    inlier = np.zeros(n, dtype=bool)
    inlier[:40] = True  # First 40 are inliers
    residuals = np.linalg.norm(matches.ref_xy - matches.tgt_xy, axis=1)
    inl_res = residuals[inlier]
    n_inl = int(inlier.sum())
    return GeometryResult(
        model=AffineTransform(matrix=[[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        status=STATUS_OK,
        inlier_mask=inlier,
        n_raw=n,
        n_inlier=n_inl,
        inlier_ratio=n_inl / n,
        residual_median=float(np.median(inl_res)) if n_inl > 0 else float("nan"),
        residual_rmse=float(np.sqrt(np.mean(inl_res ** 2))) if n_inl > 0 else float("nan"),
        residual_p90=float(np.percentile(inl_res, 90)) if n_inl > 1 else float("nan"),
        residual_p95=float(np.percentile(inl_res, 95)) if n_inl > 1 else float("nan"),
        residual_max=float(np.max(inl_res)) if n_inl > 0 else float("nan"),
        affine_scale_x=1.0,
        affine_scale_y=1.0,
        affine_rotation_deg=0.0,
        affine_shear_deg=0.0,
        residual_threshold=2.0,
    )


class TestMethodDiagnostics:
    """Integration tests for diagnostic outputs."""

    def test_all_outputs_created(self, tmp_dir):
        pair = _make_fake_pair()
        matches = _make_fake_matches(50)
        geom = _make_fake_geom(matches)

        # Simulate registered target (slightly shifted)
        registered = pair.tgt_raw.copy()
        registered_valid = pair.tgt_valid.copy()

        diag = MethodDiagnostics(str(tmp_dir), "phase", pair, matches, geom)
        diag.run_all(registered, registered_valid)

        out = Path(tmp_dir) / "phase"

        # Check all expected files exist
        expected = [
            "matches_raw.png",
            "matches_inliers.png",
            "coverage.png",
            "checkerboard_before.png",
            "checkerboard_after.png",
            "registered_target.tif",
            "mosaic_source_selection.tif",
            "matches.csv",
            "metrics.json",
        ]
        for fname in expected:
            assert (out / fname).exists(), f"Missing: {fname}"

    def test_metrics_json_has_required_keys(self, tmp_dir):
        pair = _make_fake_pair()
        matches = _make_fake_matches(50)
        geom = _make_fake_geom(matches)

        registered = pair.tgt_raw.copy()
        registered_valid = pair.tgt_valid.copy()

        diag = MethodDiagnostics(str(tmp_dir), "sift", pair, matches, geom)
        diag.run_all(registered, registered_valid)

        with open(Path(tmp_dir) / "sift" / "metrics.json") as f:
            data = json.load(f)

        required = [
            "method", "raw_matches", "inliers", "inlier_ratio", "coverage",
            "residual_median", "residual_rmse", "residual_p90", "residual_p95",
            "residual_max", "gradient_ncc_before", "gradient_ncc_after",
            "verification_dx", "verification_dy", "verification_magnitude",
            "verification_confidence", "verification_status",
            "match_runtime_sec", "status",
        ]
        for key in required:
            assert key in data, f"Missing key: {key}"

    def test_matches_csv_has_header(self, tmp_dir):
        pair = _make_fake_pair()
        matches = _make_fake_matches(20)
        geom = _make_fake_geom(matches)

        registered = pair.tgt_raw.copy()
        registered_valid = pair.tgt_valid.copy()

        diag = MethodDiagnostics(str(tmp_dir), "phase", pair, matches, geom)
        diag.run_all(registered, registered_valid)

        csv_path = Path(tmp_dir) / "phase" / "matches.csv"
        with open(csv_path) as f:
            header = f.readline().strip()
        assert "ref_x" in header
        assert "tgt_x" in header
        assert "inlier" in header

    def test_handles_zero_matches(self, tmp_dir):
        """Diagnostics should not crash with zero matches."""
        pair = _make_fake_pair()
        matches = MatchSet(
            method="empty",
            ref_xy=np.empty((0, 2)),
            tgt_xy=np.empty((0, 2)),
            confidence=np.empty(0),
            runtime_sec=0.0,
        )
        geom = GeometryResult(
            model=None,
            status="TOO_FEW_INLIERS",
            inlier_mask=np.zeros(0, dtype=bool),
            n_raw=0,
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
            residual_threshold=2.0,
        )

        registered = pair.tgt_raw.copy()
        registered_valid = pair.tgt_valid.copy()

        diag = MethodDiagnostics(str(tmp_dir), "phase", pair, matches, geom)
        # Must not raise
        diag.run_all(registered, registered_valid)

        # At minimum, CSV, JSON, and registered GeoTIFF should exist
        assert (Path(tmp_dir) / "phase" / "matches.csv").exists()
        assert (Path(tmp_dir) / "phase" / "metrics.json").exists()
        assert (Path(tmp_dir) / "phase" / "registered_target.tif").exists()