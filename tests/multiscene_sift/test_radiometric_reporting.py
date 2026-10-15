"""Tests for radiometric metrics reporting."""

import json
import numpy as np

from src.multiscene_sift.radiometric import BandRadiometricResult
from src.multiscene_sift.radiometric_reporting import (
    collect_radiometric_metrics,
    save_radiometric_metrics,
    save_normalization_info,
)


def _make_fake_result(band: str) -> BandRadiometricResult:
    """Create a minimal BandRadiometricResult for testing."""
    return BandRadiometricResult(
        band=band,
        original_metrics={},
        bagrn_metrics={"adm": 5.0, "adsd": 3.0, "cd": 2.0, "gl": 0.01, "rdoa": 3.33, "ave": 2.5},
        volrn_metrics={"adm": 2.0, "adsd": 1.5, "cd": 1.0, "gl": 0.005, "rdoa": 1.5, "ave": 1.13},
        theta_mu=np.array([0.5, 0.6, 0.7]),
        theta_sigma=np.array([0.1, 0.1, 0.1]),
        bagrn_runtime_sec=0.5,
        volrn_runtime_sec=1.2,
        normalized_arrays=[np.ones((1, 32, 32))],
        nodata_values=[None],
        corrected_transforms=[],
        corrected_bounds=[],
        overlaps=[{"idx_i": 0, "idx_j": 1}],
        registered_metrics={"adm": 10.0, "adsd": 6.0, "cd": 5.0, "gl": 0.0, "rdoa": 7.0, "ave": 5.25},
    )


class TestRadiometricReporting:
    def test_csv_has_all_rows(self, tmp_path):
        """CSV should have 3 bands × 3 stages = 9 rows."""
        results = {
            "B14": _make_fake_result("B14"),
            "B8": _make_fake_result("B8"),
            "B5": _make_fake_result("B5"),
        }

        save_radiometric_metrics(results, tmp_path)

        csv_path = tmp_path / "radiometric_metrics.csv"
        assert csv_path.exists()

        with open(csv_path) as f:
            lines = f.readlines()
        # 1 header + 9 data rows
        assert len(lines) == 10

    def test_json_structure(self, tmp_path):
        """JSON should have per_band and mean_over_bands keys."""
        results = {"B14": _make_fake_result("B14")}

        save_radiometric_metrics(results, tmp_path)

        json_path = tmp_path / "radiometric_metrics.json"
        with open(json_path) as f:
            data = json.load(f)

        assert "per_band" in data
        assert "B14" in data["per_band"]
        assert "Registered" in data["per_band"]["B14"]
        assert "BAGRN" in data["per_band"]["B14"]
        assert "VOLRN" in data["per_band"]["B14"]
        assert "mean_over_bands" in data

    def test_normalization_info_saved(self, tmp_path):
        """Normalization info should have theta shapes and stats."""
        results = {"B14": _make_fake_result("B14")}

        save_normalization_info(results, 0, "scene_0", tmp_path)

        json_path = tmp_path / "radiometric_normalization_info.json"
        with open(json_path) as f:
            data = json.load(f)

        assert data["reference_idx"] == 0
        assert data["reference_name"] == "scene_0"
        band_info = data["per_band"]["B14"]
        assert band_info["theta_mu_shape"] == [3]
        assert "theta_mu_stats" in band_info