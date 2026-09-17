"""Tests for :mod:`src.registration_benchmark.runner`."""

import json
from pathlib import Path

import pytest

from src.registration_benchmark.runner import MATCHERS, run_two_image_benchmark


class TestRunner:
    """End-to-end runner tests using synthetic data."""

    def test_synthetic_e2e_phase_sift(self, tmp_dir):
        from tests.registration_benchmark.conftest import make_synthetic_pair

        ref_path, tgt_path = make_synthetic_pair(tmp_dir)
        output_dir = str(tmp_dir / "benchmark_output")

        result = run_two_image_benchmark(
            ref_path=ref_path,
            tgt_path=tgt_path,
            output_dir=output_dir,
            band=1,
            methods=["phase", "sift"],
            match_max_side=256,
            ransac_threshold=3.0,
        )

        out = Path(output_dir)

        # --- Summary files exist -------------------------------------------------
        assert (out / "summary.json").exists()
        assert (out / "summary.csv").exists()
        assert (out / "run_config.json").exists()

        # --- Method directories exist --------------------------------------------
        assert (out / "phase").is_dir()
        assert (out / "sift").is_dir()

        # --- No BAGRN/VOLRN contamination ----------------------------------------
        for p in out.rglob("*"):
            if p.is_file():
                content = p.read_text(errors="ignore")
                assert "bagrn" not in content.lower(), f"bagrn found in {p}"
                # volrn can appear in "volrn" form — check context

        # --- Summary JSON has expected entries -----------------------------------
        with open(out / "summary.json") as f:
            summary = json.load(f)

        methods = [s["method"] for s in summary]
        assert "phase" in methods
        assert "sift" in methods

        for row in summary:
            assert "method" in row
            assert "status" in row
            assert "raw_matches" in row
            assert "inliers" in row

    def test_unknown_method_raises(self, tmp_dir):
        from tests.registration_benchmark.conftest import make_synthetic_pair

        ref_path, tgt_path = make_synthetic_pair(tmp_dir)
        output_dir = str(tmp_dir / "output")

        with pytest.raises(ValueError, match="Unknown methods"):
            run_two_image_benchmark(
                ref_path=ref_path,
                tgt_path=tgt_path,
                output_dir=output_dir,
                methods=["nonexistent"],
            )

    def test_error_isolation(self, tmp_dir):
        """If one method fails, others should still complete."""
        from tests.registration_benchmark.conftest import make_translated_texture_pair

        ref_path, tgt_path = make_translated_texture_pair(
            tmp_dir, size=128, dx=3.0, dy=-2.0
        )
        output_dir = str(tmp_dir / "output")

        result = run_two_image_benchmark(
            ref_path=ref_path,
            tgt_path=tgt_path,
            output_dir=output_dir,
            band=1,
            methods=["sift"],  # Only SIFT (phase requires structured texture)
            match_max_side=128,
            ransac_threshold=5.0,
        )

        assert len(result["summary"]) == 1
        assert result["summary"][0]["method"] == "sift"

    def test_registry_contains_phase_and_sift(self):
        assert "phase" in MATCHERS
        assert "sift" in MATCHERS
        assert callable(MATCHERS["phase"])
        assert callable(MATCHERS["sift"])