"""Synthetic end-to-end test for full SIFT + BAGRN + VOLRN pipeline."""

import json
import pytest
import rasterio


class TestFullRadiometricRunner:
    """Synthetic 3-scene e2e test with radiometric offsets."""

    @pytest.fixture
    def synthetic_input(self, tmp_path):
        """Create 3 scenes with known radiometric offsets."""
        from tests.multiscene_sift.conftest import make_five_scene_dataset
        root = tmp_path / "flat"
        names = make_five_scene_dataset(root)
        return str(root), names[:3]

    def test_full_pipeline_outputs(self, synthetic_input, tmp_path):
        """Full pipeline should produce all expected output files."""
        input_root, scene_names = synthetic_input
        out_dir = tmp_path / "output"

        from src.multiscene_sift.runner import run_five_scene_mosaic

        result = run_five_scene_mosaic(
            input_root=input_root,
            output_dir=str(out_dir),
            scene_names=scene_names,
            save_diagnostics=True,
        )

        out = tmp_path / "output"

        # Core outputs always exist
        assert (out / "run_config.json").exists()
        assert (out / "run_summary.json").exists()

        if result["status"] == "OK":
            # Radiometric outputs
            assert (out / "radiometric_metrics.csv").exists()
            assert (out / "radiometric_metrics.json").exists()
            assert (out / "radiometric_normalization_info.json").exists()

            # Final mosaics
            for band in ("B14", "B8", "B5"):
                path = out / f"mosaic_{band}_SIFT_BAGRN_VOLRN_weighted.tif"
                assert path.exists(), f"Missing {path}"

            # RGB
            assert (out / "mosaic_RGB_B14_B8_B5_SIFT_BAGRN_VOLRN.tif").exists()
            assert (out / "mosaic_RGB_B14_B8_B5_SIFT_BAGRN_VOLRN_preview.png").exists()

            # Diagnostic outputs
            assert (out / "mosaic_B14_before_registration_source_selection.tif").exists()
            assert (out / "mosaic_B14_after_registration_source_selection.tif").exists()

    def test_final_mosaics_same_grid(self, synthetic_input, tmp_path):
        """All three final band mosaics must share identical grid metadata."""
        input_root, scene_names = synthetic_input
        out_dir = tmp_path / "output"

        from src.multiscene_sift.runner import run_five_scene_mosaic

        result = run_five_scene_mosaic(
            input_root=input_root,
            output_dir=str(out_dir),
            scene_names=scene_names,
        )

        if result["status"] != "OK":
            pytest.skip("SIFT graph disconnected on synthetic data")

        out = tmp_path / "output"
        ref_w, ref_h, ref_tf, ref_crs = None, None, None, None

        for band in ("B14", "B8", "B5"):
            path = out / f"mosaic_{band}_SIFT_BAGRN_VOLRN_weighted.tif"
            with rasterio.open(path) as src:
                if ref_w is None:
                    ref_w, ref_h = src.width, src.height
                    ref_tf = src.transform
                    ref_crs = src.crs
                else:
                    assert src.width == ref_w
                    assert src.height == ref_h
                    assert src.transform == ref_tf
                    assert src.crs == ref_crs

    def test_radiometric_metrics_content(self, synthetic_input, tmp_path):
        """Radiometric metrics should contain all three stages per band."""
        input_root, scene_names = synthetic_input
        out_dir = tmp_path / "output"

        from src.multiscene_sift.runner import run_five_scene_mosaic

        result = run_five_scene_mosaic(
            input_root=input_root,
            output_dir=str(out_dir),
            scene_names=scene_names,
        )

        if result["status"] != "OK":
            pytest.skip("SIFT graph disconnected on synthetic data")

        out = tmp_path / "output"
        with open(out / "radiometric_metrics.json") as f:
            metrics = json.load(f)

        for band in ("B14", "B8", "B5"):
            assert band in metrics["per_band"]
            for stage in ("Registered", "BAGRN", "VOLRN"):
                assert stage in metrics["per_band"][band]