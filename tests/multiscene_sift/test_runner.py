"""Synthetic end-to-end tests for five-scene SIFT mosaic pipeline."""

from __future__ import annotations

import json
import pytest

from src.multiscene_sift.runner import run_five_scene_mosaic


class TestE2ESynthetic:
    """Synthetic 3-scene miniature end-to-end test."""

    @pytest.fixture
    def synthetic_input(self, tmp_path):
        """Create a synthetic 3-scene dataset."""
        from tests.multiscene_sift.conftest import make_five_scene_dataset
        root = tmp_path / "flat"
        names = make_five_scene_dataset(root)
        # Use only first 3 scenes (a linear chain)
        return str(root), names[:3]

    def test_pipeline_runs_and_produces_structure(self, synthetic_input, tmp_path):
        """Pipeline should run without exception and produce expected outputs.

        On tiny 128×128 synthetic images SIFT may or may not succeed —
        the test verifies the infrastructure regardless.
        """
        input_root, scene_names = synthetic_input
        out_dir = tmp_path / "output"

        result = run_five_scene_mosaic(
            input_root=input_root,
            output_dir=str(out_dir),
            scene_names=scene_names,
        )

        out = tmp_path / "output"

        # Always produced
        assert (out / "dataset_manifest.json").exists()
        assert (out / "overlap_graph.json").exists()
        assert (out / "pairwise_summary.json").exists()
        assert (out / "run_config.json").exists()
        assert (out / "run_summary.json").exists()

        if result["status"] == "OK":
            # Full pipeline outputs
            assert (out / "global_registration.json").exists()
            assert (out / "spanning_tree.json").exists()
            assert (out / "global_transforms.json").exists()
            assert (out / "global_edge_consistency.json").exists()
            assert (out / "mosaic_B14_before_registration_source_selection.tif").exists()
            assert (out / "mosaic_B14_after_registration_source_selection.tif").exists()
            assert (out / "mosaic_B14_weighted.tif").exists()
            assert (out / "mosaic_B8_weighted.tif").exists()
            assert (out / "mosaic_B5_weighted.tif").exists()
            assert (out / "mosaic_RGB_B14_B8_B5.tif").exists()
            assert (out / "mosaic_RGB_B14_B8_B5_preview.png").exists()
        else:
            # Disconnected: should have components file
            assert (out / "accepted_graph_components.json").exists()

    def test_run_summary_has_required_keys(self, synthetic_input, tmp_path):
        """Run summary must have required keys regardless of status."""
        input_root, scene_names = synthetic_input
        out_dir = tmp_path / "output"

        run_five_scene_mosaic(
            input_root=input_root,
            output_dir=str(out_dir),
            scene_names=scene_names,
        )

        with open(out_dir / "run_summary.json") as f:
            summary = json.load(f)

        required_keys = [
            "status", "n_scenes", "geographic_edges",
            "accepted_sift_edges", "failed_sift_edges",
            "runtime_sec",
        ]
        for key in required_keys:
            assert key in summary, f"Missing key in run_summary: {key}"

    def test_disconnected_graph_is_reported(self, tmp_path):
        """Non-overlapping scenes should report DISCONNECTED status."""
        root = tmp_path / "flat"
        root.mkdir(parents=True)
        from tests.multiscene_sift.conftest import _make_band

        for name, ox in [("scene_a", 400000), ("scene_b", 600000)]:
            d = root / name
            d.mkdir(parents=True)
            for band in ("B14", "B8", "B5"):
                _make_band(d / f"{name}_{band}.TIF", origin_x=ox)

        out_dir = tmp_path / "output"

        from src.multiscene_sift.dataset import discover_five_scenes
        scenes, _ = discover_five_scenes(str(root), ["scene_a", "scene_b"])

        with pytest.raises(ValueError, match="DISCONNECTED"):
            from src.multiscene_sift.overlap_graph import build_geographic_overlap_graph
            build_geographic_overlap_graph(scenes)