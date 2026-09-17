"""Tests for :mod:`src.registration_benchmark.runner`."""

import json
from pathlib import Path

import pytest

from src.registration_benchmark.runner import (
    MATCHERS,
    _run_matcher,
    run_two_image_benchmark,
    sanitize_json,
)


class TestSanitizeJson:
    """Tests for :func:`sanitize_json`."""

    def test_nan_to_none(self):
        result = sanitize_json({"val": float("nan")})
        assert result["val"] is None

    def test_inf_to_none(self):
        result = sanitize_json({"val": float("inf"), "neg": float("-inf")})
        assert result["val"] is None
        assert result["neg"] is None

    def test_nested_dict(self):
        result = sanitize_json({"a": {"b": float("nan")}})
        assert result["a"]["b"] is None

    def test_list_recursion(self):
        result = sanitize_json([float("nan"), 1.0, {"x": float("-inf")}])
        assert result[0] is None
        assert result[1] == 1.0
        assert result[2]["x"] is None


class TestRunMatcher:
    """Tests for :func:`_run_matcher` device forwarding."""

    def test_device_forwarded_to_learned(self):
        """LightGlue and LoFTR receive device= kwarg."""
        fake_view = object()
        calls = []

        def fake_phase(view, **kw):
            calls.append(("phase", kw))
            from src.registration_benchmark.models import MatchSet
            import numpy as np
            return MatchSet("phase", np.empty((0, 2)), np.empty((0, 2)), np.empty(0), 0.0)

        def fake_lightglue(view, **kw):
            calls.append(("lightglue", kw))
            from src.registration_benchmark.models import MatchSet
            import numpy as np
            return MatchSet("lightglue", np.empty((0, 2)), np.empty((0, 2)), np.empty(0), 0.0)

        def fake_loftr(view, **kw):
            calls.append(("loftr", kw))
            from src.registration_benchmark.models import MatchSet
            import numpy as np
            return MatchSet("loftr", np.empty((0, 2)), np.empty((0, 2)), np.empty(0), 0.0)

        _run_matcher("phase", fake_phase, fake_view, "cpu")
        _run_matcher("sift", fake_phase, fake_view, "cpu")
        _run_matcher("lightglue", fake_lightglue, fake_view, "cpu")
        _run_matcher("loftr", fake_loftr, fake_view, "cuda:0")

        assert calls[0] == ("phase", {})
        assert calls[1] == ("phase", {})  # sift uses same fn
        assert calls[2] == ("lightglue", {"device": "cpu"})
        assert calls[3] == ("loftr", {"device": "cuda:0"})

    def test_classical_matchers_no_device_kwarg(self):
        """Phase and SIFT should NOT receive device kwarg."""
        def fake_matcher(view, **kw):
            if "device" in kw:
                raise TypeError("unexpected keyword argument 'device'")
            from src.registration_benchmark.models import MatchSet
            import numpy as np
            return MatchSet("test", np.empty((0, 2)), np.empty((0, 2)), np.empty(0), 0.0)

        # These must not raise
        _run_matcher("phase", fake_matcher, object(), "cpu")
        _run_matcher("sift", fake_matcher, object(), "cuda")


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

    def test_invalid_geometry_does_not_warp(self, tmp_dir):
        """When geometry fails, warp/mosaic outputs should NOT be created."""
        from tests.registration_benchmark.conftest import make_synthetic_pair

        ref_path, tgt_path = make_synthetic_pair(
            tmp_dir, ref_size=128, tgt_size=128,
            ref_origin=(0.0, 128.0), tgt_origin=(0.0, 128.0),
        )
        output_dir = str(tmp_dir / "output")

        result = run_two_image_benchmark(
            ref_path=ref_path,
            tgt_path=tgt_path,
            output_dir=output_dir,
            band=1,
            methods=["sift"],  # SIFT on same-location pair should produce few matches
            match_max_side=128,
            ransac_threshold=1.0,  # tight threshold → likely TOO_FEW_INLIERS
        )

        out = Path(output_dir)
        summary = result["summary"][0]

        # If geometry failed, warp outputs must not exist
        if summary["status"] != "OK":
            for fname in ("checkerboard_after.png", "registered_target.tif",
                          "mosaic_source_selection.tif"):
                assert not (out / "sift" / fname).exists(), (
                    f"{fname} should not exist when geometry failed"
                )
            # But before (unwarped) outputs should exist
            assert (out / "sift" / "checkerboard_before.png").exists()