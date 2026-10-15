import pytest

from src.benchmark_v1_spectralmatch_phase2.pif_diagnostic import (
    assert_support_identity,
    evaluation_support_identity,
    validate_pif_method,
)


def test_only_two_predeclared_pif_modes_are_accepted():
    assert validate_pif_method("flood_from_match_points") == "flood_from_match_points"
    assert validate_pif_method("entire") == "entire"
    with pytest.raises(ValueError, match="unsupported PIF method"):
        validate_pif_method("best_result")


def test_pif_mode_cannot_change_frozen_evaluation_support():
    assert_support_identity("abc123", "abc123")
    with pytest.raises(ValueError, match="evaluation support changed"):
        assert_support_identity("abc123", "different")


def test_pair_support_identity_uses_exact_mask_intersection():
    import numpy as np

    mask_i = np.array([[1, 1], [0, 0]], dtype=np.uint8)
    mask_j = np.array([[1, 0], [1, 0]], dtype=np.uint8)
    first = evaluation_support_identity(mask_i, mask_j)
    same = evaluation_support_identity(mask_i.copy(), mask_j.copy())
    assert first == same
    assert first[1] == 1


def test_adapter_passes_explicit_mode_to_upstream_global_regression(tmp_path):
    pytest.importorskip("rasterio")
    from src.benchmark_v1_spectralmatch_phase2.adapter import execute_fixed_grid_stages

    class Seamline:
        @staticmethod
        def voronoi(**kwargs):
            return kwargs["output_mask"]

    class FakeApi:
        def __init__(self):
            self.global_kwargs = None
            self.Seamline = Seamline

        def global_regression(self, *, input_images, output_images, **kwargs):
            self.global_kwargs = kwargs
            return output_images

        def local_block_adjustment(self, *, input_images, output_images, **kwargs):
            return output_images

        def create_footprints(self, *, output_polygons, **kwargs):
            return output_polygons

        def mask_rasters(self, *, output_images, **kwargs):
            return output_images

        def merge_rasters(self, *, output_image_path, **kwargs):
            return output_image_path

    api = FakeApi()
    execute_fixed_grid_stages(api, ["i.tif", "j.tif"], tmp_path / "out", pif_method="entire")
    assert api.global_kwargs["pif_method"] == "entire"
