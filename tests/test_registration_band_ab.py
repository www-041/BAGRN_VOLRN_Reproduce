import json
from types import SimpleNamespace

import numpy as np
import pytest
from rasterio.transform import from_origin


def test_band_ab_configs_only_differ_in_declared_treatment_fields():
    import yaml
    from src.registration_band_ab import compare_band_ab_configs

    with open("configs/dz01_registration_band_ab_b14.yaml", encoding="utf-8") as handle:
        b14 = yaml.safe_load(handle)
    with open("configs/dz01_registration_band_ab_b12.yaml", encoding="utf-8") as handle:
        b12 = yaml.safe_load(handle)
    result = compare_band_ab_configs(b14, b12)
    assert result["valid"] is True
    assert set(result["differing_paths"]) == {"experiment_name", "registration_band"}


def test_band_ab_configs_load_both_b12_and_b14_for_every_scene():
    import yaml

    for name in ("b14", "b12"):
        with open(f"configs/dz01_registration_band_ab_{name}.yaml", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        assert config["selected_bands"] == ["B12", "B14"]
        assert all(set(scene["bands"]) == {"B12", "B14"} for scene in config["scenes"])


def test_band_ab_registration_params_are_exactly_equal():
    import yaml

    with open("configs/dz01_registration_band_ab_b14.yaml", encoding="utf-8") as handle:
        b14 = yaml.safe_load(handle)
    with open("configs/dz01_registration_band_ab_b12.yaml", encoding="utf-8") as handle:
        b12 = yaml.safe_load(handle)
    assert b14["registration_params"] == b12["registration_params"]


def test_reserve_validation_windows_override_never_resamples():
    from src.coregistration import reserve_validation_windows

    common = np.ones((64, 64), dtype=bool)
    override = [{"row": 3, "col": 5, "height": 16, "width": 16}]
    result = reserve_validation_windows(
        common, block_size_candidates=[32, 16], final_min_blocks=1,
        step=32, reserved_windows_override=override,
    )
    assert result["selected_block_size"] == 16
    assert [(item["row"], item["col"]) for item in result["reserved_windows"]] == [(3, 5)]


def test_holdout_manifest_rejects_window_below_current_valid_ratio():
    from src.coregistration import reserve_validation_windows

    common = np.ones((32, 32), dtype=bool)
    common[8:24, 8:24] = False
    with pytest.raises(ValueError, match="common_valid_ratio"):
        reserve_validation_windows(
            common, block_size_candidates=[16], final_min_blocks=1,
            reserved_windows_override=[{"row": 8, "col": 8, "height": 16, "width": 16}],
        )


def test_manifest_roundtrip_preserves_window_keys_and_block_size(tmp_path):
    from src.registration_band_ab import build_holdout_manifest, load_holdout_manifest, manifest_to_pair_overrides

    registration = {
        "scene_ids": ["a", "b"],
        "diagnostics": {"holdout": {"0-1": {"validation_reservation": {
            "selected_block_size": 384,
            "reserved_windows": [{"row": 2, "col": 4, "height": 384, "width": 384}],
        }}}},
    }
    manifest = build_holdout_manifest(
        registration, source_commit="abc", source_registration_band="B14", validation_band="B14"
    )
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    loaded = load_holdout_manifest(path)
    assert loaded["pairs"]["0-1"]["selected_block_size"] == 384
    assert manifest_to_pair_overrides(loaded, ["a", "b"])[(0, 1)][0]["col"] == 4


def test_validation_band_defaults_to_registration_band():
    from src.multiband_pipeline import _validate_final_registration_arrays

    seen = {}
    import src.coregistration as coregistration
    monkey = pytest.MonkeyPatch()
    def fake_validation(*args, **kwargs):
        seen["ref"] = args[0]
        return {"blocks": [], "stats": None, "coverage": None, "failure_reason": "synthetic"}
    monkey.setattr(coregistration, "validate_registration_independent_grid", fake_validation)
    monkey.setattr(coregistration, "aggregate_final_validation_quality", lambda *args, **kwargs: {
        "quality": "fail", "rmse": 1, "p95": 1, "median": 1, "mean_confidence": 0, "n_blocks": 0
    })
    arrays = [np.stack([np.ones((8, 8)), np.full((8, 8), 2)]), np.stack([np.ones((8, 8)), np.full((8, 8), 3)])]
    _validate_final_registration_arrays(
        arrays, 0, [from_origin(0, 8, 1, 1)] * 2, [None, None], [(0, 1)], [], {"final_min_blocks": 1}
    )
    assert seen["ref"].mean() == 1
    monkey.undo()


def test_b12_registration_can_validate_on_loaded_b14(monkeypatch):
    from src.multiband_pipeline import _validate_final_registration_arrays
    import src.coregistration as coregistration

    seen = {}
    def fake_validation(*args, **kwargs):
        seen["ref"] = args[0]
        return {"blocks": [], "stats": None, "coverage": None, "failure_reason": "synthetic"}
    monkeypatch.setattr(coregistration, "validate_registration_independent_grid", fake_validation)
    monkeypatch.setattr(coregistration, "aggregate_final_validation_quality", lambda *args, **kwargs: {
        "quality": "fail", "rmse": 1, "p95": 1, "median": 1, "mean_confidence": 0, "n_blocks": 0
    })
    arrays = [np.stack([np.ones((8, 8)), np.full((8, 8), 2)]), np.stack([np.ones((8, 8)), np.full((8, 8), 3)])]
    _validate_final_registration_arrays(
        arrays, 0, [from_origin(0, 8, 1, 1)] * 2, [None, None], [(0, 1)], [], {"final_min_blocks": 1},
        validation_band_idx=1,
    )
    assert seen["ref"].mean() == 2


def test_unknown_validation_band_fails_fast():
    from src.multiband_pipeline import MultibandPipeline

    pipeline = MultibandPipeline.__new__(MultibandPipeline)
    pipeline.common_bands = ["B12", "B14"]
    pipeline.registration_band_idx = 0
    pipeline.config = SimpleNamespace(registration_params={})
    with pytest.raises(ValueError, match="not loaded"):
        pipeline.register_scenes({"arrays": [np.zeros((2, 2, 2))], "transforms": [], "nodata_values": []}, [], diagnostic_validation_band="B11")


def test_compare_band_runs_rejects_different_validation_band():
    from src.registration_band_ab import compare_registration_band_runs

    base = {"scene_ids": ["a", "b"], "registration_band_name": "B14", "validation_band_name": "B14", "final_validation": {"edges": []}, "quality": {}}
    other = dict(base, registration_band_name="B12", validation_band_name="B12")
    result = compare_registration_band_runs(base, other)
    assert result["comparison_available"] is False
    assert result["conclusion"] is None


def test_compare_band_runs_classifies_directionally_better_b12():
    from src.registration_band_ab import compare_registration_band_runs

    def diag(band, rmse, p95):
        return {"scene_ids": ["a", "b"], "registration_band_name": band,
                "validation_band_name": "B14", "quality": {"rmse": rmse, "p95": p95},
                "final_validation": {"edges": [{"idx_i": 0, "idx_j": 1, "blocks": [{"validation_row": 1, "validation_col": 2, "block_size": 4}]}]}}
    result = compare_registration_band_runs(diag("B14", 2, 3), diag("B12", 1, 2))
    assert result["comparison_available"] is True
    assert result["delta_b14_minus_b12"]["rmse"] == 1
    assert result["conclusion"] == "B12_directionally_better"


def test_band_ab_metadata_rejects_transform_mismatch(monkeypatch):
    from src.registration_band_ab import validate_band_ab_metadata

    class FakeRaster:
        def __init__(self, path):
            self.crs = "EPSG:32649"
            self.width = self.height = 10
            self.transform = from_origin(0 if path.endswith("B14") else 1, 10, 1, 1)
            self.nodata = None
            self.count = 1
            self.dtypes = ["float32"]
        def __enter__(self): return self
        def __exit__(self, *args): return False

    monkeypatch.setattr("os.path.isfile", lambda path: True)
    monkeypatch.setattr("rasterio.open", lambda path: FakeRaster(path))
    configs = ({"scenes": [{"id": "a", "bands": {"B12": "aB12", "B14": "aB14"}},
                            {"id": "b", "bands": {"B12": "bB12", "B14": "bB14"}}]},
               {"scenes": [{"id": "a", "bands": {"B12": "aB12", "B14": "aB14"}},
                            {"id": "b", "bands": {"B12": "bB12", "B14": "bB14"}}]})
    result = validate_band_ab_metadata(*configs)
    assert result["valid"] is False
    assert any(item["field"] == "transform" for item in result["failures"])


def test_band_ab_metadata_rejects_missing_b12_path(monkeypatch):
    from src.registration_band_ab import validate_band_ab_metadata

    monkeypatch.setattr("os.path.isfile", lambda path: str(path).endswith("B14"))
    class FakeRaster:
        crs = "EPSG:32649"; width = height = 10; transform = from_origin(0, 10, 1, 1)
        nodata = None; count = 1; dtypes = ["float32"]
        def __enter__(self): return self
        def __exit__(self, *args): return False
    monkeypatch.setattr("rasterio.open", lambda path: FakeRaster())
    config = {"scenes": [{"id": "a", "bands": {"B12": "aB12", "B14": "aB14"}},
                         {"id": "b", "bands": {"B12": "bB12", "B14": "bB14"}}]}
    result = validate_band_ab_metadata(config, config)
    assert result["valid"] is False
    assert {item["band"] for item in result["failures"]} == {"B12"}


def test_band_ab_metadata_accepts_same_scene_geometry(monkeypatch):
    from src.registration_band_ab import validate_band_ab_metadata

    class FakeRaster:
        crs = "EPSG:32649"; width = height = 10; transform = from_origin(0, 10, 1, 1)
        nodata = None; count = 1; dtypes = ["float32"]
        def __enter__(self): return self
        def __exit__(self, *args): return False

    monkeypatch.setattr("os.path.isfile", lambda path: True)
    monkeypatch.setattr("rasterio.open", lambda path: FakeRaster())
    config = {"scenes": [{"id": "a", "bands": {"B12": "aB12", "B14": "aB14"}},
                         {"id": "b", "bands": {"B12": "bB12", "B14": "bB14"}}]}
    result = validate_band_ab_metadata(config, config)
    assert result["valid"] is True
