import json

import pytest

from src.multiscene_sift.radiometric_protocol import (
    GEOMETRY_SPECS,
    RADIOMETRIC_METHODS,
    build_task10_config,
    write_run_metadata,
)


def test_task10_config_freezes_dataset_geometry_and_run_matrix(tmp_path):
    config = build_task10_config(tmp_path)

    assert config["dataset"] == "B9"
    assert config["manifest_indices"] == [2, 3, 5, 8, 10]
    assert config["pixel_size_m"] == 14.0
    assert tuple(config["radiometric_methods"]) == RADIOMETRIC_METHODS
    assert tuple(config["geometry_runs"]) == tuple(GEOMETRY_SPECS)
    assert config["output_root"] == str(
        tmp_path / "data" / "output" / "b9_five_scene_validation" / "radiometric_runs_1024"
    )


def test_write_run_metadata_persists_required_task10_contract(tmp_path):
    config = build_task10_config(tmp_path)
    run_dir = tmp_path / "run"

    write_run_metadata(
        run_dir,
        config,
        geometry_source={"matcher": "sift", "global_method": "mst"},
        radiometric_method={"method": "RAW", "cloud_mask_enabled": False},
    )

    for name in ("run_config.json", "geometry_source.json", "radiometric_method.json"):
        assert (run_dir / name).is_file()
    assert json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))["dataset"] == "B9"
    assert json.loads((run_dir / "geometry_source.json").read_text(encoding="utf-8"))["global_method"] == "mst"
    assert json.loads((run_dir / "radiometric_method.json").read_text(encoding="utf-8"))["method"] == "RAW"


def test_task10_config_rejects_unknown_method():
    config = build_task10_config("C:/repo")
    config["radiometric_methods"] = ["RAW", "VOLRN"]

    from src.multiscene_sift.radiometric_protocol import validate_task10_config

    with pytest.raises(ValueError, match="radiometric_methods"):
        validate_task10_config(config)
