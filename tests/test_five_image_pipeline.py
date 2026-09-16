import json
from pathlib import Path

import numpy as np
import pytest
import rasterio


def _make_scene_dir(root: Path, name: str) -> Path:
    scene_dir = root / name
    scene_dir.mkdir()
    (scene_dir / f"{name}_B14.TIF").touch()
    return scene_dir


def test_discover_scene_paths_keeps_reference_first_and_finds_five_b14_files(tmp_path):
    from scripts.five_image_pipeline import discover_scene_paths

    names = [
        "DZ01V_L2_E113.6_N36.3_20260616031133_01_T1",
        "DZ01V_L2_E113.4_N36.6_20260810030932_01_T1",
        "DZ01V_L2_E113.0_N36.4_20260222031837_01_T1",
        "DZ01V_L2_E114.0_N36.4_20260714030410_01_T1",
        "DZ01V_L2_E113.7_N36.6_20260616031127_01_T1",
    ]
    for name in names:
        _make_scene_dir(tmp_path, name)

    paths = discover_scene_paths(
        tmp_path,
        band="B14",
        reference_scene="DZ01V_L2_E113.4_N36.6_20260810030932_01_T1",
        expected_count=5,
    )

    assert len(paths) == 5
    assert paths[0].parent.name == (
        "DZ01V_L2_E113.4_N36.6_20260810030932_01_T1"
    )
    assert [path.parent.name for path in paths[1:]] == sorted(
        name for name in names if name != paths[0].parent.name
    )


def test_discover_scene_paths_rejects_missing_reference_or_wrong_scene_count(tmp_path):
    from scripts.five_image_pipeline import discover_scene_paths

    for name in ["scene_a", "scene_b"]:
        _make_scene_dir(tmp_path, name)

    with pytest.raises(ValueError, match="expected 5"):
        discover_scene_paths(
            tmp_path,
            band="B14",
            reference_scene="scene_a",
            expected_count=5,
        )

    with pytest.raises(ValueError, match="reference scene"):
        discover_scene_paths(
            tmp_path,
            band="B14",
            reference_scene="missing_reference",
            expected_count=2,
        )


def test_run_pipeline_uses_reference_plus_four_registered_scenes(tmp_path, monkeypatch):
    from scripts import five_image_pipeline as pipeline

    scene_names = ["reference", "target_b", "target_c", "target_d", "target_e"]
    paths = [tmp_path / f"{name}_B14.TIF" for name in scene_names]
    for path in paths:
        path.touch()

    transforms = [rasterio.Affine(1, 0, index * 8, 0, -1, 100) for index in range(5)]

    def fake_load_scene(path, band):
        index = scene_names.index(Path(path).stem.removesuffix("_B14"))
        return pipeline.SceneData(
            name=scene_names[index],
            path=str(path),
            array=np.full((8, 12), index + 1, dtype=np.float32),
            transform=transforms[index],
            crs="EPSG:4326",
            nodata=None,
        )

    registration_calls = []

    def fake_register(reference, target):
        registration_calls.append((reference.name, target.name))
        return target, {
            "target_scene": target.name,
            "model_used": "translation",
            "offset": {"dx_pixels": 0.0, "dy_pixels": 0.0},
            "matching": {
                "accepted_matches": 4,
                "candidate_blocks": 4,
            },
        }

    overlap_pairs = [
        {
            "idx_i": 0,
            "idx_j": 1,
            "window_i": (0, 4, 0, 4),
            "window_j": (0, 4, 0, 4),
            "pixel_count": 16,
        }
    ]
    monkeypatch.setattr(pipeline, "load_scene", fake_load_scene)
    monkeypatch.setattr(pipeline, "register_scene_to_reference", fake_register)
    monkeypatch.setattr(pipeline, "detect_multi_overlap", lambda *args, **kwargs: overlap_pairs)

    calls = {"bagrn": None, "volrn": None, "mosaics": []}

    def fake_compute_all(arrays, normalized, nodatas, overlaps, controls):
        return {"ave": float(len(arrays)), "adm": 0.0}

    def fake_bagrn(arrays, nodatas, overlaps, control_idx):
        calls["bagrn"] = (len(arrays), control_idx, len(overlaps))
        return list(arrays), {"coefficients": np.array([1.0, 2.0])}, {"iterations": 1}

    def fake_volrn(arrays, transforms, bounds, nodatas, **kwargs):
        calls["volrn"] = (len(arrays), len(transforms), len(bounds), len(nodatas))
        return list(arrays), {"coefficients": np.array([3.0, 4.0])}

    def fake_mosaic(arrays, transforms, crs, nodatas, path):
        calls["mosaics"].append((len(arrays), Path(path).name))
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).touch()

    monkeypatch.setattr(pipeline, "compute_all", fake_compute_all)
    monkeypatch.setattr(pipeline, "bagrn_normalize", fake_bagrn)
    monkeypatch.setattr(pipeline, "volrn_normalize", fake_volrn)
    monkeypatch.setattr(pipeline, "create_mosaic", fake_mosaic)

    result = pipeline.run_pipeline(paths, tmp_path / "output", band="B14")

    assert registration_calls == [
        ("reference", "target_b"),
        ("reference", "target_c"),
        ("reference", "target_d"),
        ("reference", "target_e"),
    ]
    assert calls["bagrn"] == (5, 0, 1)
    assert calls["volrn"] == (5, 5, 5, 5)
    assert calls["mosaics"] == [(5, "mosaic_B14_bagrn.tif"), (5, "mosaic_B14_volrn.tif")]
    assert result["scene_count"] == 5
    assert result["reference_scene"] == "reference"
    assert len(result["registration"]) == 4

    summary_path = tmp_path / "output" / "B14" / "five_image_summary.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["scene_count"] == 5
    assert summary["overlap_count"] == 1
    registration_dir = tmp_path / "output" / "B14" / "registration"
    assert len(list(registration_dir.glob("*_matches.csv"))) == 4
    assert len(list(registration_dir.glob("*_metrics.json"))) == 4
