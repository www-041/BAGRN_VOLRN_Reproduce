from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
import rasterio
from affine import Affine

from benchmark_v1_radiometric_r15.inputs import load_pair_inputs


PAIR = "00_01"
SCENES = ("scene-a", "scene-b")


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _fixture(root: Path, *, overlap: int = 100, omit_b9: str = "", duplicate: bool = False):
    docs = root / "docs/benchmark/v1"
    data = root / "fixture"
    data.mkdir(parents=True, exist_ok=True)
    scene_rows = []
    band_rows = []
    pair_rows = [{"pair_id": PAIR, "scene_id_i": SCENES[0], "scene_id_j": SCENES[1],
                  "valid_overlap_pixels": str(overlap), "quality_i": "PASS", "quality_j": "PASS"}]
    manifest_rows = []
    for scene_index, scene in enumerate(SCENES):
        bands = []
        for band_index in range(1, 17):
            band = f"B{band_index}"
            if band == omit_b9 and scene_index == 1:
                continue
            path = data / f"{scene}_{band}.tif"
            with rasterio.open(path, "w", driver="GTiff", width=16, height=16, count=1,
                               dtype="uint16", crs="EPSG:32650",
                               transform=Affine(14, 0, 500000 + scene_index * 14, 0, -14, 4000000),
                               nodata=0) as dst:
                dst.write(np.ones((16, 16), dtype=np.uint16), 1)
            bands.append(band)
            band_rows.append({"scene_id": scene, "product_id": f"product-{scene_index}",
                              "band": band, "file_path": str(path), "wavelength_center_nm": "UNKNOWN",
                              "data_unit": "UNKNOWN", "quality_status_tag": "PASS"})
            if band == "B9":
                manifest_rows.append({"scene_id": f"product-{scene_index}", "band": "B9",
                                      "source_path": str(path), "source_file_size_bytes": path.stat().st_size,
                                      "recorded_source_sha256_task15": "frozen-sha"})
        scene_rows.append({"scene_id": scene, "product_id": f"product-{scene_index}",
                           "v_bands": ";".join(bands), "b9_quality_status_tag": "PASS"})
    if duplicate:
        band_rows.append(dict(band_rows[0]))
    _write_csv(docs / "scene_inventory.csv", scene_rows)
    _write_csv(docs / "v_band_inventory.csv", band_rows)
    _write_csv(docs / "temporal_pair_inventory.csv", pair_rows)
    _write_csv(docs / "phase2/input_identity_manifest.csv", manifest_rows)
    return root


def test_only_four_frozen_pairs_are_accepted(tmp_path):
    _fixture(tmp_path)
    with pytest.raises(ValueError, match="not in frozen R1.5 pair allowlist"):
        load_pair_inputs("00_02", tmp_path)


def test_product_id_join_returns_two_scenes_and_all_sixteen_bands(tmp_path):
    _fixture(tmp_path)
    pair = load_pair_inputs(PAIR, tmp_path)
    assert [scene.scene_id for scene in pair.scenes] == list(SCENES)
    assert len(pair.scenes[0].bands) == 16
    assert pair.scenes[0].bands["B9"].path.name == "scene-a_B9.tif"


def test_duplicate_scene_band_input_is_rejected(tmp_path):
    _fixture(tmp_path, duplicate=True)
    with pytest.raises(ValueError, match="duplicate.*B1"):
        load_pair_inputs(PAIR, tmp_path)


def test_missing_b9_is_fatal_but_missing_auxiliary_band_is_recorded(tmp_path):
    _fixture(tmp_path, omit_b9="B9")
    with pytest.raises(ValueError, match="B9.*missing"):
        load_pair_inputs(PAIR, tmp_path)


def test_nonpositive_overlap_is_rejected_before_opening_rasters(tmp_path, monkeypatch):
    _fixture(tmp_path, overlap=0)
    def forbidden(*args, **kwargs):
        raise AssertionError("raster opened before overlap validation")
    monkeypatch.setattr(rasterio, "open", forbidden)
    with pytest.raises(ValueError, match="no verified B9 overlap"):
        load_pair_inputs(PAIR, tmp_path)


def test_units_wavelengths_and_manifest_fingerprint_stay_explicit(tmp_path):
    _fixture(tmp_path)
    pair = load_pair_inputs(PAIR, tmp_path)
    band = pair.scenes[0].bands["B9"]
    assert band.data_unit == "UNKNOWN"
    assert band.wavelength_center_nm is None
    assert band.manifest_sha256 == "frozen-sha"
    assert band.header_fingerprint["crs"] == "EPSG:32650"
    assert band.header_fingerprint["width"] == 16
    assert band.usable


def test_auxiliary_band_missing_path_is_excluded_with_reason(tmp_path):
    _fixture(tmp_path)
    rows = list(csv.DictReader((tmp_path / "docs/benchmark/v1/v_band_inventory.csv").open(encoding="utf-8")))
    rows[0]["file_path"] = str(tmp_path / "missing.tif")
    _write_csv(tmp_path / "docs/benchmark/v1/v_band_inventory.csv", rows)
    pair = load_pair_inputs(PAIR, tmp_path)
    assert not pair.scenes[0].bands["B1"].usable
    assert pair.scenes[0].bands["B1"].exclusion_reason == "FILE_MISSING"
