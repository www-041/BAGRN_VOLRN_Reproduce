import csv

from src.benchmark_v1_radiometric_r1.band_readiness import summarize_v_bands


def _write(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def test_band_readiness_preserves_unknown_units_and_only_uses_frozen_hashes(tmp_path):
    scenes = [{"scene_id": "s1", "product_id": "p1"}, {"scene_id": "s2", "product_id": "p2"}]
    bands = []
    for scene in scenes:
        for band in range(1, 17):
            bands.append({"scene_id": scene["scene_id"], "product_id": scene["product_id"], "band": str(band),
                          "wavelength_center_nm": "700" if band == 9 else "", "bandwidth_nm": "20" if band == 9 else "",
                          "data_unit": "UNKNOWN", "quality_status_tag": "PASS", "crs": "EPSG:32650",
                          "pixel_size_x": "14", "pixel_size_y": "14", "width_px": "100", "height_px": "100",
                          "affine_c": "500000", "affine_f": "3300000", "dtype": "uint16", "nodata": "0"})
    _write(tmp_path / "bands.csv", bands)
    _write(tmp_path / "scenes.csv", scenes)
    _write(tmp_path / "manifest.csv", [{"scene_id": "p1", "band": "B9", "source_file_sha256": "a" * 64}, {"scene_id": "p2", "band": "B9", "source_file_sha256": "b" * 64}])
    rows, summary = summarize_v_bands(tmp_path / "bands.csv", tmp_path / "scenes.csv", tmp_path / "manifest.csv")
    assert summary["scene_count"] == 2 and summary["band_rows"] == 32
    assert summary["complete_scene_count"] == 2
    assert all(row["data_unit"] == "UNKNOWN" for row in rows)
    b9 = [row for row in rows if row["band"] == "9"]
    assert [row["source_file_sha256"] for row in b9] == ["a" * 64, "b" * 64]
    assert all(row["source_file_sha256"] == "" for row in rows if row["band"] != "9")
    assert summary["wavelength_metadata_rows"] == 2


def test_band_readiness_reports_missing_band_without_inventing_it(tmp_path):
    scenes = [{"scene_id": "s1", "product_id": "p1"}]
    bands = [{"scene_id": "s1", "product_id": "p1", "band": str(n), "wavelength_center_nm": "", "bandwidth_nm": "", "data_unit": "UNKNOWN", "quality_status_tag": "PASS", "crs": "EPSG:32650", "pixel_size_x": "14", "pixel_size_y": "14", "width_px": "10", "height_px": "10", "affine_c": "0", "affine_f": "0", "dtype": "uint16", "nodata": "0"} for n in range(1, 16)]
    _write(tmp_path / "bands.csv", bands); _write(tmp_path / "scenes.csv", scenes)
    _write(tmp_path / "manifest.csv", [{"scene_id": "p1", "band": "B9", "source_file_sha256": ""}])
    _, summary = summarize_v_bands(tmp_path / "bands.csv", tmp_path / "scenes.csv", tmp_path / "manifest.csv")
    assert summary["complete_scene_count"] == 0
    assert summary["missing_bands"] == {"s1": ["16"]}
