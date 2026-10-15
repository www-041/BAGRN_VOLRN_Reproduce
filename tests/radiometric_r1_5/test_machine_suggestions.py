from __future__ import annotations

import json

import pytest
from shapely.geometry import box, mapping


def test_conservative_advice_keeps_semantic_labels_unknown_without_independent_evidence():
    from benchmark_v1_radiometric_r15.machine_suggestions import advise_fields

    advice = advise_fields(
        band_features={
            3: {"common_valid_fraction": 1.0, "texture_i": 0.2, "texture_j": 0.21},
            4: {"common_valid_fraction": 1.0, "texture_i": 0.3, "texture_j": 0.29},
            7: {"common_valid_fraction": 1.0, "texture_i": 0.4, "texture_j": 0.41},
            9: {"common_valid_fraction": 1.0, "texture_i": 0.5, "texture_j": 0.48},
            14: {"common_valid_fraction": 1.0, "texture_i": 0.6, "texture_j": 0.61},
        },
        available_bands=list(range(1, 17)),
        missing_auxiliary=["DEM", "CLOUD_MASK", "INDEPENDENT_CHANGE_REFERENCE"],
        data_unit="UNKNOWN",
        geometry_verified=False,
    )

    expected = {
        "suggested_landcover_type": "UNKNOWN",
        "suggested_terrain_type": "UNKNOWN",
        "suggested_temporal_status": "UNKNOWN",
        "suggested_interference_type": "UNKNOWN",
    }
    for field, label in expected.items():
        item = advice[field]
        assert item["label"] == label
        assert item["confidence"] in {"LOW", "MEDIUM"}
        assert item["evidence"]
        assert item["uncertainty_reasons"]
        assert set(item["missing_auxiliary_data"]) >= set(
            ["DEM", "CLOUD_MASK", "INDEPENDENT_CHANGE_REFERENCE"]
        )
    assert advice["suggested_temporal_status"]["label"] != "CONFIRMED_CHANGE"


def test_machine_suggestion_files_are_identity_checked_and_never_contain_human_approval(tmp_path):
    from benchmark_v1_radiometric_r15.machine_suggestions import (
        load_machine_suggestion_map,
        write_machine_suggestion_artifacts,
    )

    feature = {
        "type": "Feature",
        "geometry": mapping(box(10, 10, 20, 20)),
        "properties": {
            "candidate_id": "05_07_c1",
            "pair_id": "05_07",
            "role": "DEVELOPMENT",
            "machine_suggestion_status": "SUGGESTED",
            "suggested_landcover_type": {"label": "UNKNOWN", "confidence": "LOW",
                                        "evidence": ["B3 texture_i=0.2"], "uncertainty_reasons": ["unknown DN"],
                                        "missing_auxiliary_data": ["land-cover reference"]},
            "suggested_terrain_type": {"label": "UNKNOWN", "confidence": "MEDIUM",
                                      "evidence": ["DEM unavailable"], "uncertainty_reasons": ["no DEM"],
                                      "missing_auxiliary_data": ["DEM"]},
            "suggested_temporal_status": {"label": "UNKNOWN", "confidence": "MEDIUM",
                                          "evidence": ["geometry not verified"], "uncertainty_reasons": ["no truth"],
                                          "missing_auxiliary_data": ["change reference"]},
            "suggested_interference_type": {"label": "UNKNOWN", "confidence": "MEDIUM",
                                            "evidence": ["cloud mask unavailable"], "uncertainty_reasons": ["no mask"],
                                            "missing_auxiliary_data": ["cloud mask"]},
        },
    }
    json_path, csv_path = write_machine_suggestion_artifacts(
        tmp_path, "05_07", [feature], source_candidate_sha256="source-hash", run_id="machine-run")
    loaded = load_machine_suggestion_map(json_path, expected_pair_id="05_07")
    assert loaded[("05_07", "05_07_c1")]["suggested_landcover_type"]["label"] == "UNKNOWN"
    assert loaded[("05_07", "05_07_c1")]["_source_candidate_sha256"] == "source-hash"
    document = json.loads(json_path.read_text(encoding="utf-8"))
    props = document["features"][0]["properties"]
    assert "multiband_review_decision" not in props
    assert "reviewer" not in props
    assert "review_time_utc" not in props
    assert csv_path.is_file()
    with pytest.raises(FileExistsError):
        write_machine_suggestion_artifacts(tmp_path, "05_07", [feature])


def test_agreement_summary_uses_only_explicit_human_approved_rows():
    from benchmark_v1_radiometric_r15.machine_suggestions import summarize_human_agreement

    machine = [{"type": "Feature", "geometry": mapping(box(10, 10, 20, 20)),
                "properties": {"pair_id": "05_07", "candidate_id": "c1",
                "suggested_landcover_type": {"label": "UNKNOWN"},
                "suggested_terrain_type": {"label": "UNKNOWN"},
                "suggested_temporal_status": {"label": "UNKNOWN"},
                "suggested_interference_type": {"label": "UNKNOWN"}}}]
    human = [
        {"pair_id": "05_07", "candidate_id": "c1", "multiband_review_decision": "UNVERIFIED",
         "landcover_type": "VEGETATION", "terrain_type": "SLOPE", "temporal_status": "NO_CHANGE",
         "interference_type": "NONE"}
    ]
    summary = summarize_human_agreement(machine, human)
    assert summary["status"] == "NO_HUMAN_APPROVED_LABELS"
    assert summary["eligible_human_record_count"] == 0

    human[0]["multiband_review_decision"] = "APPROVED"
    summary = summarize_human_agreement(machine, human)
    assert summary["status"] == "COMPARISON_ONLY_NOT_ACCURACY"
    assert summary["eligible_human_record_count"] == 1
    assert summary["fields"]["suggested_landcover_type"]["agreement_count"] == 0
    assert summary["fields"]["suggested_terrain_type"]["agreement_count"] == 0


def test_human_review_sample_covers_pair_and_role_without_using_scores():
    from benchmark_v1_radiometric_r15.machine_suggestions import select_human_review_sample

    rows = []
    for pair, role, candidate, x in (
        ("00_01", "TEST", "a", 1), ("00_01", "UNASSIGNED", "b", 2),
        ("05_07", "DEVELOPMENT", "c", 3), ("05_07", "TEST", "d", 4),
        ("05_07", "TEST", "e", 5), ("08_10", "UNASSIGNED", "f", 6),
    ):
        rows.append({"type": "Feature", "geometry": mapping(box(x, 0, x + 1, 1)),
                     "properties": {"pair_id": pair, "candidate_id": candidate,
                                    "role_at_generation": role, "score": 100 - x,
                                    **{field: {"label": "UNKNOWN"} for field in
                                       ("suggested_landcover_type", "suggested_terrain_type",
                                        "suggested_temporal_status", "suggested_interference_type")}}})
    selected = select_human_review_sample(rows, max_samples=5)
    again = select_human_review_sample(list(reversed(rows)), max_samples=5)
    assert [(r["pair_id"], r["candidate_id"]) for r in selected] == [
        (r["pair_id"], r["candidate_id"]) for r in again]
    assert {r["pair_id"] for r in selected} == {"00_01", "05_07", "08_10"}
    assert len({(r["pair_id"], r["role"]) for r in selected}) == len(selected)


def test_pair_roi_evidence_uses_bounded_geographic_windows_and_records_raw_dn(tmp_path):
    import numpy as np
    import rasterio
    from affine import Affine

    from benchmark_v1_radiometric_r15.machine_suggestions import extract_pair_roi_evidence

    transform_i = Affine(14, 0, 1000, 0, -14, 2000)
    transform_j = Affine(14, 0, 1014, 0, -14, 2000)
    paths = [{}, {}]
    arrays = []
    for index, transform in enumerate((transform_i, transform_j)):
        for band in (3, 9, 14):
            path = tmp_path / f"scene{index}_B{band}.tif"
            data = (np.arange(32 * 32, dtype=np.uint16).reshape(32, 32) + 100 * index + band)
            if index == 1:
                data[:, 10:12] = 0
            with rasterio.open(path, "w", driver="GTiff", width=32, height=32, count=1,
                               dtype="uint16", crs="EPSG:32650", transform=transform, nodata=0) as dst:
                dst.write(data, 1)
            paths[index][band] = path
    from shapely.geometry import box
    result = extract_pair_roi_evidence(box(1150, 1700, 1374, 1924), paths[0], paths[1])
    assert set(result["bands"]) == {"3", "9", "14"}
    assert result["bands"]["9"]["common_valid_fraction"] < 1.0
    assert result["bands"]["9"]["common_scene_i"]["median_raw_dn"] is not None
    assert result["data_unit"] == "UNKNOWN_RAW_DN"
    assert "scene_j warped" in result["grid_mapping"]


def test_gui_machine_prefill_is_draft_only_and_never_creates_review_event(tmp_path):
    import json
    import tkinter as tk

    import numpy as np
    import rasterio
    from affine import Affine

    from benchmark_v1_radiometric_r15.storage import write_candidate_artifacts
    from benchmark_v1_radiometric_r15.gui import AutoRoiReviewer
    from benchmark_v1_radiometric_r15.multiband_review import MultiBandReviewStore
    from shapely.geometry import box, mapping

    geometry = box(56, 56, 168, 168)
    candidate = tmp_path / "05_07.geojson"
    write_candidate_artifacts(candidate, [{
        "type": "Feature", "geometry": mapping(geometry),
        "properties": {"pair_id": "05_07", "candidate_id": "05_07_C1", "role": "DEVELOPMENT",
                       "suggested_label": "UNKNOWN", "human_label": "UNKNOWN"},
    }])
    paths = [{}, {}]
    for scene in range(2):
        for band in (3, 4, 7, 9, 14):
            path = tmp_path / f"scene{scene}_B{band}.tif"
            data = (np.arange(16 * 16, dtype=np.uint16).reshape(16, 16) + 1 + band)
            with rasterio.open(path, "w", driver="GTiff", width=16, height=16, count=1,
                               dtype="uint16", crs="EPSG:32650",
                               transform=Affine(14, 0, 0, 0, -14, 224), nodata=0) as dst:
                dst.write(data, 1)
            paths[scene][band] = path
    store = MultiBandReviewStore(candidate, tmp_path / "human" / "snapshot.json",
                                 tmp_path / "human" / "events.jsonl")
    suggestions = {("05_07", "05_07_C1"): {
        "suggested_landcover_type": {"label": "VEGETATION", "confidence": "LOW",
                                     "evidence": ["B14 spatial texture only"]},
        "suggested_terrain_type": {"label": "UNKNOWN", "confidence": "MEDIUM",
                                   "evidence": ["DEM unavailable"]},
        "suggested_temporal_status": {"label": "UNKNOWN", "confidence": "MEDIUM",
                                       "evidence": ["no independent change reference"]},
        "suggested_interference_type": {"label": "UNKNOWN", "confidence": "MEDIUM",
                                        "evidence": ["no quality mask"]},
    }}
    try:
        root = tk.Tk(); root.withdraw()
    except tk.TclError as exc:
        pytest.skip(f"Tk display unavailable: {exc}")
    try:
        gui = AutoRoiReviewer(root, "05_07", store, paths[0], paths[1], "reviewer-a",
                              multiband_mode=True, machine_suggestions=suggestions)
        root.update_idletasks(); root.update()
        assert gui.multiband_label_vars["landcover_type"].get() == "VEGETATION"
        assert gui.multiband_label_vars["terrain_type"].get() == "UNKNOWN"
        assert gui.multiband_decision_var.get() == "UNVERIFIED"
        assert "LOW" in gui.machine_suggestion_var.get()
        assert not (tmp_path / "human" / "events.jsonl").exists()
        gui.multiband_label_vars["landcover_type"].set("UNKNOWN")
        assert gui.multiband_label_vars["landcover_type"].get() == "UNKNOWN"
        gui._capture_multiband_draft()
        gui._populate(gui.keys[gui.index])
        assert gui.multiband_label_vars["landcover_type"].get() == "UNKNOWN"
        assert not (tmp_path / "human" / "events.jsonl").exists()
    finally:
        root.destroy()
