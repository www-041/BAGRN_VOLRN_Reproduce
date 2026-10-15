"""Tests for the metadata-only B9 manifest audit."""

from __future__ import annotations

import json

from scripts.audit_b9_dataset import MANIFEST_FIELDS, write_manifest


def _record(scene_id: str) -> dict:
    return {
        "scene_id": scene_id,
        "time_dir": "20250101",
        "scene_dir": f"D:/B9/{scene_id}",
        "b9_path": f"D:/B9/{scene_id}/{scene_id}_B9.TIF",
        "mtl_path": f"D:/B9/{scene_id}/{scene_id}_MTL.txt",
        "acquisition_time": "2025-01-01T00:00:00",
        "crs": "EPSG:32650",
        "resolution_x_m": 14.0,
        "resolution_y_m": 14.0,
        "width": 120,
        "height": 100,
        "left": 500000.0,
        "bottom": 3998600.0,
        "right": 501680.0,
        "top": 4000000.0,
        "cloud_cover": 0.1,
    }


def test_manifest_writes_required_csv_and_json_schema(tmp_path):
    output_dir = tmp_path / "dataset_audit"

    result = write_manifest([_record("scene_a")], output_dir, expected_count=13)

    assert result["status"] == "AUDIT_SCENE_COUNT_MISMATCH"
    assert result["n_scenes"] == 1
    csv_text = (output_dir / "01_scene_manifest.csv").read_text(encoding="utf-8")
    assert csv_text.splitlines()[0].split(",") == MANIFEST_FIELDS
    payload = json.loads(
        (output_dir / "01_scene_manifest.json").read_text(encoding="utf-8")
    )
    assert payload["scenes"][0]["scene_index"] == 0
    assert payload["scenes"][0]["pixel_size_x_m"] == 14.0
