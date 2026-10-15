from __future__ import annotations

import json

import pytest

from benchmark_v1_radiometric_r15.cli import execute_generation
from benchmark_v1_radiometric_r15.proposals import ProposalConfig
from test_inputs import _fixture


def test_invalid_pair_is_rejected_before_creating_run_output(tmp_path):
    with pytest.raises(ValueError, match="allowlist"):
        execute_generation(tmp_path, ["00_02"], "r15_test", max_candidates=3)
    assert not (tmp_path / "data/output/benchmark_v1_radiometric_r1_5/auto_roi_candidates/r15_test").exists()


def test_missing_b9_fails_before_writing_pair_artifacts(tmp_path):
    _fixture(tmp_path)
    import csv
    path = tmp_path / "docs/benchmark/v1/v_band_inventory.csv"
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    rows = [r for r in rows if not (r["scene_id"] == "scene-b" and r["band"] == "B9")]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    with pytest.raises(ValueError, match="B9 missing"):
        execute_generation(tmp_path, ["00_01"], "r15_missing", max_candidates=3)
    assert not (tmp_path / "data/output/benchmark_v1_radiometric_r1_5/auto_roi_candidates/r15_missing/00_01.geojson").exists()


def test_run_manifest_is_stable_and_new_pair_can_append_without_overwrite(tmp_path):
    _fixture(tmp_path)
    output = execute_generation(tmp_path, ["00_01"], "r15_test", max_candidates=3,
                                config=ProposalConfig(window_px=8, grid_rows=1, grid_cols=1,
                                                      max_candidates=3, texture_min=1e9))
    manifest_before = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    candidates = json.loads((output / "00_01.geojson").read_text(encoding="utf-8"))
    assert manifest_before["generated_pairs"] == ["00_01"]
    assert all(f["properties"]["review_status"] == "UNVERIFIED" for f in candidates["features"])
    with pytest.raises(FileExistsError, match="already exists"):
        execute_generation(tmp_path, ["00_01"], "r15_test", max_candidates=3,
                           config=ProposalConfig(window_px=8, grid_rows=1, grid_cols=1,
                                                 max_candidates=3, texture_min=1e9))


def test_run_id_path_traversal_is_rejected(tmp_path):
    _fixture(tmp_path)
    with pytest.raises(ValueError, match="run-id"):
        execute_generation(tmp_path, ["00_01"], "../unsafe", max_candidates=3)
