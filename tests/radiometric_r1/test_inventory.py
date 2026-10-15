from __future__ import annotations

import csv
from pathlib import Path

import pytest

from src.benchmark_v1_radiometric_r1.inventory import (
    build_pair_inventory,
    select_representative_pairs,
    stratify_pair,
)


def row(pair_id: str, days: float, valid: int, quality_i="PASS", quality_j="PASS"):
    return {
        "pair_id": pair_id,
        "days_apart": str(days),
        "same_date": str(days == 0),
        "valid_overlap_pixels": str(valid),
        "bbox_overlap_pixels": str(max(valid, 100)),
        "quality_i": quality_i,
        "quality_j": quality_j,
        "radiometric_temporal_suitability": "NO_VALID_B9_OVERLAP" if valid == 0 else "CANDIDATE_REQUIRES_STABLE_CHANGE_REVIEW",
        "scene_i": f"scene_{pair_id}_i",
        "scene_j": f"scene_{pair_id}_j",
        "acquisition_i": "2025-01-01T00:00:00",
        "acquisition_j": "2025-01-01T00:00:00",
        "common_v_bands": "B1;B9;B16",
        "bbox_overlap_area_m2": "14000",
        "registration_status_task15": "NOT_USED_FOR_SELECTION",
    }


@pytest.mark.parametrize(
    ("item", "expected"),
    [
        (row("x", 0, 500), "T0"),
        (row("x", 6.99, 500), "T1"),
        (row("x", 7, 500), "T2"),
        (row("x", 29.99, 500), "T2"),
        (row("x", 30, 500), "T3"),
        (row("x", 12, 0), "NO_VALID_OVERLAP"),
    ],
)
def test_stratify_pair_uses_frozen_edges_and_real_support(item, expected):
    assert stratify_pair(item) == expected


def test_shortlist_keeps_required_pairs_and_is_deterministic():
    raw = [row("00_01", 0, 10), row("04_11", 37, 11), row("02_10", 55, 12), row("00_10", 61, 13)]
    raw += [row("t0", 0, 20), row("a", 4, 100), row("b", 10, 200), row("c", 40, 300), row("d", 20, 0)]
    data = [{**item, "temporal_bin": stratify_pair(item)} for item in raw]
    first = select_representative_pairs(data, target=8)
    assert {r["pair_id"] for r in first} >= {"00_01", "04_11", "02_10", "00_10"}
    assert [r["pair_id"] for r in first] == [r["pair_id"] for r in select_representative_pairs(data, target=8)]
    assert all(int(r["valid_overlap_pixels"]) > 0 for r in first)
    assert len(first) == 8


def test_build_inventory_retains_all_pairs_without_calling_them_failures(tmp_path: Path):
    candidates = [row(f"p{i:02}", i, 0 if i < 30 else 10 + i) for i in range(78)]
    temporal = [r for r in candidates if int(r["valid_overlap_pixels"]) > 0]
    scene_rows = [{"scene_id": f"scene_{i:02}", "b9_quality_status_tag": "PASS"} for i in range(13)]
    paths = []
    for name, rows in (("pairs.csv", candidates), ("temporal.csv", temporal), ("scenes.csv", scene_rows)):
        path = tmp_path / name
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        paths.append(path)
    result = build_pair_inventory(*paths)
    assert len(result) == 78
    assert sum(r["temporal_bin"] == "NO_VALID_OVERLAP" for r in result) == 30
    assert {r["pair_id"]: r["status"] for r in result}["p00"] == "NO_VALID_OVERLAP"
    assert all(r["status"] != "FAIL" for r in result)
