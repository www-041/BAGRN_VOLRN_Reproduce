"""Synthetic tests for the B9 metadata-only overlap audit."""

from __future__ import annotations

import json

import pytest

from src.multiscene_sift.b9_overlap import (
    build_b9_overlap_graph,
    connected_components,
    write_overlap_outputs,
)


def _record(scene_id: str, bounds: tuple[float, float, float, float]) -> dict:
    left, bottom, right, top = bounds
    return {
        "scene_id": scene_id,
        "left": left,
        "bottom": bottom,
        "right": right,
        "top": top,
    }


def test_overlap_uses_symmetric_ratio_and_preserves_disconnected_components():
    records = [
        _record("a", (0, 0, 100, 100)),
        _record("b", (50, 0, 150, 100)),
        _record("c", (100, 0, 200, 100)),
        _record("d", (400, 0, 500, 100)),
    ]

    pairs = build_b9_overlap_graph(records)

    assert len(pairs) == 6
    ab = next(pair for pair in pairs if pair["scene_i"] == "a" and pair["scene_j"] == "b")
    assert ab["intersection_area"] == pytest.approx(5000.0)
    assert ab["overlap_area_i_ratio"] == pytest.approx(0.5)
    assert ab["overlap_area_j_ratio"] == pytest.approx(0.5)
    assert ab["symmetric_overlap_ratio"] == pytest.approx(0.5)
    assert ab["has_overlap"] is True
    assert connected_components(4, pairs) == [[0, 1, 2], [3]]


def test_overlap_outputs_include_required_artifacts(tmp_path):
    records = [
        _record("a", (0, 0, 100, 100)),
        _record("b", (50, 0, 150, 100)),
    ]
    pairs = build_b9_overlap_graph(records)

    payload = write_overlap_outputs(records, pairs, tmp_path)

    assert payload["connected"] is True
    assert payload["n_edges"] == 1
    for name in (
        "02_overlap_edges.csv",
        "02_overlap_graph.json",
        "02_overlap_matrix.csv",
        "02_overlap_graph.png",
    ):
        assert (tmp_path / name).exists()
    saved = json.loads((tmp_path / "02_overlap_graph.json").read_text(encoding="utf-8"))
    assert saved["edges"][0]["symmetric_overlap_ratio"] == pytest.approx(0.5)

