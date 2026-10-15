from __future__ import annotations

import json
import tkinter as tk

import numpy as np
import pytest
import rasterio
from affine import Affine
from shapely.geometry import box, mapping

from benchmark_v1_radiometric_r15.review_store import AutoReviewStore
from benchmark_v1_radiometric_r15.gui import AutoRoiReviewer


def _collection(role="DEVELOPMENT"):
    return {"type": "FeatureCollection", "features": [{
        "type": "Feature", "geometry": mapping(box(1050, 1050, 2000, 2000)),
        "properties": {"pair_id": "00_01", "candidate_id": "00_01_R15_00_00",
                       "scene_i": "s1", "scene_j": "s2", "role": role,
                       "suggested_label": "UNKNOWN", "suggestion_confidence": "LOW",
                       "review_status": "UNVERIFIED"}}]}


def test_generated_suggestions_never_load_as_approved(tmp_path):
    source = tmp_path / "candidates.geojson"
    source.write_text(json.dumps(_collection()), encoding="utf-8")
    store = AutoReviewStore(source, tmp_path / "review.json", tmp_path / "events.jsonl")
    row = store.get("00_01", "00_01_R15_00_00")
    assert row["review_status"] == "UNVERIFIED"
    assert row["reviewer"] == ""
    assert row["human_label"] == "UNKNOWN"


def test_approval_requires_evidence_and_real_actor(tmp_path):
    source = tmp_path / "candidates.geojson"
    source.write_text(json.dumps(_collection()), encoding="utf-8")
    store = AutoReviewStore(source, tmp_path / "review.json", tmp_path / "events.jsonl")
    with pytest.raises(ValueError, match="evidence"):
        store.record_review("00_01", "00_01_R15_00_00", reviewer="alice",
                            status="APPROVED", label="STABLE", evidence="")
    result = store.record_review("00_01", "00_01_R15_00_00", reviewer="alice",
                                 status="APPROVED", label="STABLE", evidence="same visible road junction")
    assert result["review_status"] == "APPROVED"
    assert result["reviewer"] == "alice"
    assert result["suggested_label"] == "UNKNOWN"


def test_test_roi_requires_distinct_explicit_second_review(tmp_path):
    source = tmp_path / "candidates.geojson"
    source.write_text(json.dumps(_collection(role="TEST")), encoding="utf-8")
    store = AutoReviewStore(source, tmp_path / "review.json", tmp_path / "events.jsonl")
    store.record_review("00_01", "00_01_R15_00_00", reviewer="alice", status="APPROVED",
                        label="STABLE", evidence="independently inspected both dates")
    with pytest.raises(ValueError, match="different"):
        store.record_review("00_01", "00_01_R15_00_00", reviewer="alice", status="APPROVED",
                            label="STABLE", evidence="again", second_review=True)
    result = store.record_review("00_01", "00_01_R15_00_00", reviewer="bob", status="APPROVED",
                                 label="STABLE", evidence="second independent review", second_review=True)
    assert result["second_review_status"] == "APPROVED"
    assert result["second_reviewer"] == "bob"


def test_disagreement_stays_unresolved_and_history_survives_reload(tmp_path):
    source = tmp_path / "candidates.geojson"
    source.write_text(json.dumps(_collection(role="TEST")), encoding="utf-8")
    snapshot, events = tmp_path / "review.json", tmp_path / "events.jsonl"
    store = AutoReviewStore(source, snapshot, events)
    store.record_review("00_01", "00_01_R15_00_00", reviewer="alice", status="APPROVED",
                        label="STABLE", evidence="first review")
    result = store.record_review("00_01", "00_01_R15_00_00", reviewer="bob", status="DOUBTFUL",
                                 label="REAL_CHANGE", evidence="possible seasonal change", second_review=True)
    assert result["dispute_status"] == "UNRESOLVED_DISAGREEMENT"
    reloaded = AutoReviewStore(source, snapshot, events)
    assert reloaded.get("00_01", "00_01_R15_00_00")["second_review_status"] == "DOUBTFUL"
    lines = events.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[-1])["before"]["human_label"] == "STABLE"


def test_append_only_events_recover_when_snapshot_lags(tmp_path):
    source = tmp_path / "candidates.geojson"
    source.write_text(json.dumps(_collection()), encoding="utf-8")
    snapshot, events = tmp_path / "review.json", tmp_path / "events.jsonl"
    store = AutoReviewStore(source, snapshot, events)
    store.record_review("00_01", "00_01_R15_00_00", reviewer="alice", status="DOUBTFUL",
                        label="UNKNOWN", evidence="needs a second look")
    snapshot.write_text(json.dumps({"schema_version": 1, "records": []}), encoding="utf-8")
    restored = AutoReviewStore(source, snapshot, events)
    assert restored.get("00_01", "00_01_R15_00_00")["review_status"] == "DOUBTFUL"


def test_geometry_edit_recomputes_and_invalidates_approval(tmp_path):
    source = tmp_path / "candidates.geojson"
    source.write_text(json.dumps(_collection()), encoding="utf-8")
    store = AutoReviewStore(source, tmp_path / "review.json", tmp_path / "events.jsonl")
    store.record_review("00_01", "00_01_R15_00_00", reviewer="alice", status="APPROVED",
                        label="STABLE", evidence="reviewed")
    result = store.edit_geometry("00_01", "00_01_R15_00_00", box(9500, 1500, 10500, 2450),
                                 support_validator=lambda geom: {"s1_B9": 1.0, "s2_B9": .98},
                                 all_candidate_features=[store.get("00_01", "00_01_R15_00_00")],
                                 editor="alice")
    assert result["role"] == "UNASSIGNED"
    assert result["valid_support_recomputed"] is True
    assert result["review_status"] == "UNVERIFIED"
    assert result["eligible_for_test"] is False
    assert result["pairwise_leakage_audit"] == "PENDING_ALL_PAIR_REAUDIT"
    assert result["relative_spectral_shape_similarity_diagnostic"] is None
    assert result["multiband_feature_status"] == "INVALIDATED_BY_GEOMETRY_EDIT"


def test_reviewer_progress_counts_include_pending_and_second_review(tmp_path):
    source = tmp_path / "candidates.geojson"
    c = _collection(role="TEST")
    c["features"].append({**c["features"][0], "properties": {**c["features"][0]["properties"], "candidate_id": "second"}})
    source.write_text(json.dumps(c), encoding="utf-8")
    store = AutoReviewStore(source, tmp_path / "review.json", tmp_path / "events.jsonl")
    assert store.progress() == {"pending": 2, "reviewed": 0, "second_pending": 0, "second_approved": 0,
                                "doubtful": 0, "rejected": 0}
    store.record_review("00_01", "00_01_R15_00_00", reviewer="alice", status="APPROVED",
                        label="STABLE", evidence="review")
    assert store.progress()["second_pending"] == 1


def test_gui_exposes_candidate_queue_dual_views_and_review_controls(tmp_path):
    source = tmp_path / "00_01.geojson"
    source.write_text(json.dumps(_collection()), encoding="utf-8")
    store = AutoReviewStore(source, tmp_path / "review.json", tmp_path / "events.jsonl")
    paths = []
    for n in (1, 2):
        path = tmp_path / f"scene{n}.tif"
        with rasterio.open(path, "w", driver="GTiff", width=512, height=512, count=1,
                           dtype="uint16", crs="EPSG:32650",
                           transform=Affine(14, 0, 0, 0, -14, 4096), nodata=0) as dst:
            dst.write(np.arange(512*512, dtype=np.uint16).reshape(512, 512), 1)
        paths.append(path)
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"Tk display unavailable: {exc}")
    try:
        root.withdraw()
        gui = AutoRoiReviewer(root, "00_01", store, {9: paths[0]}, {9: paths[1]}, "reviewer")
        root.update_idletasks()
        assert gui.listbox.size() == 1
        assert gui.band_box.get() == "9"
        assert gui.canvas_i.winfo_exists() and gui.canvas_j.winfo_exists()
        assert gui.review_button.winfo_exists() and gui.geometry_button.winfo_exists()
        assert "suggestion=UNKNOWN" in gui.listbox.get(0)
    finally:
        root.destroy()
