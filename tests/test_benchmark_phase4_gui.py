"""Second-review GUI must show frozen roles without editing first-review data."""

from pathlib import Path
import os
import sys
import tkinter as tk
from tkinter import messagebox, ttk

import numpy as np
import rasterio
from affine import Affine
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import annotate_benchmark_v1 as annotator
from benchmark_phase4_freeze import FrozenTrial, load_frozen_trial, sha256_file
from benchmark_phase4_review import load_review_events


def _descendants(widget):
    for child in widget.winfo_children():
        yield child
        yield from _descendants(child)


def test_gui_tcl_runtime_uses_matching_python_installation(monkeypatch):
    monkeypatch.delenv("TCL_LIBRARY", raising=False)
    monkeypatch.delenv("TK_LIBRARY", raising=False)
    annotator.configure_tk_runtime()
    if sys.platform == "win32":
        assert Path(os.environ["TCL_LIBRARY"]) == Path(sys.base_prefix) / "tcl/tcl8.6"
        assert Path(os.environ["TK_LIBRARY"]) == Path(sys.base_prefix) / "tcl/tk8.6"
        assert (Path(os.environ["TCL_LIBRARY"]) / "init.tcl").is_file()
        assert (Path(os.environ["TK_LIBRARY"]) / "tk.tcl").is_file()


def test_gui_recovers_one_transient_tcl_library_read_failure(monkeypatch):
    sentinel = object()
    calls = []

    def transient():
        calls.append(1)
        if len(calls) == 1:
            raise tk.TclError("Can't find a usable tk.tcl: transient read failure")
        return sentinel

    monkeypatch.setattr(tk, "Tk", transient)
    root = annotator.create_tk_root(tk)
    assert len(calls) == 2
    assert root is sentinel


def test_second_review_gui_shows_frozen_test_without_editable_split(monkeypatch, tmp_path):
    snapshot = load_frozen_trial(ROOT)
    source = ROOT / "docs/benchmark/v1/phase2/auto_candidates/00_01.csv"
    assignment = ROOT / "docs/benchmark/v1/phase3/checkpoint_split_assignments.csv"
    before = (sha256_file(source), sha256_file(assignment))
    observed = []

    def inspect(root):
        try:
            root.update()
            widgets = list(_descendants(root))
            labels = [w.cget("text") for w in widgets if isinstance(w, ttk.Label)]
            buttons = [w.cget("text") for w in widgets if isinstance(w, ttk.Button)]
            boxes = [w for w in widgets if isinstance(w, tk.Listbox)]
            assert any("Frozen TEST" in text for text in labels)
            assert any("Pending" in text for text in labels)
            assert "Approve second review" in buttons
            assert "Reject" in buttons and "Doubtful" in buttons
            assert boxes and boxes[0].size() == 19
            assert not any(isinstance(w, ttk.Combobox) and "Split" in str(w.cget("values")) for w in widgets)
            observed.append(True)
        finally:
            root.destroy()

    monkeypatch.setattr(tk.Tk, "mainloop", inspect)
    annotator.run_second_review_gui("00_01", "independent-researcher", snapshot,
                                    tmp_path / "events.jsonl")
    assert observed == [True]
    assert not (tmp_path / "events.jsonl").exists()
    assert (sha256_file(source), sha256_file(assignment)) == before


def test_second_review_gui_releases_window_when_mainloop_returns(monkeypatch, tmp_path):
    windows = []
    monkeypatch.setattr(tk.Tk, "mainloop", lambda root: windows.append(root))
    annotator.run_second_review_gui("00_01", "independent-researcher", load_frozen_trial(ROOT),
                                    tmp_path / "events.jsonl")
    assert len(windows) == 1
    with pytest.raises(tk.TclError, match="application has been destroyed"):
        windows[0].winfo_exists()


def test_synthetic_gui_explicit_approval_saves_only_disposable_event(monkeypatch, tmp_path):
    key = ("00_01", "SYN_C1")
    transform = Affine.translation(650500, 3280500)
    scenes = []
    for name in ("SYN_L", "SYN_R"):
        path = tmp_path / f"{name}.tif"
        with rasterio.open(path, "w", driver="GTiff", width=32, height=32, count=1,
                           dtype="uint8", crs="EPSG:32650", transform=transform, nodata=0) as dst:
            dst.write(np.ones((32, 32), dtype=np.uint8), 1)
        scenes.append({"scene_id": name, "b9_path": str(path)})
    frozen = {"pair_id": "00_01", "candidate_id": "SYN_C1", "split": "TEST",
              "grid_cell_id": "g65_328", "scene_i": "SYN_L", "scene_j": "SYN_R",
              "source_review_status": "APPROVED", "effective_x_i": "650501",
              "effective_y_i": "3280501", "effective_x_j": "650501", "effective_y_j": "3280501"}
    source = {"pair_id": "00_01", "candidate_id": "SYN_C1", "scene_i": "SYN_L", "scene_j": "SYN_R",
              "primary_reviewer": "first", "reviewer": "first", "human_review_status": "APPROVED",
              "human_adjusted": "False", "confidence": "HIGH", "row_i": "0.5", "col_i": "0.5",
              "row_j": "0.5", "col_j": "0.5"}
    snapshot = FrozenTrial(ROOT, {key: frozen}, {key: source}, {}, {"00_01": "syntheticsha"}, "syntheticphase3")
    monkeypatch.setattr(annotator, "pair_paths", lambda pair: scenes)
    monkeypatch.setattr(messagebox, "askyesno", lambda *args, **kwargs: True)

    def inspect_and_save(root):
        try:
            root.update()
            widgets = list(_descendants(root))
            for check in (w for w in widgets if isinstance(w, ttk.Checkbutton)):
                check.invoke()
            entries = [w for w in widgets if isinstance(w, ttk.Entry)]
            entries[0].insert(0, "My estimate 0.7 px")
            approve = next(w for w in widgets if isinstance(w, ttk.Button)
                           and w.cget("text") == "Approve second review")
            approve.invoke()
            assert not (tmp_path / "events.jsonl").exists()
            box = next(w for w in widgets if isinstance(w, tk.Listbox))
            box.selection_set(0)
            box.event_generate("<<ListboxSelect>>")
            root.update()
            for check in (w for w in widgets if isinstance(w, ttk.Checkbutton)):
                check.invoke()
            entries[0].insert(0, "My estimate 0.7 px")
            approve.invoke()
        finally:
            root.destroy()

    monkeypatch.setattr(tk.Tk, "mainloop", inspect_and_save)
    event_path = tmp_path / "events.jsonl"
    annotator.run_second_review_gui("00_01", "second", snapshot, event_path, smoke=True)
    events = load_review_events(event_path, snapshot)
    assert len(events) == 1 and events[0]["human_outcome"] == "APPROVED"
    assert events[0]["reviewer"] == "second"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["SYN_L.tif", "SYN_R.tif", "events.jsonl"]


def test_synthetic_second_review_locate_adjust_reject_save_and_reload(monkeypatch, tmp_path):
    key = ("00_01", "SYN_C1")
    transform = Affine.translation(650500, 3280500)
    scenes = []
    for name in ("SYN_L", "SYN_R"):
        path = tmp_path / f"{name}.tif"
        with rasterio.open(path, "w", driver="GTiff", width=32, height=32, count=1,
                           dtype="uint8", crs="EPSG:32650", transform=transform, nodata=0) as dst:
            dst.write(np.ones((32, 32), dtype=np.uint8), 1)
        scenes.append({"scene_id": name, "b9_path": str(path)})
    frozen = {"pair_id": "00_01", "candidate_id": "SYN_C1", "split": "TEST",
              "grid_cell_id": "g65_328", "scene_i": "SYN_L", "scene_j": "SYN_R"}
    source = {"pair_id": "00_01", "candidate_id": "SYN_C1", "scene_i": "SYN_L", "scene_j": "SYN_R",
              "primary_reviewer": "first", "reviewer": "first", "human_review_status": "APPROVED",
              "human_adjusted": "False", "row_i": "0.5", "col_i": "0.5",
              "row_j": "0.5", "col_j": "0.5"}
    snapshot = FrozenTrial(ROOT, {key: frozen}, {key: source}, {}, {"00_01": "syntheticsha"}, "syntheticphase3")
    monkeypatch.setattr(annotator, "pair_paths", lambda pair: scenes)
    monkeypatch.setattr(messagebox, "askyesno", lambda *args, **kwargs: True)
    event_path = tmp_path / "events.jsonl"

    def review(root):
        try:
            root.update()
            widgets = list(_descendants(root))
            box = next(w for w in widgets if isinstance(w, tk.Listbox))
            box.selection_set(0)
            box.event_generate("<<ListboxSelect>>")
            root.update()
            assert "SYN_C1" in box.get(0)
            canvases = [w for w in widgets if isinstance(w, tk.Canvas)
                        and int(w.cget("width")) == annotator.PANEL_SIZE]
            assert len(canvases) == 2
            checks = [w for w in widgets if isinstance(w, ttk.Checkbutton)]
            for check in checks:
                check.invoke()
            entries = [w for w in widgets if isinstance(w, ttk.Entry)]
            entries[0].insert(0, "Independent estimate: 0.7 px")
            buttons = {w.cget("text"): w for w in widgets if isinstance(w, ttk.Button)}
            buttons["Approve second review"].invoke()
            canvases[0].event_generate("<Button-1>", x=258, y=258)
            root.update()
            entries[1].insert(0, "Distinct landmark corner")
            checks[0].invoke()
            buttons["Reject"].invoke()
        finally:
            root.destroy()

    monkeypatch.setattr(tk.Tk, "mainloop", review)
    annotator.run_second_review_gui("00_01", "second", snapshot, event_path, smoke=True)
    events = load_review_events(event_path, snapshot)
    assert [event["human_outcome"] for event in events] == ["APPROVED", "REJECTED"]
    assert events[1]["adjusted"] is True
    assert events[1]["adjustment_reason"] == "Distinct landmark corner"
    assert events[1]["after"]["left"]["row"] != events[1]["before"]["left"]["row"]

    def reopened(root):
        try:
            root.update()
            box = next(w for w in _descendants(root) if isinstance(w, tk.Listbox))
            assert "REJECTED" in box.get(0)
            box.selection_set(0)
            box.event_generate("<<ListboxSelect>>")
            root.update()
            assert "SYN_C1" in box.get(0)
        finally:
            root.destroy()

    monkeypatch.setattr(tk.Tk, "mainloop", reopened)
    annotator.run_second_review_gui("00_01", "second", snapshot, event_path, smoke=True)
    assert len(load_review_events(event_path, snapshot)) == 2


def test_second_review_refuses_manifest_image_different_from_frozen_identity(monkeypatch, tmp_path):
    snapshot = load_frozen_trial(ROOT)
    paths = annotator.pair_paths("00_01")
    wrong = [{**paths[0], "b9_path": str(tmp_path / "wrong.tif")}, paths[1]]
    monkeypatch.setattr(annotator, "pair_paths", lambda pair: wrong)
    with pytest.raises(ValueError, match="frozen B9 identity"):
        annotator.run_second_review_gui("00_01", "another-human", snapshot,
                                        tmp_path / "events.jsonl")
