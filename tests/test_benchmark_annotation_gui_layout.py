"""Regression checks for the actual Tk candidate-review layout."""

from pathlib import Path
import csv
import json
import subprocess
import sys
import tkinter as tk
from tkinter import ttk

import numpy as np
import pytest
import rasterio
from affine import Affine


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import annotate_benchmark_v1 as annotator
from benchmark_candidate_core import CandidateReviewStore


@pytest.fixture
def trial_data(tmp_path, monkeypatch):
    """Independent 40-point UNVERIFIED trial; never reads or rewrites real review CSVs."""
    transform = Affine.translation(650000, 3280000) * Affine.scale(14, -14)
    scenes = []
    for scene_id in ("SYN_L", "SYN_R"):
        image = tmp_path / f"{scene_id}.tif"
        with rasterio.open(image, "w", driver="GTiff", width=640, height=640,
                           count=1, dtype="uint8", nodata=0, crs="EPSG:32650",
                           transform=transform) as dst:
            dst.write(np.ones((640, 640), dtype=np.uint8), 1)
        scenes.append({"scene_id": scene_id, "b9_path": str(image)})
    manifest = tmp_path / "scene_manifest.csv"
    inventory = tmp_path / "scene_inventory.csv"
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["scene_index", "scene_id", "b9_path", "acquisition_time"])
        writer.writeheader()
        for index, scene in enumerate(scenes):
            writer.writerow({**scene, "scene_index": index,
                             "acquisition_time": f"2025-{'11' if index == 0 else '12'}-14T02:38:47"})
    with inventory.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["product_id", "b9_quality_status_tag"])
        writer.writeheader()
        for scene in scenes:
            writer.writerow({"product_id": scene["scene_id"], "b9_quality_status_tag": "PASS"})
    monkeypatch.setattr(annotator, "MANIFEST", manifest)
    monkeypatch.setattr(annotator, "SCENE_INVENTORY", inventory)
    candidate_path = tmp_path / "00_01.csv"
    store = CandidateReviewStore(candidate_path, "00_01", "SYN_L", "SYN_R")
    suggestions = []
    for index in range(40):
        row = float(100 + index * 10)
        col = float(120 + index * 9)
        x, y = transform * (col + .5, row + .5)
        suggestions.append({"candidate_id": f"00_01_C{index+1:04d}", "pair_id": "00_01",
                            "scene_i": "SYN_L", "scene_j": "SYN_R", "row_i": row,
                            "col_i": col, "x_i": x, "y_i": y, "row_j": row,
                            "col_j": col, "x_j": x, "y_j": y,
                            "recommendation_method": "synthetic_trial", "matching_score": .9,
                            "forward_backward_error": 0, "grid_cell_id": "r03_c02"})
    store.replace_suggestions(suggestions)
    store.save()
    return candidate_path


def _descendants(widget):
    for child in widget.winfo_children():
        yield child
        yield from _descendants(child)


@pytest.mark.parametrize("window_size", ["1200x720+0+0", "800x500+0+0"])
def test_trial_candidates_and_review_controls_fit_window(monkeypatch, trial_data, tmp_path, window_size):
    """A clipped candidate section must fail even if widgets were constructed."""
    seen = []

    def inspect(root):
        try:
            root.geometry(window_size)
            root.update()
            widgets = list(_descendants(root))
            lists = [w for w in widgets if isinstance(w, tk.Listbox)]
            assert len(lists) >= 2
            candidates = max(lists, key=lambda w: w.size())
            assert candidates.size() == 40
            filters = [w for w in widgets if isinstance(w, ttk.Combobox)
                       and tuple(w.cget("values")) == ("PENDING", "REVIEWED", "ALL")]
            assert len(filters) == 1 and filters[0].winfo_ismapped()
            buttons = {w.cget("text"): w for w in widgets if isinstance(w, ttk.Button)}
            for name in ("Previous", "Next", "Adjust candidate", "Apply candidate review"):
                assert name in buttons
                button = buttons[name]
                assert button.winfo_ismapped()
                assert button.winfo_rooty() + button.winfo_height() <= root.winfo_rooty() + root.winfo_height()
            assert candidates.winfo_ismapped()
            assert candidates.winfo_rooty() + candidates.winfo_height() <= root.winfo_rooty() + root.winfo_height()
            buttons["Next"].invoke()
            root.update()
            assert candidates.curselection() == (0,)
            seen.append(True)
        finally:
            root.destroy()

    monkeypatch.setattr(tk.Tk, "mainloop", inspect)
    annotator.run_gui(
        "00_01", "layout-test",
        tmp_path / "manual.csv", trial_data, tmp_path / "truth.csv",
        tmp_path / "provenance.csv",
    )
    assert seen == [True]


def test_missing_candidate_file_is_reported_before_manual_gui(monkeypatch, tmp_path, capsys):
    """A wrong --candidates path cannot silently start manual-only annotation."""
    missing = tmp_path / "missing.csv"
    dialogs = []
    monkeypatch.setattr(sys, "argv", ["annotate_benchmark_v1.py", "--pair", "00_01",
                                  "--reviewer", "layout-test", "--candidates", str(missing)])
    monkeypatch.setattr("tkinter.messagebox.showerror", lambda title, message: dialogs.append((title, message)))
    monkeypatch.setattr(tk.Tk, "mainloop", lambda root: pytest.fail("manual-only GUI opened"))
    result = annotator.main()
    assert result == 2
    assert str(missing) in capsys.readouterr().err
    assert len(dialogs) == 1 and str(missing) in dialogs[0][1]


def test_default_candidate_path_is_independent_of_working_directory(tmp_path):
    result = subprocess.run([sys.executable, str(ROOT / "scripts/annotate_benchmark_v1.py"),
                             "--pair", "00_01", "--check-env"], cwd=tmp_path,
                            text=True, capture_output=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert "Loaded 40 auto candidates" in result.stdout
    assert "GUI_CREATE_AND_RENDER_OK pair=00_01" in result.stdout


def test_gui_shows_objective_context_unknown_suggestions_and_review_counts(monkeypatch, tmp_path, trial_data):
    candidate_path = trial_data

    def inspect(root):
        try:
            root.update()
            widgets = list(_descendants(root))
            buttons = {w.cget("text"): w for w in widgets if isinstance(w, ttk.Button)}
            assert "Bulk labels..." in buttons
            buttons["Next"].invoke()
            root.update()
            labels = [root.getvar(w.cget("textvariable")) for w in widgets
                      if isinstance(w, ttk.Label) and w.cget("textvariable")]
            assert any("PAIR_GRID:00_01:" in value and "2025-11-14" in value
                       and "PASS" in value for value in labels)
            assert any("TERRAIN_UNKNOWN" in value and "UNKNOWN" in value
                       and "NO_VERIFIED" in value for value in labels)
            assert any("pending=40" in value and "second=0" in value for value in labels)
            review_boxes = [w for w in widgets if isinstance(w, ttk.Combobox)
                            and "APPROVED" in tuple(w.cget("values"))]
            assert len(review_boxes) == 1 and "DOUBTFUL" in tuple(review_boxes[0].cget("values"))
            split_boxes = [w for w in widgets if isinstance(w, ttk.Combobox)
                           and tuple(w.cget("values")) == ("UNASSIGNED",)]
            assert len(split_boxes) == 1
        finally:
            root.destroy()

    monkeypatch.setattr(tk.Tk, "mainloop", inspect)
    annotator.run_gui("00_01", "gui-test", tmp_path / "manual.csv", candidate_path,
                      tmp_path / "truth.csv", tmp_path / "provenance.csv")


def test_first_review_gui_releases_window_when_mainloop_returns(monkeypatch, tmp_path, trial_data):
    windows = []
    monkeypatch.setattr(tk.Tk, "mainloop", lambda root: windows.append(root))
    annotator.run_gui("00_01", "gui-test", tmp_path / "manual.csv", trial_data,
                      tmp_path / "truth.csv", tmp_path / "provenance.csv")
    assert len(windows) == 1
    with pytest.raises(tk.TclError, match="application has been destroyed"):
        windows[0].winfo_exists()


def test_first_review_gui_filters_pending_reviewed_and_all_independently(monkeypatch, tmp_path, trial_data):
    store = CandidateReviewStore(trial_data, "00_01", "SYN_L", "SYN_R")
    store.review("00_01_C0001", "APPROVED", "first-human", "UNASSIGNED", "MEDIUM", "0.5 px")
    store.save()
    sizes = []

    def inspect(root):
        try:
            root.update()
            widgets = list(_descendants(root))
            combo = next(w for w in widgets if isinstance(w, ttk.Combobox)
                         and tuple(w.cget("values")) == ("PENDING", "REVIEWED", "ALL"))
            box = next(w for w in widgets if isinstance(w, tk.Listbox) and w.master is root)
            for value in ("PENDING", "REVIEWED", "ALL"):
                combo.set(value)
                root.update()
                sizes.append(box.size())
        finally:
            root.destroy()

    monkeypatch.setattr(tk.Tk, "mainloop", inspect)
    annotator.run_gui("00_01", "another-human", tmp_path / "manual.csv", trial_data,
                      tmp_path / "truth.csv", tmp_path / "provenance.csv")
    assert sizes == [39, 1, 40]
    assert CandidateReviewStore(trial_data, "00_01", "SYN_L", "SYN_R")._row(
        "00_01_C0001")["human_review_status"] == "APPROVED"


def test_gui_bulk_edits_selected_unverified_points_and_saves_audit(monkeypatch, tmp_path, trial_data):
    candidate_path = trial_data
    monkeypatch.setattr("tkinter.messagebox.askyesno", lambda *args, **kwargs: True)
    monkeypatch.setattr("tkinter.messagebox.showerror",
                        lambda title, message, **kwargs: pytest.fail(f"{title}: {message}"))

    def inspect(root):
        try:
            root.update()
            widgets = list(_descendants(root))
            buttons = {w.cget("text"): w for w in widgets if isinstance(w, ttk.Button)}
            candidate_box = max((w for w in widgets if isinstance(w, tk.Listbox)), key=lambda w: w.size())
            candidate_box.selection_set(0, 1)
            root.update()
            buttons["Bulk labels..."].invoke()
            root.update()
            dialogs = [w for w in root.winfo_children() if isinstance(w, tk.Toplevel)]
            assert len(dialogs) == 1
            dialog_widgets = list(_descendants(dialogs[0]))
            terrain_boxes = [w for w in dialog_widgets if isinstance(w, ttk.Combobox)
                             and "MOUNTAIN" in tuple(w.cget("values"))]
            assert len(terrain_boxes) == 1
            terrain_boxes[0].set("MOUNTAIN")
            apply = [w for w in dialog_widgets if isinstance(w, ttk.Button)
                     and w.cget("text") == "Apply selected labels"]
            assert len(apply) == 1
            apply[0].invoke()
            buttons["Save CSV"].invoke()
        finally:
            root.destroy()

    monkeypatch.setattr(tk.Tk, "mainloop", inspect)
    annotator.run_gui("00_01", "gui-test", tmp_path / "manual.csv", candidate_path,
                      tmp_path / "truth.csv", tmp_path / "provenance.csv")
    first_two = CandidateReviewStore(candidate_path, "00_01",
                                     annotator.pair_paths("00_01")[0]["scene_id"],
                                     annotator.pair_paths("00_01")[1]["scene_id"]).rows[:2]
    assert [r["terrain_class"] for r in first_two] == ["MOUNTAIN", "MOUNTAIN"]
    assert all(r["human_review_status"] == "UNVERIFIED" for r in first_two)
    assert candidate_path.with_suffix(".baseline.csv").is_file()
    assert candidate_path.with_suffix(".audit.jsonl").is_file()


def test_cli_rejects_unfrozen_split_before_opening_gui(monkeypatch, tmp_path, capsys):
    split = tmp_path / "split.geojson"
    split.write_text(json.dumps({"type": "FeatureCollection", "metadata": {"status": "DRAFT",
                                 "version": "v1", "crs": "EPSG:32650"}, "features": []}), encoding="utf-8")
    dialogs = []
    monkeypatch.setattr(sys, "argv", ["annotate_benchmark_v1.py", "--pair", "00_01", "--reviewer", "human",
                                  "--frozen-split", str(split)])
    monkeypatch.setattr("tkinter.messagebox.showerror", lambda title, message: dialogs.append(message))
    monkeypatch.setattr(tk.Tk, "mainloop", lambda root: pytest.fail("GUI opened with draft split"))
    assert annotator.main() == 2
    assert "FROZEN" in capsys.readouterr().err
    assert len(dialogs) == 1 and "FROZEN" in dialogs[0]


def test_count_distinguishes_human_approved_unassigned_from_formal_truth(monkeypatch, tmp_path, capsys, trial_data):
    candidate_path = trial_data
    left, right = annotator.pair_paths("00_01")
    store = CandidateReviewStore(candidate_path, "00_01", left["scene_id"], right["scene_id"])
    store.review(store.rows[0]["candidate_id"], "APPROVED", "human", "UNASSIGNED", "MEDIUM", "0.5 px")
    store.save()
    monkeypatch.setattr(sys, "argv", ["annotate_benchmark_v1.py", "--pair", "00_01", "--count-approved",
                                  "--candidates", str(candidate_path), "--output", str(tmp_path / "manual.csv")])
    assert annotator.main() is None
    output = capsys.readouterr().out
    assert "auto_APPROVED_UNASSIGNED=1" in output
    assert "auto_formal_APPROVED=0" in output


def test_candidate_review_flow_distinguishes_manual_apply_and_saves_candidate(monkeypatch, tmp_path, trial_data):
    """The manual Apply must direct a candidate reviewer to the correct control."""
    candidate_path = trial_data
    candidate_id = CandidateReviewStore(candidate_path, "00_01",
                                        *[scene["scene_id"] for scene in annotator.pair_paths("00_01")]).rows[0]["candidate_id"]

    def inspect(root):
        try:
            root.update()
            widgets = list(_descendants(root))
            buttons = {w.cget("text"): w for w in widgets if isinstance(w, ttk.Button)}
            candidate_box = max((w for w in widgets if isinstance(w, tk.Listbox)), key=lambda w: w.size())
            candidate_box.selection_set(0)
            candidate_box.event_generate("<<ListboxSelect>>")
            root.update()
            buttons["Apply manual-point review"].invoke()
            root.update()
            labels = [root.getvar(w.cget("textvariable")) for w in widgets
                      if isinstance(w, ttk.Label) and w.cget("textvariable")]
            assert any("Apply candidate review" in value for value in labels)
            review_box = next(w for w in widgets if isinstance(w, ttk.Combobox)
                              and "DOUBTFUL" in tuple(w.cget("values")))
            review_box.set("DOUBTFUL")
            review_box.focus_set()
            root.update()
            buttons["Apply candidate review"].invoke()
            buttons["Save CSV"].invoke()
        finally:
            root.destroy()

    monkeypatch.setattr(tk.Tk, "mainloop", inspect)
    annotator.run_gui("00_01", "reviewer-test", tmp_path / "manual.csv", candidate_path,
                      tmp_path / "truth.csv", tmp_path / "provenance.csv")
    left, right = annotator.pair_paths("00_01")
    saved = CandidateReviewStore(candidate_path, "00_01", left["scene_id"], right["scene_id"])
    assert saved._row(candidate_id)["human_review_status"] == "DOUBTFUL"


def test_candidate_review_accepts_viewed_single_candidate_after_list_highlight_disappears(monkeypatch, tmp_path, trial_data):
    """Editing the form must not invalidate the sole candidate shown in both image panels."""
    candidate_path = trial_data
    monkeypatch.setattr("tkinter.messagebox.askyesno", lambda *args, **kwargs: True)
    monkeypatch.setattr("tkinter.messagebox.showerror",
                        lambda title, message, **kwargs: pytest.fail(f"{title}: {message}"))

    def inspect(root):
        try:
            root.update()
            widgets = list(_descendants(root))
            buttons = {w.cget("text"): w for w in widgets if isinstance(w, ttk.Button)}
            candidate_box = max((w for w in widgets if isinstance(w, tk.Listbox)), key=lambda w: w.size())
            candidate_box.selection_set(0)
            candidate_box.event_generate("<<ListboxSelect>>")
            root.update()
            assert candidate_box.curselection() == (0,)
            review_box = next(w for w in widgets if isinstance(w, ttk.Combobox)
                              and "DOUBTFUL" in tuple(w.cget("values")))
            review_box.set("APPROVED")
            confidence_box = next(w for w in widgets if isinstance(w, ttk.Combobox)
                                  and "MEDIUM" in tuple(w.cget("values")))
            confidence_box.set("MEDIUM")
            precision_entry = next(w for w in widgets if type(w) is ttk.Entry)
            precision_entry.delete(0, "end")
            precision_entry.insert(0, "0.5 px estimated by reviewer")
            assert [(tuple(w.cget("values")), w.get()) for w in widgets if isinstance(w, ttk.Combobox)
                    and ("DOUBTFUL" in tuple(w.cget("values")) or "MEDIUM" in tuple(w.cget("values")))] == [
                        (("UNKNOWN", "LOW", "MEDIUM", "HIGH"), "MEDIUM"),
                        (("UNVERIFIED", "APPROVED", "REJECTED", "DOUBTFUL"), "APPROVED")]
            assert precision_entry.get() == "0.5 px estimated by reviewer"
            candidate_box.selection_clear(0, "end")  # screenshot: candidate visible but no highlight
            root.update()
            assert candidate_box.curselection() == ()
            buttons["Apply candidate review"].invoke()
            buttons["Save CSV"].invoke()
        finally:
            root.destroy()

    monkeypatch.setattr(tk.Tk, "mainloop", inspect)
    annotator.run_gui("00_01", "reviewer-test", tmp_path / "manual.csv", candidate_path,
                      tmp_path / "truth.csv", tmp_path / "provenance.csv")
    left, right = annotator.pair_paths("00_01")
    saved = CandidateReviewStore(candidate_path, "00_01", left["scene_id"], right["scene_id"])
    assert saved._row("00_01_C0001")["human_review_status"] == "APPROVED"
    assert saved._row("00_01_C0001")["split"] == "UNASSIGNED"
    assert len(saved.approved_truth()) == 0
