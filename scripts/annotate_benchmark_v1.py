"""Human-only DZ01V B9 tie-point annotator. No matcher data are loaded."""

from __future__ import annotations

import argparse
import csv
import gc
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image, ImageTk
from rasterio.windows import Window

from benchmark_phase2_core import Viewport, load_truth, new_row, save_truth
from benchmark_candidate_core import CandidateReviewStore
from benchmark_candidate_metadata import (LANDCOVER_VALUES, TERRAIN_VALUES, apply_batch_labels,
                                          build_candidate_context, load_frozen_split,
                                          load_scene_metadata, save_candidate_session)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/output/final_pipeline/b9_13scene_formal_v2/scene_manifest.csv"
SCENE_INVENTORY = ROOT / "docs/benchmark/v1/scene_inventory.csv"
DEFAULT_OUT = ROOT / "docs/benchmark/v1/phase2/annotations"
DEFAULT_CANDIDATES = ROOT / "docs/benchmark/v1/phase2/auto_candidates"
DEFAULT_AUTO_TRUTH = ROOT / "docs/benchmark/v1/phase2/auto_reviewed_truth"
DEFAULT_PROVENANCE = ROOT / "docs/benchmark/v1/phase2/truth_provenance"
TRIAL_PAIRS = {"00_01", "02_03", "05_07", "08_10"}
PANEL_SIZE = 512


def configure_tk_runtime() -> None:
    """Bind Tcl/Tk script libraries to this Python installation on Windows."""
    if sys.platform != "win32":
        return
    base = Path(sys.base_prefix) / "tcl"
    locations = {"TCL_LIBRARY": base / "tcl8.6", "TK_LIBRARY": base / "tk8.6"}
    required = {"TCL_LIBRARY": "init.tcl", "TK_LIBRARY": "tk.tcl"}
    for name, path in locations.items():
        if not (path / required[name]).is_file():
            raise RuntimeError(f"Missing Tcl/Tk resource for active Python: {path / required[name]}")
        os.environ[name] = path.as_posix()


def create_tk_root(tk_module):
    """Retry only transient Tcl/Tk resource reads, not arbitrary GUI errors."""
    configure_tk_runtime()
    for attempt in range(3):
        try:
            return tk_module.Tk()
        except tk_module.TclError as exc:
            if "Can't find a usable" not in str(exc) or attempt == 2:
                raise
            gc.collect()
    raise AssertionError("unreachable")


def destroy_tk_root_if_alive(root, tk_module) -> None:
    """Tk may invalidate `winfo` itself after the user already destroyed the app."""
    try:
        alive = root.winfo_exists()
    except tk_module.TclError as exc:
        if "application has been destroyed" not in str(exc):
            raise
        return
    if alive:
        root.destroy()


def load_required_candidates(path: Path, pair_id: str, scene_i: str, scene_j: str) -> CandidateReviewStore:
    if not path.is_file():
        raise FileNotFoundError(f"Candidate CSV not found: {path.resolve()}")
    store = CandidateReviewStore(path, pair_id, scene_i, scene_j)
    if not store.rows:
        raise ValueError(f"Candidate CSV has no rows: {path.resolve()}")
    return store


def pair_paths(pair_id: str):
    i, j = (int(x) for x in pair_id.split("_"))
    with MANIFEST.open(newline="", encoding="utf-8-sig") as stream:
        manifest = {int(r["scene_index"]): r for r in csv.DictReader(stream)}
    return manifest[i], manifest[j]


def stretch(data: np.ndarray, mask: np.ndarray) -> Image.Image:
    valid = data[mask != 0]
    if valid.size == 0:
        return Image.fromarray(np.zeros(data.shape, dtype=np.uint8))
    low, high = np.percentile(valid, (2, 98))
    if high <= low:
        high = low + 1
    out = np.clip((data.astype(np.float32) - low) * 255 / (high - low), 0, 255).astype(np.uint8)
    out[mask == 0] = 0
    return Image.fromarray(out)


def run_gui(pair_id: str, reviewer: str, out_path: Path, candidate_path: Path,
            auto_truth_path: Path, provenance_path: Path, *, smoke: bool = False,
            frozen_split: dict | None = None) -> None:
    configure_tk_runtime()
    import tkinter as tk
    from tkinter import messagebox, ttk

    from benchmark_phase2_core import projected_to_pixel

    scene_i, scene_j = pair_paths(pair_id)
    with rasterio.open(scene_i["b9_path"]) as a, rasterio.open(scene_j["b9_path"]) as b:
        if a.crs != b.crs or a.crs.to_epsg() != 32650:
            raise ValueError("Annotation tool requires common EPSG:32650 B9 images")
        rows = load_truth(out_path, pair_id, scene_i["scene_id"], scene_j["scene_id"])
        candidates = load_required_candidates(candidate_path, pair_id, scene_i["scene_id"], scene_j["scene_id"])
        scene_metadata = load_scene_metadata(MANIFEST, SCENE_INVENTORY)
        candidate_saved_rows = {row["candidate_id"]: dict(row) for row in candidates.rows}
        manual_dirty = [False]
        candidate_dirty = [False]
        x = (max(a.bounds.left, b.bounds.left) + min(a.bounds.right, b.bounds.right)) / 2
        y = (max(a.bounds.bottom, b.bounds.bottom) + min(a.bounds.top, b.bounds.top)) / 2
        centers = [[x, y], [x, y]]
        scale = [1.0]
        viewport = [None, None]
        images = [None, None]
        pending = [None, None]
        edit_mode = [False]
        candidate_edit_mode = [False]
        manual_add_mode = [True]
        selected_candidate_id = [None]
        candidate_viewed = set()
        root = create_tk_root(tk)
        root.title(f"DZ01V independent B9 annotation: {pair_id} | {reviewer}")
        root.geometry(f"{min(1200, root.winfo_screenwidth()-60)}x{min(900, root.winfo_screenheight()-80)}")
        candidate_bar = ttk.Frame(root)
        candidate_bar.pack(fill="x")
        ttk.Label(candidate_bar, text=f"Candidates ({len(candidates.rows)} loaded):").pack(side="left")
        candidate_filter = tk.StringVar(value="PENDING")
        ttk.Combobox(candidate_bar, textvariable=candidate_filter,
                     values=("PENDING", "REVIEWED", "ALL"), state="readonly", width=12).pack(side="left")
        candidate_score = tk.StringVar(value="No candidate selected")
        ttk.Label(candidate_bar, textvariable=candidate_score).pack(side="left", padx=8)
        candidate_actions = ttk.Frame(root)
        candidate_actions.pack(fill="x")
        batch_actions = ttk.Frame(root)
        batch_actions.pack(fill="x")
        candidate_box = tk.Listbox(root, height=6, selectmode=tk.EXTENDED)
        candidate_box.pack(fill="x")
        review_counts = tk.StringVar()
        ttk.Label(root, textvariable=review_counts).pack(fill="x")
        visible_candidate_ids = []

        workspace = ttk.Frame(root)
        workspace.pack(fill="both", expand=True)
        body_canvas = tk.Canvas(workspace, highlightthickness=0)
        y_scroll = ttk.Scrollbar(workspace, orient="vertical", command=body_canvas.yview)
        x_scroll = ttk.Scrollbar(workspace, orient="horizontal", command=body_canvas.xview)
        body_canvas.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)
        y_scroll.pack(side="right", fill="y")
        x_scroll.pack(side="bottom", fill="x")
        body_canvas.pack(side="left", fill="both", expand=True)
        body = ttk.Frame(body_canvas)
        body_canvas.create_window((0, 0), window=body, anchor="nw")
        body.bind("<Configure>", lambda event: body_canvas.configure(scrollregion=body_canvas.bbox("all")))
        objective_info = tk.StringVar(value="Select a candidate to inspect objective metadata.")
        suggestions_info = tk.StringVar(value="Suggestions are not human review or geometric truth.")
        ttk.Label(body, textvariable=objective_info, wraplength=1080, justify="left").pack(fill="x")
        ttk.Label(body, textvariable=suggestions_info, wraplength=1080, justify="left").pack(fill="x")
        frame = ttk.Frame(body)
        frame.pack(fill="both", expand=True)
        canvases = []
        for idx, scene in enumerate((scene_i, scene_j)):
            col = ttk.Frame(frame)
            col.grid(row=0, column=idx, padx=4)
            ttk.Label(col, text=f"{idx+1}: {scene['scene_id']}").pack()
            c = tk.Canvas(col, width=PANEL_SIZE, height=PANEL_SIZE, bg="black")
            c.pack()
            canvases.append(c)
        toolbar = ttk.Frame(body)
        toolbar.pack(fill="x")
        status = tk.StringVar(value="Click matching feature in LEFT and RIGHT; only human review can APPROVE.")
        ttk.Label(body, textvariable=status).pack(fill="x")
        meta = ttk.Frame(body)
        meta.pack(fill="x")
        landcover = tk.StringVar(value="UNKNOWN")
        terrain = tk.StringVar(value="TERRAIN_UNKNOWN")
        confidence = tk.StringVar(value="UNKNOWN")
        review = tk.StringVar(value="UNVERIFIED")
        split = tk.StringVar(value="UNASSIGNED")
        precision_note = tk.StringVar(value="mouse_click_subpixel_coordinate_not_accuracy_claim")
        notes = tk.StringVar(value="")
        split_combo = None
        for label, variable, values in (
            ("Landcover", landcover, ("UNKNOWN", "BUILDING", "ROAD", "BARE", "VEGETATION", "WATER", "OTHER")),
            ("Terrain", terrain, ("TERRAIN_UNKNOWN", "FLAT", "HILLY", "MOUNTAIN")),
            ("Confidence", confidence, ("UNKNOWN", "LOW", "MEDIUM", "HIGH")),
            ("Review", review, ("UNVERIFIED", "APPROVED", "REJECTED", "DOUBTFUL")),
            ("Split", split, ("UNASSIGNED", "DEVELOPMENT", "TEST")),
        ):
            ttk.Label(meta, text=label).pack(side="left")
            combo = ttk.Combobox(meta, textvariable=variable, values=values, width=17, state="readonly")
            combo.pack(side="left")
            if label == "Split":
                split_combo = combo
        ttk.Label(body, text="Coordinate precision / estimated uncertainty:").pack(anchor="w")
        ttk.Entry(body, textvariable=precision_note).pack(fill="x")
        ttk.Label(body, text="Notes:").pack(anchor="w")
        ttk.Entry(body, textvariable=notes).pack(fill="x")
        listbox = tk.Listbox(body, height=7)
        listbox.pack(fill="x")

        def refresh_list(select_id=None):
            listbox.delete(0, "end")
            for n, row in enumerate(rows):
                listbox.insert("end", f"{row['point_id']}  {row['review_status']}  {row['split']}  {row['landcover_type']}")
                if row["point_id"] == select_id:
                    listbox.selection_set(n)

        def selected_index():
            selection = listbox.curselection()
            return selection[0] if selection else None

        def refresh_candidates(select_id=None):
            candidate_box.delete(0, "end")
            visible_candidate_ids.clear()
            for row in candidates.rows:
                state = row["human_review_status"]
                if candidate_filter.get() == "PENDING" and state not in ("UNVERIFIED", "PENDING_SECOND_REVIEW", "DOUBTFUL", "SKIPPED"):
                    continue
                if candidate_filter.get() == "REVIEWED" and state in ("UNVERIFIED", "PENDING_SECOND_REVIEW", "SKIPPED"):
                    continue
                visible_candidate_ids.append(row["candidate_id"])
                candidate_box.insert("end", f"{row['candidate_id']}  {state}  score={float(row['matching_score']):.3f}  {row['grid_cell_id']}")
                if row["candidate_id"] == select_id:
                    candidate_box.selection_set(len(visible_candidate_ids)-1)

        def refresh_counts():
            states = [row["human_review_status"] for row in candidates.rows]
            reviewed = sum(state in ("APPROVED", "REJECTED", "DOUBTFUL") for state in states)
            pending = sum(state in ("UNVERIFIED", "SKIPPED") for state in states)
            second = states.count("PENDING_SECOND_REVIEW")
            review_counts.set(f"Candidate progress: reviewed={reviewed} pending={pending} second={second}; "
                              "only explicit human review can approve")

        def context_for(row):
            return build_candidate_context(row, scene_metadata, frozen_split)

        def display_context(row):
            context = context_for(row)
            objective_info.set(
                f"Pair {context['pair_id']} | {context['scene_i']} / {context['scene_j']} | "
                f"dates {context['acquisition_time_i']} / {context['acquisition_time_j']} "
                f"({context['elapsed_days']:.3f} days) | quality {context['quality_i']} / {context['quality_j']} | "
                f"{context['geographic_zone']} (original candidate grid). "
                f"Original i(row,col,x,y)=({context['original_row_i']},{context['original_col_i']},"
                f"{context['original_x_i']},{context['original_y_i']}); "
                f"j=({context['original_row_j']},{context['original_col_j']},"
                f"{context['original_x_j']},{context['original_y_j']}). "
                f"Source={context['candidate_source']}; method={context['recommendation_method']}")
            suggestions_info.set(
                f"Suggestions: Terrain={context['terrain_suggestion']}, "
                f"Landcover={context['landcover_suggestion']} "
                f"({context['label_suggestion_source']}); Split={context['split_suggestion']} "
                f"({context['split_source']}). Human labels remain separately editable.")
            return context

        def current_candidate():
            return candidates._row(selected_candidate_id[0]) if selected_candidate_id[0] else None

        def render():
            for idx, ds in enumerate((a, b)):
                row_center, col_center = projected_to_pixel(ds.transform, *centers[idx])
                size = max(16, int(round(PANEL_SIZE / scale[0])))
                col0 = int(round(col_center - size / 2))
                row0 = int(round(row_center - size / 2))
                window = Window(col0, row0, size, size)
                data = ds.read(1, window=window, boundless=True, fill_value=0)
                mask = ds.read_masks(1, window=window, boundless=True)
                img = stretch(data, mask).resize((PANEL_SIZE, PANEL_SIZE), Image.Resampling.BILINEAR)
                images[idx] = ImageTk.PhotoImage(img)
                c = canvases[idx]
                c.delete("all")
                c.create_image(0, 0, image=images[idx], anchor="nw")
                viewport[idx] = Viewport(col0, row0, PANEL_SIZE / size, PANEL_SIZE, PANEL_SIZE)
                for n, row in enumerate(rows):
                    rr, cc = float(row[f"row_{'i' if idx == 0 else 'j'}"]), float(row[f"col_{'i' if idx == 0 else 'j'}"])
                    px, py = viewport[idx].native_to_screen(rr, cc)
                    if viewport[idx].contains(px, py):
                        color = "lime" if row["review_status"] == "APPROVED" else "yellow"
                        c.create_oval(px-3, py-3, px+3, py+3, outline=color, width=2)
                        c.create_text(px+5, py-7, text=row["point_id"], fill=color, anchor="w")
                selected = current_candidate()
                if selected is not None:
                    suffix = "i" if idx == 0 else "j"
                    prefix = "adjusted_" if selected["human_adjusted"] == "True" else ""
                    rr = float(selected[prefix + f"row_{suffix}"])
                    cc = float(selected[prefix + f"col_{suffix}"])
                    px, py = viewport[idx].native_to_screen(rr, cc)
                    c.create_oval(px-9, py-9, px+9, py+9, outline="cyan", width=2)
                    c.create_line(px-13, py, px+13, py, fill="cyan", width=2)
                    c.create_line(px, py-13, px, py+13, fill="cyan", width=2)
                    c.create_text(px+11, py-11, text=selected["candidate_id"], fill="cyan", anchor="w")
                if pending[idx] is not None:
                    px, py = viewport[idx].native_to_screen(*pending[idx])
                    c.create_line(px-8, py, px+8, py, fill="red", width=2)
                    c.create_line(px, py-8, px, py+8, fill="red", width=2)

        def click(idx, event):
            if not viewport[idx].contains(event.x, event.y):
                return
            rr, cc = viewport[idx].screen_to_native(event.x, event.y)
            ds = (a, b)[idx]
            if not (0 <= rr < ds.height and 0 <= cc < ds.width):
                status.set("Click outside source bounds; ignored.")
                return
            row_index, col_index = int(round(rr)), int(round(cc))
            if not (0 <= row_index < ds.height and 0 <= col_index < ds.width) or not ds.read_masks(
                1, window=Window(col_index, row_index, 1, 1)
            )[0, 0]:
                status.set("NoData at click; ignored.")
                return
            pending[idx] = (rr, cc)
            if all(v is not None for v in pending):
                if candidate_edit_mode[0]:
                    selected = current_candidate()
                    if selected is None:
                        status.set("Select a candidate before adjustment.")
                        return
                    candidates.adjust(selected["candidate_id"], pending[0], pending[1],
                                      a.transform, b.transform, reviewer=reviewer)
                    candidate_edit_mode[0] = False
                    candidate_dirty[0] = True
                    status.set("Candidate adjusted; original suggestion retained; review reset to UNVERIFIED.")
                    refresh_candidates(selected["candidate_id"])
                    display_context(candidates._row(selected["candidate_id"]))
                    refresh_counts()
                elif edit_mode[0]:
                    n = selected_index()
                    if n is None:
                        status.set("Select a row before edit.")
                        return
                    old = rows[n]
                    revised = new_row(pair_id, scene_i["scene_id"], scene_j["scene_id"], reviewer,
                                      old["point_id"], pending[0], pending[1], a.transform, b.transform)
                    revised.update(landcover_type=old["landcover_type"], terrain_class=old["terrain_class"],
                                   confidence=old["confidence"], notes=old["notes"], split=old["split"])
                    rows[n] = revised  # edited coordinates invalidate previous approval
                    manual_dirty[0] = True
                    edit_mode[0] = False
                    status.set("Edited coordinates; status reset to UNVERIFIED. Save to persist.")
                    refresh_list(revised["point_id"])
                elif manual_add_mode[0]:
                    used = {r["point_id"] for r in rows}
                    number = 1
                    while f"{pair_id}_P{number:04d}" in used:
                        number += 1
                    row = new_row(pair_id, scene_i["scene_id"], scene_j["scene_id"], reviewer,
                                  f"{pair_id}_P{number:04d}", pending[0], pending[1], a.transform, b.transform)
                    rows.append(row)
                    manual_dirty[0] = True
                    refresh_list(row["point_id"])
                    status.set("Point added UNVERIFIED. Add metadata, review, and Save.")
                else:
                    status.set("Candidate selected: use 'Adjust candidate' or 'Manual add' before clicking.")
                pending[:] = [None, None]
            render()

        def move(dx, dy):
            for center in centers:
                center[0] += dx * 14 * (PANEL_SIZE / scale[0]) / 2
                center[1] += dy * 14 * (PANEL_SIZE / scale[0]) / 2
            render()

        def zoom(factor):
            scale[0] = max(0.5, min(8.0, scale[0] * factor))
            render()

        for idx, c in enumerate(canvases):
            c.bind("<Button-1>", lambda event, which=idx: click(which, event))
            c.bind("<MouseWheel>", lambda event: zoom(2 if event.delta > 0 else 0.5))
            c.bind("<Button-2>", lambda event: status.set("Use arrow buttons to pan common map center."))
        for label, command in (("←", lambda: move(-1, 0)), ("→", lambda: move(1, 0)),
                               ("↑", lambda: move(0, 1)), ("↓", lambda: move(0, -1)),
                               ("Zoom +", lambda: zoom(2)), ("Zoom -", lambda: zoom(.5))):
            ttk.Button(toolbar, text=label, command=command).pack(side="left")

        def clear_pending():
            pending[:] = [None, None]
            edit_mode[0] = False
            candidate_edit_mode[0] = False
            render()
            status.set("Skipped pending click; no point created.")

        def edit_selected():
            if selected_index() is None:
                status.set("Select a point first.")
                return
            edit_mode[0] = True
            candidate_edit_mode[0] = False
            manual_add_mode[0] = True
            pending[:] = [None, None]
            status.set("Edit mode: click new location in both images; approval will reset.")

        def delete_selected():
            n = selected_index()
            if n is not None and messagebox.askyesno("Delete annotation", "Delete selected point?", parent=root):
                rows.pop(n)
                manual_dirty[0] = True
                refresh_list()
                render()
                status.set("Point deleted in memory; Save to persist.")

        def load_selected(event=None):
            n = selected_index()
            if n is None:
                return
            row = rows[n]
            split_combo.configure(values=("UNASSIGNED", "DEVELOPMENT", "TEST"))
            landcover.set(row["landcover_type"])
            terrain.set(row["terrain_class"])
            confidence.set(row["confidence"])
            review.set(row["review_status"])
            split.set(row["split"])
            precision_note.set(row["coordinate_precision_note"])
            notes.set(row["notes"])

        def apply_review():
            n = selected_index()
            if n is None:
                status.set("No manual point selected. For an auto candidate, select it in the TOP list and click Apply candidate review.")
                return
            if review.get() == "APPROVED":
                if (split.get() not in ("DEVELOPMENT", "TEST") or confidence.get() == "UNKNOWN"
                        or not precision_note.get().strip()
                        or precision_note.get() == "mouse_click_subpixel_coordinate_not_accuracy_claim"):
                    messagebox.showerror("Incomplete review", "APPROVED requires split, confidence, and human coordinate uncertainty note.", parent=root)
                    return
                if not messagebox.askyesno("Independent review", "Confirm this point was manually checked and is independent of matcher output?", parent=root):
                    return
            rows[n].update(landcover_type=landcover.get(), terrain_class=terrain.get(),
                           confidence=confidence.get(), review_status=review.get(), split=split.get(),
                           coordinate_precision_note=precision_note.get(), notes=notes.get(), reviewer=reviewer,
                           review_time_utc=datetime.now(timezone.utc).isoformat())
            manual_dirty[0] = True
            refresh_list(rows[n]["point_id"])
            render()
            status.set("Review updated in memory; Save to persist.")

        def mark_doubtful():
            n = selected_index()
            if n is not None:
                rows[n].update(review_status="UNVERIFIED", confidence="LOW",
                               notes=(rows[n]["notes"] + " | DOUBTFUL").strip(" |"), reviewer=reviewer,
                               review_time_utc=datetime.now(timezone.utc).isoformat())
                manual_dirty[0] = True
                refresh_list(rows[n]["point_id"])
                render()

        def select_candidate(candidate_id):
            row = candidates._row(candidate_id)
            context = display_context(row)
            allowed_splits = ["UNASSIGNED"]
            if context["split_suggestion"] != "UNASSIGNED":
                allowed_splits.append(context["split_suggestion"])
            if row["split"] and row["split"] not in allowed_splits:
                allowed_splits.append(row["split"])  # preserve already reviewed records
            split_combo.configure(values=allowed_splits)
            selected_candidate_id[0] = candidate_id
            manual_add_mode[0] = False
            edit_mode[0] = False
            candidate_edit_mode[0] = False
            pending[:] = [None, None]
            prefix = "adjusted_" if row["human_adjusted"] == "True" else ""
            centers[0][:] = [float(row[prefix + "x_i"]), float(row[prefix + "y_i"])]
            centers[1][:] = [float(row[prefix + "x_j"]), float(row[prefix + "y_j"])]
            scale[0] = 4.0
            landcover.set(row["landcover_type"] or "UNKNOWN")
            terrain.set(row["terrain_class"] or "TERRAIN_UNKNOWN")
            confidence.set(row["confidence"] or "UNKNOWN")
            review.set(row["human_review_status"] if row["human_review_status"] in
                       ("APPROVED", "REJECTED", "DOUBTFUL", "UNVERIFIED") else "UNVERIFIED")
            split.set(row["split"] if row["split"] and row["split"] != "UNASSIGNED"
                      else context["split_suggestion"])
            precision_note.set(row["coordinate_precision_note"] or "")
            notes.set(row["review_notes"] or "")
            candidate_score.set(f"{candidate_id}: score={float(row['matching_score']):.3f}, "
                                f"FB={float(row['forward_backward_error']):.2f}px, "
                                f"grid={row['grid_cell_id']} (not accuracy)")
            render()
            candidate_viewed.add(candidate_id)
            status.set("Inspect BOTH cyan markers at native-scale zoom. Candidate is not truth until explicit review.")

        def select_candidate_from_list(event=None):
            selection = candidate_box.curselection()
            if selection:
                select_candidate(visible_candidate_ids[selection[0]])

        def step_candidate(step):
            if not visible_candidate_ids:
                return
            if selected_candidate_id[0] in visible_candidate_ids:
                pos = visible_candidate_ids.index(selected_candidate_id[0])
                pos = max(0, min(len(visible_candidate_ids)-1, pos + step))
            else:
                pos = 0
            candidate_box.selection_clear(0, "end")
            candidate_box.selection_set(pos)
            candidate_box.see(pos)
            select_candidate(visible_candidate_ids[pos])

        def skip_candidate():
            row = current_candidate()
            if row is None:
                step_candidate(1)
                return
            current_index = visible_candidate_ids.index(row["candidate_id"]) if row["candidate_id"] in visible_candidate_ids else -1
            candidates.review(row["candidate_id"], "SKIPPED", reviewer, row["split"] or "UNASSIGNED",
                              row["confidence"] or "UNKNOWN", row["coordinate_precision_note"],
                              landcover=row["landcover_type"], terrain=row["terrain_class"],
                              notes=row["review_notes"])
            candidate_dirty[0] = True
            refresh_candidates(row["candidate_id"])
            refresh_counts()
            if visible_candidate_ids:
                next_index = min(len(visible_candidate_ids)-1, max(0, current_index + 1))
                select_candidate(visible_candidate_ids[next_index])
            status.set("Candidate marked SKIPPED (not truth). Save to persist.")

        def adjust_candidate():
            if current_candidate() is None:
                status.set("Select a candidate first.")
                return
            candidate_edit_mode[0] = True
            manual_add_mode[0] = False
            edit_mode[0] = False
            pending[:] = [None, None]
            status.set("Adjust candidate: click corrected location in BOTH panels; original suggestion remains in CSV.")

        def manual_add():
            manual_add_mode[0] = True
            split_combo.configure(values=("UNASSIGNED", "DEVELOPMENT", "TEST"))
            candidate_edit_mode[0] = False
            edit_mode[0] = False
            pending[:] = [None, None]
            status.set("Manual source: click the same feature in both panels. This will not alter candidates.")

        def apply_candidate_review():
            row = current_candidate()
            if row is None or row["candidate_id"] not in candidate_viewed or manual_add_mode[0]:
                status.set("Select and inspect a candidate in both images first.")
                return
            if len(candidate_box.curselection()) > 1:
                status.set("Select exactly one candidate for review; bulk approval is prohibited.")
                return
            state = review.get()
            if state == "APPROVED":
                if not messagebox.askyesno("Human-only approval", "I inspected BOTH original-resolution windows and confirmed the same stable feature. Candidate score is NOT accuracy. Continue?", parent=root):
                    return
            try:
                candidates.review(row["candidate_id"], state, reviewer, split.get(),
                                  confidence.get(), precision_note.get(),
                                  second_reviewer=reviewer, landcover=landcover.get(),
                                  terrain=terrain.get(), notes=notes.get())
            except ValueError as exc:
                messagebox.showerror("Review rejected", str(exc), parent=root)
                return
            candidate_dirty[0] = True
            state = candidates._row(row["candidate_id"])["human_review_status"]
            refresh_candidates(row["candidate_id"])
            refresh_counts()
            render()
            status.set(f"Candidate review={state}; TEST requires an independent second reviewer. Save to persist.")

        def bulk_labels():
            positions = candidate_box.curselection()
            if not positions:
                status.set("Ctrl/Shift-select candidate rows before Bulk labels.")
                return
            selected_ids = [visible_candidate_ids[index] for index in positions]
            dialog = tk.Toplevel(root)
            dialog.title(f"Batch non-review labels: {len(selected_ids)} selected")
            dialog.transient(root)
            ttk.Label(dialog, text="Only selected UNVERIFIED candidates; Review, Confidence and Split are never batched.").pack(fill="x")
            terrain_choice = tk.StringVar(value="KEEP")
            landcover_choice = tk.StringVar(value="KEEP")
            for label, variable, values in (
                ("Terrain", terrain_choice, ("KEEP", *sorted(TERRAIN_VALUES))),
                ("Landcover", landcover_choice, ("KEEP", *sorted(LANDCOVER_VALUES))),
            ):
                line = ttk.Frame(dialog)
                line.pack(fill="x")
                ttk.Label(line, text=label).pack(side="left")
                ttk.Combobox(line, textvariable=variable, values=values, state="readonly").pack(side="left")

            def apply_selected_labels():
                target_terrain = None if terrain_choice.get() == "KEEP" else terrain_choice.get()
                target_landcover = None if landcover_choice.get() == "KEEP" else landcover_choice.get()
                if target_terrain is None and target_landcover is None:
                    status.set("Choose Terrain or Landcover; no batch edit occurred.")
                    return
                if not messagebox.askyesno("Confirm batch labels",
                                           f"Apply Terrain={target_terrain or 'KEEP'}, "
                                           f"Landcover={target_landcover or 'KEEP'} to exactly "
                                           f"{len(selected_ids)} selected UNVERIFIED candidates? "
                                           "No approval, confidence or split will change.", parent=dialog):
                    return
                try:
                    changes = apply_batch_labels(candidates, selected_ids, reviewer=reviewer,
                                                 terrain=target_terrain, landcover=target_landcover)
                except (ValueError, KeyError) as exc:
                    messagebox.showerror("Batch labels rejected", str(exc), parent=dialog)
                    return
                if changes:
                    candidate_dirty[0] = True
                refresh_candidates(selected_ids[0])
                select_candidate(selected_ids[0])
                status.set(f"Updated non-review labels on {len(changes)} candidates in memory; Save CSV to persist.")
                dialog.destroy()

            ttk.Button(dialog, text="Apply selected labels", command=apply_selected_labels).pack(fill="x")

        def save():
            changed = manual_dirty[0] or candidate_dirty[0]
            if manual_dirty[0]:
                save_truth(out_path, rows)
                manual_dirty[0] = False
            if candidate_dirty[0]:
                try:
                    context_snapshot = {row["candidate_id"]: context_for(row) for row in candidates.rows}
                    save_candidate_session(candidates, candidate_saved_rows, context_snapshot,
                                           auto_truth_path, reviewer=reviewer)
                except (OSError, ValueError, KeyError) as exc:
                    messagebox.showerror("Candidate save failed", str(exc), parent=root)
                    return
                candidate_saved_rows.clear()
                candidate_saved_rows.update({row["candidate_id"]: dict(row) for row in candidates.rows})
                candidate_dirty[0] = False
            if changed:
                provenance_path.parent.mkdir(parents=True, exist_ok=True)
                with provenance_path.open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.DictWriter(stream, fieldnames=["point_id", "pair_id", "point_source", "origin_candidate_id"])
                    writer.writeheader()
                    writer.writerows([{"point_id": r["point_id"], "pair_id": pair_id,
                                      "point_source": "MANUAL", "origin_candidate_id": ""} for r in rows] +
                                     [{"point_id": r["point_id"], "pair_id": pair_id,
                                       "point_source": "AUTO_SUGGESTED", "origin_candidate_id": r["origin_candidate_id"]}
                                      for r in candidates.approved_truth()])
            status.set(f"Saved changes. Manual={len(rows)}; candidate reviewed="
                       f"{sum(r['human_review_status'] != 'UNVERIFIED' for r in candidates.rows)}; "
                       f"approved auto truth={len(candidates.approved_truth())}.")

        for label, command in (("Skip", clear_pending), ("Manual add", manual_add),
                               ("Edit selected", edit_selected),
                               ("Delete selected", delete_selected), ("Doubtful", mark_doubtful),
                               ("Apply manual-point review", apply_review), ("Save CSV", save)):
            ttk.Button(toolbar, text=label, command=command).pack(side="left")
        for label, command in (("Previous", lambda: step_candidate(-1)),
                               ("Next", lambda: step_candidate(1)),
                               ("Skip candidate", skip_candidate),
                               ("Adjust candidate", adjust_candidate),
                               ("Apply candidate review", apply_candidate_review)):
            ttk.Button(candidate_actions, text=label, command=command).pack(side="left")
        ttk.Button(batch_actions, text="Bulk labels...", command=bulk_labels).pack(side="left")
        ttk.Label(batch_actions, text="Ctrl/Shift-select; Terrain/Landcover only.").pack(side="left")
        listbox.bind("<<ListboxSelect>>", load_selected)
        candidate_box.bind("<<ListboxSelect>>", select_candidate_from_list)
        candidate_filter.trace_add("write", lambda *_: refresh_candidates(selected_candidate_id[0]))
        root.protocol("WM_DELETE_WINDOW", lambda: root.destroy() if messagebox.askyesno(
            "Close", "Close window? Unsaved changes will be lost.", parent=root) else None)
        refresh_list()
        refresh_candidates()
        refresh_counts()
        render()
        if smoke and candidates.rows:
            def smoke_once():
                select_candidate(candidates.rows[0]["candidate_id"])
                adjust_candidate()
                first = current_candidate()
                for idx, suffix in ((0, "i"), (1, "j")):
                    rr = float(first[f"row_{suffix}"]) + .25
                    cc = float(first[f"col_{suffix}"]) + .25
                    px, py = viewport[idx].native_to_screen(rr, cc)
                    click(idx, type("Event", (), {"x": int(round(px)), "y": int(round(py))})())
                review.set("REJECTED")
                apply_candidate_review()
                save()
                root.destroy()
            root.after(250, smoke_once)
            root.after(10000, root.destroy)
        try:
            root.mainloop()
        finally:
            destroy_tk_root_if_alive(root, tk)


def run_second_review_gui(pair_id: str, reviewer: str, snapshot, event_path: Path,
                          *, smoke: bool = False) -> None:
    """Isolated TEST second review; never saves Phase2 candidates or Phase3 rows."""
    configure_tk_runtime()
    import tkinter as tk
    from tkinter import messagebox, ttk

    from benchmark_phase2_core import projected_to_pixel
    from benchmark_phase4_freeze import load_frozen_trial, verify_b9_identity
    from benchmark_phase4_review import (ImageGeometry, ReviewError, append_review_event,
                                         load_review_events, propose_review, review_state,
                                         validate_event_destination, validate_review_geometry_history)

    keys = sorted(key for key, row in snapshot.assignments.items()
                  if key[0] == pair_id and row["split"] == "TEST")
    if not keys:
        raise ValueError(f"No frozen TEST candidates for {pair_id}")
    event_path = validate_event_destination(event_path, snapshot)
    left, right = pair_paths(pair_id)
    if not smoke:
        verify_b9_identity(snapshot)
        expected = {item["scene_id"]: Path(item["path"]).resolve()
                    for item in snapshot.audit["b9_verified_inputs"]}
        for scene in (left, right):
            if (scene["scene_id"] not in expected or
                    Path(scene["b9_path"]).resolve() != expected[scene["scene_id"]]):
                raise ValueError("Second-review GUI image path differs from frozen B9 identity")
    with rasterio.open(left["b9_path"]) as ds_i, rasterio.open(right["b9_path"]) as ds_j:
        if ds_i.crs.to_epsg() != 32650 or ds_j.crs.to_epsg() != 32650:
            raise ValueError("Second review requires EPSG:32650 B9 inputs")
        def mask_check(ds):
            def valid(row, col):
                r, c = int(round(row)), int(round(col))
                return (0 <= r < ds.height and 0 <= c < ds.width
                        and bool(ds.read_masks(1, window=Window(c, r, 1, 1))[0, 0]))
            return valid
        geometry = {
            left["scene_id"]: ImageGeometry(ds_i.transform, ds_i.height, ds_i.width, mask_check(ds_i)),
            right["scene_id"]: ImageGeometry(ds_j.transform, ds_j.height, ds_j.width, mask_check(ds_j)),
        }
        events = load_review_events(event_path, snapshot)
        validate_review_geometry_history(snapshot, events, geometry)
        root = create_tk_root(tk)
        root.title(f"DZ01V independent second review: {pair_id} | {reviewer}")
        root.geometry(f"{min(1200, root.winfo_screenwidth()-60)}x{min(900, root.winfo_screenheight()-80)}")
        ttk.Label(root, text=f"Frozen TEST — {pair_id}; role and global cell cannot be edited").pack(fill="x")
        counts_label = ttk.Label(root, text=f"Pending: {len(keys)}; approved-valid: 0; doubtful: 0; rejected: 0")
        counts_label.pack(fill="x")
        info = tk.StringVar(value="Select one TEST candidate. Inspect BOTH original-resolution views independently.")
        ttk.Label(root, textvariable=info).pack(fill="x")
        candidate_box = tk.Listbox(root, height=6, exportselection=False)
        candidate_box.pack(fill="x")
        controls = ttk.Frame(root)
        controls.pack(fill="x")
        uncertain = tk.StringVar(value="")
        adjustment_reason = tk.StringVar(value="")
        disagreement = tk.StringVar(value="NO")
        disagreement_note = tk.StringVar(value="")
        checked_i, checked_j = tk.BooleanVar(value=False), tk.BooleanVar(value=False)
        ttk.Checkbutton(controls, text="I inspected LEFT at original resolution", variable=checked_i).pack(side="left")
        ttk.Checkbutton(controls, text="I inspected RIGHT at original resolution", variable=checked_j).pack(side="left")
        fields = ttk.Frame(root)
        fields.pack(fill="x")
        for label, variable in (("New human uncertainty", uncertain), ("Adjustment reason", adjustment_reason),
                                ("Disagreement explanation", disagreement_note)):
            ttk.Label(fields, text=label).pack(side="left")
            ttk.Entry(fields, textvariable=variable, width=20).pack(side="left")
        ttk.Label(fields, text="Material disagreement?").pack(side="left")
        ttk.Combobox(fields, textvariable=disagreement, values=("NO", "YES"),
                     state="readonly", width=5).pack(side="left")
        actions = ttk.Frame(root)
        actions.pack(fill="x")
        workspace = ttk.Frame(root)
        workspace.pack(fill="both", expand=True)
        body = tk.Canvas(workspace, highlightthickness=0)
        vertical = ttk.Scrollbar(workspace, orient="vertical", command=body.yview)
        horizontal = ttk.Scrollbar(workspace, orient="horizontal", command=body.xview)
        body.configure(yscrollcommand=vertical.set, xscrollcommand=horizontal.set)
        vertical.pack(side="right", fill="y")
        horizontal.pack(side="bottom", fill="x")
        body.pack(side="left", fill="both", expand=True)
        panels = ttk.Frame(body)
        body.create_window((0, 0), window=panels, anchor="nw")
        panels.bind("<Configure>", lambda _: body.configure(scrollregion=body.bbox("all")))
        canvases = []
        for index, scene in enumerate((left, right)):
            column = ttk.Frame(panels)
            column.grid(row=0, column=index, padx=4)
            ttk.Label(column, text=scene["scene_id"]).pack()
            canvas = tk.Canvas(column, width=PANEL_SIZE, height=PANEL_SIZE, bg="black")
            canvas.pack()
            canvases.append(canvas)
        selected = [None]
        positions = [(0.0, 0.0), (0.0, 0.0)]
        images = [None, None]
        viewports = [None, None]
        window_size = [PANEL_SIZE]

        def current_rowcol(key):
            past = [event for event in events if (event["pair_id"], event["candidate_id"]) == key]
            if past:
                after = past[-1]["after"]
                return [(after["left"]["row"], after["left"]["col"]),
                        (after["right"]["row"], after["right"]["col"])]
            source = snapshot.candidates[key]
            prefix = "adjusted_" if source.get("human_adjusted", "").lower() == "true" else ""
            return [(float(source[prefix + "row_i"]), float(source[prefix + "col_i"])),
                    (float(source[prefix + "row_j"]), float(source[prefix + "col_j"]))]

        def render():
            for index, ds in enumerate((ds_i, ds_j)):
                row, col = positions[index]
                size = window_size[0]
                col0 = int(round(col - size / 2))
                row0 = int(round(row - size / 2))
                window = Window(col0, row0, size, size)
                data = ds.read(1, window=window, boundless=True, fill_value=0)
                mask = ds.read_masks(1, window=window, boundless=True)
                preview = stretch(data, mask).resize((PANEL_SIZE, PANEL_SIZE), Image.Resampling.BILINEAR)
                images[index] = ImageTk.PhotoImage(preview)
                canvas = canvases[index]
                canvas.delete("all")
                canvas.create_image(0, 0, image=images[index], anchor="nw")
                viewports[index] = Viewport(col0, row0, PANEL_SIZE / size, PANEL_SIZE, PANEL_SIZE)
                px, py = viewports[index].native_to_screen(row, col)
                canvas.create_line(px-12, py, px+12, py, fill="cyan", width=2)
                canvas.create_line(px, py-12, px, py+12, fill="cyan", width=2)

        def refresh():
            states = review_state(snapshot, events)
            candidate_box.delete(0, "end")
            for n, key in enumerate(keys):
                frozen = snapshot.assignments[key]
                state = states[key]
                candidate_box.insert("end", f"{key[1]}  {frozen['grid_cell_id']}  {state['human_outcome']}  {state['spatial_status']}")
                if key == selected[0]:
                    candidate_box.selection_set(n)
            this = [states[key] for key in keys]
            counts_label.configure(text=(
                f"Pending: {sum(s['human_outcome']=='PENDING' for s in this)}; "
                f"approved-valid: {sum(s['eligible'] for s in this)}; "
                f"doubtful: {sum(s['human_outcome']=='DOUBTFUL' for s in this)}; "
                f"rejected: {sum(s['human_outcome']=='REJECTED' for s in this)}; "
                f"ineligible: {sum(s['spatial_status']=='NOT_ELIGIBLE_AFTER_COORDINATE_ADJUSTMENT' for s in this)}; "
                f"disputes: {sum(s['dispute_pending'] for s in this)}"))

        def choose(index):
            selected[0] = keys[index]
            positions[:] = current_rowcol(selected[0])
            checked_i.set(False)
            checked_j.set(False)
            uncertain.set("")
            adjustment_reason.set("")
            disagreement.set("NO")
            disagreement_note.set("")
            info.set(f"Frozen TEST {selected[0][1]} | {snapshot.assignments[selected[0]]['grid_cell_id']} | score is NOT accuracy")
            refresh()
            render()

        def select_list(_=None):
            index = candidate_box.curselection()
            if index:
                choose(index[0])

        def move(delta):
            choose((keys.index(selected[0]) + delta) % len(keys) if selected[0] else 0)

        def click(index, event):
            viewport = viewports[index]
            if not viewport.contains(event.x, event.y):
                return
            row, col = viewport.screen_to_native(event.x, event.y)
            positions[index] = (row, col)
            (checked_i if index == 0 else checked_j).set(False)
            info.set("Position adjusted in memory. Enter reason, reinspect BOTH views, then choose outcome.")
            render()

        def save_decision(outcome):
            if selected[0] is None:
                info.set("Select one frozen TEST candidate before second-review action.")
                return
            if outcome == "APPROVED" and not messagebox.askyesno(
                    "Independent second review", "I independently inspected BOTH original-resolution images; this is my own human judgment. Continue?", parent=root):
                return
            try:
                event = propose_review(snapshot, selected[0], reviewer, outcome, positions[0], positions[1],
                                       {"left_original_resolution": checked_i.get(),
                                        "right_original_resolution": checked_j.get()},
                                       uncertain.get(), adjustment_reason.get(), disagreement.get(),
                                       disagreement_note.get(), geometry, events)
                append_review_event(event_path, event)
                events[:] = load_review_events(event_path, snapshot)
                validate_review_geometry_history(snapshot, events, geometry)
                refresh()
                info.set(f"Saved {outcome}; spatial={event['spatial_status']}; reasons={event['spatial_reasons']}")
            except (ReviewError, OSError, ValueError) as exc:
                messagebox.showerror("Second review not saved", str(exc), parent=root)

        candidate_box.bind("<<ListboxSelect>>", select_list)
        for index, canvas in enumerate(canvases):
            canvas.bind("<Button-1>", lambda event, index=index: click(index, event))
        for name, command in (("Previous", lambda: move(-1)), ("Next", lambda: move(1)),
                              ("Zoom +", lambda: (window_size.__setitem__(0, max(64, window_size[0]//2)), render())),
                              ("Zoom -", lambda: (window_size.__setitem__(0, min(2048, window_size[0]*2)), render())),
                              ("Approve second review", lambda: save_decision("APPROVED")),
                              ("Reject", lambda: save_decision("REJECTED")),
                              ("Doubtful", lambda: save_decision("DOUBTFUL"))):
            ttk.Button(actions, text=name, command=command).pack(side="left")
        refresh()
        try:
            root.mainloop()
        finally:
            destroy_tk_root_if_alive(root, tk)
            if not smoke:  # synthetic GUI tests have no on-disk frozen snapshot
                reloaded = load_frozen_trial(snapshot.root)
                if (reloaded.candidate_sha256 != snapshot.candidate_sha256
                        or reloaded.assignment_sha256 != snapshot.assignment_sha256):
                    raise RuntimeError("Frozen source bytes changed during second-review GUI session")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", choices=sorted(TRIAL_PAIRS), default="00_01")
    parser.add_argument("--reviewer", default="")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--candidates", type=Path)
    parser.add_argument("--frozen-split", type=Path,
                        help="Optional pre-frozen EPSG:32650 GeoJSON spatial split; never use draft regions")
    parser.add_argument("--second-review", action="store_true",
                        help="Independent Phase4 TEST second review; Phase2/Phase3 files stay read-only")
    parser.add_argument("--second-review-events", type=Path,
                        help="Alternative second-review JSONL destination (use disposable path for tests)")
    parser.add_argument("--check-env", action="store_true", help="Check files and GUI creation without running annotation")
    parser.add_argument("--count-approved", action="store_true", help="Read-only count of reviewed independent points")
    parser.add_argument("--smoke-candidates", action="store_true", help="Exercise GUI clicks on a disposable CSV copy only")
    args = parser.parse_args()
    if args.second_review:
        from benchmark_phase4_freeze import FrozenInputError, load_frozen_trial
        if not args.reviewer.strip():
            parser.error("--second-review requires --reviewer")
        try:
            frozen = load_frozen_trial(ROOT)
            event_path = args.second_review_events or ROOT / "docs/benchmark/v1/phase4/second_reviews" / f"{args.pair}.jsonl"
            run_second_review_gui(args.pair, args.reviewer.strip(), frozen, event_path)
        except (FrozenInputError, ValueError, OSError) as exc:
            message = f"Phase4 frozen second review cannot open: {exc}"
            print(message, file=sys.stderr)
            from tkinter import messagebox
            messagebox.showerror("Second-review freeze loading failed", message)
            return 2
        return 0
    left, right = pair_paths(args.pair)
    out_path = args.output or DEFAULT_OUT / f"{args.pair}.csv"
    candidate_path = args.candidates or DEFAULT_CANDIDATES / f"{args.pair}.csv"
    try:
        store = load_required_candidates(candidate_path, args.pair, left["scene_id"], right["scene_id"])
    except (OSError, ValueError, UnicodeError) as exc:
        message = f"Cannot load auto candidates for {args.pair} from {candidate_path.resolve()}: {exc}"
        print(message, file=sys.stderr)
        from tkinter import messagebox
        messagebox.showerror("Benchmark candidate loading failed", message)
        return 2
    try:
        frozen_split = load_frozen_split(args.frozen_split)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        message = f"Cannot use frozen spatial split {args.frozen_split}: {exc}"
        print(message, file=sys.stderr)
        from tkinter import messagebox
        messagebox.showerror("Benchmark split loading failed", message)
        return 2
    print(f"Loaded {len(store.rows)} auto candidates for {args.pair} from {candidate_path.resolve()}")
    if args.count_approved:
        manual = load_truth(out_path, args.pair, left["scene_id"], right["scene_id"])
        unassigned_approved = sum(
            row["human_review_status"] == "APPROVED" and row["split"] == "UNASSIGNED"
            for row in store.rows
        )
        print(f"pair={args.pair} manual_APPROVED={sum(r['review_status']=='APPROVED' for r in manual)} "
              f"auto_APPROVED={len(store.approved_truth())} "
              f"auto_APPROVED_UNASSIGNED={unassigned_approved} "
              f"auto_formal_APPROVED={len(store.approved_truth())} "
              f"auto_APPROVED_TEST_second_reviewed={len(store.approved_test_truth())} "
              f"auto_PENDING_SECOND_REVIEW={sum(r['human_review_status']=='PENDING_SECOND_REVIEW' for r in store.rows)}")
        return
    for scene in (left, right):
        with rasterio.open(scene["b9_path"]) as ds:
            if ds.count != 1 or ds.crs.to_epsg() != 32650:
                raise RuntimeError("Input is not a single-band EPSG:32650 B9 image")
    if args.check_env:
        configure_tk_runtime()
        import tkinter as tk
        app = create_tk_root(tk)
        app.withdraw()
        with rasterio.open(left["b9_path"]) as ds:
            preview = stretch(ds.read(1, window=Window(0, 0, 32, 32)),
                              ds.read_masks(1, window=Window(0, 0, 32, 32)))
            rendered = ImageTk.PhotoImage(preview.resize((64, 64), Image.Resampling.BILINEAR), master=app)
            if rendered.width() != 64:
                raise RuntimeError("Tk image rendering failed")
        app.update()
        app.destroy()
        print(f"GUI_CREATE_AND_RENDER_OK pair={args.pair} input_headers=2 candidates={len(store.rows)}; no annotation saved")
        return
    if args.smoke_candidates:
        if not store.rows:
            parser.error("--smoke-candidates requires an existing candidate CSV")
        with tempfile.TemporaryDirectory(prefix="dz01v_candidate_gui_") as temp:
            tempdir = Path(temp)
            disposable_candidates = tempdir / "candidates.csv"
            shutil.copy2(candidate_path, disposable_candidates)
            disposable_manual = tempdir / "manual.csv"
            if out_path.is_file():
                shutil.copy2(out_path, disposable_manual)
            original_manual = disposable_manual.read_bytes() if disposable_manual.is_file() else None
            run_gui(args.pair, args.reviewer or "SMOKE_REVIEWER", disposable_manual,
                    disposable_candidates, tempdir / "auto_truth.csv", tempdir / "provenance.csv",
                    smoke=True, frozen_split=frozen_split)
            checked = CandidateReviewStore(disposable_candidates, args.pair, left["scene_id"], right["scene_id"])
            first = checked.rows[0]
            if first["human_adjusted"] != "True" or first["human_review_status"] != "REJECTED":
                raise RuntimeError("GUI smoke failed to adjust/review/save candidate")
            if (disposable_manual.read_bytes() if disposable_manual.is_file() else None) != original_manual:
                raise RuntimeError("GUI candidate smoke modified manual annotations")
            print(f"GUI_CANDIDATE_SMOKE_OK pair={args.pair} adjusted=1 reviewed=REJECTED "
                  "manual_unchanged=True disposable_only=True")
        return
    if not args.reviewer.strip():
        parser.error("--reviewer is required for annotation")
    run_gui(args.pair, args.reviewer.strip(), out_path, candidate_path,
            DEFAULT_AUTO_TRUTH / f"{args.pair}.csv", DEFAULT_PROVENANCE / f"{args.pair}.csv",
            frozen_split=frozen_split)


if __name__ == "__main__":
    main()
