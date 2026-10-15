from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Any

import numpy as np
import rasterio
from PIL import Image, ImageTk
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window
from shapely import wkt
from shapely.geometry import mapping, shape

from benchmark_v1_radiometric_r1.roi_gui import preview_stretch
from .review_store import AutoReviewStore
from .spatial import audit_candidate_sets
from .multiband_view import (COMPOSITES, difference_preview, read_window_padded,
                             read_raster_roi_stats, render_composite, signed_difference_rgb,
                             stretch_pair)
from .multiband_review import (INTERFERENCE_TYPES, LANDCOVER_TYPES, TEMPORAL_STATUSES,
                               TERRAIN_TYPES)


LABELS = ("UNKNOWN", "STABLE", "REAL_CHANGE", "CLOUD", "SHADOW")
STATES = ("APPROVED", "DOUBTFUL", "REJECTED")


class AutoRoiReviewer:
    """Small read-only pair viewer with explicit per-ROI human review actions."""

    def __init__(self, root: tk.Tk, pair_id: str, store: AutoReviewStore,
                 paths_i: dict[int, Path], paths_j: dict[int, Path], reviewer: str,
                 *, second_review: bool = False, audit_features: list[dict[str, Any]] | None = None,
                 all_pairs_available: bool = False, multiband_mode: bool = False,
                 band_centers_nm: dict[int, float | None] | None = None,
                 machine_suggestions: dict[tuple[str, str], dict[str, Any]] | None = None):
        self.root, self.pair_id, self.store = root, pair_id, store
        self.paths_i, self.paths_j, self.reviewer = paths_i, paths_j, reviewer.strip()
        self.second_review = second_review
        self.audit_features = audit_features or []
        self.all_pairs_available = all_pairs_available
        self.multiband_mode = multiband_mode
        self.band_centers_nm = band_centers_nm or {}
        self.machine_suggestions = machine_suggestions or {}
        self.keys = [key for key in store.keys() if key[0] == pair_id]
        self.index = 0
        self.images: list[ImageTk.PhotoImage] = []
        self.zoom_px = 256
        self.render_window = None
        self.render_transform = None
        self.band = tk.IntVar(value=9 if 9 in paths_i and 9 in paths_j else min(set(paths_i) & set(paths_j)))
        self.filter_var = tk.StringVar(value="PENDING")
        self.label_var = tk.StringVar(value="UNKNOWN")
        self.status_var = tk.StringVar(value="DOUBTFUL")
        self.wkt_var = tk.StringVar(value="")
        self.evidence_var = tk.StringVar(value="")
        self.summary_var = tk.StringVar(value="")
        self.composite_var = tk.StringVar(value="GRAY")
        self.stretch_var = tk.StringVar(value="INDEPENDENT")
        self.display_var = tk.StringVar(value="SIDE_BY_SIDE")
        self.warning_var = tk.StringVar(value="")
        self.multiband_label_vars = {name: tk.StringVar(value="UNKNOWN") for name in
                                     ("landcover_type", "terrain_type", "temporal_status", "interference_type")}
        self.multiband_decision_var = tk.StringVar(value="UNVERIFIED")
        self.multiband_evidence_var = tk.StringVar(value="")
        self.legacy_var = tk.StringVar(value="")
        self.machine_suggestion_var = tk.StringVar(value="")
        self._multiband_drafts: dict[tuple[str, str], dict[str, str]] = {}
        self.render_mode_description = ""
        self._blink_photos: list[ImageTk.PhotoImage] = []
        self._blink_job = None
        self._build()
        self._refresh()

    def _build(self):
        self.root.title(f"R1.5 radiometric ROI review — {self.pair_id}")
        self.root.geometry("1540x980" if self.multiband_mode else "1160x820")
        outer = ttk.Frame(self.root, padding=6); outer.pack(fill="both", expand=True)
        toolbar = ttk.Frame(outer); toolbar.pack(fill="x")
        ttk.Label(toolbar, text="Queue").pack(side="left")
        filter_box = ttk.Combobox(toolbar, textvariable=self.filter_var,
                                  values=("PENDING", "REVIEWED", "ALL"), width=10, state="readonly")
        filter_box.pack(side="left", padx=4); filter_box.bind("<<ComboboxSelected>>", lambda _e: self._refresh())
        ttk.Button(toolbar, text="Previous", command=lambda: self._move(-1)).pack(side="left", padx=3)
        ttk.Button(toolbar, text="Next", command=lambda: self._move(1)).pack(side="left", padx=3)
        ttk.Label(toolbar, text="Band").pack(side="left", padx=(10, 2))
        bands = sorted(set(self.paths_i) & set(self.paths_j))
        self.band_box = ttk.Combobox(toolbar, textvariable=self.band, values=bands, width=5, state="readonly")
        self.band_box.pack(side="left"); self.band_box.bind("<<ComboboxSelected>>", lambda _e: self._render())
        ttk.Button(toolbar, text="Zoom +", command=lambda: self._change_zoom(0.5)).pack(side="left", padx=(8, 2))
        ttk.Button(toolbar, text="Zoom -", command=lambda: self._change_zoom(2.0)).pack(side="left", padx=2)
        if self.multiband_mode:
            ttk.Label(toolbar, text="Composition").pack(side="left", padx=(8, 2))
            composite_box = ttk.Combobox(toolbar, textvariable=self.composite_var,
                values=("GRAY", "TRUE_COLOR", "NIR_FALSE_COLOR"), width=15, state="readonly")
            composite_box.pack(side="left"); composite_box.bind("<<ComboboxSelected>>", lambda _e: self._render())
            ttk.Label(toolbar, text="Stretch").pack(side="left", padx=(4, 2))
            stretch_box = ttk.Combobox(toolbar, textvariable=self.stretch_var,
                values=("INDEPENDENT", "SHARED"), width=12, state="readonly")
            stretch_box.pack(side="left"); stretch_box.bind("<<ComboboxSelected>>", lambda _e: self._render())
            ttk.Label(toolbar, text="Two-date").pack(side="left", padx=(4, 2))
            display_box = ttk.Combobox(toolbar, textvariable=self.display_var,
                values=("SIDE_BY_SIDE", "BLINK", "DIFFERENCE"), width=15, state="readonly")
            display_box.pack(side="left"); display_box.bind("<<ComboboxSelected>>", lambda _e: self._render())
        self.listbox = tk.Listbox(outer, height=6, exportselection=False)
        self.listbox.pack(fill="x", pady=(5, 3)); self.listbox.bind("<<ListboxSelect>>", self._select)
        self.suggestion_var = tk.StringVar(value="")
        ttk.Label(outer, textvariable=self.suggestion_var, wraplength=1120).pack(fill="x")
        if self.multiband_mode:
            ttk.Label(outer, textvariable=self.machine_suggestion_var, wraplength=1120,
                      foreground="#315c25").pack(fill="x")
            ttk.Label(outer, textvariable=self.warning_var, wraplength=1120, foreground="#9c2600").pack(fill="x")
            ttk.Label(outer, textvariable=self.legacy_var, wraplength=1120).pack(fill="x")
        self.summary_label = ttk.Label(outer, textvariable=self.summary_var); self.summary_label.pack(fill="x")
        self.canvases = ttk.Frame(outer); self.canvases.pack(fill="both", expand=True, pady=4)
        self.canvas_i = tk.Canvas(self.canvases, width=540, height=430, background="#222")
        self.canvas_j = tk.Canvas(self.canvases, width=540, height=430, background="#222")
        self.canvas_i.pack(side="left", fill="both", expand=True, padx=(0, 3))
        self.canvas_j.pack(side="left", fill="both", expand=True, padx=(3, 0))
        if self.multiband_mode:
            fields = ttk.LabelFrame(outer, text="New visual labels — independent schema v2")
            fields.pack(fill="x", pady=3)
            options = {"landcover_type": sorted(LANDCOVER_TYPES), "terrain_type": sorted(TERRAIN_TYPES),
                       "temporal_status": sorted(TEMPORAL_STATUSES), "interference_type": sorted(INTERFERENCE_TYPES)}
            for name, values in options.items():
                ttk.Label(fields, text=name.replace("_", " ")).pack(side="left", padx=(4, 1))
                ttk.Combobox(fields, textvariable=self.multiband_label_vars[name], values=values,
                             width=14, state="readonly").pack(side="left", padx=(0, 3))
            ttk.Label(fields, text="Decision").pack(side="left", padx=(4, 1))
            ttk.Combobox(fields, textvariable=self.multiband_decision_var,
                         values=("UNVERIFIED", "APPROVED", "DOUBTFUL", "REJECTED"), width=11,
                         state="readonly").pack(side="left", padx=2)
            ttk.Label(fields, text="Evidence").pack(side="left", padx=(4, 1))
            ttk.Entry(fields, textvariable=self.multiband_evidence_var).pack(side="left", fill="x", expand=True, padx=2)
            self.multiband_review_button = ttk.Button(fields, text="Save new labels", command=self._apply_multiband_review)
            self.multiband_review_button.pack(side="left", padx=4)
            ttk.Label(outer, text="NO_CHANGE is a native-resolution observation, not formal STABLE truth. "
                "CONFIRMED_CHANGE requires evidence entered as 'source: …' and 'reference: …'; a radiometric difference alone is insufficient.",
                wraplength=1120, foreground="#7a4500").pack(fill="x")
        edit = ttk.LabelFrame(outer, text="Geometry edit — EPSG:32650 WKT; applying recomputes support/role and invalidates previous review")
        edit.pack(fill="x", pady=3)
        self.wkt_entry = ttk.Entry(edit, textvariable=self.wkt_var)
        self.wkt_entry.pack(side="left", fill="x", expand=True, padx=4, pady=4)
        self.geometry_button = ttk.Button(edit, text="Apply geometry", command=self._apply_geometry)
        self.geometry_button.pack(side="left", padx=4)
        review = ttk.LabelFrame(outer, text="Legacy Label (read-only)" if self.multiband_mode else "Human review (one selected ROI only)")
        if not self.multiband_mode:
            review.pack(fill="x", pady=3)
        else:
            self.legacy_review_frame = review
            return
        ttk.Label(review, text="Label").pack(side="left")
        ttk.Combobox(review, textvariable=self.label_var, values=LABELS, width=12, state="readonly").pack(side="left", padx=3)
        ttk.Label(review, text="Decision").pack(side="left")
        ttk.Combobox(review, textvariable=self.status_var, values=STATES, width=12, state="readonly").pack(side="left", padx=3)
        ttk.Label(review, text="Evidence").pack(side="left")
        ttk.Entry(review, textvariable=self.evidence_var).pack(side="left", fill="x", expand=True, padx=3)
        self.review_button = ttk.Button(review, text="Apply human review", command=self._apply_review)
        self.review_button.pack(side="left", padx=4)

    def _visible_keys(self):
        rows = []
        for key in self.keys:
            row = self.store.get(*key)
            if self.multiband_mode:
                pending = (row.get("role") == "TEST" and row.get("multiband_review_decision") == "APPROVED"
                           and row.get("multiband_second_review_status") == "UNVERIFIED") if self.second_review else (
                           row.get("multiband_review_decision") == "UNVERIFIED")
            else:
                pending = row.get("review_status") == "UNVERIFIED" if not self.second_review else (
                    row.get("role") == "TEST" and row.get("review_status") == "APPROVED" and row.get("second_review_status") == "UNVERIFIED")
            reviewed = not pending
            if self.filter_var.get() == "ALL" or (self.filter_var.get() == "PENDING" and pending) or (self.filter_var.get() == "REVIEWED" and reviewed):
                rows.append(key)
        return rows

    def _refresh(self):
        visible = self._visible_keys()
        current = self.keys[self.index] if self.keys else None
        self.listbox.delete(0, "end")
        for key in visible:
            row = self.store.get(*key)
            if self.multiband_mode:
                status = row.get("multiband_second_review_status") if self.second_review else row.get("multiband_review_decision")
            else:
                status = row.get("second_review_status") if self.second_review else row.get("review_status")
            self.listbox.insert("end", f"{key[1]}  {status}  suggestion={row.get('suggested_label')} / {row.get('suggestion_confidence')}")
        if visible:
            self.index = self.keys.index(current) if current in visible else self.keys.index(visible[0])
            visible_index = visible.index(self.keys[self.index])
            self.listbox.selection_set(visible_index); self.listbox.see(visible_index)
            self._populate(self.keys[self.index])
        counts = self.store.progress()
        self.summary_var.set(" | ".join(f"{k}={v}" for k, v in counts.items()))

    def _select(self, _event=None):
        selection = self.listbox.curselection()
        visible = self._visible_keys()
        if selection and selection[0] < len(visible):
            self._capture_multiband_draft()
            self.index = self.keys.index(visible[selection[0]])
            self._populate(self.keys[self.index])

    def _capture_multiband_draft(self):
        if self.multiband_mode and self.keys:
            key = self.keys[self.index]
            self._multiband_drafts[key] = {name: variable.get()
                                           for name, variable in self.multiband_label_vars.items()}

    def _populate(self, key):
        row = self.store.get(*key)
        self.wkt_var.set(shape(row["geometry"]).wkt)
        if self.multiband_mode:
            machine = self.machine_suggestions.get(key)
            current_labels = {name: row.get(name, "UNKNOWN") for name in
                              ("landcover_type", "terrain_type", "temporal_status", "interference_type")}
            may_prefill = (row.get("multiband_review_decision") == "UNVERIFIED" and
                           all(value == "UNKNOWN" for value in current_labels.values()) and
                           key not in self._multiband_drafts)
            if machine:
                for name, variable in self.multiband_label_vars.items():
                    recommendation = machine.get(f"suggested_{name}", {})
                    if may_prefill and current_labels[name] == "UNKNOWN":
                        variable.set(recommendation.get("label", "UNKNOWN"))
                self.machine_suggestion_var.set("Machine-only draft (not reviewed): " + " | ".join(
                    f"{name}={machine.get(f'suggested_{name}', {}).get('label', 'UNKNOWN')} "
                    f"[{machine.get(f'suggested_{name}', {}).get('confidence', 'UNKNOWN')}]; "
                    f"evidence={'; '.join(machine.get(f'suggested_{name}', {}).get('evidence', []))}; "
                    f"uncertainty={'; '.join(machine.get(f'suggested_{name}', {}).get('uncertainty_reasons', []))}"
                    for name in ("landcover_type", "terrain_type", "temporal_status", "interference_type")))
            else:
                self.machine_suggestion_var.set("No machine suggestion file loaded.")
            for name, variable in self.multiband_label_vars.items():
                if key in self._multiband_drafts:
                    variable.set(self._multiband_drafts[key].get(name, "UNKNOWN"))
                elif self.second_review:
                    variable.set(row.get(f"second_{name}", "UNKNOWN"))
                elif not (machine and may_prefill and current_labels[name] == "UNKNOWN"):
                    variable.set(row.get(name, "UNKNOWN"))
            decision = row.get("multiband_second_review_status") if self.second_review else row.get("multiband_review_decision")
            self.multiband_decision_var.set(decision if decision in ("UNVERIFIED", *STATES) else "UNVERIFIED")
            self.multiband_evidence_var.set(row.get("multiband_second_evidence", "") if self.second_review
                                             else row.get("multiband_evidence", ""))
            self.legacy_var.set(f"Legacy Label/history (read-only): {row.get('legacy_human_label', row.get('human_label', 'UNKNOWN'))} / "
                                f"{row.get('legacy_review_status', row.get('review_status', 'UNVERIFIED'))} — no automatic schema mapping.")
        self.label_var.set(row.get("human_label", "UNKNOWN"))
        status = row.get("second_review_status") if self.second_review else row.get("review_status")
        self.status_var.set(status if status in STATES else "DOUBTFUL")
        self.evidence_var.set(row.get("second_review_notes", "") if self.second_review else row.get("evidence_notes", ""))
        self.suggestion_var.set(
            f"{key[1]} | suggested={row.get('suggested_label')} ({row.get('suggestion_confidence')}); "
            f"evidence={row.get('suggestion_evidence')}; uncertainty={row.get('uncertainty_reason')}; "
            f"role={row.get('role')} cell={row.get('grid_cell_id')} support={row.get('valid_support_fraction', 'not evaluated')} "
            f"status={status}. Suggestion is not truth; all generated items begin UNVERIFIED."
        )
        self.warning_var.set("WARNING: cross-scene geometry is not independently verified; apparent differences may be caused by residual offset, view conditions, or masks. "
                             "Map: EPSG:32650 affine coordinates; J→I preview resampling: bilinear; values/units: raw DN / UNKNOWN; DEM: NOT_AVAILABLE." if self.multiband_mode else "")
        self._render()

    def _render(self):
        if self.multiband_mode:
            self._render_multiband()
            return
        self._render_legacy()

    def _render_legacy(self):
        if not self.keys:
            return
        row = self.store.get(*self.keys[self.index])
        geometry = shape(row["geometry"]); center = geometry.centroid
        band = int(self.band.get())
        path_i, path_j = self.paths_i[band], self.paths_j[band]
        with rasterio.open(path_i) as di, rasterio.open(path_j) as dj:
            col, row_px = (~di.transform) @ (center.x, center.y)
            size = self.zoom_px
            win = Window(round(col - size / 2), round(row_px - size / 2), size, size)
            self.render_window, self.render_transform = win, di.transform
            left = np.zeros((size, size), dtype=np.float32)
            right = np.zeros_like(left)
            valid_left = np.zeros((size, size), dtype=bool)
            valid_right = np.zeros_like(valid_left)
            w = win.round_offsets().round_lengths()
            c0, r0 = max(0, int(w.col_off)), max(0, int(w.row_off))
            c1, r1 = min(di.width, int(w.col_off) + size), min(di.height, int(w.row_off) + size)
            if c1 > c0 and r1 > r0:
                source_win = Window(c0, r0, c1-c0, r1-r0)
                dst_r, dst_c = r0-int(w.row_off), c0-int(w.col_off)
                sl = (slice(dst_r, dst_r+r1-r0), slice(dst_c, dst_c+c1-c0))
                left[sl] = di.read(1, window=source_win); valid_left[sl] = di.read_masks(1, window=source_win) > 0
            with WarpedVRT(dj, crs=di.crs, transform=di.transform, width=di.width, height=di.height,
                           resampling=Resampling.bilinear) as vrt:
                c0, r0 = max(0, int(w.col_off)), max(0, int(w.row_off))
                c1, r1 = min(vrt.width, int(w.col_off)+size), min(vrt.height, int(w.row_off)+size)
                if c1 > c0 and r1 > r0:
                    source_win = Window(c0, r0, c1-c0, r1-r0)
                    dst_r, dst_c = r0-int(w.row_off), c0-int(w.col_off)
                    sl = (slice(dst_r, dst_r+r1-r0), slice(dst_c, dst_c+c1-c0))
                    right[sl] = vrt.read(1, window=source_win); valid_right[sl] = vrt.read_masks(1, window=source_win) > 0
        self.images = []
        for canvas, image, mask in ((self.canvas_i, left, valid_left), (self.canvas_j, right, valid_right)):
            stretched = preview_stretch(image, mask)
            photo = ImageTk.PhotoImage(Image.fromarray(stretched).resize((520, 410), Image.Resampling.NEAREST))
            self.images.append(photo); canvas.delete("all"); canvas.create_image(0, 0, anchor="nw", image=photo)
            canvas.create_text(12, 12, anchor="nw", fill="cyan", text=f"B{band} — review only; display stretch")
            coords = []
            for x, y in geometry.exterior.coords:
                pixel_x, pixel_y = (~self.render_transform) @ (x, y)
                coords.extend(((pixel_x - win.col_off) * 520 / size,
                               (pixel_y - win.row_off) * 410 / size))
            if len(coords) >= 6:
                canvas.create_line(*coords, fill="#00ffff", width=2, joinstyle="round")

    def _render_multiband(self):
        if not self.keys:
            return
        row = self.store.get(*self.keys[self.index])
        geometry = shape(row["geometry"])
        composite = self.composite_var.get()
        view = self.display_var.get()
        mode = self.stretch_var.get().lower()
        if view == "DIFFERENCE":
            bands = (int(self.band.get()),)
        elif composite == "GRAY":
            bands = (int(self.band.get()),)
        else:
            bands = COMPOSITES[composite]
        wavelength_tag = ", ".join(f"B{b}:{self.band_centers_nm.get(b, 'UNKNOWN')}nm" for b in bands)
        reference_path = self.paths_i.get(9)
        if not reference_path:
            self._show_render_error("B9 reference input is missing")
            return
        size = self.zoom_px
        try:
            with rasterio.open(reference_path) as ref:
                if ref.crs is None or ref.crs.to_string() != "EPSG:32650":
                    raise ValueError("B9 reference must be EPSG:32650")
                col, row_px = (~ref.transform) @ (geometry.centroid.x, geometry.centroid.y)
                window = Window(round(col - size / 2), round(row_px - size / 2), size, size)
                self.render_window, self.render_transform = window, ref.transform
                height, width = int(window.height), int(window.width)
                scene_arrays: list[dict[int, np.ndarray]] = [{}, {}]
                scene_masks: list[dict[int, np.ndarray]] = [{}, {}]
                for band in sorted(set(bands) | {9}):
                    if band not in self.paths_i or band not in self.paths_j:
                        raise ValueError(f"B{band} is unavailable in one scene")
                    with rasterio.open(self.paths_i[band]) as src_i:
                        if (src_i.crs != ref.crs or src_i.transform != ref.transform or
                                src_i.width != ref.width or src_i.height != ref.height):
                            raise ValueError(f"scene-i B{band} is not on the verified in-scene B9 grid")
                        scene_arrays[0][band], scene_masks[0][band] = read_window_padded(src_i, window)
                    with rasterio.open(self.paths_j[band]) as src_j:
                        ref_j = self.paths_j[9]
                        with rasterio.open(ref_j) as ref_j_ds:
                            if (src_j.crs != ref_j_ds.crs or src_j.transform != ref_j_ds.transform or
                                    src_j.width != ref_j_ds.width or src_j.height != ref_j_ds.height):
                                raise ValueError(f"scene-j B{band} is not on its verified in-scene B9 grid")
                        with WarpedVRT(src_j, crs=ref.crs, transform=ref.transform,
                                       width=ref.width, height=ref.height,
                                       resampling=Resampling.bilinear) as vrt:
                            scene_arrays[1][band], scene_masks[1][band] = read_window_padded(vrt, window)
            if any(a.shape != (height, width) for scene in scene_arrays for a in scene.values()):
                raise ValueError("window-to-array shape mismatch")
            self._layout_for_view(view)
            self._stop_blink()
            if view == "DIFFERENCE":
                diff, valid = difference_preview(scene_arrays[0][bands[0]], scene_arrays[1][bands[0]],
                    scene_masks[0][bands[0]], scene_masks[1][bands[0]])
                images = [Image.fromarray(signed_difference_rgb(diff, valid))]
                self.render_mode_description = (f"scene_i-scene_j raw DN difference B{bands[0]}; "
                    "scene_j warped to scene_i EPSG:32650 affine window using bilinear resampling; common valid mask")
                self._draw_multiband(self.canvas_i, images[0], "Signed raw-DN difference; blue<0, white=0, red>0",
                                     geometry, window, ref.transform)
            else:
                display_bands = bands if len(bands) == 3 else (bands[0],) * 3
                channel_images = []
                for channel_band in display_bands:
                    left_range, right_range = stretch_pair(
                        scene_arrays[0][channel_band], scene_arrays[1][channel_band],
                        scene_masks[0][channel_band], scene_masks[1][channel_band], mode=mode)
                    channel_images.append((left_range, right_range))
                images = []
                for scene_idx in (0, 1):
                    if len(bands) == 3:
                        ranges = [channel_images[i][scene_idx] for i in range(3)]
                        rgb = render_composite(scene_arrays[scene_idx], scene_masks[scene_idx], bands, ranges)
                    else:
                        low, high = channel_images[0][scene_idx]
                        data = scene_arrays[scene_idx][bands[0]]
                        valid = scene_masks[scene_idx][bands[0]] & np.isfinite(data)
                        gray = np.round(np.clip((data-low)/(high-low), 0, 1)*255).astype(np.uint8)
                        gray[~valid] = 0
                        rgb = np.repeat(gray[:, :, None], 3, axis=2)
                    images.append(Image.fromarray(rgb))
                tag = ("B" + "/B".join(map(str, bands))) if len(bands) == 3 else f"B{bands[0]}"
                self.render_mode_description = (f"{composite if view != 'DIFFERENCE' else 'GRAY'} {tag}; "
                    f"{mode} percentile stretch (preview only); scene_j→scene_i affine mapping, bilinear display resampling")
                if view == "BLINK":
                    self._blink_photos = []
                    self._blink_images = images
                    self._blink_index = 0
                    self._draw_multiband(self.canvas_i, images[0], f"BLINK: scene_i | {composite} {tag}", geometry, window, ref.transform)
                    self._schedule_blink(geometry, window, ref.transform)
                else:
                    self._draw_multiband(self.canvas_i, images[0], f"scene_i | {composite} {tag}", geometry, window, ref.transform)
                    self._draw_multiband(self.canvas_j, images[1], f"scene_j | {composite} {tag}", geometry, window, ref.transform)
            stats_band = int(self.band.get())
            stat_parts = []
            for band in sorted({9, stats_band}):
                for scene_name, paths in (("i", self.paths_i), ("j", self.paths_j)):
                    stat_parts.append(f"{scene_name}.B{band}={self._read_roi_stats(paths.get(band), geometry)}")
            self.summary_var.set(self.render_mode_description + "; bands=" + wavelength_tag + " | " + " | ".join(stat_parts) +
                                 " | Raw values; units UNKNOWN; display warp does not validate geometry.")
        except Exception as exc:
            self._show_render_error(f"Multiband preview unavailable: {type(exc).__name__}: {exc}")

    def _read_roi_stats(self, path: Path | None, geometry) -> str:
        if path is None:
            return "NOT_AVAILABLE"
        try:
            with rasterio.open(path) as ds:
                stats = read_raster_roi_stats(ds, geometry)
        except Exception as exc:
            return f"RASTER_READ_FAILED:{type(exc).__name__}"
        if stats["median"] is None:
            return str(stats.get("status", "NO_VALID_PIXELS"))
        return (f"valid={stats['valid_fraction']:.3f},median={stats['median']:.3f},"
                f"MAD={stats['mad']:.3f},std={stats['std']:.3f}")

    def _draw_multiband(self, canvas, image: Image.Image, title: str, geometry, window, transform):
        photo = ImageTk.PhotoImage(image.resize((520, 410), Image.Resampling.NEAREST))
        if not hasattr(self, "_multiband_photo_refs"):
            self._multiband_photo_refs = {}
        self._multiband_photo_refs[canvas] = photo
        canvas.delete("all"); canvas.create_image(0, 0, anchor="nw", image=photo)
        canvas.create_text(12, 12, anchor="nw", fill="cyan", text=title)
        coords = []
        for x, y in geometry.exterior.coords:
            px, py = (~transform) @ (x, y)
            coords.extend(((px-window.col_off)*520/window.width, (py-window.row_off)*410/window.height))
        if len(coords) >= 6:
            canvas.create_line(*coords, fill="#00ffff", width=2, joinstyle="round")

    def _layout_for_view(self, view: str):
        self.canvas_i.pack_forget(); self.canvas_j.pack_forget()
        if view == "SIDE_BY_SIDE":
            self.canvas_i.pack(side="left", fill="both", expand=True, padx=(0, 3))
            self.canvas_j.pack(side="left", fill="both", expand=True, padx=(3, 0))
        else:
            self.canvas_i.pack(fill="both", expand=True)

    def _schedule_blink(self, geometry, window, transform):
        if self._blink_job is not None:
            self.root.after_cancel(self._blink_job)
        def tick():
            self._blink_index = 1 - self._blink_index
            scene = "scene_i" if self._blink_index == 0 else "scene_j"
            self._draw_multiband(self.canvas_i, self._blink_images[self._blink_index], scene, geometry, window, transform)
            if self.display_var.get() == "BLINK":
                self._blink_job = self.root.after(600, tick)
        self._blink_job = self.root.after(600, tick)

    def _stop_blink(self):
        if self._blink_job is not None:
            try:
                self.root.after_cancel(self._blink_job)
            except tk.TclError:
                pass
            self._blink_job = None

    def _show_render_error(self, message: str):
        self.canvas_i.delete("all"); self.canvas_j.delete("all")
        self.canvas_i.create_text(10, 10, anchor="nw", fill="white", text=message, width=520)
        self.summary_var.set(message)

    def _move(self, delta):
        visible = self._visible_keys()
        if not visible:
            return
        self._capture_multiband_draft()
        current = self.keys[self.index]
        pos = visible.index(current) if current in visible else 0
        self.index = self.keys.index(visible[(pos + delta) % len(visible)])
        self._refresh()

    def _change_zoom(self, scale):
        self.zoom_px = max(64, min(1024, int(self.zoom_px * scale)))
        self._render()

    def _support(self, geometry):
        from benchmark_v1_radiometric_r1.roi_gui import RoiPairReader
        reader = RoiPairReader(self.paths_i, self.paths_j)
        try:
            result = {}
            for band in sorted(reader.bands):
                left, right = reader.fractions(band, geometry)
                result[f"B{band}"] = {"scene_i_valid_fraction": left, "scene_j_valid_fraction": right,
                                       "common_lower_bound": min(left, right)}
            return result
        finally:
            reader.close()

    def _apply_geometry(self):
        if not self.keys:
            return
        try:
            geometry = wkt.loads(self.wkt_var.get())
            feature_rows = [self.store.get(*key) for key in self.keys]
            feature_rows.extend(f for f in self.audit_features
                                if f.get("properties", {}).get("pair_id") != self.pair_id)
            updated = self.store.edit_geometry(*self.keys[self.index], geometry,
                support_validator=self._support, all_candidate_features=feature_rows, editor=self.reviewer,
                all_pairs_available=self.all_pairs_available, preserve_frozen_role=self.multiband_mode)
            self._multiband_drafts.pop(self.keys[self.index], None)
            self._populate(self.keys[self.index]); self._refresh()
            messagebox.showinfo("Geometry updated", f"Frozen role={updated.get('frozen_role', updated['role'])}; "
                f"spatial edit eligibility={updated.get('spatial_edit_eligibility', 'recomputed')}; scoring eligibility cleared.", parent=self.root)
        except Exception as exc:
            messagebox.showerror("Geometry not applied", str(exc), parent=self.root)

    def _apply_review(self):
        if not self.keys:
            return
        key = self.keys[self.index]
        if self.status_var.get() == "APPROVED":
            if not messagebox.askyesno("Human visual review", "I inspected both image windows. The automatic score is not accuracy. Continue?", parent=self.root):
                return
        try:
            updated = self.store.record_review(*key, reviewer=self.reviewer, status=self.status_var.get(),
                label=self.label_var.get(), evidence=self.evidence_var.get(), second_review=self.second_review)
            self._refresh()
            self.suggestion_var.set(f"Saved human review: {updated.get('review_status')} / second={updated.get('second_review_status')} / dispute={updated.get('dispute_status')}")
        except Exception as exc:
            messagebox.showerror("Review not saved", str(exc), parent=self.root)

    def _apply_multiband_review(self):
        if not self.keys or not self.multiband_mode:
            return
        key = self.keys[self.index]
        decision = self.multiband_decision_var.get()
        if decision == "APPROVED" and not messagebox.askyesno(
                "Human-only label approval",
                "You reviewed both dates and all selected bands. NO_CHANGE is not formal STABLE truth; "
                "a radiometric difference alone cannot confirm change. Save this visual review?", parent=self.root):
            return
        values = {name: variable.get() for name, variable in self.multiband_label_vars.items()}
        try:
            updated = self.store.record_labels(*key, reviewer=self.reviewer, decision=decision,
                evidence=self.multiband_evidence_var.get(), second_review=self.second_review, **values)
            self._multiband_drafts.pop(key, None)
            self._refresh()
            self.suggestion_var.set(f"New schema v2 saved: {decision}; formal stable truth eligible="
                                    f"{updated.get('formal_stable_truth_eligible', False)}; test eligible="
                                    f"{updated.get('eligible_for_test', False)}")
        except Exception as exc:
            messagebox.showerror("New labels not saved", str(exc), parent=self.root)


def launch_auto_roi_reviewer(root: tk.Tk, pair_id: str, candidate_path: Path,
                             paths_i: dict[int, Path], paths_j: dict[int, Path],
                             snapshot_path: Path, events_path: Path, reviewer: str,
                             *, second_review: bool = False):
    from .inputs import ALLOWED_PAIRS
    from .storage import load_candidate_collection
    siblings = {}
    for other_pair in ALLOWED_PAIRS:
        sibling = candidate_path.parent / f"{other_pair}.geojson"
        if sibling.is_file():
            siblings[other_pair] = load_candidate_collection(sibling, other_pair)["features"]
    all_pairs_available = set(siblings) == set(ALLOWED_PAIRS)
    audit_features = [feature for rows in siblings.values() for feature in rows]
    pairwise_audit = audit_candidate_sets([rows for rows in siblings.values()]) if all_pairs_available else None
    store = AutoReviewStore(candidate_path, snapshot_path, events_path,
                            pairwise_audit=pairwise_audit, all_pairs_available=all_pairs_available)
    gui = AutoRoiReviewer(root, pair_id, store, paths_i, paths_j, reviewer,
                          second_review=second_review, audit_features=audit_features,
                          all_pairs_available=all_pairs_available)
    root.protocol("WM_DELETE_WINDOW", root.destroy)
    return gui


def launch_multiband_roi_reviewer(root: tk.Tk, pair_id: str, candidate_path: Path,
                                  paths_i: dict[int, Path], paths_j: dict[int, Path],
                                  review_root: Path, legacy_review_root: Path, reviewer: str,
                                  band_centers_nm: dict[int, float | None],
                                  *, second_review: bool = False,
                                  machine_suggestions: dict[tuple[str, str], dict[str, Any]] | None = None):
    """Open schema-v2 review state in a separate directory; legacy files are read-only."""
    from .inputs import ALLOWED_PAIRS
    from .multiband_review import MultiBandReviewStore
    from .storage import load_candidate_collection

    siblings = {}
    for other_pair in ALLOWED_PAIRS:
        sibling = candidate_path.parent / f"{other_pair}.geojson"
        if sibling.is_file():
            siblings[other_pair] = load_candidate_collection(sibling, other_pair)["features"]
    all_pairs_available = set(siblings) == set(ALLOWED_PAIRS)
    audit_features = [feature for rows in siblings.values() for feature in rows]
    pairwise_audit = audit_candidate_sets([rows for rows in siblings.values()]) if all_pairs_available else None
    store = MultiBandReviewStore(candidate_path, review_root / f"{pair_id}_snapshot_v2.json",
        review_root / f"{pair_id}_events_v2.jsonl", pairwise_audit=pairwise_audit,
        all_pairs_available=all_pairs_available,
        legacy_snapshot_path=legacy_review_root / f"{pair_id}_review.json",
        legacy_events_path=legacy_review_root / f"{pair_id}_review_events.jsonl")
    gui = AutoRoiReviewer(root, pair_id, store, paths_i, paths_j, reviewer,
        second_review=second_review, audit_features=audit_features,
        all_pairs_available=all_pairs_available, multiband_mode=True,
        band_centers_nm=band_centers_nm, machine_suggestions=machine_suggestions)
    root.protocol("WM_DELETE_WINDOW", lambda: (gui._stop_blink(), root.destroy()))
    return gui
