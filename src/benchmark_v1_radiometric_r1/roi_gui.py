from __future__ import annotations

import math
import tkinter as tk
from datetime import datetime, timezone
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Any

import numpy as np
import rasterio
from PIL import Image, ImageTk
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window, from_bounds
from shapely import wkt
from shapely.geometry import Polygon, box, mapping, shape
from shapely.ops import transform as geometry_transform
from rasterio.warp import transform_geom

from .roi import classify_roi_role, replace_roi_geometry, to_projected, valid_fraction_for_geometry
from .roi_store import LABELS, PURPOSES, REVIEW_STATES, ReviewStore, is_formal_test_eligible


CANVAS_SIZE = 512
MIN_WINDOW = 128
MAX_WINDOW = 1024


def canvas_to_world(x: float, y: float, window: Window, transform, *, canvas_size: int = CANVAS_SIZE) -> tuple[float, float]:
    col = float(window.col_off) + float(x) * float(window.width) / canvas_size
    row = float(window.row_off) + float(y) * float(window.height) / canvas_size
    world_x, world_y = transform @ (col, row)
    return float(world_x), float(world_y)


def world_to_canvas(x: float, y: float, window: Window, transform, *, canvas_size: int = CANVAS_SIZE) -> tuple[float, float]:
    col, row = (~transform) @ (x, y)
    return (
        (float(col) - float(window.col_off)) * canvas_size / float(window.width),
        (float(row) - float(window.row_off)) * canvas_size / float(window.height),
    )


def preview_stretch(data: np.ndarray, valid: np.ndarray, low_pct: float = 2, high_pct: float = 98) -> np.ndarray:
    """Create display-only grayscale pixels; the input raster samples are never changed."""
    source = np.asarray(data)
    mask = np.asarray(valid, dtype=bool) & np.isfinite(source)
    display = np.zeros(source.shape, dtype=np.uint8)
    if not np.any(mask):
        return display
    low, high = np.percentile(source[mask].astype(np.float64), (low_pct, high_pct))
    if high <= low:
        high = low + 1.0
    display[mask] = np.clip((source[mask].astype(np.float64) - low) * (255.0 / (high - low)), 0, 255).astype(np.uint8)
    return display


def candidate_list_entries(candidates: list[dict[str, Any]], saved_records: dict[str, Any], pair_id: str) -> list[tuple[str, str]]:
    """Show a saved annotation instead of a duplicate read-only proposal with the same ID."""
    entries = []
    for feature in candidates:
        props = feature.get("properties", {})
        roi_id = str(props.get("roi_id", ""))
        if props.get("pair_id") == pair_id and roi_id and roi_id not in saved_records:
            entries.append(("candidate", roi_id))
    entries.extend(("saved", roi_id) for roi_id, row in sorted(saved_records.items()) if row.get("pair_id") == pair_id)
    return entries


def _read_padded(dataset, band: int, window: Window, *, fill_value=0, read_mask: bool = False) -> np.ndarray:
    """Read only the in-bounds window and pad the requested viewport without VRT boundless reads."""
    window = window.round_offsets().round_lengths()
    width, height = int(window.width), int(window.height)
    out_shape = (height, width)
    result = np.zeros(out_shape, dtype=np.uint8) if read_mask else np.full(out_shape, fill_value, dtype=dataset.dtypes[band - 1])
    left, top = int(window.col_off), int(window.row_off)
    right, bottom = left + width, top + height
    src_left, src_top = max(0, left), max(0, top)
    src_right, src_bottom = min(dataset.width, right), min(dataset.height, bottom)
    if src_left >= src_right or src_top >= src_bottom:
        return result
    clipped = Window(src_left, src_top, src_right - src_left, src_bottom - src_top)
    data = dataset.read_masks(band, window=clipped) if read_mask else dataset.read(band, window=clipped)
    dst_left, dst_top = src_left - left, src_top - top
    result[dst_top:dst_top + data.shape[0], dst_left:dst_left + data.shape[1]] = data
    return result


class RoiPairReader:
    """On-demand paired raster reader; scene-j is warped into scene-i's georeferenced grid."""

    def __init__(self, paths_i: dict[int, Path], paths_j: dict[int, Path]):
        if not paths_i or not paths_j:
            raise ValueError("both scenes need at least one available V band")
        self.paths_i = {int(k): Path(v) for k, v in paths_i.items()}
        self.paths_j = {int(k): Path(v) for k, v in paths_j.items()}
        self.bands = sorted(set(self.paths_i) & set(self.paths_j))
        if not self.bands:
            raise ValueError("the image pair has no common V band")

    def read_window(self, band: int, window: Window) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if band not in self.paths_i or band not in self.paths_j:
            raise ValueError(f"band B{band} is not available in both scenes")
        with rasterio.open(self.paths_i[band]) as src_i, rasterio.open(self.paths_j[band]) as src_j:
            if src_i.crs is None or src_j.crs is None:
                raise ValueError("both rasters require a CRS")
            with WarpedVRT(
                src_j,
                crs=src_i.crs,
                transform=src_i.transform,
                width=src_i.width,
                height=src_i.height,
                resampling=Resampling.bilinear,
                nodata=src_j.nodata,
            ) as aligned_j:
                with WarpedVRT(
                    src_j,
                    crs=src_i.crs,
                    transform=src_i.transform,
                    width=src_i.width,
                    height=src_i.height,
                    resampling=Resampling.nearest,
                    nodata=src_j.nodata,
                ) as mask_j:
                    a = _read_padded(src_i, 1, window, fill_value=src_i.nodata if src_i.nodata is not None else 0)
                    b = _read_padded(aligned_j, 1, window, fill_value=src_j.nodata if src_j.nodata is not None else 0)
                    ma = _read_padded(src_i, 1, window, read_mask=True) > 0
                    mb = _read_padded(mask_j, 1, window, read_mask=True) > 0
            return a, b, ma & np.isfinite(a), mb & np.isfinite(b)

    def fractions(self, band: int, geometry_projected) -> tuple[float, float]:
        with rasterio.open(self.paths_i[band]) as src_i, rasterio.open(self.paths_j[band]) as src_j:
            with WarpedVRT(src_j, crs=src_i.crs, transform=src_i.transform, width=src_i.width, height=src_i.height, resampling=Resampling.nearest, nodata=src_j.nodata) as aligned_j:
                return valid_fraction_for_geometry(src_i, geometry_projected), valid_fraction_for_geometry(aligned_j, geometry_projected)

    def reference_grid(self, band: int):
        with rasterio.open(self.paths_i[band]) as src:
            return src.transform, src.crs, src.width, src.height

    def close(self) -> None:
        return None


def _project_wgs84_geom(geometry):
    return to_projected(geometry, "EPSG:4326")


def _feature_geometries(geometry):
    if geometry.geom_type == "Polygon":
        yield list(geometry.exterior.coords)
        for interior in geometry.interiors:
            yield list(interior.coords)
    elif geometry.geom_type == "MultiPolygon":
        for polygon in geometry.geoms:
            yield from _feature_geometries(polygon)


class RoiAnnotator:
    def __init__(self, root, pair_id: str, scene_i: str, scene_j: str, reader: RoiPairReader,
                 candidates: list[dict[str, Any]], store: ReviewStore, reviewer: str, second_review: bool = False):
        self.root, self.pair_id, self.scene_i, self.scene_j = root, pair_id, scene_i, scene_j
        self.reader, self.candidates, self.store = reader, candidates, store
        self.reviewer, self.second_review = reviewer.strip(), second_review
        self.band = 9 if 9 in reader.bands else reader.bands[0]
        self.transform, self.crs, self.raster_width, self.raster_height = reader.reference_grid(self.band)
        self.window = Window(max(0, self.raster_width // 2 - 256), max(0, self.raster_height // 2 - 256), 512, 512)
        self.current_geometry = None
        self.current_record: dict[str, Any] | None = None
        self.current_photo_i = self.current_photo_j = None
        self.draw_mode = ""
        self.editing_roi_id: str | None = None
        self.vertices: list[tuple[float, float]] = []
        self.rectangle_start = None
        self.band_var = tk.IntVar(value=self.band)
        self.label_var = tk.StringVar(value="UNKNOWN")
        self.second_label_var = tk.StringVar(value="UNKNOWN")
        self.review_var = tk.StringVar(value="UNVERIFIED")
        self.second_state_var = tk.StringVar(value="UNVERIFIED")
        self.purpose_var = tk.StringVar(value="UNASSIGNED")
        self.notes_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="Select a candidate or draw a new ROI.")
        self.valid_var = tk.StringVar(value="Valid support: not calculated")
        self._build_ui()
        self._populate_list()
        self._refresh_counts()
        self.render()

    def _build_ui(self):
        self.root.title(f"DZ01V Radiometric ROI review — {self.pair_id}")
        self.root.geometry("1260x900")
        toolbar = ttk.Frame(self.root, padding=5); toolbar.pack(fill="x")
        ttk.Label(toolbar, text=f"Pair {self.pair_id}  Reviewer: {self.reviewer}  Mode: {'SECOND REVIEW' if self.second_review else 'FIRST REVIEW'}").pack(side="left")
        ttk.Label(toolbar, text="Band").pack(side="left", padx=(20, 3))
        ttk.Combobox(toolbar, textvariable=self.band_var, values=self.reader.bands, width=5, state="readonly").pack(side="left")
        self.band_var.trace_add("write", self._band_changed)
        ttk.Button(toolbar, text="Zoom +", command=lambda: self._zoom(0.5)).pack(side="left", padx=3)
        ttk.Button(toolbar, text="Zoom −", command=lambda: self._zoom(2.0)).pack(side="left", padx=3)
        for label, dx, dy in (("←", -1, 0), ("→", 1, 0), ("↑", 0, -1), ("↓", 0, 1)):
            ttk.Button(toolbar, text=label, width=3, command=lambda x=dx, y=dy: self._pan(x, y)).pack(side="left", padx=1)

        main = ttk.Frame(self.root); main.pack(fill="both", expand=True, padx=6)
        image_frame = ttk.Frame(main); image_frame.pack(side="left", fill="both", expand=True)
        canvases = ttk.Frame(image_frame); canvases.pack()
        self.canvas_i = tk.Canvas(canvases, width=CANVAS_SIZE, height=CANVAS_SIZE, background="black", highlightthickness=1)
        self.canvas_j = tk.Canvas(canvases, width=CANVAS_SIZE, height=CANVAS_SIZE, background="black", highlightthickness=1)
        self.canvas_i.grid(row=0, column=0, padx=3); self.canvas_j.grid(row=0, column=1, padx=3)
        ttk.Label(canvases, text=self.scene_i).grid(row=1, column=0, sticky="w")
        ttk.Label(canvases, text=self.scene_j).grid(row=1, column=1, sticky="w")
        self.canvas_i.bind("<Button-1>", self._canvas_click); self.canvas_j.bind("<Button-1>", self._canvas_click)
        self.canvas_i.bind("<ButtonRelease-1>", self._rectangle_release); self.canvas_j.bind("<ButtonRelease-1>", self._rectangle_release)
        self.canvas_i.bind("<ButtonPress-1>", self._rectangle_press, add="+"); self.canvas_j.bind("<ButtonPress-1>", self._rectangle_press, add="+")

        side = ttk.LabelFrame(main, text="Candidate / saved ROI", padding=5); side.pack(side="right", fill="y", padx=5)
        self.listbox = tk.Listbox(side, width=39, height=24); self.listbox.pack(fill="y", expand=True)
        self.listbox.bind("<<ListboxSelect>>", self._select_list_item)
        ttk.Button(side, text="New polygon", command=self._start_polygon).pack(fill="x", pady=2)
        ttk.Button(side, text="Finish polygon", command=self._finish_polygon).pack(fill="x", pady=2)
        ttk.Button(side, text="New rectangle (drag)", command=self._start_rectangle).pack(fill="x", pady=2)
        ttk.Button(side, text="Delete saved ROI", command=self._delete_selected).pack(fill="x", pady=2)
        ttk.Button(side, text="Edit selected geometry", command=self._edit_selected_geometry).pack(fill="x", pady=2)

        fields = ttk.LabelFrame(self.root, text="Review metadata", padding=5); fields.pack(fill="x", padx=8, pady=5)
        label_display_var = self.second_label_var if self.second_review else self.label_var
        self.label_combo = ttk.Combobox(fields, textvariable=label_display_var, values=sorted(LABELS), state="readonly", width=15)
        self.label_field_label = ttk.Label(fields, text="Independent label" if self.second_review else "Land cover label")
        self.label_field_label.grid(row=0, column=0, sticky="w"); self.label_combo.grid(row=0, column=1, sticky="w", padx=4)
        ttk.Label(fields, text="Purpose").grid(row=0, column=2, sticky="w")
        self.purpose_combo = ttk.Combobox(fields, textvariable=self.purpose_var, values=sorted(PURPOSES), state="readonly", width=15)
        self.purpose_combo.grid(row=0, column=3, sticky="w", padx=4)
        ttk.Label(fields, text="First review").grid(row=1, column=0, sticky="w")
        self.review_combo = ttk.Combobox(fields, textvariable=self.review_var, values=sorted(REVIEW_STATES), state="readonly", width=15)
        self.review_combo.grid(row=1, column=1, sticky="w", padx=4)
        ttk.Label(fields, text="TEST second review").grid(row=1, column=2, sticky="w")
        self.second_combo = ttk.Combobox(fields, textvariable=self.second_state_var, values=sorted(REVIEW_STATES), state="readonly", width=15)
        self.second_combo.grid(row=1, column=3, sticky="w", padx=4)
        ttk.Label(fields, text="Evidence / notes").grid(row=2, column=0, sticky="w")
        self.notes_entry = ttk.Entry(fields, textvariable=self.notes_var, width=105); self.notes_entry.grid(row=2, column=1, columnspan=3, sticky="ew", padx=4)
        self.apply_button = ttk.Button(fields, text="Apply human review", command=self._apply_review); self.apply_button.grid(row=0, column=4, rowspan=2, padx=12)
        ttk.Button(fields, text="Save ROI", command=self._save_roi).grid(row=2, column=4, padx=12)
        ttk.Label(fields, textvariable=self.valid_var).grid(row=3, column=0, columnspan=5, sticky="w", pady=3)
        ttk.Label(fields, textvariable=self.status_var).grid(row=4, column=0, columnspan=5, sticky="w")
        if self.second_review:
            self.review_combo.configure(state="disabled"); self.purpose_combo.configure(state="disabled")
            self.review_var.set("UNVERIFIED")
        else:
            self.second_combo.configure(state="disabled")

    def _populate_list(self):
        self.list_items = candidate_list_entries(self.candidates, self.store.records, self.pair_id)
        self.listbox.delete(0, "end")
        features = {str(f.get("properties", {}).get("roi_id", "")): f for f in self.candidates}
        for kind, roi_id in self.list_items:
            if kind == "candidate":
                props = features[roi_id]["properties"]
                text = f"{roi_id}  {props.get('label','UNKNOWN')}  UNVERIFIED candidate"
            else:
                record = self.store.records[roi_id]
                text = f"{roi_id}  {record.get('label')}  {record.get('review_status')}"
            self.listbox.insert("end", text)

    def _select_list_item(self, _event=None):
        selected = self.listbox.curselection()
        if not selected:
            return
        kind, roi_id = self.list_items[selected[0]]
        if kind == "candidate":
            feature = next(f for f in self.candidates if f["properties"].get("roi_id") == roi_id)
            geom_wgs = shape(feature["geometry"])
            geom_projected = _project_wgs84_geom(geom_wgs)
            props = feature["properties"]
            role, cell, spatial_status = classify_roi_role(geom_projected, "EPSG:32650")
            self.label_var.set("UNKNOWN"); self.review_var.set("UNVERIFIED")
            self.second_state_var.set("UNVERIFIED"); self.purpose_var.set("UNASSIGNED"); self.notes_var.set("")
            self.current_record = {
                "roi_id": roi_id, "pair_id": self.pair_id, "scene_i": self.scene_i, "scene_j": self.scene_j,
                "geometry": geom_projected.wkt, "geometry_crs": "EPSG:32650", "days_apart": props.get("days_apart", ""),
                "label": "UNKNOWN", "review_status": "UNVERIFIED", "reviewer": "", "review_time_utc": "",
                "evidence_notes": "", "role": role, "grid_cell_id": cell, "spatial_status": spatial_status,
                "purpose": "UNASSIGNED", "second_reviewer": "", "second_review_status": "UNVERIFIED",
                "second_review_time_utc": "", "second_review_label": "UNKNOWN", "second_review_notes": "",
                "candidate_source": props.get("source_region_id", roi_id),
                "spatial_overlap_audit": props.get("spatial_overlap_audit", "PENDING_PAIRWISE_AUDIT"),
                "display_band": self.band,
                "source_sha256_i": props.get("source_sha256_i", ""), "source_sha256_j": props.get("source_sha256_j", ""),
                "candidate_geometry_source": props.get("candidate_geometry_source", "EXISTING_UNVERIFIED_REGION"),
                "source_crs_resolution": props.get("source_crs_resolution", ""),
                "quality_i": props.get("quality_i", "UNKNOWN"), "quality_j": props.get("quality_j", "UNKNOWN"),
                "temporal_bin": props.get("temporal_bin", "UNKNOWN"),
            }
        else:
            self.current_record = dict(self.store.records[roi_id])
            geom_projected = wkt.loads(self.current_record["geometry"])
            if self.second_review:
                self.label_var.set("UNKNOWN"); self.review_var.set("UNVERIFIED")
                self.second_label_var.set(self.current_record.get("second_review_label", "UNKNOWN"))
                self.second_state_var.set(self.current_record["second_review_status"])
                self.notes_var.set(self.current_record.get("second_review_notes", ""))
            else:
                self.label_var.set(self.current_record["label"]); self.review_var.set(self.current_record["review_status"])
                self.second_state_var.set(self.current_record["second_review_status"]); self.purpose_var.set(self.current_record["purpose"])
                self.notes_var.set(self.current_record["evidence_notes"])
        self.current_geometry = geom_projected
        self._center_on_geometry(geom_projected)
        self._refresh_valid_fraction()
        self.render()

    def _center_on_geometry(self, geometry):
        bounds_window = from_bounds(*geometry.bounds, transform=self.transform).round_offsets().round_lengths()
        size = min(MAX_WINDOW, max(MIN_WINDOW, int(max(bounds_window.width, bounds_window.height) * 1.5)))
        center_col = bounds_window.col_off + bounds_window.width / 2
        center_row = bounds_window.row_off + bounds_window.height / 2
        self.window = Window(int(center_col - size / 2), int(center_row - size / 2), size, size)

    def _band_changed(self, *_):
        self.band = int(self.band_var.get())
        self.transform, self.crs, self.raster_width, self.raster_height = self.reader.reference_grid(self.band)
        if self.current_geometry is not None:
            self._center_on_geometry(self.current_geometry)
        self.render()
        self._refresh_valid_fraction()

    def _pan(self, dx: int, dy: int):
        shift = max(1, int(self.window.width / 4))
        self.window = Window(int(self.window.col_off + dx * shift), int(self.window.row_off + dy * shift), int(self.window.width), int(self.window.height))
        self.render()

    def _zoom(self, factor: float):
        size = min(MAX_WINDOW, max(MIN_WINDOW, int(self.window.width * factor)))
        cx = self.window.col_off + self.window.width / 2; cy = self.window.row_off + self.window.height / 2
        self.window = Window(int(cx - size / 2), int(cy - size / 2), size, size)
        self.render()

    def render(self):
        band = int(self.band_var.get())
        a, b, ma, mb = self.reader.read_window(band, self.window)
        display_a, display_b = preview_stretch(a, ma), preview_stretch(b, mb)
        self.current_photo_i = ImageTk.PhotoImage(Image.fromarray(display_a).resize((CANVAS_SIZE, CANVAS_SIZE)))
        self.current_photo_j = ImageTk.PhotoImage(Image.fromarray(display_b).resize((CANVAS_SIZE, CANVAS_SIZE)))
        for canvas, photo in ((self.canvas_i, self.current_photo_i), (self.canvas_j, self.current_photo_j)):
            canvas.delete("all"); canvas.create_image(0, 0, image=photo, anchor="nw")
        if self.current_geometry is not None:
            for canvas in (self.canvas_i, self.canvas_j):
                for coords in _feature_geometries(self.current_geometry):
                    points = [world_to_canvas(x, y, self.window, self.transform) for x, y in coords]
                    flat = [v for point in points for v in point]
                    canvas.create_line(*flat, fill="#00ffff", width=2)
        if self.vertices:
            pts = [world_to_canvas(x, y, self.window, self.transform) for x, y in self.vertices]
            for canvas in (self.canvas_i, self.canvas_j):
                for x, y in pts:
                    canvas.create_oval(x-3, y-3, x+3, y+3, outline="#ffcc00", width=2)
                if len(pts) > 1:
                    canvas.create_line(*[v for point in pts for v in point], fill="#ffcc00", width=2)

    def _start_polygon(self):
        self.draw_mode = "polygon"; self.vertices = []; self.status_var.set("Click polygon vertices on either panel; Finish polygon closes it.")

    def _start_rectangle(self):
        self.draw_mode = "rectangle"; self.rectangle_start = None; self.status_var.set("Drag on either panel to create a shared map-coordinate rectangle.")

    def _canvas_click(self, event):
        if self.draw_mode == "polygon":
            self.vertices.append(canvas_to_world(event.x, event.y, self.window, self.transform))
            self.render()

    def _rectangle_press(self, event):
        if self.draw_mode == "rectangle":
            self.rectangle_start = canvas_to_world(event.x, event.y, self.window, self.transform)

    def _rectangle_release(self, event):
        if self.draw_mode != "rectangle" or self.rectangle_start is None:
            return
        end = canvas_to_world(event.x, event.y, self.window, self.transform)
        geom = box(min(self.rectangle_start[0], end[0]), min(self.rectangle_start[1], end[1]), max(self.rectangle_start[0], end[0]), max(self.rectangle_start[1], end[1]))
        if geom.area <= 0:
            self.status_var.set("Rectangle is empty; draw a larger one."); return
        self._set_new_geometry(geom)
        self.draw_mode = ""; self.rectangle_start = None

    def _finish_polygon(self):
        if len(self.vertices) < 3:
            self.status_var.set("At least three vertices are required."); return
        geom = Polygon(self.vertices)
        if not geom.is_valid or geom.area <= 0:
            messagebox.showerror("Invalid ROI", "The polygon is invalid or has zero area.", parent=self.root); return
        self._set_new_geometry(geom)
        self.vertices = []; self.draw_mode = ""

    def _set_new_geometry(self, geometry):
        if self.editing_roi_id and self.editing_roi_id in self.store.records:
            roi_id = self.editing_roi_id
            self.current_record = replace_roi_geometry(self.store.records[roi_id], geometry)
            self.current_record["candidate_geometry_source"] = "MANUAL_GEOMETRY_EDIT"
            self.current_record["display_band"] = self.band
            self.editing_roi_id = None
            self.label_var.set("UNKNOWN"); self.review_var.set("UNVERIFIED")
            self.second_label_var.set("UNKNOWN"); self.second_state_var.set("UNVERIFIED")
            self.purpose_var.set("UNASSIGNED"); self.notes_var.set("")
            self.current_geometry = geometry
            self._center_on_geometry(geometry); self._refresh_valid_fraction(); self.render()
            self.status_var.set(f"Geometry replaced for {roi_id}; prior approvals invalidated and retained in audit history. Save ROI to commit the edit.")
            return
        role, cell, spatial_status = classify_roi_role(geometry, "EPSG:32650")
        roi_id = f"{self.pair_id}_ROI_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
        self.current_geometry = geometry
        self.current_record = {
            "roi_id": roi_id, "pair_id": self.pair_id, "scene_i": self.scene_i, "scene_j": self.scene_j,
            "geometry": geometry.wkt, "geometry_crs": "EPSG:32650", "label": "UNKNOWN", "review_status": "UNVERIFIED",
            "reviewer": "", "review_time_utc": "", "evidence_notes": "", "role": role,
            "grid_cell_id": cell, "spatial_status": spatial_status, "purpose": "UNASSIGNED",
            "second_reviewer": "", "second_review_status": "UNVERIFIED", "second_review_time_utc": "",
            "candidate_source": "MANUAL_DRAW", "spatial_overlap_audit": "PENDING_PAIRWISE_AUDIT", "display_band": self.band,
        }
        self.label_var.set("UNKNOWN"); self.review_var.set("UNVERIFIED"); self.second_state_var.set("UNVERIFIED")
        self.purpose_var.set("UNASSIGNED"); self.notes_var.set("")
        self._center_on_geometry(geometry); self._refresh_valid_fraction(); self.render()

    def _edit_selected_geometry(self):
        if self.current_record is None or self.current_record["roi_id"] not in self.store.records:
            messagebox.showinfo("Select saved ROI", "Select an already saved ROI before replacing its geometry.", parent=self.root)
            return
        self.editing_roi_id = self.current_record["roi_id"]
        self.status_var.set("Choose New polygon or New rectangle; replacing geometry will reset review approvals and preserve prior state in the audit log.")

    def _refresh_valid_fraction(self):
        if self.current_geometry is None:
            return
        try:
            fi, fj = self.reader.fractions(self.band, self.current_geometry)
            self.current_record["valid_fraction_i"] = f"{fi:.8f}"; self.current_record["valid_fraction_j"] = f"{fj:.8f}"
            self.valid_var.set(f"Valid fraction in scene i/j: {fi:.4%} / {fj:.4%}; role={self.current_record['role']} {self.current_record['grid_cell_id']} ({self.current_record['spatial_status']})")
        except Exception as exc:
            self.valid_var.set(f"Valid support unavailable: {exc}")

    def _apply_review(self):
        if self.current_record is None:
            messagebox.showinfo("Select ROI", "Select or draw an ROI first.", parent=self.root); return
        row = dict(self.current_record)
        now = datetime.now(timezone.utc).isoformat()
        if self.second_review:
            if row["role"] != "TEST":
                messagebox.showerror("Not a TEST ROI", "Second review applies only to frozen TEST-role ROIs.", parent=self.root); return
            if not row.get("reviewer") or row.get("review_status") != "APPROVED":
                messagebox.showerror("First review required", "A first human approval is required before second review.", parent=self.root); return
            if self.reviewer.casefold() == row["reviewer"].strip().casefold():
                messagebox.showerror("Independent reviewer required", "The second reviewer must be a different real person.", parent=self.root); return
            state = self.second_state_var.get()
            second_label = self.second_label_var.get()
            if state == "APPROVED" and (second_label == "UNKNOWN" or second_label != row["label"]):
                messagebox.showerror("Review disagreement", "Independent labels differ. Mark DOUBTFUL and record the discrepancy; do not approve.", parent=self.root); return
            if state == "APPROVED" and not self.notes_var.get().strip():
                messagebox.showerror("Evidence required", "Record independent visual evidence before approval.", parent=self.root); return
            if state == "APPROVED" and not messagebox.askyesno("Independent ROI review", "I independently inspected this ROI on both dates and confirm the recorded class and test role. Continue?", parent=self.root):
                return
            row["second_reviewer"] = self.reviewer; row["second_review_status"] = state; row["second_review_time_utc"] = now
            row["second_review_label"] = second_label; row["second_review_notes"] = self.notes_var.get()
            event = "SECOND_REVIEW_" + state
        else:
            state = self.review_var.get(); label = self.label_var.get()
            if state == "APPROVED":
                if label == "UNKNOWN" or not self.reviewer:
                    messagebox.showerror("Review required", "Choose a determinate visual label and provide your own reviewer identity before approval.", parent=self.root); return
                if not self.notes_var.get().strip():
                    messagebox.showerror("Evidence required", "Record visual evidence in the notes field before approval.", parent=self.root); return
                if not messagebox.askyesno("Human-only ROI approval", "I visually reviewed the relevant dates/bands and support this label from image evidence, not pixel similarity alone. Continue?", parent=self.root):
                    return
                row["reviewer"] = self.reviewer; row["review_time_utc"] = now
            elif self.reviewer:
                row["reviewer"] = self.reviewer; row["review_time_utc"] = now
            row["label"] = label; row["review_status"] = state; row["purpose"] = self.purpose_var.get()
            row["evidence_notes"] = self.notes_var.get(); event = "FIRST_REVIEW_" + state
        try:
            self.store.upsert(row, actor=self.reviewer, event_type=event); self.store.save()
            self.current_record = row
            self._populate_list(); self._refresh_counts()
            self.status_var.set(f"Saved {event}; formal TEST eligibility={is_formal_test_eligible(row)}")
        except ValueError as exc:
            messagebox.showerror("Invalid review record", str(exc), parent=self.root)

    def _save_roi(self):
        if self.current_record is None:
            messagebox.showinfo("Select ROI", "Select or draw an ROI first.", parent=self.root); return
        row = dict(self.current_record)
        row["display_band"] = str(self.band)
        self.store.upsert(row, actor=self.reviewer, event_type="SAVE")
        self.store.save(); self.current_record = row
        self._populate_list(); self._refresh_counts()
        self.status_var.set(f"Saved {row['roi_id']} (human status {row['review_status']}; TEST eligible={is_formal_test_eligible(row)}).")

    def _delete_selected(self):
        if self.current_record is None or self.current_record["roi_id"] not in self.store.records:
            messagebox.showinfo("Nothing to delete", "Only saved annotations can be deleted; candidate source files are read-only.", parent=self.root); return
        roi_id = self.current_record["roi_id"]
        if not messagebox.askyesno("Delete annotation", f"Delete saved ROI {roi_id}? An audit event is retained.", parent=self.root):
            return
        self.store.delete(roi_id, actor=self.reviewer); self.store.save()
        self.current_record = None; self.current_geometry = None; self._populate_list(); self.render()

    def _refresh_counts(self):
        rows = list(self.store.records.values())
        first_approved = sum(r.get("review_status") == "APPROVED" for r in rows)
        stable = sum(r.get("review_status") == "APPROVED" and r.get("label") == "STABLE" for r in rows)
        changed = sum(r.get("review_status") == "APPROVED" and r.get("label") == "REAL_CHANGE" for r in rows)
        eligible = sum(is_formal_test_eligible(r) for r in rows)
        test_rows = [r for r in rows if r.get("role") == "TEST" and r.get("purpose") == "EVALUATION"]
        pending_second = sum(r.get("review_status") == "APPROVED" and r.get("second_review_status") == "UNVERIFIED" for r in test_rows)
        second_approved = sum(r.get("second_review_status") == "APPROVED" for r in test_rows)
        doubtful = sum(r.get("review_status") == "DOUBTFUL" or r.get("second_review_status") == "DOUBTFUL" for r in rows)
        rejected = sum(r.get("review_status") == "REJECTED" or r.get("second_review_status") == "REJECTED" for r in rows)
        self.status_var.set(f"Saved={len(rows)} first-approved={first_approved} stable={stable} real-change={changed} TEST-pending-second={pending_second} second-approved={second_approved} doubtful={doubtful} rejected={rejected} formal-TEST-eligible={eligible}")


def launch_roi_annotator(root, pair_id: str, scene_i: str, scene_j: str, paths_i: dict[int, Path], paths_j: dict[int, Path],
                         candidate_collection: dict[str, Any], output_csv: Path, events_jsonl: Path, reviewer: str,
                         second_review: bool = False):
    candidates = candidate_collection.get("features", [])
    store = ReviewStore(output_csv, events_jsonl)
    reader = RoiPairReader(paths_i, paths_j)
    gui = RoiAnnotator(root, pair_id, scene_i, scene_j, reader, candidates, store, reviewer, second_review)
    root.protocol("WM_DELETE_WINDOW", lambda: (reader.close(), root.destroy()))
    return gui

