from __future__ import annotations

from dataclasses import dataclass
from math import ceil, floor
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window, bounds as window_bounds, from_bounds
from scipy.ndimage import binary_dilation, sobel
from shapely.geometry import box, mapping

from .inputs import PairInputs


@dataclass(frozen=True)
class ProposalConfig:
    window_px: int = 32
    grid_rows: int = 8
    grid_cols: int = 8
    search_subdivisions: int = 4
    valid_fraction_min: float = 0.95
    strong_edge_fraction_max: float = 0.10
    edge_buffer_px: int = 2
    max_candidates: int = 30
    texture_min: float = 1e-6
    structural_corr_stable_min: float = 0.90
    spectral_shape_corr_stable_min: float = 0.95
    stable_max_days: float = 0.01
    geometry_verified: bool = False
    band_coreg_verified: bool = False
    auxiliary_bands: tuple[str, ...] = ("B1", "B9", "B16")


def window_quality(valid_i: np.ndarray, valid_j: np.ndarray, edge_i: np.ndarray,
                   edge_j: np.ndarray, *, valid_min: float = 0.95,
                   edge_max: float = 0.10, edge_buffer_px: int = 2) -> dict[str, Any]:
    if not (valid_i.shape == valid_j.shape == edge_i.shape == edge_j.shape):
        raise ValueError("validity and edge masks must have identical shapes")
    if valid_i.size == 0:
        raise ValueError("empty ROI window")
    joint = valid_i.astype(bool) & valid_j.astype(bool)
    valid_fraction = float(np.count_nonzero(joint) / joint.size)
    edges = edge_i.astype(bool) | edge_j.astype(bool)
    if edge_buffer_px > 0 and np.any(edges):
        edges = binary_dilation(edges, iterations=edge_buffer_px)
    edge_fraction = float(np.count_nonzero(edges) / edges.size)
    if valid_fraction < valid_min:
        reason = f"B9_VALID_SUPPORT_BELOW_{valid_min:.2f}"
    elif edge_fraction > edge_max:
        reason = f"STRONG_EDGE_FRACTION_ABOVE_{edge_max:.2f}"
    else:
        reason = ""
    return {"valid_fraction": valid_fraction, "edge_fraction": edge_fraction,
            "accepted": not reason, "exclusion_reason": reason}


def recommend_label(*, days_apart: float | None, gradient_correlation: float | None,
                    spectral_shape_correlation: float | None, geometry_verified: bool,
                    band_coreg_verified: bool, texture_strength: float,
                    brightness_similarity: float | None = None,
                    cloud_evidence: bool = False, shadow_evidence: bool = False,
                    coherent_change_evidence: bool = False) -> dict[str, str]:
    label, confidence, reason = "UNKNOWN", "LOW", "INSUFFICIENT_INDEPENDENT_EVIDENCE"
    if coherent_change_evidence and (not geometry_verified or not band_coreg_verified):
        reason = "GEOMETRY_OR_COREG_UNVERIFIED"
    elif coherent_change_evidence:
        label, confidence, reason = "REAL_CHANGE", "MEDIUM", "COHERENT_STRUCTURAL_AND_SPECTRAL_SHAPE_CUE"
    elif cloud_evidence and band_coreg_verified:
        label, confidence, reason = "CLOUD", "LOW", "CONSERVATIVE_MULTIBAND_CUE_REQUIRES_HUMAN_CHECK"
    elif shadow_evidence and band_coreg_verified:
        label, confidence, reason = "SHADOW", "LOW", "CONTEXTUAL_MULTIBAND_CUE_REQUIRES_HUMAN_CHECK"
    elif (days_apart is not None and days_apart <= 0.01 and geometry_verified and band_coreg_verified
          and texture_strength > 1e-6 and gradient_correlation is not None
          and spectral_shape_correlation is not None and gradient_correlation >= 0.90
          and spectral_shape_correlation >= 0.95):
        label, confidence, reason = "STABLE", "MEDIUM", "SAME_DATE_STRUCTURE_AND_MULTISPECTRAL_SHAPE_CUE"
    elif brightness_similarity is not None and brightness_similarity > 0.95:
        reason = "BRIGHTNESS_SIMILARITY_ALONE_IS_NOT_STABILITY_EVIDENCE"
    elif not geometry_verified or not band_coreg_verified:
        reason = "GEOMETRY_OR_COREG_UNVERIFIED"
    elif texture_strength <= 1e-6:
        reason = "LOW_TEXTURE"
    return {"suggested_label": label, "suggestion_confidence": confidence,
            "uncertainty_reason": reason}


def select_spatially(candidates: list[dict[str, Any]], maximum: int) -> list[dict[str, Any]]:
    """Select a deterministic score-ranked item per spatial cell before taking extras."""
    by_cell: dict[str, list[dict[str, Any]]] = {}
    for row in candidates:
        by_cell.setdefault(str(row["grid_cell_id"]), []).append(row)
    for rows in by_cell.values():
        rows.sort(key=lambda item: (-float(item.get("score", 0.0)), str(item["candidate_id"])))
    selected: list[dict[str, Any]] = []
    depth = 0
    cells = sorted(by_cell)
    while len(selected) < maximum:
        added = False
        for cell in cells:
            if depth < len(by_cell[cell]):
                selected.append(by_cell[cell][depth])
                added = True
                if len(selected) == maximum:
                    break
        if not added:
            break
        depth += 1
    return selected


def _edge_strength(image: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, float]:
    values = image.astype(np.float32)
    if np.any(valid):
        fill = float(np.median(values[valid]))
        values = np.where(valid, values, fill)
    gx, gy = sobel(values, axis=1), sobel(values, axis=0)
    magnitude = np.hypot(gx, gy)
    sample = magnitude[valid]
    if sample.size == 0:
        return np.zeros(image.shape, dtype=bool), 0.0
    med = float(np.median(sample))
    mad = float(np.median(np.abs(sample - med)))
    threshold = med + max(4.0 * mad, float(np.quantile(sample, 0.95)) * 0.25, 1e-6)
    edge = (magnitude > threshold) & valid
    p90 = float(np.quantile(sample, 0.90))
    texture = float(np.std(sample))
    return edge, max(p90, texture)


def _corr(a: np.ndarray, b: np.ndarray) -> float | None:
    aa, bb = np.asarray(a, dtype=np.float64).ravel(), np.asarray(b, dtype=np.float64).ravel()
    if aa.size < 4 or np.std(aa) <= 1e-9 or np.std(bb) <= 1e-9:
        return None
    value = float(np.corrcoef(aa, bb)[0, 1])
    return value if np.isfinite(value) else None


def _window_band_support(inputs: PairInputs, window: list[int]) -> dict[str, Any]:
    row, col, height, width = window
    win = Window(col, row, width, height)
    result = {}
    for band in inputs.scenes[0].bands:
        left, right = inputs.scenes[0].bands[band], inputs.scenes[1].bands[band]
        if not left.usable or not right.usable or not left.path or not right.path:
            result[band] = {"available": False, "reason": left.exclusion_reason or right.exclusion_reason or "MISSING_AUXILIARY"}
            continue
        try:
            with rasterio.open(left.path) as ds_i, rasterio.open(right.path) as ds_j:
                with WarpedVRT(ds_j, crs=ds_i.crs, transform=ds_i.transform,
                               width=ds_i.width, height=ds_i.height,
                               resampling=Resampling.nearest) as aligned_j:
                    values_i = ds_i.read(1, window=win, masked=True).astype(np.float64).filled(np.nan)
                    values_j = aligned_j.read(1, window=win, masked=True).astype(np.float64).filled(np.nan)
                    valid_i = ds_i.read_masks(1, window=win) > 0
                    valid_j = aligned_j.read_masks(1, window=win) > 0
                    valid_i &= np.isfinite(values_i)
                    valid_j &= np.isfinite(values_j)
                    # Window clipping can return a smaller array at raster edges; no padding is
                    # interpreted as valid support.
                    denominator = height * width
                    left_fraction = float(valid_i.sum() / denominator)
                    right_fraction = float(valid_j.sum() / denominator)
                    joint_fraction = 0.0
                    if valid_i.shape == valid_j.shape:
                        joint_fraction = float((valid_i & valid_j).sum() / denominator)
                    ref = inputs.scenes[0].bands["B9"].header_fingerprint
                    header = left.header_fingerprint
                    same_grid = (header.get("crs") == ref.get("crs") and
                                 header.get("width") == ref.get("width") and
                                 header.get("height") == ref.get("height") and
                                 header.get("transform") == ref.get("transform"))
                    median_i = float(np.median(values_i[valid_i])) if np.any(valid_i) else None
                    median_j = float(np.median(values_j[valid_j])) if np.any(valid_j) else None
                    mad_i = float(np.median(np.abs(values_i[valid_i] - median_i))) if median_i is not None else None
                    mad_j = float(np.median(np.abs(values_j[valid_j] - median_j))) if median_j is not None else None
                    result[band] = {"available": True, "scene_i_valid_fraction": left_fraction,
                                    "scene_j_valid_fraction": right_fraction,
                                    "joint_valid_fraction": joint_fraction,
                                    "scene_i_pixel_median_diagnostic": median_i,
                                    "scene_j_pixel_median_diagnostic": median_j,
                                    "scene_i_pixel_mad_diagnostic": mad_i,
                                    "scene_j_pixel_mad_diagnostic": mad_j,
                                    "unit": left.data_unit if left.data_unit == right.data_unit else "UNKNOWN",
                                    "wavelength_center_nm_i": left.wavelength_center_nm,
                                    "wavelength_center_nm_j": right.wavelength_center_nm,
                                    "scene_i_grid_matches_b9": same_grid,
                                    "cross_scene_coregistered": False}
        except Exception as exc:
            result[band] = {"available": False, "reason": f"MASK_READ_ERROR:{type(exc).__name__}"}
    common = [band for band, item in result.items() if item.get("available") and
              item.get("scene_i_pixel_median_diagnostic") is not None and
              item.get("scene_j_pixel_median_diagnostic") is not None]
    if len(common) >= 2:
        vector_i = np.asarray([result[b]["scene_i_pixel_median_diagnostic"] for b in common], dtype=float)
        vector_j = np.asarray([result[b]["scene_j_pixel_median_diagnostic"] for b in common], dtype=float)
        denominator = float(np.linalg.norm(vector_i) * np.linalg.norm(vector_j))
        similarity = float(np.dot(vector_i, vector_j) / denominator) if denominator > 1e-12 else None
    else:
        similarity = None
    result["_cross_scene_relative_spectral_shape_similarity_diagnostic"] = similarity
    result["_diagnostic_only"] = True
    return result


def _grid_windows(ds: rasterio.io.DatasetReader, other_bounds: tuple[float, ...], cfg: ProposalConfig):
    left = max(ds.bounds.left, other_bounds[0])
    bottom = max(ds.bounds.bottom, other_bounds[1])
    right = min(ds.bounds.right, other_bounds[2])
    top = min(ds.bounds.top, other_bounds[3])
    if right <= left or top <= bottom:
        return
    raw = from_bounds(left, bottom, right, top, ds.transform)
    c0, r0 = max(0, floor(raw.col_off)), max(0, floor(raw.row_off))
    c1, r1 = min(ds.width, ceil(raw.col_off + raw.width)), min(ds.height, ceil(raw.row_off + raw.height))
    w, h = cfg.window_px, cfg.window_px
    if c1 - c0 < w or r1 - r0 < h:
        return
    seen = set()
    for gy in range(cfg.grid_rows):
        for gx in range(cfg.grid_cols):
            cell_c0 = c0 + gx * (c1 - c0) / cfg.grid_cols
            cell_c1 = c0 + (gx + 1) * (c1 - c0) / cfg.grid_cols
            cell_r0 = r0 + gy * (r1 - r0) / cfg.grid_rows
            cell_r1 = r0 + (gy + 1) * (r1 - r0) / cfg.grid_rows
            for sy in range(cfg.search_subdivisions):
                for sx in range(cfg.search_subdivisions):
                    cx = cell_c0 + (sx + 0.5) * (cell_c1 - cell_c0) / cfg.search_subdivisions
                    cy = cell_r0 + (sy + 0.5) * (cell_r1 - cell_r0) / cfg.search_subdivisions
                    col, row = int(round(cx - w / 2)), int(round(cy - h / 2))
                    col = min(max(col, c0), c1 - w)
                    row = min(max(row, r0), r1 - h)
                    key = (row, col)
                    if key in seen:
                        continue
                    seen.add(key)
                    yield gy, gx, sy, sx, Window(col, row, w, h)


def generate_pair_proposals(inputs: PairInputs, config: ProposalConfig | None = None,
                            diagnostics: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Read only fixed-size candidate windows on the native B9-i grid."""
    cfg = config or ProposalConfig()
    a, b = inputs.scenes
    pa, pb = a.bands["B9"].path, b.bands["B9"].path
    if pa is None or pb is None:
        raise ValueError("B9 is mandatory for proposal generation")
    records: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    total_windows = 0
    with rasterio.open(pa) as ref, rasterio.open(pb) as source:
        if ref.crs is None or source.crs is None:
            raise ValueError("both B9 inputs require CRS")
        with WarpedVRT(source, crs=ref.crs, transform=ref.transform, width=ref.width,
                       height=ref.height, resampling=Resampling.bilinear) as aligned:
            for gy, gx, sy, sx, win in _grid_windows(ref, tuple(source.bounds), cfg) or ():
                total_windows += 1
                image_i = ref.read(1, window=win, masked=True).filled(0)
                image_j = aligned.read(1, window=win, masked=True).filled(0)
                valid_i = (ref.read_masks(1, window=win) > 0) & np.isfinite(image_i)
                valid_j = (aligned.read_masks(1, window=win) > 0) & np.isfinite(image_j)
                edge_i, texture_i = _edge_strength(image_i, valid_i)
                edge_j, texture_j = _edge_strength(image_j, valid_j)
                quality = window_quality(valid_i, valid_j, edge_i, edge_j,
                                         valid_min=cfg.valid_fraction_min,
                                         edge_max=cfg.strong_edge_fraction_max,
                                         edge_buffer_px=cfg.edge_buffer_px)
                texture = min(texture_i, texture_j)
                reason = quality["exclusion_reason"]
                if not reason and texture <= cfg.texture_min:
                    reason = "LOW_TEXTURE"
                grid_id = f"r{gy:02d}_c{gx:02d}"
                if reason:
                    rejected.append({"grid_cell_id": grid_id, "search_subcell_id": f"s{sy:02d}_c{sx:02d}", "reason": reason,
                                     "valid_fraction": quality["valid_fraction"],
                                     "edge_fraction": quality["edge_fraction"],
                                     "texture_strength": texture})
                    continue
                grad_corr = _corr(edge_i.astype(float), edge_j.astype(float))
                # The current evidence does not independently verify cross-scene geometry or
                # cross-band co-registration, so semantic change/stability is deliberately suppressed.
                label = recommend_label(days_apart=inputs.days_apart,
                                        gradient_correlation=grad_corr,
                                        spectral_shape_correlation=None,
                                        geometry_verified=cfg.geometry_verified,
                                        band_coreg_verified=cfg.band_coreg_verified,
                                        texture_strength=texture)
                roi = box(*window_bounds(win, ref.transform))
                support = float(quality["valid_fraction"])
                score = support * min(texture / (texture + 100.0), 1.0) * (1.0 - quality["edge_fraction"])
                records.append({
                    "candidate_id": f"{inputs.pair_id}_R15_{gy:02d}_{gx:02d}_s{sy:02d}{sx:02d}",
                    "pair_id": inputs.pair_id, "scene_i": a.scene_id, "scene_j": b.scene_id,
                    "geometry": mapping(roi), "geometry_crs": "EPSG:32650",
                    "pixel_window_i": [int(win.row_off), int(win.col_off), cfg.window_px, cfg.window_px],
                    "valid_support_fraction": support,
                    "strong_edge_fraction": quality["edge_fraction"],
                    "texture_strength": texture, "grid_cell_id": grid_id,
                    "search_subcell_id": f"s{sy:02d}_c{sx:02d}",
                    "score": float(score), "suggested_label": label["suggested_label"],
                    "suggested_landcover_type": "UNKNOWN",
                    "suggestion_confidence": label["suggestion_confidence"],
                    "suggestion_evidence": "BOUNDED_B9_TEXTURE_AND_MASK_SCREEN; NO_ALGORITHM_OUTPUTS",
                    "uncertainty_reason": label["uncertainty_reason"],
                    "review_status": "UNVERIFIED", "data_unit": "UNKNOWN",
                    "resampling": {"scene_j_values": "BILINEAR", "scene_j_mask": "NEAREST"},
                    "eligible_for_test": False,
                })
    selected = select_spatially(records, cfg.max_candidates)
    accepted_pool = [{key: row[key] for key in ("candidate_id", "grid_cell_id", "search_subcell_id",
                                                "valid_support_fraction", "strong_edge_fraction", "texture_strength")}
                     for row in records]
    for candidate in selected:
        candidate["support_by_scene_band"] = _window_band_support(inputs, candidate["pixel_window_i"])
        candidate["cross_scene_geometry_verified"] = False
        candidate["cross_band_coreg_verified"] = False
        candidate["available_common_bands"] = [band for band, details in candidate["support_by_scene_band"].items()
                                                if isinstance(details, dict) and details.get("available")]
        candidate["relative_spectral_shape_similarity_diagnostic"] = candidate["support_by_scene_band"].get(
            "_cross_scene_relative_spectral_shape_similarity_diagnostic")
        candidate["multiband_features_diagnostic_only"] = True
        candidate["input_data_units"] = {band: details.get("unit", "UNKNOWN")
                                         for band, details in candidate["support_by_scene_band"].items()
                                         if isinstance(details, dict)}
    if diagnostics is not None:
        diagnostics.update({"windows_examined": total_windows, "quality_accepted_before_cap": len(records),
                            "selected_count": len(selected), "rejected_windows": rejected,
                            "accepted_windows": accepted_pool,
                            "selected_candidate_ids": [r["candidate_id"] for r in selected]})
    return selected
