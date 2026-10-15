from __future__ import annotations

import math
from typing import Any

import numpy as np
import rasterio
from rasterio.features import geometry_mask
from rasterio.windows import Window, from_bounds
from rasterio.warp import transform_geom
from shapely.geometry import mapping, shape


CELL_M = 10_000.0
BUFFER_M = 500.0


def to_projected(geometry, source_crs: str):
    geom = shape(geometry) if isinstance(geometry, dict) else geometry
    if not geom.is_valid or geom.is_empty:
        raise ValueError("ROI geometry must be non-empty and valid")
    normalized = source_crs.upper().replace(" ", "")
    if normalized in {"EPSG:32650", "URN:OGC:DEF:CRS:EPSG::32650"}:
        return geom
    projected = transform_geom(source_crs, "EPSG:32650", mapping(geom))
    return shape(projected)


def classify_roi_role(geometry, geometry_crs: str = "EPSG:32650") -> tuple[str, str, str]:
    """Apply the frozen Phase3 origin/cell/role formula to a whole ROI."""
    projected = to_projected(geometry, geometry_crs)
    minx, miny, maxx, maxy = projected.bounds
    ix = math.floor(minx / CELL_M)
    iy = math.floor(miny / CELL_M)
    cell_id = f"g{ix}_{iy}"
    safe = (
        math.floor(maxx / CELL_M) == ix
        and math.floor(maxy / CELL_M) == iy
        and minx - ix * CELL_M > BUFFER_M
        and miny - iy * CELL_M > BUFFER_M
        and (ix + 1) * CELL_M - maxx > BUFFER_M
        and (iy + 1) * CELL_M - maxy > BUFFER_M
    )
    if not safe:
        return "UNASSIGNED", cell_id, "CROSSES_CELL_OR_500M_BUFFER"
    role = "DEVELOPMENT" if (ix + 2 * iy) % 3 == 0 else "TEST"
    return role, cell_id, "SAFE_SINGLE_CELL"


def replace_roi_geometry(record: dict[str, Any], geometry) -> dict[str, Any]:
    """Replace geometry and invalidate prior human decisions; ReviewStore retains the old snapshot in its event log."""
    projected = to_projected(geometry, "EPSG:32650")
    role, cell_id, status = classify_roi_role(projected, "EPSG:32650")
    updated = dict(record)
    updated.update({
        "geometry": projected.wkt, "geometry_crs": "EPSG:32650", "role": role,
        "grid_cell_id": cell_id, "spatial_status": status, "label": "UNKNOWN",
        "review_status": "UNVERIFIED", "reviewer": "", "review_time_utc": "",
        "evidence_notes": "", "purpose": "UNASSIGNED", "second_reviewer": "",
        "second_review_status": "UNVERIFIED", "second_review_time_utc": "",
        "second_review_label": "UNKNOWN", "second_review_notes": "",
        "spatial_overlap_audit": "PENDING_PAIRWISE_AUDIT",
    })
    return updated


def build_roi_candidates(region_collections: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge candidate polygons while forcing every human label/status to unverified."""
    features: list[dict[str, Any]] = []
    seen: set[str] = set()
    for collection in region_collections:
        for source_feature in collection.get("features", []):
            geometry = shape(source_feature["geometry"])
            props = dict(source_feature.get("properties", {}))
            source_id = str(props.get("region_id") or props.get("pair_id") or len(features))
            roi_id = f"R1_{source_id}"
            if roi_id in seen:
                raise ValueError(f"duplicate ROI candidate ID: {roi_id}")
            seen.add(roi_id)
            source_crs = str(props.get("geometry_crs") or props.get("source_crs") or "EPSG:4326")
            try:
                role, cell_id, spatial_status = classify_roi_role(geometry, source_crs)
            except (ValueError, Exception) as exc:
                role, cell_id, spatial_status = "UNASSIGNED", "", f"INVALID_GEOMETRY:{type(exc).__name__}"
            geojson_geometry = geometry if source_crs.upper().replace(" ", "") == "EPSG:4326" else shape(
                transform_geom(source_crs, "EPSG:4326", mapping(geometry))
            )
            props.update({
                "roi_id": roi_id,
                "source_region_id": source_id,
                "geometry_crs": "EPSG:4326",
                "label": "UNKNOWN",
                "review_status": "UNVERIFIED",
                "reviewer": "",
                "review_time_utc": "",
                "second_reviewer": "",
                "second_review_status": "UNVERIFIED",
                "second_review_time_utc": "",
                "spatial_role": role,
                "grid_cell_id": cell_id,
                "spatial_status": spatial_status,
                "purpose": "UNASSIGNED",
                "calibration_or_evaluation": "UNASSIGNED",
                "spatial_overlap_audit": "PENDING_PAIRWISE_AUDIT",
            })
            features.append({"type": "Feature", "geometry": mapping(geojson_geometry), "properties": props})
    return {"type": "FeatureCollection", "name": "radiometric_roi_candidates_unverified", "crs": {"type": "name", "properties": {"name": "EPSG:4326"}}, "features": features}


def valid_fraction_for_geometry(dataset: rasterio.io.DatasetReader, geometry) -> float:
    """Window-read the mask and calculate valid ROI-center fraction for one raster."""
    geom = shape(geometry) if isinstance(geometry, dict) else geometry
    if dataset.crs is None:
        raise ValueError("dataset CRS is required")
    if not geom.is_valid or geom.is_empty:
        raise ValueError("ROI geometry must be non-empty and valid")
    from shapely.geometry import box
    bounds = dataset.bounds
    if not geom.intersects(box(bounds.left, bounds.bottom, bounds.right, bounds.top)):
        return 0.0
    roi_geom = geom
    # Callers pass geometry in the dataset CRS; ROI GUI normalizes its display geometry first.
    raw = from_bounds(*roi_geom.bounds, transform=dataset.transform)
    col_start = math.floor(raw.col_off)
    row_start = math.floor(raw.row_off)
    col_stop = math.ceil(raw.col_off + raw.width)
    row_stop = math.ceil(raw.row_off + raw.height)
    out_width, out_height = col_stop - col_start, row_stop - row_start
    if out_width <= 0 or out_height <= 0:
        return 0.0
    if out_width * out_height > 4_000_000:
        raise ValueError("ROI support exceeds bounded 4M-pixel validation window")
    full_window = Window(col_start, row_start, out_width, out_height)
    inside = geometry_mask([mapping(roi_geom)], out_shape=(out_height, out_width), transform=dataset.window_transform(full_window), invert=True, all_touched=False)
    support = int(np.count_nonzero(inside))
    if not support:
        return 0.0
    col0, row0 = max(0, col_start), max(0, row_start)
    col1, row1 = min(dataset.width, col_stop), min(dataset.height, row_stop)
    valid_in_full = np.zeros((out_height, out_width), dtype=bool)
    if col1 > col0 and row1 > row0:
        source_window = Window(col0, row0, col1 - col0, row1 - row0)
        target_rows = slice(row0 - row_start, row1 - row_start)
        target_cols = slice(col0 - col_start, col1 - col_start)
        valid_in_full[target_rows, target_cols] = dataset.read_masks(1, window=source_window) > 0
    return float(np.count_nonzero(inside & valid_in_full) / support)

