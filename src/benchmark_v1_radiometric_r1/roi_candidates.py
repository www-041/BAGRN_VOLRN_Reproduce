from __future__ import annotations

from typing import Any

from shapely.geometry import box, mapping
from shapely.geometry import shape

from .roi import build_roi_candidates


def _scene_map(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    ids = [str(row["scene_id"]) for row in rows]
    if len(set(ids)) != len(rows):
        raise ValueError("duplicate scene_id in scene inventory")
    result = {}
    for row in rows:
        result[str(row["scene_id"])] = row
        if row.get("product_id"):
            result[str(row["product_id"])] = row
    return result


def _bbox_candidate(pair: dict[str, Any], scenes: dict[str, dict[str, Any]]) -> dict[str, Any]:
    try:
        left_scene, right_scene = scenes[str(pair["scene_i"])], scenes[str(pair["scene_j"])]
        left = max(float(left_scene["b9_bounds_left"]), float(right_scene["b9_bounds_left"]))
        bottom = max(float(left_scene["b9_bounds_bottom"]), float(right_scene["b9_bounds_bottom"]))
        right = min(float(left_scene["b9_bounds_right"]), float(right_scene["b9_bounds_right"]))
        top = min(float(left_scene["b9_bounds_top"]), float(right_scene["b9_bounds_top"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"cannot derive candidate bounds for {pair.get('pair_id')}: {exc}") from exc
    width, height = right - left, top - bottom
    if width <= 0 or height <= 0:
        raise ValueError(f"selected pair {pair.get('pair_id')} has no bounding-box overlap")
    side = min(7168.0, width, height)
    cx, cy = (left + right) / 2, (bottom + top) / 2
    geometry = box(cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2)
    return {
        "type": "Feature", "geometry": mapping(geometry),
        "properties": {
            "pair_id": pair["pair_id"], "region_id": f"R1_BBOX_{pair['pair_id']}",
            "source_crs": "EPSG:32650", "candidate_geometry_source": "BBOX_INTERSECTION_UNVERIFIED",
            "selection_note": "Bounding-box-only proposal; raster valid masks and human suitability are not established.",
        },
    }


def _resolve_region_crs(feature: dict[str, Any], pair: dict[str, Any], scenes: dict[str, dict[str, Any]]) -> tuple[str, str]:
    props = feature.get("properties", {})
    declared = str(props.get("geometry_crs") or props.get("source_crs") or "EPSG:4326")
    normalized = declared.upper().replace(" ", "")
    geometry = shape(feature["geometry"])
    minx, miny, maxx, maxy = geometry.bounds
    if normalized in {"EPSG:32650", "URN:OGC:DEF:CRS:EPSG::32650"} and -180 <= minx <= maxx <= 180 and -90 <= miny <= maxy <= 90:
        try:
            from shapely.geometry import box as geographic_box
            bounds = []
            for scene_key in (str(pair["scene_i"]), str(pair["scene_j"])):
                row = scenes[scene_key]
                bounds.append(geographic_box(float(row["b9_lon_min"]), float(row["b9_lat_min"]), float(row["b9_lon_max"]), float(row["b9_lat_max"])))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"CRS conflict for {pair['pair_id']} cannot be resolved without both scene geographic extents") from exc
        if all(geometry.intersects(scene_bounds) for scene_bounds in bounds):
            return "EPSG:4326", "EPSG:4326_CONFIRMED_BY_SCENE_EXTENTS"
        raise ValueError(f"CRS conflict for {pair['pair_id']}: coordinates look geographic but do not overlap both scene extents")
    return declared, "DECLARED_SOURCE_CRS"


def build_r1_roi_candidates(
    selected_pairs: list[dict[str, Any]],
    region_collections: list[dict[str, Any]],
    extended_collections: list[dict[str, Any]],
    scene_rows: list[dict[str, Any]],
    source_hashes: dict[str, str],
) -> dict[str, Any]:
    """Build unapproved proposals for the frozen pre-method shortlist only."""
    scenes = _scene_map(scene_rows)
    available: dict[str, dict[str, Any]] = {}
    for collection in (*region_collections, *extended_collections):
        for feature in collection.get("features", []):
            pair_id = str(feature.get("properties", {}).get("pair_id", ""))
            if pair_id and pair_id not in available:
                available[pair_id] = feature

    raw_features = []
    pair_by_id = {}
    for pair in selected_pairs:
        pair_id = str(pair["pair_id"])
        try:
            valid_pixels = int(float(pair.get("valid_overlap_pixels", 0)))
        except (TypeError, ValueError):
            valid_pixels = 0
        if valid_pixels <= 0:
            raise ValueError(f"selected pair {pair_id} has no effective valid-mask overlap")
        if not scenes_by_pair(pair, scenes):
            raise ValueError(f"selected pair {pair_id} references missing scene metadata")
        feature = available.get(pair_id) or _bbox_candidate(pair, scenes)
        properties = dict(feature.get("properties", {}))
        input_crs, crs_resolution = _resolve_region_crs(feature, pair, scenes) if pair_id in available else ("EPSG:32650", "BBOX_FROM_SCENE_METRICS")
        properties.update({
            "pair_id": pair_id, "region_id": str(properties.get("region_id") or f"R1_{pair_id}"),
            "scene_i": pair.get("scene_i", ""), "scene_j": pair.get("scene_j", ""),
            "days_apart": pair.get("days_apart", ""), "quality_i": pair.get("quality_i", "UNKNOWN"),
            "quality_j": pair.get("quality_j", "UNKNOWN"), "temporal_bin": pair.get("temporal_bin", "UNKNOWN"),
            "source_sha256_i": source_hashes.get(str(pair.get("scene_i", "")), ""),
            "source_sha256_j": source_hashes.get(str(pair.get("scene_j", "")), ""),
            "geometry_crs": input_crs, "source_crs_resolution": crs_resolution,
        })
        feature = {"type": "Feature", "geometry": feature["geometry"], "properties": properties}
        raw_features.append(feature)
        pair_by_id[pair_id] = pair

    collection = build_roi_candidates([{"type": "FeatureCollection", "features": raw_features}])
    for feature in collection["features"]:
        props = feature["properties"]
        pair_id = props["pair_id"]
        pair = pair_by_id[pair_id]
        props.update({
            "scene_i": pair.get("scene_i", ""), "scene_j": pair.get("scene_j", ""),
            "days_apart": pair.get("days_apart", ""), "quality_i": pair.get("quality_i", "UNKNOWN"),
            "quality_j": pair.get("quality_j", "UNKNOWN"), "temporal_bin": pair.get("temporal_bin", "UNKNOWN"),
            "source_sha256_i": source_hashes.get(str(pair.get("scene_i", "")), ""),
            "source_sha256_j": source_hashes.get(str(pair.get("scene_j", "")), ""),
            "candidate_geometry_source": props.get("candidate_geometry_source", "EXISTING_UNVERIFIED_REGION"),
        })
    return collection


def scenes_by_pair(pair: dict[str, Any], scenes: dict[str, dict[str, Any]]) -> tuple[str, str]:
    scene_i, scene_j = str(pair.get("scene_i", "")), str(pair.get("scene_j", ""))
    if scene_i not in scenes or scene_j not in scenes:
        return ()
    return scene_i, scene_j
