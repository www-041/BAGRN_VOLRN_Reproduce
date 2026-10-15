"""Objective candidate context and guarded human-label edits for Benchmark V1.

Suggestions are never review decisions or independent geometric truth.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

from shapely.geometry import Point, shape

from benchmark_candidate_core import CandidateReviewStore


LANDCOVER_VALUES = {"UNKNOWN", "BUILDING", "ROAD", "BARE", "VEGETATION", "WATER", "OTHER"}
TERRAIN_VALUES = {"TERRAIN_UNKNOWN", "FLAT", "HILLY", "MOUNTAIN"}
COORDINATE_FIELDS = ("row_i", "col_i", "x_i", "y_i", "row_j", "col_j", "x_j", "y_j")


def load_scene_metadata(manifest_path: Path, inventory_path: Path) -> dict[str, dict]:
    """Join exact full scene IDs; do not infer quality or dates from filenames."""
    with inventory_path.open(newline="", encoding="utf-8-sig") as stream:
        inventory = {row["product_id"]: row for row in csv.DictReader(stream)}
    with manifest_path.open(newline="", encoding="utf-8-sig") as stream:
        manifest = list(csv.DictReader(stream))
    result = {}
    for row in manifest:
        scene_id = row["scene_id"]
        if scene_id not in inventory:
            raise ValueError(f"Scene missing from benchmark inventory: {scene_id}")
        result[scene_id] = {
            "acquisition_time": row["acquisition_time"],
            "quality_status": inventory[scene_id]["b9_quality_status_tag"] or "UNKNOWN",
            "date_source": str(manifest_path), "quality_source": str(inventory_path),
        }
    return result


def load_frozen_split(path: Path | None) -> dict | None:
    """Only explicit frozen polygons in the native metre CRS may suggest a role."""
    if path is None:
        return None
    raw = path.read_bytes()
    document = json.loads(raw)
    meta = document.get("metadata", {})
    if meta.get("status") != "FROZEN" or not meta.get("version") or meta.get("crs") != "EPSG:32650":
        raise ValueError("Split GeoJSON requires metadata status=FROZEN, version and crs=EPSG:32650")
    features = []
    for feature in document.get("features", []):
        properties = feature.get("properties", {})
        role = properties.get("role")
        split_id = properties.get("split_id", "")
        polygon = shape(feature["geometry"])
        if (role not in {"DEVELOPMENT", "TEST"} or not split_id or polygon.geom_type not in
                {"Polygon", "MultiPolygon"} or not polygon.is_valid or polygon.is_empty):
            raise ValueError("Frozen split has invalid role, split_id or polygon")
        if any(other_role != role and polygon.intersects(other_polygon)
               for other_polygon, other_role, _ in features):
            raise ValueError("Frozen DEVELOPMENT and TEST polygons overlap or touch")
        features.append((polygon, role, split_id))
    if not features:
        raise ValueError("Frozen split has no polygons")
    return {"features": features, "version": meta["version"],
            "sha256": hashlib.sha256(raw).hexdigest()}


def build_candidate_context(row: dict, scene_meta: dict[str, dict], frozen_split: dict | None = None) -> dict:
    """Compute display-only facts; UNKNOWN labels stay UNKNOWN without verified references."""
    left, right = scene_meta[row["scene_i"]], scene_meta[row["scene_j"]]
    date_i = datetime.fromisoformat(left["acquisition_time"])
    date_j = datetime.fromisoformat(right["acquisition_time"])
    context = {
        "candidate_id": row["candidate_id"], "pair_id": row["pair_id"],
        "scene_i": row["scene_i"], "scene_j": row["scene_j"],
        "candidate_source": "AUTO_SUGGESTED", "recommendation_method": row["recommendation_method"],
        **{f"original_{name}": row[name] for name in COORDINATE_FIELDS},
        "acquisition_time_i": left["acquisition_time"], "acquisition_time_j": right["acquisition_time"],
        "elapsed_days": abs((date_j - date_i).total_seconds()) / 86400.0,
        "quality_i": left["quality_status"], "quality_j": right["quality_status"],
        "date_source": left.get("date_source", "SCENE_MANIFEST"),
        "quality_source": left.get("quality_source", "SCENE_INVENTORY"),
        "grid_cell_id": row["grid_cell_id"],
        "geographic_zone": f"PAIR_GRID:{row['pair_id']}:{row['grid_cell_id']}",
        "zone_source": "PAIR_LOCAL_8X8_CANDIDATE_GRID_NOT_FROZEN_SPLIT",
        "landcover_suggestion": "UNKNOWN", "terrain_suggestion": "TERRAIN_UNKNOWN",
        "label_suggestion_source": "NO_VERIFIED_DEM_OR_LANDCOVER_MAP",
        "split_suggestion": "UNASSIGNED", "split_source": "NO_FROZEN_SPATIAL_SPLIT",
        "split_id": "",
    }
    if frozen_split is not None:
        coordinate_prefix = "adjusted_" if row.get("human_adjusted") == "True" else ""
        left_point = Point(float(row[coordinate_prefix + "x_i"]), float(row[coordinate_prefix + "y_i"]))
        right_point = Point(float(row[coordinate_prefix + "x_j"]), float(row[coordinate_prefix + "y_j"]))
        match_i = [(role, split_id) for polygon, role, split_id in frozen_split["features"]
                   if polygon.contains(left_point)]
        match_j = [(role, split_id) for polygon, role, split_id in frozen_split["features"]
                   if polygon.contains(right_point)]
        if len(match_i) == len(match_j) == 1 and match_i[0] == match_j[0]:
            context["split_suggestion"], context["split_id"] = match_i[0]
            context["split_source"] = f"FROZEN:{frozen_split['version']}:{frozen_split['sha256']}"
    return context


def apply_batch_labels(store: CandidateReviewStore, candidate_ids: list[str], *, reviewer: str,
                       terrain: str | None = None, landcover: str | None = None) -> list[dict]:
    """Explicit human labels only; reject the whole batch if any target is reviewed."""
    if not reviewer.strip() or not candidate_ids or len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("Batch requires reviewer and distinct selected candidates")
    if terrain is None and landcover is None:
        raise ValueError("Choose at least one non-review label")
    if terrain is not None and terrain not in TERRAIN_VALUES:
        raise ValueError("Invalid terrain label")
    if landcover is not None and landcover not in LANDCOVER_VALUES:
        raise ValueError("Invalid landcover label")
    rows = [store._row(candidate_id) for candidate_id in candidate_ids]
    if any(row["human_review_status"] != "UNVERIFIED" for row in rows):
        raise ValueError("Batch labels may change only UNVERIFIED candidates; no point was changed")
    changes = []
    for row in rows:
        before = dict(row)
        if terrain is not None:
            row["terrain_class"] = terrain
        if landcover is not None:
            row["landcover_type"] = landcover
        if row != before:
            changes.append({"candidate_id": row["candidate_id"], "before": before,
                            "after": dict(row), "reviewer": reviewer})
    return changes


def save_candidate_session(store: CandidateReviewStore, previously_saved: dict[str, dict],
                           context: dict[str, dict], auto_truth_path: Path, *, reviewer: str) -> None:
    """Retain the pre-feature baseline and per-save before/after candidate records."""
    if not store.path.is_file():
        raise FileNotFoundError(f"Cannot audit missing candidate CSV: {store.path}")
    baseline_path = store.path.with_suffix(".baseline.csv")
    if not baseline_path.exists():
        shutil.copy2(store.path, baseline_path)
    changes = []
    now = datetime.now(timezone.utc).isoformat()
    for row in store.rows:
        before = previously_saved.get(row["candidate_id"])
        if before is None:
            raise ValueError(f"Candidate ID was not present at load: {row['candidate_id']}")
        if before != row:
            changes.append({"candidate_id": row["candidate_id"], "reviewer": reviewer,
                            "saved_at_utc": now, "before": before, "after": dict(row)})
    context_path = store.path.with_suffix(".context.json")
    prior_context = json.loads(context_path.read_text(encoding="utf-8")) if context_path.is_file() else None
    context_path.write_text(json.dumps(context, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    store.save()
    store.save_approved_truth(auto_truth_path)
    if prior_context is not None and prior_context != context:
        changes.append({"candidate_id": None, "reviewer": reviewer, "saved_at_utc": now,
                        "before": prior_context, "after": context, "action": "CONTEXT_UPDATED"})
    if changes:
        with store.path.with_suffix(".audit.jsonl").open("a", encoding="utf-8") as stream:
            for change in changes:
                stream.write(json.dumps(change, ensure_ascii=False) + "\n")
