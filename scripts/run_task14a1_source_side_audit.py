"""Read-only Task14A.1 source-side topology audit.

This script consumes only frozen Task13A/Task14 artifacts.  It does not call
the matcher, seam optimizer, local-moment estimator, labeler, or mosaic code.
All outputs are written below ``audit_task14a1_source_side``.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from affine import Affine
from rasterio.features import shapes as raster_shapes
from rasterio.windows import Window
from shapely.geometry import box
from shapely.geometry import shape as shapely_shape
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parents[1]
OUT13 = ROOT / "data/output/b9_13scene_task14"
OUT5 = ROOT / "data/output/b9_five_scene_validation"
STAGE07 = OUT13 / "stages/07_pairwise_seam_local"
AUDIT = OUT13 / "audit_task14a1_source_side"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.seam_local.seam import SeamResult
from src.seam_local.source_side import resolve_source_sides


RESOLVABLE = {"EXCLUSIVE_CONTACT_RESOLVABLE", "CENTROID_RESOLVABLE"}
STAGE_STATUS = {"PASS", "REQUIRES_MULTISCENE_LABELING", "UNSTABLE_LOCAL_GAIN"}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=_json_default) + "\n", encoding="utf-8")


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def footprint_polygon(mask: np.ndarray, transform: Affine):
    polygons = [
        shapely_shape(geometry)
        for geometry, value in raster_shapes(
            np.asarray(mask, dtype=np.uint8),
            mask=np.asarray(mask, dtype=bool),
            transform=transform,
        )
        if int(value) == 1
    ]
    if not polygons:
        raise ValueError("empty frozen valid footprint")
    return unary_union(polygons)


def load_masks(paths: list[Path]) -> tuple[list[np.ndarray], Affine, int, int]:
    masks: list[np.ndarray] = []
    transform = None
    height = width = 0
    for path in paths:
        with rasterio.open(path) as src:
            mask = src.read_masks(1) > 0
            if transform is None:
                transform = src.transform
                height, width = src.height, src.width
            if mask.shape != (height, width) or src.transform != transform:
                raise ValueError(f"canonical grid mismatch: {path}")
            masks.append(mask)
    if transform is None:
        raise ValueError("no scenes")
    return masks, transform, height, width


def scene_paths_from_manifest(manifest_path: Path) -> list[Path]:
    manifest = read_json(manifest_path)
    return [ROOT / row["path"] for row in manifest["scenes"]]


def outer_crop(mask_a: np.ndarray, mask_b: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
    joint = np.asarray(mask_a, dtype=bool) & np.asarray(mask_b, dtype=bool)
    rows, cols = np.where(joint)
    if rows.size == 0:
        raise ValueError("no shared bounding-box crop")
    r0, r1, c0, c1 = int(rows.min()), int(rows.max()) + 1, int(cols.min()), int(cols.max()) + 1
    if r1 - r0 >= c1 - c0:
        support = (np.asarray(mask_a, dtype=bool) | np.asarray(mask_b, dtype=bool))[r0:r1]
        transverse = np.flatnonzero(np.any(support, axis=0))
        return (r0, int(transverse[0])), (r1 - r0, int(transverse[-1]) + 1 - int(transverse[0]))
    support = (np.asarray(mask_a, dtype=bool) | np.asarray(mask_b, dtype=bool))[:, c0:c1]
    transverse = np.flatnonzero(np.any(support, axis=1))
    return (int(transverse[0]), c0), (int(transverse[-1]) + 1 - int(transverse[0]), c1 - c0)


def read_pair_arrays(paths: list[Path], masks: list[np.ndarray], i: int, j: int,
                     origin: tuple[int, int] | None = None,
                     shape: tuple[int, int] | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[int, int], tuple[int, int]]:
    if origin is None or shape is None:
        origin, shape = outer_crop(masks[i], masks[j])
    r0, c0 = origin
    height, width = shape
    window = Window(c0, r0, width, height)
    with rasterio.open(paths[i]) as src_a, rasterio.open(paths[j]) as src_b:
        a = src_a.read(1, window=window)
        b = src_b.read(1, window=window)
        va = src_a.read_masks(1, window=window) > 0
        vb = src_b.read_masks(1, window=window) > 0
    return a, b, va, vb, origin, shape


def load_saved_seam(path: Path, crop_origin: tuple[int, int], transform: Affine,
                    shape: tuple[int, int]) -> SeamResult:
    record = read_json(path)
    feature = record["features"][0]
    properties = feature["properties"]
    coordinates = np.asarray(feature["geometry"]["coordinates"], dtype=np.float64)
    inverse = ~transform
    pixels = np.asarray([inverse * (float(x), float(y)) for x, y in coordinates])
    cols = np.rint(pixels[:, 0] - 0.5).astype(np.int64) - int(crop_origin[1])
    rows = np.rint(pixels[:, 1] - 0.5).astype(np.int64) - int(crop_origin[0])
    path_rc = np.column_stack((rows, cols))
    orientation = properties["orientation"]
    expected = shape[0] if orientation == "vertical" else shape[1]
    line = rows if orientation == "vertical" else cols
    if path_rc.shape != (expected, 2) or not np.array_equal(line, np.arange(expected)):
        raise ValueError(f"saved seam does not span local crop: {path}")
    if (rows < 0).any() or (rows >= shape[0]).any() or (cols < 0).any() or (cols >= shape[1]).any():
        raise ValueError(f"saved seam outside local crop: {path}")
    return SeamResult(
        orientation,
        path_rc,
        float(properties["total_cost"]),
        float(properties["mean_cost"]),
        float(properties["p95_cost"]),
        "OK",
        properties["search_mode"],
    )


def seam_topology(seam: SeamResult | None, shape: tuple[int, int]) -> dict[str, Any]:
    if seam is None:
        return {"valid": False, "orientation": None, "length": None, "endpoint_1": None,
                "endpoint_2": None, "touches_expected_boundaries": False}
    path = np.asarray(seam.row_col_path)
    if path.size == 0:
        return {"valid": False, "orientation": seam.orientation, "length": 0, "endpoint_1": None,
                "endpoint_2": None, "touches_expected_boundaries": False}
    h, w = shape
    expected = h if seam.orientation == "vertical" else w
    line = path[:, 0] if seam.orientation == "vertical" else path[:, 1]
    valid = bool(path.shape == (expected, 2) and np.array_equal(line, np.arange(expected)))
    if seam.orientation == "vertical":
        touches = bool(valid and path[0, 0] == 0 and path[-1, 0] == h - 1 and np.all((path[:, 1] >= 0) & (path[:, 1] < w)))
    elif seam.orientation == "horizontal":
        touches = bool(valid and path[0, 1] == 0 and path[-1, 1] == w - 1 and np.all((path[:, 0] >= 0) & (path[:, 0] < h)))
    else:
        touches = False
    return {"valid": valid, "orientation": seam.orientation, "length": int(len(path)),
            "endpoint_1": [int(path[0, 0]), int(path[0, 1])],
            "endpoint_2": [int(path[-1, 0]), int(path[-1, 1])],
            "touches_expected_boundaries": touches}


def resolver_summary(initial: Any, refined: Any) -> dict[str, Any]:
    initial_ok = initial is not None and initial.status in RESOLVABLE
    refined_ok = refined is not None and refined.status in RESOLVABLE
    agree = bool(initial_ok and refined_ok and
                 initial.side_1_source == refined.side_1_source and
                 initial.side_2_source == refined.side_2_source)
    if agree:
        status = "PASS"
        method = refined.method
    elif initial is not None and refined is not None and (
        initial.side_1_source != refined.side_1_source or initial.side_2_source != refined.side_2_source or initial.status != refined.status
    ):
        status = "REQUIRES_MULTISCENE_LABELING"
        method = "INITIAL_REFINED_CONFLICT"
    else:
        status = "REQUIRES_MULTISCENE_LABELING"
        method = refined.method if refined is not None else "UNRESOLVED_GEOMETRY"
    return {
        "status": status,
        "method": method,
        "initial_status": initial.status if initial is not None else None,
        "refined_status": refined.status if refined is not None else None,
        "initial_method": initial.method if initial is not None else None,
        "refined_method": refined.method if refined is not None else None,
        "assignment_agreement": agree,
        "side_1_source": refined.side_1_source if agree else None,
        "side_2_source": refined.side_2_source if agree else None,
    }


def geometry_class(initial: Any, refined: Any, va: np.ndarray, vb: np.ndarray,
                   refined_topology: dict[str, Any]) -> tuple[str, str]:
    a_only = int(np.count_nonzero(va & ~vb))
    b_only = int(np.count_nonzero(vb & ~va))
    if refined_topology["valid"] is False:
        return "SEAM_TOPOLOGY_NOT_TWO_SIDE", "missing_or_invalid_saved_seam"
    if a_only == 0 and b_only == 0:
        return "TRUE_CONTAINMENT", "identical_shared_support"
    if a_only == 0:
        return "MISSING_A_EXCLUSIVE", "no_a_only_valid_support"
    if b_only == 0:
        return "MISSING_B_EXCLUSIVE", "no_b_only_valid_support"
    inter = int(np.count_nonzero(va & vb))
    ra = inter / max(int(np.count_nonzero(va)), 1)
    rb = inter / max(int(np.count_nonzero(vb)), 1)
    if max(ra, rb) >= 0.99:
        return "NEAR_CONTAINMENT", "directional_pixel_containment_ge_0.99"
    if initial is not None and refined is not None and (
        initial.status != refined.status or initial.side_1_source != refined.side_1_source or initial.side_2_source != refined.side_2_source
    ):
        return "INITIAL_REFINED_CONFLICT", "initial_refined_resolver_disagreement"
    if refined is not None and refined.status == "EXCLUSIVE_CONTACT_RESOLVABLE":
        d = refined.diagnostics
        if d.get("assignment_score_ab") == d.get("assignment_score_ba") or max(d.get("assignment_score_ab", 0), d.get("assignment_score_ba", 0)) == 0:
            return "CONTACT_AMBIGUOUS", "equal_or_zero_exclusive_contact_scores"
        return "CLEAR_TWO_SIDE_GEOMETRY", "exclusive_boundary_contact_resolved"
    if refined is not None and refined.status == "CENTROID_RESOLVABLE":
        return "CENTROID_AMBIGUOUS", "centroid_fallback_used"
    return "CONTACT_AMBIGUOUS", "resolver_requires_multiscene_labeling"


def diagnostics_row(pair: dict[str, Any], i: int, j: int, paths: list[Path], masks: list[np.ndarray],
                    transform: Affine, footprints: list[Any], bbox_footprints: list[Any],
                    five_scene: bool = False) -> tuple[dict[str, Any], dict[str, Any] | None]:
    stage_status = pair.get("status", pair.get("new_status", ""))
    if stage_status not in STAGE_STATUS:
        stage_status = "REQUIRES_MULTISCENE_LABELING"
    origin = shape = None
    if pair.get("crop_origin"):
        origin = tuple(int(v) for v in json.loads(pair["crop_origin"]) if isinstance(pair["crop_origin"], str)) if isinstance(pair["crop_origin"], str) else tuple(int(v) for v in pair["crop_origin"])
    if pair.get("crop_shape"):
        shape = tuple(int(v) for v in json.loads(pair["crop_shape"]) if isinstance(pair["crop_shape"], str)) if isinstance(pair["crop_shape"], str) else tuple(int(v) for v in pair["crop_shape"])
    a, b, va, vb, crop_origin, crop_shape = read_pair_arrays(paths, masks, i, j, origin, shape)
    inter_mask = va & vb
    valid_a = int(np.count_nonzero(va))
    valid_b = int(np.count_nonzero(vb))
    intersection = int(np.count_nonzero(inter_mask))
    a_only = int(np.count_nonzero(va & ~vb))
    b_only = int(np.count_nonzero(vb & ~va))
    initial = refined = None
    pair_dir = STAGE07 / "pairs" / pair["pair_id"] if not five_scene else OUT5 / "seam_local_task13a" / "pairs" / pair["pair_id"]
    if (pair_dir / "seam_initial.geojson").is_file():
        initial = load_saved_seam(pair_dir / "seam_initial.geojson", crop_origin, transform, crop_shape)
        refined = load_saved_seam(pair_dir / "seam_refined.geojson", crop_origin, transform, crop_shape)
    top_i = seam_topology(initial, crop_shape)
    top_r = seam_topology(refined, crop_shape)
    true_i = true_r = bbox_i = bbox_r = None
    if initial is not None and refined is not None:
        true_i = resolve_source_sides(initial, inter_mask, va, vb, footprints[i], footprints[j])
        true_r = resolve_source_sides(refined, inter_mask, va, vb, footprints[i], footprints[j])
        bbox_i = resolve_source_sides(initial, inter_mask, va, vb, bbox_footprints[i], bbox_footprints[j])
        bbox_r = resolve_source_sides(refined, inter_mask, va, vb, bbox_footprints[i], bbox_footprints[j])
    true_summary = resolver_summary(true_i, true_r)
    bbox_summary = resolver_summary(bbox_i, bbox_r)
    primary, primary_reason = geometry_class(true_i, true_r, va, vb, top_r)
    flags: list[str] = []
    if max(intersection / max(valid_a, 1), intersection / max(valid_b, 1)) >= 0.99 and a_only and b_only:
        flags.append("NEAR_CONTAINMENT")
    if true_summary["method"] == "INITIAL_REFINED_CONFLICT":
        flags.append("INITIAL_REFINED_CONFLICT")
    if top_r["touches_expected_boundaries"] is False:
        flags.append("SEAM_TOPOLOGY_NOT_TWO_SIDE")
    if true_r is not None:
        d = true_r.diagnostics
        if d.get("assignment_score_ab") == d.get("assignment_score_ba") or max(d.get("assignment_score_ab", 0), d.get("assignment_score_ba", 0)) == 0:
            flags.append("CONTACT_AMBIGUOUS")
        if true_r.method == "FOOTPRINT_CENTROID_PROJECTION":
            flags.append("CENTROID_AMBIGUOUS")
        if d.get("nested_or_contained_footprints"):
            flags.append("TRUE_OR_NEAR_CONTAINMENT")
    if stage_status == "UNSTABLE_LOCAL_GAIN":
        flags.append("UNSTABLE_LOCAL_GAIN")
    if not flags:
        flags.append(primary)
    fmin = min(valid_a, valid_b)
    area_a = float(footprints[i].area)
    area_b = float(footprints[j].area)
    area_inter = float(footprints[i].intersection(footprints[j]).area)
    seam_status = pair.get("refined_seam_status")
    if five_scene:
        seam_status = pair.get("v2_status", pair.get("status"))
    row = {
        "pair_id": pair["pair_id"], "scene_i": i, "scene_j": j,
        "registration_edge_accepted": pair.get("registration_edge_accepted", True),
        "shared_valid_pixel_count": intersection, "valid_a": valid_a, "valid_b": valid_b,
        "intersection_over_min_scene": intersection / max(fmin, 1),
        "a_only": a_only, "b_only": b_only,
        "a_only_ratio_of_a": a_only / max(valid_a, 1), "b_only_ratio_of_b": b_only / max(valid_b, 1),
        "a_in_b_pixel_ratio": intersection / max(valid_a, 1), "b_in_a_pixel_ratio": intersection / max(valid_b, 1),
        "footprint_intersection_over_a_area": area_inter / max(area_a, 1e-12),
        "footprint_intersection_over_b_area": area_inter / max(area_b, 1e-12),
        "near_containment_flag": bool(max(intersection / max(valid_a, 1), intersection / max(valid_b, 1)) >= 0.99),
        "true_containment_flag": bool(a_only == 0 or b_only == 0),
        "initial_orientation": top_i["orientation"], "initial_length": top_i["length"],
        "initial_endpoint_1": json.dumps(top_i["endpoint_1"]), "initial_endpoint_2": json.dumps(top_i["endpoint_2"]),
        "refined_orientation": top_r["orientation"], "refined_length": top_r["length"],
        "refined_endpoint_1": json.dumps(top_r["endpoint_1"]), "refined_endpoint_2": json.dumps(top_r["endpoint_2"]),
        "seam_touches_expected_overlap_boundaries": top_r["touches_expected_boundaries"],
        "old_status": stage_status, "old_initial_source_side_status": pair.get("initial_source_side_status"),
        "old_refined_source_side_status": pair.get("refined_source_side_status"),
        "exclusive_contact_a_side1": true_r.diagnostics.get("exclusive_contact_a_side1") if true_r else None,
        "exclusive_contact_a_side2": true_r.diagnostics.get("exclusive_contact_a_side2") if true_r else None,
        "exclusive_contact_b_side1": true_r.diagnostics.get("exclusive_contact_b_side1") if true_r else None,
        "exclusive_contact_b_side2": true_r.diagnostics.get("exclusive_contact_b_side2") if true_r else None,
        "assignment_score_ab": true_r.diagnostics.get("assignment_score_ab") if true_r else None,
        "assignment_score_ba": true_r.diagnostics.get("assignment_score_ba") if true_r else None,
        "centroid_a_x": true_r.diagnostics.get("centroid_a", (None, None))[0] if true_r else None,
        "centroid_a_y": true_r.diagnostics.get("centroid_a", (None, None))[1] if true_r else None,
        "centroid_b_x": true_r.diagnostics.get("centroid_b", (None, None))[0] if true_r else None,
        "centroid_b_y": true_r.diagnostics.get("centroid_b", (None, None))[1] if true_r else None,
        "centroid_projection_separation": true_r.diagnostics.get("projection_separation") if true_r else None,
        "initial_resolver_status": true_summary["initial_status"], "initial_resolver_method": true_summary["initial_method"],
        "refined_resolver_status": true_summary["refined_status"], "refined_resolver_method": true_summary["refined_method"],
        "task13a1_resolver_dryrun_status": true_summary["status"], "task13a1_resolver_method": true_summary["method"],
        "task13a1_assignment_agreement": true_summary["assignment_agreement"],
        "bbox_resolver_dryrun_status": bbox_summary["status"], "bbox_resolver_dryrun_method": bbox_summary["method"],
        "task14a_stage07_status": stage_status,
        "status_equal": bool(stage_status == true_summary["status"]),
        "adapter_logic_mismatch": bool(true_summary["status"] == "PASS" and stage_status == "REQUIRES_MULTISCENE_LABELING"),
        "primary_failure_class": primary, "primary_failure_reason": primary_reason,
        "secondary_flags": ";".join(sorted(set(flags))),
        "final_failure_reason": "PASS" if stage_status == "PASS" else ("UNSTABLE_LOCAL_GAIN" if stage_status == "UNSTABLE_LOCAL_GAIN" else ("ADAPTER_LOGIC_MISMATCH" if true_summary["status"] == "PASS" else primary_reason)),
        "local_gain_status": stage_status if stage_status == "UNSTABLE_LOCAL_GAIN" else "STABLE_OR_NOT_FLAGGED",
        "crop_origin": json.dumps(list(crop_origin)), "crop_shape": json.dumps(list(crop_shape)),
        "_crop_image_a": a, "_crop_image_b": b, "_crop_va": va, "_crop_vb": vb, "_seam": refined,
    }
    return row, None


def parse_pair_row(row: dict[str, str]) -> dict[str, Any]:
    out: dict[str, Any] = dict(row)
    out["scene_i"] = int(row["scene_i"])
    out["scene_j"] = int(row["scene_j"])
    out["registration_edge_accepted"] = str(row.get("registration_edge_accepted", "True")).lower() == "true"
    return out


def five_rows() -> list[dict[str, Any]]:
    metrics = {r["pair_id"]: r for r in read_csv(OUT5 / "seam_local_task13a" / "pair_metrics.csv")}
    audit = {r["pair_id"]: r for r in read_csv(OUT5 / "seam_local_task13a1_source_side" / "source_side_audit.csv")}
    rows: list[dict[str, Any]] = []
    for pair_id, metric in metrics.items():
        m = parse_pair_row(metric)
        m["registration_edge_accepted"] = True
        m["status"] = audit.get(pair_id, {}).get("new_status", metric.get("status", "REQUIRES_MULTISCENE_LABELING"))
        m["initial_source_side_status"] = audit.get(pair_id, {}).get("initial_resolver_status")
        m["refined_source_side_status"] = audit.get(pair_id, {}).get("refined_resolver_status")
        rows.append(m)
    return rows


def compare_rows(scene_set: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    def vals(key: str) -> np.ndarray:
        return np.asarray([float(r[key]) for r in rows if r.get(key) not in (None, "", "nan")], dtype=float)
    return {
        "scene_set": scene_set, "pair_count": len(rows),
        "median_shared_valid": float(np.median(vals("shared_valid_pixel_count"))),
        "median_intersection_over_min_scene": float(np.median(vals("intersection_over_min_scene"))),
        "near_containment_fraction": float(np.mean([bool(r["near_containment_flag"]) for r in rows])),
        "both_exclusive_support_count": sum(int(r["a_only"] > 0 and r["b_only"] > 0) for r in rows),
        "missing_a_exclusive_count": sum(int(r["a_only"] == 0) for r in rows),
        "missing_b_exclusive_count": sum(int(r["b_only"] == 0) for r in rows),
        "ambiguous_contact_count": sum(int("CONTACT_AMBIGUOUS" in str(r["secondary_flags"]) or r["primary_failure_class"] == "CONTACT_AMBIGUOUS") for r in rows),
        "exclusive_contact_resolved_count": sum(int(r["task13a1_resolver_method"] == "EXCLUSIVE_BOUNDARY_CONTACT") for r in rows),
        "centroid_fallback_resolved_count": sum(int(r["task13a1_resolver_method"] == "FOOTPRINT_CENTROID_PROJECTION") for r in rows),
        "requires_multiscene_count": sum(int(r["task13a1_resolver_dryrun_status"] != "PASS") for r in rows),
        "pass_fraction": float(np.mean([r["task13a1_resolver_dryrun_status"] == "PASS" for r in rows])),
    }


def make_figure(row: dict[str, Any], output: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    va = row["_crop_va"]
    vb = row["_crop_vb"]
    inter = va & vb
    category = np.zeros(va.shape, dtype=np.uint8)
    category[va & ~vb] = 1
    category[vb & ~va] = 2
    category[inter] = 3
    scale = max(1, int(max(category.shape) / 900))
    if scale > 1:
        h = category.shape[0] // scale * scale
        w = category.shape[1] // scale * scale
        category = category[:h, :w].reshape(h // scale, scale, w // scale, scale).max(axis=(1, 3))
    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    ax.imshow(category, interpolation="nearest", cmap="viridis", vmin=0, vmax=3)
    seam = row.get("_seam")
    if seam is not None:
        path = np.asarray(seam.row_col_path)
        ax.plot(path[:, 1] / scale, path[:, 0] / scale, color="white", linewidth=0.7)
    ax.set_title(f"{row['pair_id']} | {row['primary_failure_class']} | {row['task14a_stage07_status']}\n"
                 f"shared={row['shared_valid_pixel_count']:,}, A-only={row['a_only']:,}, B-only={row['b_only']:,}")
    ax.set_axis_off()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=140)
    plt.close(fig)


def unstable_audit(rows: list[dict[str, Any]], masks: list[np.ndarray], paths: list[Path]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        if row["task14a_stage07_status"] != "UNSTABLE_LOCAL_GAIN":
            continue
        i, j = int(row["scene_i"]), int(row["scene_j"])
        crop = (row["_crop_va"], row["_crop_vb"])
        va, vb = crop
        shared = va & vb
        h, w = shared.shape
        aspect = max(h, w) / max(min(h, w), 1)
        edge = np.zeros_like(shared)
        edge[0, :] = edge[-1, :] = True
        edge[:, 0] = edge[:, -1] = True
        edge_fraction = float(np.count_nonzero(shared & edge) / max(np.count_nonzero(shared), 1))
        cause = "UNKNOWN"
        if aspect >= 4.0:
            cause = "NARROW_OVERLAP"
        elif np.count_nonzero(shared) < 4096:
            cause = "SMALL_SUPPORT"
        elif edge_fraction >= 0.5:
            cause = "EDGE_DOMINATED"
        out.append({
            "pair_id": row["pair_id"], "scene_i": i, "scene_j": j,
            "shared_valid_pixels": int(np.count_nonzero(shared)), "segment_count": "NOT_SAVED",
            "valid_segment_count": "NOT_SAVED", "unstable_segment_count": "NOT_SAVED",
            "gain_min": "NOT_SAVED", "gain_max": "NOT_SAVED", "gain_median": "NOT_SAVED",
            "offset_min": "NOT_SAVED", "offset_max": "NOT_SAVED", "offset_median": "NOT_SAVED",
            "source_std_a": "NOT_SAVED", "source_std_b": "NOT_SAVED", "target_std": "NOT_SAVED",
            "low_variance_segment_count": "NOT_SAVED", "min_segment_pixels": "NOT_SAVED",
            "overlap_height": h, "overlap_width": w, "overlap_aspect": aspect,
            "near_nodata_edge_fraction": edge_fraction, "near_nodata_edge_flag": edge_fraction >= 0.5,
            "extreme_overlap_flag": aspect >= 4.0, "anomalous_radiometry": "NOT_ASSESSED",
            "classification": cause,
            "evidence_boundary": "Stage07 stopped before saving local_coefficients.csv; no gain/moment re-estimation performed",
        })
    return out


def main() -> None:
    AUDIT.mkdir(parents=True, exist_ok=True)
    (AUDIT / "figures").mkdir(parents=True, exist_ok=True)
    manifest13 = OUT13 / "stages/06_bagrn/bagrn/normalized_scenes_manifest.json"
    paths13 = scene_paths_from_manifest(manifest13)
    masks13, transform13, _, _ = load_masks(paths13)
    footprints13 = [footprint_polygon(mask, transform13) for mask in masks13]
    bbox13 = [box(*rasterio.transform.array_bounds(mask.shape[0], mask.shape[1], transform13)) for mask in masks13]
    stage_rows = [parse_pair_row(r) for r in read_json(STAGE07 / "pairwise_results.json")]
    audit_rows: list[dict[str, Any]] = []
    for pair in stage_rows:
        if int(pair.get("shared_valid_pixel_count", 0)) <= 0:
            continue
        row, _ = diagnostics_row(pair, pair["scene_i"], pair["scene_j"], paths13, masks13, transform13, footprints13, bbox13)
        audit_rows.append(row)
    clean_rows = [{k: v for k, v in row.items() if not k.startswith("_")} for row in audit_rows]
    write_csv(AUDIT / "source_side_failure_summary.csv", clean_rows)
    audited_by_id = {row["pair_id"]: row for row in clean_rows}
    resolver_all_rows: list[dict[str, Any]] = []
    for pair in stage_rows:
        pid = pair["pair_id"]
        if pid in audited_by_id:
            selected = audited_by_id[pid]
            resolver_all_rows.append({
                "pair_id": pid,
                "scene_i": pair["scene_i"], "scene_j": pair["scene_j"],
                "task14a_stage07_status": selected["task14a_stage07_status"],
                "task13a1_resolver_dryrun_status": selected["task13a1_resolver_dryrun_status"],
                "task13a1_resolver_method": selected["task13a1_resolver_method"],
                "status_equal": selected["status_equal"],
                "adapter_logic_mismatch": selected["adapter_logic_mismatch"],
                "geometry_audit_scope": "FINAL_SHARED_SUPPORT",
            })
        else:
            resolver_all_rows.append({
                "pair_id": pid,
                "scene_i": pair["scene_i"], "scene_j": pair["scene_j"],
                "task14a_stage07_status": pair.get("status", "NO_FINAL_SHARED_SUPPORT"),
                "task13a1_resolver_dryrun_status": "NO_FINAL_SHARED_SUPPORT",
                "task13a1_resolver_method": "NOT_APPLICABLE_NO_SAVED_SEAM",
                "status_equal": True,
                "adapter_logic_mismatch": False,
                "geometry_audit_scope": "NO_FINAL_SHARED_SUPPORT",
            })
    write_csv(AUDIT / "resolver_dryrun_all_60_pairs.csv", resolver_all_rows)

    five_manifest = OUT5 / "seam_local_task13a" / "protocol.json"
    protocol5 = read_json(five_manifest)
    paths5 = [ROOT / scene["path"] for scene in protocol5["scenes"]]
    masks5, transform5, _, _ = load_masks(paths5)
    footprints5 = [footprint_polygon(mask, transform5) for mask in masks5]
    bbox5 = [box(*rasterio.transform.array_bounds(mask.shape[0], mask.shape[1], transform5)) for mask in masks5]
    five_audit_rows: list[dict[str, Any]] = []
    for pair in five_rows():
        row, _ = diagnostics_row(pair, pair["scene_i"], pair["scene_j"], paths5, masks5, transform5, footprints5, bbox5, five_scene=True)
        five_audit_rows.append(row)
    write_csv(AUDIT / "five_scene_geometry_audit.csv", [{k: v for k, v in row.items() if not k.startswith("_")} for row in five_audit_rows])
    comparisons = [compare_rows("five_scene", five_audit_rows), compare_rows("thirteen_scene_shared_support", audit_rows)]
    write_csv(AUDIT / "five_vs_thirteen_geometry_comparison.csv", comparisons)

    # Select figures only after all classifications are known.
    selected: dict[str, dict[str, Any]] = {}
    for row in audit_rows:
        for key in ("adapter_mismatch", row["primary_failure_class"]):
            if key is True:
                key = "ADAPTER_LOGIC_MISMATCH"
            if isinstance(key, str) and key not in selected:
                selected[key] = row
    for index, (category, row) in enumerate(list(selected.items())[:6]):
        make_figure(row, AUDIT / "figures" / f"{index:02d}_{row['pair_id']}_{category}.png")

    status_counts = Counter(str(r["task14a_stage07_status"]) for r in audit_rows)
    dry_counts = Counter(str(r["task13a1_resolver_dryrun_status"]) for r in audit_rows)
    primary_counts = Counter(str(r["primary_failure_class"]) for r in audit_rows)
    mismatch_count = sum(bool(r["adapter_logic_mismatch"]) for r in audit_rows)
    unstable = unstable_audit(audit_rows, masks13, paths13)
    write_csv(AUDIT / "unstable_gain_audit.csv", unstable)

    resolver_path = {
        "same_resolver_function": True,
        "resolver_module": "src.seam_local.source_side",
        "resolver_function": "resolve_source_sides",
        "five_scene_entrypoint": "scripts/run_task13a1_source_side.py::_replay_pair",
        "thirteen_scene_stage07_entrypoint": "scripts/run_task14a_resume_13.py::_stage07",
        "five_scene_footprint_path": "scripts/run_task13a1_source_side.py::_footprint_polygon(actual_valid_mask)",
        "thirteen_scene_stage07_footprint_path": "scripts/run_task14a_resume_13.py::_stage07(box(dataset_bounds))",
        "thirteen_scene_audit_dryrun_footprint_path": "scripts/run_task14a1_source_side_audit.py::footprint_polygon(actual_valid_mask)",
        "footprint_geometry_path_equal": False,
        "stage07_source_side_status_not_rewritten": True,
        "adapter_logic_mismatch_pair_count": mismatch_count,
        "adapter_logic_mismatch_pairs": [r["pair_id"] for r in audit_rows if r["adapter_logic_mismatch"]],
        "source_side_module_sha256": sha256(ROOT / "src/seam_local/source_side.py"),
        "stage07_script_sha256": sha256(ROOT / "scripts/run_task14a_resume_13.py"),
        "task13a1_script_sha256": sha256(ROOT / "scripts/run_task13a1_source_side.py"),
    }
    write_json(AUDIT / "resolver_path_audit.json", resolver_path)

    decision = "TOPOLOGY_LIMITATION_DOMINANT"
    if mismatch_count >= max(1, len(audit_rows) // 2):
        decision = "IMPLEMENTATION_LIMITATION_DOMINANT"
    elif mismatch_count > 0:
        decision = "MIXED_CAUSE"
    summary = {
        "task": "Task14A.1",
        "scope": "read-only source-side topology audit; only 48 13-scene pairs with final shared support",
        "stage07_status_counts": dict(status_counts),
        "task13a1_true_footprint_dryrun_counts": dict(dry_counts),
        "primary_failure_class_counts": dict(primary_counts),
        "adapter_logic_mismatch_count": mismatch_count,
        "decision": decision,
        "unstable_pair_count": len(unstable),
        "no_stage07_12_rerun": True,
        "no_task14a2": True,
        "source_side_failure_summary_csv": str(AUDIT / "source_side_failure_summary.csv"),
        "resolver_dryrun_all_60_pairs_csv": str(AUDIT / "resolver_dryrun_all_60_pairs.csv"),
        "five_vs_thirteen_geometry_comparison_csv": str(AUDIT / "five_vs_thirteen_geometry_comparison.csv"),
    }
    write_json(AUDIT / "source_side_failure_summary.json", summary)

    # Keep the report self-contained and explicitly answer the decision questions.
    near = sum(bool(r["near_containment_flag"]) for r in audit_rows)
    true = sum(bool(r["true_containment_flag"]) for r in audit_rows)
    miss_a = sum(r["a_only"] == 0 for r in audit_rows)
    miss_b = sum(r["b_only"] == 0 for r in audit_rows)
    contact = sum("CONTACT_AMBIGUOUS" in str(r["secondary_flags"]) for r in audit_rows)
    centroid = sum("CENTROID_AMBIGUOUS" in str(r["secondary_flags"]) for r in audit_rows)
    conflict = sum("INITIAL_REFINED_CONFLICT" in str(r["secondary_flags"]) for r in audit_rows)
    report = f"""# Task14A.1 — 13-Scene Source-Side Topology Audit

## Scope and hard boundary

This is a read-only audit of frozen Stage06/Stage07 artifacts. It covers the 48 pairs with final shared valid support. It does not rerun matching, RANSAC, Translation-L2, canonical warp, EDT, BAGRN, seam search, local moments, source labeling, or mosaics. Stage07–12 artifacts were not overwritten, and Task14A.2 was not started.

## Executive result

- Frozen Stage07 status: `{dict(status_counts)}`.
- Task13A.1 resolver dry-run with actual valid-mask footprints: `{dict(dry_counts)}`.
- Adapter-logic mismatch (`dry-run PASS` but frozen Stage07 `REQUIRES_MULTISCENE_LABELING`): **{mismatch_count}** pairs.
- Decision: **{decision}**.
- The same resolver function is used by both paths: `src.seam_local.source_side.resolve_source_sides`.
- The footprint path is not the same: the five-scene replay uses polygonized valid support; frozen 13-scene Stage07 uses dataset bounding boxes.

## Direct answers

1. **Why are there 0/13 final PASS pairs?** The frozen Stage07 source-side gate requires both initial and refined resolver assignments to be resolvable and agree. The audit separates this from local-gain instability and records the exact per-pair cause in `source_side_failure_summary.csv`.
2. **What explains the 46 `REQUIRES_MULTISCENE_LABELING` pairs?** `{dict(primary_counts)}`. The detailed rows include shared support, exclusive support, directional containment, contact scores, centroids, seam endpoints, and resolver dry-run status.
3. **Near/true containment:** near-containment rows = {near}; true-containment rows = {true}; missing-A-exclusive = {miss_a}; missing-B-exclusive = {miss_b}.
4. **Contact/centroid ambiguity:** contact ambiguity flags = {contact}; centroid-fallback/ambiguity flags = {centroid}.
5. **Initial/refined conflict:** {conflict} pairs are flagged; the initial/refined statuses and assignments are preserved per row.
6. **Seam topology:** every saved seam is checked for ordered full-span endpoints against its frozen local overlap crop. Any failure is reported as `SEAM_TOPOLOGY_NOT_TWO_SIDE`.
7. **Dry-run PASS versus Stage07 REQUIRES:** {mismatch_count} pairs; these are explicitly marked `ADAPTER_LOGIC_MISMATCH`.
8. **Same code path?** Yes for the resolver function; no for footprint construction. This is recorded, including source hashes, in `resolver_path_audit.json`.
9. **Five versus thirteen geometry:** see `five_vs_thirteen_geometry_comparison.csv`; it compares median support, containment, both-exclusive support, ambiguity, resolver method counts, requires count, and PASS fraction under the same audit procedure.
10. **Why the two unstable pairs?** The frozen Stage07 records `UNSTABLE_LOCAL_GAIN` for `01_08` and `08_11` but saved neither seam nor local coefficient CSV. Gain/offset/moment statistics therefore remain `NOT_SAVED`; no coefficients were re-estimated. The audit only reports frozen support/aspect/edge evidence and classifies without inventing radiometric causes.
11. **Dominant cause:** {decision}. This decision uses source-side geometry and resolver-path evidence only; boundary MAE/RDD/V2 quality is not used.
12. **Next step:** correct the adapter footprint/resolver integration only after reviewing this audit; do not abandon pairwise processing based on this audit alone. No implementation correction is made in Task14A.1.

## Artifacts

- `source_side_failure_summary.csv/json`
- `unstable_gain_audit.csv`
- `resolver_path_audit.json`
- `resolver_dryrun_all_60_pairs.csv` (the 12 no-final-support pairs are explicitly recorded as not applicable)
- `five_vs_thirteen_geometry_comparison.csv`
- `figures/`

The audit ends here. No Stage07–12 rerun and no Task14A.2.
"""
    (AUDIT / "TASK14A1_SOURCE_SIDE_TOPOLOGY_AUDIT_REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
