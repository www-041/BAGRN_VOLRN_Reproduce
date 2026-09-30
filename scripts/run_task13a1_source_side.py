"""Replay only the four Task13A ambiguous blends from frozen artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from rasterio.features import shapes as raster_shapes
from rasterio.windows import Window
from shapely.geometry import shape as shapely_shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.seam_local.blend import blend_across_seam
from src.seam_local.pipeline import _crop_for_pair
from src.seam_local.seam import SeamResult
from src.seam_local.source_side import resolve_source_sides
from src.seam_local.footprint import footprint_polygon_from_valid_mask

BASE = ROOT / "data/output/b9_five_scene_validation"
TASK13A = BASE / "seam_local_task13a"
OUTPUT = BASE / "seam_local_task13a1_source_side"
PAIR_IDS = ("00_01", "00_02", "01_02", "01_04")
SAVED_FILES = ("seam_initial.geojson", "seam_refined.geojson", "local_coefficients.csv")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_saved_seam(path: Path, crop_origin: tuple[int, int], transform: Affine,
                     shape: tuple[int, int]) -> SeamResult:
    record = json.loads(path.read_text(encoding="utf-8"))
    feature, = record["features"]
    properties = feature["properties"]
    coordinates = np.asarray(feature["geometry"]["coordinates"], dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2 or not np.isfinite(coordinates).all():
        raise ValueError(f"invalid saved seam coordinates: {path}")
    inverse = ~transform
    pixel = np.array([inverse * (float(x), float(y)) for x, y in coordinates])
    cols = np.rint(pixel[:, 0] - .5).astype(np.int64) - crop_origin[1]
    rows = np.rint(pixel[:, 1] - .5).astype(np.int64) - crop_origin[0]
    restored = np.column_stack((rows, cols))
    centers = np.array([transform * (int(col + crop_origin[1]) + .5,
                                      int(row + crop_origin[0]) + .5)
                        for row, col in restored])
    if not np.allclose(centers, coordinates, rtol=0, atol=1e-7):
        raise ValueError(f"saved seam is not on the frozen pixel-center grid: {path}")
    orientation = properties["orientation"]
    if orientation == "vertical":
        valid_path = restored.shape == (shape[0], 2) and np.array_equal(rows, np.arange(shape[0]))
    elif orientation == "horizontal":
        valid_path = restored.shape == (shape[1], 2) and np.array_equal(cols, np.arange(shape[1]))
    else:
        valid_path = False
    if not valid_path or (rows < 0).any() or (rows >= shape[0]).any() or (cols < 0).any() or (cols >= shape[1]).any():
        raise ValueError(f"saved seam does not span the frozen crop: {path}")
    return SeamResult(orientation, restored, properties["total_cost"],
                      properties["mean_cost"], properties["p95_cost"], "OK",
                      properties["search_mode"])


def _load_coefficients(path: Path) -> tuple[tuple[int, float, float, float, float, float], ...]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    coefficients = tuple((int(row["index"]), float(row["center_arc"]),
                          float(row["a_a"]), float(row["b_a"]),
                          float(row["a_b"]), float(row["b_b"])) for row in rows)
    if not coefficients or [row[0] for row in coefficients] != list(range(len(coefficients))):
        raise ValueError(f"saved local coefficients have missing segments: {path}")
    values = np.asarray([row[1:] for row in coefficients])
    if not np.isfinite(values).all() or (np.diff(values[:, 0]) <= 0).any():
        raise ValueError(f"saved local coefficients are nonfinite or unordered: {path}")
    return coefficients


def replay_saved_local_correction(
    image_a: np.ndarray, image_b: np.ndarray, valid_a: np.ndarray, valid_b: np.ndarray,
    initial_seam: SeamResult,
    coefficients: tuple[tuple[int, float, float, float, float, float], ...],
    *, half_width: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply saved affine coefficients with the frozen arc interpolation and taper."""
    a, b = np.asarray(image_a), np.asarray(image_b)
    va, vb = np.asarray(valid_a, dtype=bool), np.asarray(valid_b, dtype=bool)
    if a.ndim != 2 or 0 in a.shape or any(x.shape != a.shape for x in (b, va, vb)):
        raise ValueError("sources and validity masks must be same-shape nonempty 2D arrays")
    if initial_seam.status != "OK" or initial_seam.orientation not in {"vertical", "horizontal"}:
        raise ValueError("a saved successful monotonic seam is required")
    if half_width < 1 or not coefficients:
        raise ValueError("positive correction width and saved coefficients are required")
    from src.seam_local.local_moment import _seam_lines

    transverse, arcs = _seam_lines(initial_seam, a.shape)
    saved = np.asarray(coefficients, dtype=np.float64)
    if saved.shape[1] != 6 or not np.isfinite(saved).all() or (np.diff(saved[:, 1]) <= 0).any():
        raise ValueError("invalid saved coefficient rows")
    # Same column order as Task13A: gain/offset for A, then gain/offset for B.
    interpolated = np.column_stack([
        np.interp(arcs, saved[:, 1], saved[:, column]) for column in (2, 3, 4, 5)
    ])
    corrected_a = np.asarray(a, dtype=np.float64).copy()
    corrected_b = np.asarray(b, dtype=np.float64).copy()
    for line, center in enumerate(transverse):
        left = max(0, int(center) - half_width)
        right = min(a.shape[1 if initial_seam.orientation == "vertical" else 0], int(center) + half_width + 1)
        pixels = np.arange(left, right)
        weights = .5 * (1.0 + np.cos(np.pi * np.abs(pixels - center) / half_width))
        slot = ((line, slice(left, right)) if initial_seam.orientation == "vertical"
                else (slice(left, right), line))
        for original, corrected, valid, gain, offset in (
            (a, corrected_a, va, interpolated[line, 0], interpolated[line, 1]),
            (b, corrected_b, vb, interpolated[line, 2], interpolated[line, 3]),
        ):
            source = np.asarray(original[slot], dtype=np.float64)
            active = valid[slot] & np.isfinite(source)
            if active.any():
                # Frozen Task13A symmetric affine correction and cosine taper.
                with np.errstate(over="ignore", invalid="ignore"):
                    candidate = source[active] + weights[active] * ((gain - 1.0) * source[active] + offset)
                if not np.isfinite(candidate).all():
                    raise ValueError("saved local correction yielded nonfinite valid pixels")
                corrected[slot][active] = candidate
    return corrected_a, corrected_b


def _footprint_polygon(mask: np.ndarray, transform: Affine) -> BaseGeometry:
    """Compatibility wrapper for the shared frozen footprint semantics."""
    return footprint_polygon_from_valid_mask(mask, transform)


def _write_mosaic(path: Path, mosaic: np.ndarray, valid: np.ndarray,
                  crop_origin: tuple[int, int], grid: dict) -> None:
    if not np.isfinite(mosaic[valid]).all():
        raise ValueError(f"nonfinite valid mosaic pixels: {path}")
    transform = Affine(*grid["transform"]) * Affine.translation(crop_origin[1], crop_origin[0])
    with rasterio.open(path, "w", driver="GTiff", width=mosaic.shape[1], height=mosaic.shape[0],
                       count=1, dtype="float32", crs=grid["crs"], transform=transform,
                       nodata=np.nan, compress="deflate", predictor=3, tiled=True) as dst:
        for first in range(0, mosaic.shape[0], 256):
            last = min(mosaic.shape[0], first + 256)
            tile = np.asarray(mosaic[first:last], dtype=np.float32).copy()
            tile[~valid[first:last]] = np.nan
            if not np.isfinite(tile[valid[first:last]]).all():
                raise ValueError(f"float32 mosaic has nonfinite valid pixels: {path}")
            dst.write(tile, 1, window=Window(0, first, mosaic.shape[1], last - first))


def _read_csv(path: Path) -> dict[str, dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return {row["pair_id"]: row for row in csv.DictReader(stream)}


def _geometry_class(result, valid_a: np.ndarray, valid_b: np.ndarray) -> str:
    if result.status in {"EXCLUSIVE_CONTACT_RESOLVABLE", "CENTROID_RESOLVABLE"}:
        return result.status
    if result.diagnostics.get("nested_or_contained_footprints") or result.diagnostics.get("identical_footprints"):
        return "CONTAINMENT_OR_NESTED_FOOTPRINT"
    a_only = int(np.count_nonzero(valid_a & ~valid_b))
    b_only = int(np.count_nonzero(valid_b & ~valid_a))
    return "CONTAINMENT_OR_NESTED_FOOTPRINT" if min(a_only, b_only) == 0 else "TRUE_GEOMETRIC_AMBIGUITY"


def _replay_pair(pair: dict, protocol: dict, metrics: dict) -> dict:
    pair_id = pair["pair_id"]
    originals = {name: TASK13A / "pairs" / pair_id / name for name in SAVED_FILES}
    before = {name: _sha256(path) for name, path in originals.items()}
    scene_paths = [ROOT / protocol["scenes"][index]["path"] for index in (pair["scene_i"], pair["scene_j"])]
    with rasterio.open(scene_paths[0]) as src_a, rasterio.open(scene_paths[1]) as src_b:
        va_full, vb_full = src_a.read_masks(1) > 0, src_b.read_masks(1) > 0
        outer, _, overlap_origin = _crop_for_pair(None, None, va_full, vb_full)
        origin = (outer[0].start, outer[1].start)
        recorded_origin = tuple(json.loads(metrics["diagnostic_crop_origin"]))
        if origin != recorded_origin:
            raise ValueError(f"frozen crop origin mismatch for {pair_id}: {origin} != {recorded_origin}")
        height, width = outer[0].stop - outer[0].start, outer[1].stop - outer[1].start
        window = Window(outer[1].start, outer[0].start, width, height)
        a, b = src_a.read(1, window=window), src_b.read(1, window=window)
        va, vb = va_full[outer], vb_full[outer]
        transform = src_a.transform
        footprint_a = _footprint_polygon(va_full, transform)
        footprint_b = _footprint_polygon(vb_full, transform)
    del va_full, vb_full
    if int(np.count_nonzero(va | vb)) != int(metrics["union_valid_pixels"]):
        raise ValueError(f"frozen crop union count mismatch for {pair_id}")
    if not np.isfinite(a[va]).all() or not np.isfinite(b[vb]).all():
        raise ValueError(f"nonfinite frozen input for {pair_id}")
    initial = _load_saved_seam(originals["seam_initial.geojson"], origin, transform, a.shape)
    refined = _load_saved_seam(originals["seam_refined.geojson"], origin, transform, a.shape)
    coefficients = _load_coefficients(originals["local_coefficients.csv"])
    initial_path, refined_path = initial.row_col_path.copy(), refined.row_col_path.copy()
    # The resolver receives the shared-valid corridor; exclusive support is
    # read from the registered masks outside that corridor.
    shared_overlap = va & vb
    initial_side = resolve_source_sides(initial, shared_overlap, va, vb, footprint_a, footprint_b)
    refined_side = resolve_source_sides(refined, shared_overlap, va, vb, footprint_a, footprint_b)
    side_agreement = (initial_side.side_1_source == refined_side.side_1_source
                      and initial_side.side_2_source == refined_side.side_2_source)
    resolved = (initial_side.status in {"EXCLUSIVE_CONTACT_RESOLVABLE", "CENTROID_RESOLVABLE"}
                and refined_side.status in {"EXCLUSIVE_CONTACT_RESOLVABLE", "CENTROID_RESOLVABLE"}
                and side_agreement)
    pair_dir = OUTPUT / "pairs" / pair_id
    pair_dir.mkdir(parents=True, exist_ok=False)
    for name, path in originals.items():
        shutil.copyfile(path, pair_dir / name)
        if _sha256(pair_dir / name) != before[name]:
            raise ValueError(f"saved artifact copy changed: {pair_id}/{name}")
    new_status = "REQUIRES_MULTISCENE_LABELING"
    finite_v1 = finite_v2 = 0
    if resolved:
        v1 = blend_across_seam(a, b, initial, va, vb, source_side=initial_side)
        corrected_a, corrected_b = replay_saved_local_correction(a, b, va, vb, initial, coefficients)
        v2 = blend_across_seam(corrected_a, corrected_b, refined, va, vb, source_side=refined_side)
        if v1.status == v2.status == "OK":
            valid = va | vb
            _write_mosaic(pair_dir / "v1_seam_only.tif", v1.image, valid, origin, protocol["grid"])
            _write_mosaic(pair_dir / "v2_local_refined.tif", v2.image, valid, origin, protocol["grid"])
            finite_v1 = int(np.count_nonzero(np.isfinite(v1.image) & valid))
            finite_v2 = int(np.count_nonzero(np.isfinite(v2.image) & valid))
            new_status = "PASS"
        else:
            new_status = "NUMERICAL_INVALID"
    if not np.array_equal(initial.row_col_path, initial_path) or not np.array_equal(refined.row_col_path, refined_path):
        raise ValueError(f"replay changed saved seam path: {pair_id}")
    after = {name: _sha256(path) for name, path in originals.items()}
    if before != after:
        raise ValueError(f"replay changed Task13A artifacts: {pair_id}")
    diagnostics = refined_side.diagnostics
    return {
        "pair_id": pair_id, "old_status": metrics["status"],
        "geometry_class": _geometry_class(refined_side, va, vb),
        "orientation": refined.orientation,
        "exclusive_contact_a_side1": diagnostics["exclusive_contact_a_side1"],
        "exclusive_contact_a_side2": diagnostics["exclusive_contact_a_side2"],
        "exclusive_contact_b_side1": diagnostics["exclusive_contact_b_side1"],
        "exclusive_contact_b_side2": diagnostics["exclusive_contact_b_side2"],
        "assignment_score_ab": diagnostics["assignment_score_ab"],
        "assignment_score_ba": diagnostics["assignment_score_ba"],
        "centroid_a": json.dumps(diagnostics.get("centroid_a", (footprint_a.centroid.x, footprint_a.centroid.y))),
        "centroid_b": json.dumps(diagnostics.get("centroid_b", (footprint_b.centroid.x, footprint_b.centroid.y))),
        "projection_separation": diagnostics.get("projection_separation"),
        "resolver_method": refined_side.method,
        "initial_resolver_status": initial_side.status,
        "initial_resolver_method": initial_side.method,
        "initial_side1_source": initial_side.side_1_source,
        "initial_side2_source": initial_side.side_2_source,
        "refined_resolver_status": refined_side.status,
        "assignment_agreement": side_agreement,
        "new_status": new_status,
        "side1_source": refined_side.side_1_source if resolved else None,
        "side2_source": refined_side.side_2_source if resolved else None,
        "union_valid_pixels": int(np.count_nonzero(va | vb)),
        "v1_finite_valid_pixels": finite_v1,
        "v2_finite_valid_pixels": finite_v2,
        "saved_seam_and_coefficients_sha256": json.dumps(before, sort_keys=True),
        "artifact_hashes_unchanged": before == after,
        "seam_paths_unchanged": True,
        "local_coefficients_byte_equal": _sha256(pair_dir / "local_coefficients.csv") == before["local_coefficients.csv"],
    }


def run_replay() -> dict:
    if OUTPUT.exists():
        raise FileExistsError(f"Task13A.1 output exists; refusing to overwrite: {OUTPUT}")
    protocol_path = TASK13A / "protocol.json"
    manifest_path = TASK13A / "run_manifest.json"
    aggregate_path = TASK13A / "aggregate_summary.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    if _sha256(protocol_path) != manifest["protocol_sha256"] or manifest["status"] != "COMPLETED":
        raise ValueError("frozen Task13A protocol or run manifest mismatch")
    for scene in protocol["scenes"]:
        if _sha256(ROOT / scene["path"]) != scene["sha256"]:
            raise ValueError(f"frozen BAGRN raster hash mismatch: {scene['path']}")
    metrics_path = TASK13A / "pair_metrics.csv"
    if _sha256(metrics_path) != aggregate["input_sha256"]["pair_metrics_csv"]:
        raise ValueError("frozen Task13A metrics hash mismatch")
    metrics = _read_csv(metrics_path)
    pair_lookup = {pair["pair_id"]: pair for pair in protocol["accepted_pairs"]}
    if any(pair_id not in pair_lookup or metrics[pair_id]["status"] != "AMBIGUOUS_SOURCE_SIDE"
           for pair_id in PAIR_IDS):
        raise ValueError("frozen ambiguous-pair list changed")
    for pair_id in PAIR_IDS:
        for name in SAVED_FILES:
            path = TASK13A / "pairs" / pair_id / name
            if not path.is_file():
                raise FileNotFoundError(path)
        expected = aggregate["input_sha256"]["local_coefficients_csv_by_pair"][pair_id]
        if _sha256(TASK13A / "pairs" / pair_id / "local_coefficients.csv") != expected:
            raise ValueError(f"frozen coefficient hash mismatch: {pair_id}")
    OUTPUT.mkdir(parents=True, exist_ok=False)
    rows = [_replay_pair(pair_lookup[pair_id], protocol, metrics[pair_id]) for pair_id in PAIR_IDS]
    audit_path = OUTPUT / "source_side_audit.csv"
    with audit_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    old_pass = int(aggregate["pair_count_v1_pass"])
    new_pass = old_pass + sum(row["new_status"] == "PASS" for row in rows)
    unresolved = sum(row["new_status"] == "REQUIRES_MULTISCENE_LABELING" for row in rows)
    other_gates = {key: value for key, value in aggregate["gate"]["conditions"].items()
                   if key not in {"v1_at_least_8_of_10_pass", "v2_at_least_8_of_10_pass"}}
    decision = ("PROMISING_FOR_MULTISCENE" if new_pass >= 8 and all(other_gates.values())
                and all(row["artifact_hashes_unchanged"] and row["seam_paths_unchanged"]
                        and row["local_coefficients_byte_equal"] for row in rows)
                else "MIXED_NEEDS_REVIEW")
    summary = {
        "task": "Task13A.1 geometry-only source-side replay",
        "processed_pair_ids": list(PAIR_IDS),
        "old_pass_pairs_of_10": old_pass,
        "new_pass_pairs_of_10": new_pass,
        "requires_multiscene_labeling": unresolved,
        "still_ambiguous_source_side": sum(row["new_status"] == "AMBIGUOUS_SOURCE_SIDE" for row in rows),
        "decision": decision,
        "frozen_protocol_sha256": _sha256(protocol_path),
        "frozen_pair_metrics_sha256": _sha256(metrics_path),
        "radiometric_gate_conditions_reused_without_recomputation": other_gates,
        "all_new_mosaics_finite": all(row["v1_finite_valid_pixels"] == row["union_valid_pixels"]
                                  and row["v2_finite_valid_pixels"] == row["union_valid_pixels"]
                                  for row in rows if row["new_status"] == "PASS"),
        "frozen_structural_diagnostics_reused": True,
        "task13a_outputs_overwritten": False,
        "pairs": [{"pair_id": row["pair_id"], "geometry_class": row["geometry_class"],
                   "new_status": row["new_status"], "resolver_method": row["resolver_method"]}
                  for row in rows],
    }
    (OUTPUT / "source_side_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    lines = ["# Task13A.1 source-side resolution", "",
             f"Frozen Task13A V1/V2 PASS: {old_pass}/10. Geometry-only replay: {new_pass}/10 PASS; "
             f"{unresolved} require multiscene labeling. Decision: `{decision}`.", "",
             "The four old failures came from the blend's exclusive-support side rule: both "
             "assignments were supported or neither was supported. Saved seams and local "
             "coefficients were already available.", "",
             "| Pair | Geometry class | Refined evidence | New status | Side 1 | Side 2 |",
             "|---|---|---|---|---|---|"]
    for row in rows:
        lines.append(f"| {row['pair_id']} | {row['geometry_class']} | {row['resolver_method']} | "
                     f"{row['new_status']} | {row['side1_source'] or 'N/A'} | {row['side2_source'] or 'N/A'} |")
    lines += ["", "Ownership used only valid-mask boundary contact and registered valid-footprint "
              "centroids. No radiometric metric selected an assignment. Initial and refined "
              "assignments had to agree before composition.", "",
              "Each pair directory contains byte-identical copies of the saved seams and coefficient "
              "CSV. SHA-256 checks before and after replay and pixel-path equality checks passed. "
              "New mosaics contain finite values at every valid pixel.", "",
              "Task13B has not been started. This audit leaves the frozen Task13A outputs untouched.", ""]
    (OUTPUT / "TASK13A1_SOURCE_SIDE_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(run_replay(), indent=2))
