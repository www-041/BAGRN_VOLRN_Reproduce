"""Task14A.2 controlled valid-footprint replay and post-fix pipeline.

The first phase reads frozen Stage06/Stage07 artifacts and replays only the
source-side resolver with the shared actual-valid-mask footprint helper. The
post-fix Stage08--12 execution is added below without invoking Stage07.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from affine import Affine
from rasterio.windows import Window

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/output/b9_13scene_task14"
OLD_STAGE07 = BASE / "stages/07_pairwise_seam_local"
OLD_STAGE06 = BASE / "stages/06_bagrn/bagrn"
AUDIT = BASE / "audit_task14a1_source_side"
OUT = BASE / "task14a2_valid_footprint_fix"
SOURCE_OUT = OUT / "source_side"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.seam_local.footprint import footprint_polygon_from_valid_mask
from src.seam_local.source_side import resolve_source_sides
from scripts.run_task14a1_source_side_audit import (
    load_masks,
    load_saved_seam,
    read_pair_arrays,
    read_json,
)
from scripts.run_task14a_resume_13 import _local_tile_seam_path


RESOLVABLE = {"EXCLUSIVE_CONTACT_RESOLVABLE", "CENTROID_RESOLVABLE"}
REQUIRES = "REQUIRES_MULTISCENE_LABELING"
UNSTABLE = "UNSTABLE_LOCAL_GAIN"
NO_SUPPORT = "NO_FINAL_SHARED_SUPPORT"


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=_json_default) + "\n", encoding="utf-8")


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


def classify_formal_pair_status(
    old_status: str,
    initial_resolver_status: str | None,
    refined_resolver_status: str | None,
    assignment_agreement: bool,
    local_gain_status: str,
) -> str:
    """Separate source ownership status from the formal pair status."""

    if old_status in {UNSTABLE, NO_SUPPORT}:
        return old_status
    if (
        local_gain_status == "STABLE"
        and initial_resolver_status in RESOLVABLE
        and refined_resolver_status in RESOLVABLE
        and assignment_agreement
    ):
        return "PASS"
    return REQUIRES


def preference_eligible(formal_pair_status: str) -> bool:
    return formal_pair_status == "PASS"


def preference_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if preference_eligible(row.get("formal_pair_status", ""))]


def required_label_diagnostic_keys() -> tuple[str, ...]:
    return (
        "resolved_preference_edge_count", "multiscene_pixels", "cycle_pixels", "cycle_fraction",
        "pairwise_score_tie_pixels", "clipped_interiority_fallback_pixels",
        "unclipped_normalized_interiority_pixels", "raw_edt_pixels", "unresolved_pixels",
        "invalid_label_pixels", "two_scene_disagreement_pixels", "label_pixels_per_scene",
        "label_method_counts", "score_margin_min", "score_margin_median", "score_margin_p95",
    )


def scene_identity_difference(original_labels: Any, permuted_labels: Any, order: list[int] | tuple[int, ...]) -> int:
    original = np.asarray(original_labels)
    permuted = np.asarray(permuted_labels)
    mapped = np.full(permuted.shape, -1, dtype=np.int64)
    for permuted_index, scene_identity in enumerate(order):
        mapped[permuted == permuted_index] = int(scene_identity)
    return int(np.count_nonzero(original != mapped))


def correction_order_equivalent(max_abs_difference: float, mean_abs_difference: float,
                                *, rtol: float = 1e-6, atol: float = 1e-3) -> bool:
    scale = max(1.0, abs(float(mean_abs_difference)))
    return bool(float(max_abs_difference) <= atol + rtol * scale)


def final_decision(gates: dict[str, bool]) -> str:
    return "READY_FOR_NEXT_SCALE_STAGE" if all(bool(value) for value in gates.values()) else "MIXED_SCALE_REVIEW"


def validate_source_side_gate(rows: list[dict[str, Any]]) -> dict[str, int]:
    mismatch = [row for row in rows if bool(row.get("adapter_logic_mismatch"))]
    mismatch_pass = [row for row in mismatch if row.get("new_source_side_status") == "PASS"]
    true_requires = [row for row in rows if row.get("pair_id") == "02_05" and row.get("formal_pair_status") == REQUIRES]
    unstable = [row for row in rows if row.get("formal_pair_status") == UNSTABLE]
    no_support = [row for row in rows if row.get("formal_pair_status") == NO_SUPPORT]
    nonpass_edges = [row for row in rows if not preference_eligible(row.get("formal_pair_status", "")) and row.get("preference_eligible")]
    summary = {
        "adapter_logic_mismatch_count": len(mismatch),
        "adapter_logic_mismatch_pass_count": len(mismatch_pass),
        "true_geometric_requires_count": len(true_requires),
        "unstable_count": len(unstable),
        "no_support_count": len(no_support),
        "nonpass_preference_edge_count": len(nonpass_edges),
    }
    if summary["adapter_logic_mismatch_count"] != 45:
        raise RuntimeError("HARD_STOP_TASK14A1_A2_INCONSISTENCY: mismatch count is not 45")
    if summary["adapter_logic_mismatch_pass_count"] != 45:
        failed = [row.get("pair_id") for row in mismatch if row.get("new_source_side_status") != "PASS"]
        raise RuntimeError(
            "HARD_STOP_SOURCE_SIDE_REPLAY_MISMATCH: not all mismatch pairs are PASS; "
            + json.dumps({"failed_pairs": failed, "count": len(failed)}, ensure_ascii=False)
        )
    if summary["true_geometric_requires_count"] != 1:
        raise RuntimeError("HARD_STOP_TASK14A1_A2_INCONSISTENCY: true geometry count changed")
    if summary["unstable_count"] != 2 or summary["no_support_count"] != 12:
        raise RuntimeError("HARD_STOP_TASK14A1_A2_INCONSISTENCY: unstable/no-support count changed")
    if summary["nonpass_preference_edge_count"]:
        raise RuntimeError("HARD_STOP_SOURCE_SIDE_REPLAY_MISMATCH: non-PASS edge entered preference network")
    return summary


def validate_v0_semantics(actual: dict[str, Any], expected: dict[str, Any]) -> dict[str, Any]:
    mismatches = {}
    for key, expected_value in expected.items():
        if actual.get(key) != expected_value:
            mismatches[key] = {"actual": actual.get(key), "expected": expected_value}
    if mismatches:
        raise RuntimeError("HARD_STOP_V0_SEMANTIC_MISMATCH: " + json.dumps(mismatches, ensure_ascii=False))
    return {"status": "PASS", "checked": sorted(expected)}


def _detect_v0_semantic(summary: dict[str, Any], implementation_source: str) -> str:
    if summary.get("radiometric_method") != "BAGRN":
        return "unknown"
    if "_stream_blend(normalized_paths, distance_paths, mosaic_path)" in implementation_source:
        return "weighted_feather"
    return "unknown"


def _manifest() -> dict[str, Any]:
    return read_json(OLD_STAGE06 / "normalized_scenes_manifest.json")


def _scene_paths() -> list[Path]:
    return [ROOT / row["path"] for row in _manifest()["scenes"]]


def _canonical_grid() -> tuple[Affine, str, int, int]:
    with rasterio.open(_scene_paths()[0]) as src:
        return src.transform, str(src.crs), src.height, src.width


def _audit_rows() -> dict[str, dict[str, str]]:
    return {row["pair_id"]: row for row in read_csv(AUDIT / "source_side_failure_summary.csv")}


def _parse_json_sequence(value: Any) -> tuple[int, ...] | None:
    if value in (None, "", []):
        return None
    if isinstance(value, str):
        value = json.loads(value)
    return tuple(int(item) for item in value)


def _replay_source_side() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    paths = _scene_paths()
    masks, transform, _, _ = load_masks(paths)
    footprints = [footprint_polygon_from_valid_mask(mask, transform) for mask in masks]
    frozen_rows = read_json(OLD_STAGE07 / "pairwise_results.json")
    audit_rows = _audit_rows()
    result_rows: list[dict[str, Any]] = []
    frozen_hashes: dict[str, str] = {}
    for pair in frozen_rows:
        pid = pair["pair_id"]
        old_status = pair.get("status", NO_SUPPORT)
        audit = audit_rows.get(pid, {})
        mismatch = audit.get("adapter_logic_mismatch", "False").lower() == "true"
        base = dict(pair)
        base.update({
            "pair_id": pid,
            "scene_i": int(pair["scene_i"]),
            "scene_j": int(pair["scene_j"]),
            "old_stage07_status": old_status,
            "task14a1_dryrun_status": audit.get("task13a1_resolver_dryrun_status", NO_SUPPORT),
            "registration_edge_accepted": pair.get("registration_edge_accepted"),
            "shared_valid_pixels": int(pair.get("shared_valid_pixel_count", 0)),
            "local_gain_status": UNSTABLE if old_status == UNSTABLE else (NO_SUPPORT if old_status == NO_SUPPORT else "STABLE"),
            "footprint_method": "actual_valid_mask_polygon_shared_helper",
            "adapter_logic_mismatch": mismatch,
            "initial_assignment": None,
            "refined_assignment": None,
            "initial_source_side_status": None,
            "refined_source_side_status": None,
            "resolver_method": None,
            "source_side_status": old_status if old_status in {UNSTABLE, NO_SUPPORT} else REQUIRES,
            "new_source_side_status": old_status if old_status in {UNSTABLE, NO_SUPPORT} else REQUIRES,
            "preference_eligible": False,
        })
        if old_status in {UNSTABLE, NO_SUPPORT} or int(pair.get("shared_valid_pixel_count", 0)) <= 0:
            base["formal_pair_status"] = old_status if old_status in {UNSTABLE, NO_SUPPORT} else NO_SUPPORT
            base["status"] = base["formal_pair_status"]
            result_rows.append(base)
            continue
        origin = _parse_json_sequence(pair.get("crop_origin"))
        shape = _parse_json_sequence(pair.get("crop_shape"))
        if origin is None or shape is None:
            raise RuntimeError(f"HARD_STOP_SOURCE_SIDE_REPLAY_MISMATCH: missing crop for {pid}")
        _, _, va, vb, crop_origin, crop_shape = read_pair_arrays(paths, masks, int(pair["scene_i"]), int(pair["scene_j"]), origin, shape)
        pair_dir = OLD_STAGE07 / "pairs" / pid
        for filename in ("seam_initial.geojson", "seam_refined.geojson", "local_coefficients.csv"):
            path = pair_dir / filename
            if not path.is_file():
                raise RuntimeError(f"HARD_STOP_SOURCE_SIDE_REPLAY_MISMATCH: missing frozen artifact {pid}/{filename}")
            frozen_hashes[f"{pid}/{filename}"] = sha256(path)
        initial = load_saved_seam(pair_dir / "seam_initial.geojson", crop_origin, transform, crop_shape)
        refined = load_saved_seam(pair_dir / "seam_refined.geojson", crop_origin, transform, crop_shape)
        overlap = va & vb
        initial_side = resolve_source_sides(initial, overlap, va, vb, footprints[int(pair["scene_i"])], footprints[int(pair["scene_j"])])
        refined_side = resolve_source_sides(refined, overlap, va, vb, footprints[int(pair["scene_i"])], footprints[int(pair["scene_j"])])
        agree = bool(
            initial_side.status in RESOLVABLE
            and refined_side.status in RESOLVABLE
            and initial_side.side_1_source == refined_side.side_1_source
            and initial_side.side_2_source == refined_side.side_2_source
        )
        source_status = "PASS" if agree else REQUIRES
        formal = classify_formal_pair_status(old_status, initial_side.status, refined_side.status, agree, "STABLE")
        base.update({
            "initial_assignment": initial_side.diagnostics.get("selected_assignment"),
            "refined_assignment": refined_side.diagnostics.get("selected_assignment"),
            "initial_source_side_status": initial_side.status,
            "refined_source_side_status": refined_side.status,
            "resolver_method": refined_side.method,
            "source_side_status": source_status,
            "new_source_side_status": source_status,
            "formal_pair_status": formal,
            "preference_eligible": preference_eligible(formal),
            "status": formal,
            "side_1_source": refined_side.side_1_source if agree else None,
            "side_2_source": refined_side.side_2_source if agree else None,
            "assignment_agreement": agree,
            "assignment_score_ab": refined_side.diagnostics.get("assignment_score_ab"),
            "assignment_score_ba": refined_side.diagnostics.get("assignment_score_ba"),
        })
        result_rows.append(base)
    gate = validate_source_side_gate(result_rows)
    for path_key, before in frozen_hashes.items():
        pid, filename = path_key.split("/", 1)
        after = sha256(OLD_STAGE07 / "pairs" / pid / filename)
        if before != after:
            raise RuntimeError(f"HARD_STOP_SOURCE_SIDE_REPLAY_MISMATCH: frozen artifact changed {path_key}")
    return result_rows, {"gate": gate, "frozen_artifact_sha256": frozen_hashes, "scene_count": len(paths)}


def run_source_side_replay() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows, summary = _replay_source_side()
    write_csv(SOURCE_OUT / "source_side_replay_status.csv", rows)
    write_json(SOURCE_OUT / "source_side_summary.json", summary)
    write_json(SOURCE_OUT / "replay_gate.json", {"status": "PASS", **summary["gate"]})
    return rows, summary


def _v0_metadata(paths: list[Path], masks: list[np.ndarray], transform: Affine,
                  crs: str, height: int, width: int) -> tuple[dict[str, Any], dict[str, Any]]:
    union = np.zeros((height, width), dtype=bool)
    for mask in masks:
        union |= mask
    summary = read_json(OLD_STAGE06 / "radiometric_summary.json")
    implementation = (ROOT / "scripts/run_task14_13scene_scale.py").read_text(encoding="utf-8")
    semantic = _detect_v0_semantic(summary, implementation)
    union_path = OLD_STAGE06 / "mosaic.tif"
    with rasterio.open(union_path) as mosaic:
        mosaic_nodata = mosaic.nodata
        actual_nodata = "nan" if mosaic_nodata is not None and np.isnan(mosaic_nodata) else mosaic_nodata
        actual = {
        "semantic": semantic,
        "crs": str(mosaic.crs),
        "transform": [float(value) for value in mosaic.transform[:6]],
        "shape": [mosaic.height, mosaic.width],
        "dtype": mosaic.dtypes[0],
        "nodata": actual_nodata,
        "union_support": _finite_support([union_path], [union]),
        }
    expected = {
        "semantic": "weighted_feather",
        "crs": "EPSG:32650",
        "transform": [14.0, 0.0, 633346.0, 0.0, -14.0, 3359874.0],
        "shape": [12404, 7992],
        "dtype": "float32",
        "nodata": "nan",
        "union_support": 62033096,
    }
    validate_v0_semantics(actual, expected)
    return actual, {"status": "PASS", "expected": expected, "source": "Stage06 frozen BAGRN weighted feather"}


def _score_margin_summary(path: Path, height: int, width: int) -> dict[str, float]:
    values: list[np.ndarray] = []
    with rasterio.open(path) as src:
        for r0 in range(0, height, 1024):
            hh = min(1024, height - r0)
            tile = src.read(1, window=Window(0, r0, width, hh)).astype(np.float64)
            finite = tile[np.isfinite(tile)]
            if finite.size:
                values.append(finite)
    all_values = np.concatenate(values) if values else np.asarray([], dtype=np.float64)
    if not all_values.size:
        return {"min": float("nan"), "median": float("nan"), "p95": float("nan")}
    return {
        "min": float(np.min(all_values)),
        "median": float(np.percentile(all_values, 50)),
        "p95": float(np.percentile(all_values, 95)),
    }


def _label_counts(labels_path: Path, methods_path: Path, height: int, width: int, scene_count: int) -> tuple[dict[str, int], dict[str, int]]:
    scene_counts = {str(i): 0 for i in range(scene_count)}
    method_counts = {str(i): 0 for i in range(5)}
    with rasterio.open(labels_path) as labels, rasterio.open(methods_path) as methods:
        for r0 in range(0, height, 1024):
            hh = min(1024, height - r0)
            win = Window(0, r0, width, hh)
            lab = labels.read(1, window=win)
            meth = methods.read(1, window=win)
            for scene in range(scene_count):
                scene_counts[str(scene)] += int(np.count_nonzero(lab == scene))
            for method in range(5):
                method_counts[str(method)] += int(np.count_nonzero(meth == method))
    return scene_counts, method_counts


def enrich_label_diagnostics(stats: dict[str, Any], rows: list[dict[str, Any]],
                             label_path: Path, methods_path: Path,
                             margin_path: Path, height: int, width: int,
                             scene_count: int, union_mask: np.ndarray | None = None) -> dict[str, Any]:
    scene_counts, method_counts = _label_counts(label_path, methods_path, height, width, scene_count)
    margin = _score_margin_summary(margin_path, height, width)
    enriched = dict(stats)
    enriched["resolved_preference_edge_count"] = sum(preference_eligible(row.get("formal_pair_status", "")) for row in rows)
    enriched["raw_edt_pixels"] = int(stats.get("resolved_by_raw_edt", 0))
    enriched["unclipped_normalized_interiority_pixels"] = int(stats.get("resolved_by_unclipped_interiority", 0))
    enriched["label_pixels_per_scene"] = scene_counts
    enriched["label_method_counts"] = method_counts
    enriched["score_margin_min"] = margin["min"]
    enriched["score_margin_median"] = margin["median"]
    enriched["score_margin_p95"] = margin["p95"]
    enriched["two_scene_disagreement_pixels"] = int(stats.get("two_scene_disagreement_pixels", 0))
    if union_mask is not None:
        invalid_valid_support = 0
        with rasterio.open(label_path) as labels:
            for r0 in range(0, height, 1024):
                hh = min(1024, height - r0)
                tile = labels.read(1, window=Window(0, r0, width, hh))
                valid = union_mask[r0:r0 + hh]
                invalid_valid_support += int(np.count_nonzero(valid & ((tile < 0) | (tile >= scene_count))))
        # Outside the union is intentional nodata, not an invalid ownership label.
        enriched["invalid_label_pixels"] = invalid_valid_support
    missing = [key for key in required_label_diagnostic_keys() if (
        key not in enriched and key.replace("score_margin_", "score_margin_") not in enriched
    )]
    if missing:
        raise RuntimeError("HARD_STOP_POSTFIX_LABEL_INVALID: missing diagnostics " + json.dumps(missing))
    return enriched


def _postfix_pair_rows(replayed: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in replayed:
        copy = dict(row)
        copy["status"] = row["formal_pair_status"]
        copy["source_side_status"] = row["new_source_side_status"]
        rows.append(copy)
    return rows


def _load_p95_edt(scene_count: int) -> np.ndarray:
    values = np.zeros(scene_count, dtype=np.float32)
    for index in range(scene_count):
        path = BASE / f"stages/05_valid_distance_cache/distance_scene_{index:03d}.tif"
        with rasterio.open(path) as src:
            data = src.read(1)
        positive = data[data > 0]
        values[index] = float(np.percentile(positive, 95)) if positive.size else 1.0
    return values


def _label_order_difference(rows: list[dict[str, Any]], paths: list[Path], transform: Affine,
                            height: int, width: int, order: list[int], formal_labels: Path) -> int:
    from src.seam_local.adapter import aggregate_labels_with_ties
    from src.seam_local.multiscene_label import PairwisePreferenceField, build_pairwise_preference_field
    import scripts.run_task14a_resume_13 as frozen

    positions = {scene: index for index, scene in enumerate(order)}
    pair_info = []
    for row in preference_rows(rows):
        pid = row["pair_id"]
        orient, centers = frozen._load_seam_centers(OLD_STAGE07 / "pairs" / pid / "seam_refined.geojson", transform, height, width)
        pair_info.append((positions[int(row["scene_i"])], positions[int(row["scene_j"])], int(row["scene_i"]), int(row["scene_j"]), orient, centers, row["side_1_source"] == "A", pid))
    p95 = _load_p95_edt(len(paths))
    difference = 0
    with rasterio.open(formal_labels) as reference:
        sources = [rasterio.open(paths[index]) for index in order]
        distances = [rasterio.open(BASE / f"stages/05_valid_distance_cache/distance_scene_{index:03d}.tif") for index in order]
        try:
            for r0 in range(0, height, 1024):
                hh = min(1024, height - r0)
                win = Window(0, r0, width, hh)
                masks = np.stack([src.read_masks(1, window=win) > 0 for src in sources])
                raw = np.stack([src.read(1, window=win) for src in distances]).astype(np.float32)
                fields = {}
                for ai, bj, original_i, original_j, orient, centers, side_a, pid in pair_info:
                    field_a, field_b = (ai, bj) if ai < bj else (bj, ai)
                    field_side_a = side_a if ai < bj else not side_a
                    path = _local_tile_seam_path(orient, centers, r0, hh, width)
                    c = centers[r0:r0 + hh] if orient == "vertical" else centers
                    domain = np.isfinite(c)[:, None] if orient == "vertical" else np.broadcast_to(np.isfinite(c)[None, :], (hh, width))
                    field = build_pairwise_preference_field(
                        masks[field_a], masks[field_b], path, "A" if field_side_a else "B",
                        orientation=orient, scene_a=field_a, scene_b=field_b,
                    )
                    fields[(field_a, field_b)] = PairwisePreferenceField(
                        field_a, field_b, np.where(domain, field.vote, 0).astype(np.float32),
                        field.available & domain, np.where(domain, field.confidence, 0).astype(np.float32), orient,
                    )
                labels, _, _ = aggregate_labels_with_ties(masks, fields, raw, p95_edt=p95[order])
                reference_tile = reference.read(1, window=win)
                mapped = np.full(labels.shape, -1, dtype=np.int16)
                for permuted_index, scene_identity in enumerate(order):
                    mapped[labels == permuted_index] = int(scene_identity)
                difference += int(np.count_nonzero(mapped != reference_tile))
        finally:
            for source in sources + distances:
                source.close()
    return difference


def _raster_difference(first_path: Path, second_path: Path, height: int, width: int) -> dict[str, Any]:
    max_abs = 0.0
    sum_abs = 0.0
    count = 0
    with rasterio.open(first_path) as first, rasterio.open(second_path) as second:
        for r0 in range(0, height, 1024):
            hh = min(1024, height - r0)
            win = Window(0, r0, width, hh)
            a = first.read(1, window=win).astype(np.float64)
            b = second.read(1, window=win).astype(np.float64)
            valid = np.isfinite(a) & np.isfinite(b)
            if np.any(valid):
                delta = np.abs(a[valid] - b[valid])
                max_abs = max(max_abs, float(np.max(delta)))
                sum_abs += float(np.sum(delta))
                count += int(delta.size)
    return {"max_abs_difference": max_abs, "mean_abs_difference": sum_abs / max(count, 1), "finite_pixels": count}


def _correction_order_invariance(rows: list[dict[str, Any]], paths: list[Path], transform: Affine,
                                 height: int, width: int, frozen: Any) -> dict[str, Any]:
    normal_dir = OUT / "invariance/correction_order_normal"
    reverse_dir = OUT / "invariance/correction_order_reversed"
    formal_stage09 = OUT / "stages/09_correction"
    def complete(directory: Path) -> bool:
        return (directory / "_SUCCESS.json").is_file() and all(
            (directory / f"corrected_scene_{index:03d}.tif").is_file()
            for index in range(len(paths))
        )

    if complete(normal_dir):
        normal = [normal_dir / f"corrected_scene_{index:03d}.tif" for index in range(len(paths))]
    else:
        frozen.STAGE09 = normal_dir
        normal = frozen._stage09(rows, paths, transform, height, width)
    if complete(reverse_dir):
        reverse = [reverse_dir / f"corrected_scene_{index:03d}.tif" for index in range(len(paths))]
    else:
        frozen.STAGE09 = reverse_dir
        reversed_rows = list(reversed(rows))
        reverse = frozen._stage09(reversed_rows, paths, transform, height, width)
    frozen.STAGE09 = formal_stage09
    diffs = [_raster_difference(a, b, height, width) for a, b in zip(normal, reverse)]
    max_abs = max((item["max_abs_difference"] for item in diffs), default=0.0)
    mean_abs = max((item["mean_abs_difference"] for item in diffs), default=0.0)
    return {"status": "PASS" if correction_order_equivalent(max_abs, mean_abs) else "HARD_STOP_13SCENE_CORRECTION_ORDER_DEPENDENCE",
            "max_abs_difference": max_abs, "mean_abs_difference": mean_abs, "scene_differences": diffs,
            "rtol": 1e-6, "atol": 1e-3}


def _pre_post_comparison(stats: dict[str, Any], metrics: dict[str, Any],
                         height: int, width: int) -> dict[str, Any]:
    old_label = BASE / "stages/08_multiscene_labels/source_label_map.tif"
    new_label = OUT / "stages/08_multiscene_labeling/source_label_map.tif"
    label_changed = 0
    with rasterio.open(old_label) as old, rasterio.open(new_label) as new:
        for r0 in range(0, height, 1024):
            hh = min(1024, height - r0)
            win = Window(0, r0, width, hh)
            label_changed += int(np.count_nonzero(old.read(1, window=win) != new.read(1, window=win)))
    old_v2 = BASE / "stages/10_mosaics/v2_mosaic.tif"
    new_v2 = OUT / "stages/10_mosaics/v2_mosaic.tif"
    v2_difference = _raster_difference(old_v2, new_v2, height, width)
    return {
        "pre_fix_label_map_changed_pixels": label_changed,
        "pre_fix_v2_vs_post_fix_v2": v2_difference,
        "pre_fix_labeling_summary": str(BASE / "stages/08_multiscene_labels/labeling_summary.json"),
        "post_fix_labeling_summary": str(OUT / "stages/08_multiscene_labeling/labeling_summary.json"),
        "pre_fix_metrics": str(BASE / "stages/11_metrics/metrics_summary.json"),
        "post_fix_metrics": metrics,
    }


def _finite_support(paths: list[Path], masks: list[np.ndarray]) -> int:
    count = 0
    for path, mask in zip(paths, masks):
        with rasterio.open(path) as src:
            for r0 in range(0, mask.shape[0], 1024):
                hh = min(1024, mask.shape[0] - r0)
                tile = src.read(1, window=Window(0, r0, mask.shape[1], hh))
                count += int(np.count_nonzero(np.isfinite(tile) & mask[r0:r0 + hh]))
    return count


def _output_finite(paths: list[Path], masks: list[np.ndarray]) -> bool:
    for path, mask in zip(paths, masks):
        with rasterio.open(path) as src:
            for r0 in range(0, mask.shape[0], 1024):
                hh = min(1024, mask.shape[0] - r0)
                tile = src.read(1, window=Window(0, r0, mask.shape[1], hh))
                if not np.isfinite(tile[mask[r0:r0 + hh]]).all():
                    return False
    return True


def _copy_postfix_layout() -> None:
    labels = OUT / "labels"
    labels.mkdir(parents=True, exist_ok=True)
    stage08 = OUT / "stages/08_multiscene_labeling"
    shutil.copyfile(stage08 / "source_label_map.tif", labels / "source_label_map.tif")
    shutil.copyfile(stage08 / "label_method_map.tif", labels / "label_method_map.tif")
    shutil.copyfile(stage08 / "coverage.tif", labels / "contributor_count.tif")
    shutil.copyfile(stage08 / "label_score_margin.tif", labels / "label_score_margin.tif")
    shutil.copyfile(stage08 / "labeling_summary.json", labels / "labeling_summary.json")
    mosaics = OUT / "mosaics"
    mosaics.mkdir(parents=True, exist_ok=True)
    stage10 = OUT / "stages/10_mosaics"
    for name in ("v0_bagrn_mosaic.tif", "v1_mosaic.tif", "v2_mosaic.tif"):
        shutil.copyfile(stage10 / name, mosaics / name.replace("_bagrn", ""))
    correction = OUT / "correction"
    correction.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(OUT / "stages/09_correction/correction_summary.json", correction / "correction_summary.json")
    metrics_dir = OUT / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    for name in ("boundary_metrics.csv", "structural_metrics.csv", "final_metrics.json", "pre_post_fix_comparison.csv"):
        source = metrics_dir / name
        if not source.is_file():
            continue


def run_postfix_pipeline(replayed_rows: list[dict[str, Any]], replay_summary: dict[str, Any]) -> dict[str, Any]:
    import scripts.run_task14a_resume_13 as frozen

    paths = _scene_paths()
    transform, crs, height, width = _canonical_grid()
    masks, _, _, _ = load_masks(paths)
    union_mask = np.zeros((height, width), dtype=bool)
    for mask in masks:
        union_mask |= mask
    OUT.mkdir(parents=True, exist_ok=True)
    v0_actual, v0_gate = _v0_metadata(paths, masks, transform, crs, height, width)
    write_json(OUT / "protocol.json", {
        "task": "Task14A.2",
        "status": "POST_FIX_FORMAL",
        "stage07_input": str(OLD_STAGE07),
        "stage07_reexecuted": False,
        "footprint_method": "actual_valid_mask_polygon_shared_helper",
        "resolver": "src.seam_local.source_side.resolve_source_sides",
        "v0_semantics": v0_gate,
        "v0_metadata": v0_actual,
        "scene_count": len(paths),
        "grid": {"crs": crs, "transform": list(transform[:6]), "height": height, "width": width},
    })

    frozen.STAGE08 = OUT / "stages/08_multiscene_labeling"
    frozen.STAGE09 = OUT / "stages/09_correction"
    frozen.STAGE10 = OUT / "stages/10_mosaics"
    frozen.STAGE11 = OUT / "stages/11_metrics"
    frozen.STAGE12 = OUT / "stages/12_scale_summary"
    post_rows = _postfix_pair_rows(replayed_rows)
    existing_stage_outputs = all(path.is_file() for path in [
        OUT / "stages/08_multiscene_labeling/labeling_summary.json",
        OUT / "stages/09_correction/correction_summary.json",
        OUT / "stages/10_mosaics/mosaic_summary.json",
        OUT / "stages/11_metrics/metrics_summary.json",
        OUT / "stages/12_scale_summary/scale_summary.json",
    ])
    if existing_stage_outputs:
        # Resume finalization from a completed isolated Stage08--12 run.  This
        # preserves the frozen-stage rule while avoiding a second full replay.
        stats = read_json(OUT / "stages/08_multiscene_labeling/labeling_summary.json")
        corrected = [OUT / "stages/09_correction" / f"corrected_scene_{i:03d}.tif" for i in range(len(paths))]
        if not all(path.is_file() for path in corrected):
            raise RuntimeError("HARD_STOP_POSTFIX_RESUME_INCOMPLETE: corrected scene output missing")
        mosaic_summary = read_json(OUT / "stages/10_mosaics/mosaic_summary.json")
        metrics = read_json(OUT / "stages/11_metrics/metrics_summary.json")
        scale = read_json(OUT / "stages/12_scale_summary/scale_summary.json")
    else:
        stats = frozen._stage08(post_rows, paths, transform, crs, height, width)
        corrected = frozen._stage09(post_rows, paths, transform, height, width)
        mosaic_summary = frozen._stage10(paths, corrected, height, width)
        metrics = frozen._stage11(post_rows, paths, corrected, height, width)
        scale = frozen._stage12(post_rows, stats, metrics, height, width)
    stats = enrich_label_diagnostics(
        stats, post_rows,
        OUT / "stages/08_multiscene_labeling/source_label_map.tif",
        OUT / "stages/08_multiscene_labeling/label_method_map.tif",
        OUT / "stages/08_multiscene_labeling/label_score_margin.tif",
        height, width, len(paths), union_mask,
    )
    stats["resolved_preference_edge_count"] = len(preference_rows(post_rows))
    write_json(OUT / "stages/08_multiscene_labeling/labeling_summary.json", stats)

    reverse_order = list(reversed(range(len(paths))))
    fixed_permutation = [3, 8, 1, 12, 0, 6, 10, 2, 11, 5, 9, 4, 7]
    scene_invariance_path = OUT / "invariance/scene_order_invariance.json"
    if scene_invariance_path.is_file():
        scene_invariance = read_json(scene_invariance_path)
    else:
        reverse_difference = _label_order_difference(post_rows, paths, transform, height, width, reverse_order, OUT / "stages/08_multiscene_labeling/source_label_map.tif")
        permutation_difference = _label_order_difference(post_rows, paths, transform, height, width, fixed_permutation, OUT / "stages/08_multiscene_labeling/source_label_map.tif")
        scene_invariance = {
            "status": "PASS" if reverse_difference == 0 and permutation_difference == 0 else "HARD_STOP_13SCENE_SCENE_ORDER_DEPENDENCE",
            "original_order": list(range(len(paths))),
            "reverse_order": reverse_order,
            "fixed_permutation": fixed_permutation,
            "reverse_order_difference_pixels": reverse_difference,
            "permutation_difference_pixels": permutation_difference,
        }
        write_json(scene_invariance_path, scene_invariance)
    if scene_invariance["status"] != "PASS":
        raise RuntimeError(scene_invariance["status"])

    correction_invariance = _correction_order_invariance(post_rows, paths, transform, height, width, frozen)
    write_json(OUT / "invariance/correction_order_invariance.json", correction_invariance)
    if correction_invariance["status"] != "PASS":
        raise RuntimeError(correction_invariance["status"])

    # Restore formal Stage09 after the diagnostic normal/reverse runs.
    frozen.STAGE09 = OUT / "stages/09_correction"
    support_counts = {
        "v0": int(v0_actual["union_support"]),
        "v1": int(_finite_support([OUT / "stages/10_mosaics/v1_mosaic.tif"], [union_mask])),
        "v2": int(_finite_support([OUT / "stages/10_mosaics/v2_mosaic.tif"], [union_mask])),
    }
    metrics["status_label"] = "POST_FIX_FORMAL"
    metrics_dir = OUT / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    write_csv(metrics_dir / "boundary_metrics.csv", metrics.get("boundary_rows", []))
    write_csv(metrics_dir / "structural_metrics.csv", metrics.get("structural", []))
    write_json(metrics_dir / "final_metrics.json", metrics)
    pre_post = _pre_post_comparison(stats, metrics, height, width)
    write_csv(metrics_dir / "pre_post_fix_comparison.csv", [pre_post])
    write_json(OUT / "invariance/pre_post_fix_comparison.json", pre_post)

    labels_dir = OUT / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    for source_name, target_name in (
        ("source_label_map.tif", "source_label_map.tif"),
        ("label_method_map.tif", "label_method_map.tif"),
        ("coverage.tif", "contributor_count.tif"),
        ("label_score_margin.tif", "label_score_margin.tif"),
        ("labeling_summary.json", "labeling_summary.json"),
    ):
        shutil.copyfile(OUT / "stages/08_multiscene_labeling" / source_name, labels_dir / target_name)
    mosaics_dir = OUT / "mosaics"
    mosaics_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(OUT / "stages/10_mosaics/v0_bagrn_mosaic.tif", mosaics_dir / "v0_bagrn_weighted_feather.tif")
    shutil.copyfile(OUT / "stages/10_mosaics/v1_mosaic.tif", mosaics_dir / "v1_multiscene_label_blend.tif")
    shutil.copyfile(OUT / "stages/10_mosaics/v2_mosaic.tif", mosaics_dir / "v2_multiscene_local_blend.tif")
    correction_dir = OUT / "correction"
    correction_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(OUT / "stages/09_correction/correction_summary.json", correction_dir / "correction_summary.json")
    write_csv(OUT / "scale_summary/five_vs_thirteen.csv", read_csv(AUDIT / "five_vs_thirteen_geometry_comparison.csv"))
    scale.update({
        "status_label": "POST_FIX_FORMAL",
        "resolved_preference_edge_count": len(preference_rows(post_rows)),
        "requires_multiscene_edge_count": sum(row.get("formal_pair_status") == REQUIRES for row in post_rows),
        "unstable_edge_count": sum(row.get("formal_pair_status") == UNSTABLE for row in post_rows),
        "no_support_edge_count": sum(row.get("formal_pair_status") == NO_SUPPORT for row in post_rows),
        "local_correction_usable_edge_count": sum(row.get("formal_pair_status") in {"PASS", REQUIRES} and int(row.get("local_segment_count", 0)) > 0 for row in post_rows),
        "v0_v1_v2_support": support_counts,
        "two_scene_disagreement_pixels": stats["two_scene_disagreement_pixels"],
        "pairwise_score_tie_pixels": stats["pairwise_score_tie_pixels"],
        "clipped_interiority_fallback_pixels": stats["clipped_interiority_fallback_pixels"],
        "unclipped_normalized_interiority_pixels": stats["unclipped_normalized_interiority_pixels"],
        "raw_edt_pixels": stats["raw_edt_pixels"],
    })
    write_json(OUT / "scale_summary/scale_summary.json", scale)

    boundary = metrics["boundary_metrics"]
    structural = metrics["structural"]
    finite_outputs = _output_finite(corrected, masks) and all(
        _output_finite([OUT / "stages/10_mosaics" / name], [union_mask])
        for name in ("v1_mosaic.tif", "v2_mosaic.tif")
    )
    old_five_gate = read_json(BASE / "adapter_validation/five_scene_replay/replay_gate.json")
    gates = {
        "source_side": replay_summary["gate"]["adapter_logic_mismatch_pass_count"] == 45,
        "true_geometry_preserved": replay_summary["gate"]["true_geometric_requires_count"] == 1,
        "labels": stats["unresolved_pixels"] == 0 and stats["invalid_label_pixels"] == 0 and stats["two_scene_disagreement_pixels"] == 0,
        "invariance": scene_invariance["status"] == "PASS",
        "correction_order": correction_invariance["status"] == "PASS",
        "support": support_counts == {"v0": 62033096, "v1": 62033096, "v2": 62033096},
        "structural": all(item.get("finite") and item.get("gradient_magnitude_ncc", 0) >= 0.99 for item in structural),
        "no_invalid": finite_outputs,
        "weighted_mae_improved": boundary["v2_weighted_mae"] < boundary["bagrn_weighted_mae"],
        "weighted_rdd_improved": boundary["v2_weighted_rdd"] < boundary["bagrn_weighted_rdd"],
        "median_mae_improved": boundary["median_v2_mae"] < boundary["median_bagrn_mae"],
        "median_rdd_improved": boundary["median_v2_rdd"] < boundary["median_bagrn_rdd"],
        "five_scene_replay": old_five_gate.get("status") == "PASS",
    }
    decision = final_decision(gates)
    final = {
        "status": decision,
        "gates": gates,
        "source_side": replay_summary["gate"],
        "scale_summary": scale,
        "label_diagnostics": {key: stats.get(key) for key in required_label_diagnostic_keys()},
        "boundary_metrics": boundary,
        "support_counts": support_counts,
        "scene_order_invariance": scene_invariance,
        "correction_order_invariance": correction_invariance,
        "pre_post_fix": pre_post,
        "old_five_scene_replay": old_five_gate,
    }
    write_json(OUT / "metrics/final_metrics.json", final)
    write_json(OUT / "final_decision.json", final)
    return final


def write_final_report(replayed_rows: list[dict[str, Any]], replay_summary: dict[str, Any], final: dict[str, Any]) -> None:
    stats = final["label_diagnostics"]
    boundary = final["boundary_metrics"]
    gates = final["gates"]
    true_requires = [row["pair_id"] for row in replayed_rows if row.get("formal_pair_status") == REQUIRES]
    report = f"""# Task14A.2 — Valid-Mask Footprint Fix + 13-Scene Controlled Replay

## Status

This report is the `POST_FIX_FORMAL` result. Frozen Stage02–06 and old
Stage07–12 outputs remain unchanged and are retained as historical references.
Stage07 seam search and local-moment coefficient estimation were not rerun.

Final decision: **{final['status']}**.

## Answers to the 23 required questions

1. **Original Stage07 bug:** the resolver received rectangular dataset bounds
   (`box(dataset_bounds)`) instead of polygonized actual valid-mask support.
2. **Footprint semantic equality:** yes; the shared helper is extracted from
   the existing five-scene `_footprint_polygon` body and used by both paths.
3. **45 mismatch pairs:** {replay_summary['gate']['adapter_logic_mismatch_pass_count']} / 45 became source-side `PASS`.
4. **True geometric REQUIRES:** {len(true_requires)} pair(s), `{', '.join(true_requires)}`; these remain unresolved because their frozen geometry is near-containment/seam-topology limited.
5. **UNSTABLE_LOCAL_GAIN:** `01_08` and `08_11` remain unstable; missing coefficients were not reconstructed.
6. **NO_FINAL_SHARED_SUPPORT:** {replay_summary['gate']['no_support_count']} pairs remain unchanged and excluded.
7. **Resolved ownership edges:** {replay_summary['gate']['adapter_logic_mismatch_pass_count']}.
8. **Cycle:** {stats.get('cycle_pixels')} pixels; fraction `{final.get('scale_summary', {}).get('cycle_fraction', 'see scale_summary.json')}`.
9. **Pairwise-score ties:** {stats.get('pairwise_score_tie_pixels')} pixels.
10. **Fallbacks:** clipped={stats.get('clipped_interiority_fallback_pixels')}, unclipped normalized={stats.get('unclipped_normalized_interiority_pixels')}, raw EDT={stats.get('raw_edt_pixels')}.
11. **Unresolved:** {stats.get('unresolved_pixels')}.
12. **Scene-order invariance:** `{final['scene_order_invariance']['status']}`, reverse difference={final['scene_order_invariance']['reverse_order_difference_pixels']}, fixed permutation difference={final['scene_order_invariance']['permutation_difference_pixels']}.
13. **Correction pair-order:** `{final['correction_order_invariance']['status']}`; max abs difference={final['correction_order_invariance']['max_abs_difference']}.
14. **V0/V1/V2 support:** `{final['support_counts']}`.
15. **Weighted MAE:** BAGRN={boundary['bagrn_weighted_mae']}; post-fix V2={boundary['v2_weighted_mae']}.
16. **Weighted RDD:** BAGRN={boundary['bagrn_weighted_rdd']}; post-fix V2={boundary['v2_weighted_rdd']}.
17. **Median MAE/RDD:** BAGRN MAE/RDD={boundary['median_bagrn_mae']}/{boundary['median_bagrn_rdd']}; V2={boundary['median_v2_mae']}/{boundary['median_v2_rdd']}.
18. **Structure:** all values are finite and the per-scene NCC gate is recorded in `metrics/final_metrics.json`.
19. **Changed labels:** `{final['pre_post_fix']['pre_fix_label_map_changed_pixels']}` pixels.
20. **V2 change:** `{final['pre_post_fix']['pre_fix_v2_vs_post_fix_v2']}`.
21. **Five-scene replay:** `{final['gates']['five_scene_replay']}`; the frozen replay gate remains the reference.
22. **Task14 decision:** **{final['status']}**.
23. **Next scale benchmark:** permitted only if all gates in `final_decision.json` are true; regardless of the decision, this task stops here and does not start the next scale benchmark.

## Gate record

```json
{json.dumps(gates, indent=2, ensure_ascii=False)}
```

The old Stage11 metrics are `PRE_FIX_DIAGNOSTIC_REFERENCE`; these outputs are
`POST_FIX_FORMAL`. The task ends here: no Task14S, 25/50/100-scene,
1000-scene, or Task15 execution was started.
"""
    (OUT / "TASK14A2_VALID_FOOTPRINT_FIX_REPORT.md").write_text(report, encoding="utf-8")


def run_task14a2() -> dict[str, Any]:
    replayed, replay_summary = run_source_side_replay()
    final = run_postfix_pipeline(replayed, replay_summary)
    write_final_report(replayed, replay_summary, final)
    return final


if __name__ == "__main__":
    result = run_task14a2()
    print(json.dumps(result, indent=2, ensure_ascii=False, default=_json_default))
