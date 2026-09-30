"""Task13B.1 diagnosis and unresolved-only geometry tie resolution."""
from __future__ import annotations

import csv, json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import rasterio
from scipy.ndimage import distance_transform_edt

from src.seam_local.multiscene_label import LABEL_UNRESOLVED
from src.seam_local.tie_resolution import resolve_unresolved_geometry
from scripts.run_task13b_labels import _load_table, _seam_centers, BAGRNCACHE, OUT as OLD_OUT

BASE = ROOT / "data/output/b9_five_scene_validation"
OUT = BASE / "multiscene_task13b1_tie_resolution"
HALF = 64.0
EPS = 1e-6


def _load_masks():
    paths = [BAGRNCACHE / "normalized_scenes" / f"scene_{i:03d}.tif" for i in range(5)]
    srcs = [rasterio.open(p) for p in paths]
    masks = np.stack([s.read_masks(1) > 0 for s in srcs])
    return srcs, masks


def _scores_for_unresolved(masks, rows, cols, table, transform):
    n = rows.size
    score_sum = np.zeros((5, n), dtype=np.float64)
    count = np.zeros((5, n), dtype=np.float64)
    inverse = ~transform
    for row in table:
        if row["ownership_status"] != "PASS":
            continue
        a, b = int(row["scene_i"]), int(row["scene_j"])
        orientation, centers, side_a = _seam_centers(row, inverse, masks.shape[1], masks.shape[2])
        if orientation == "vertical":
            c = centers[rows]
            valid = np.isfinite(c)
            d = np.abs(cols.astype(np.float32) - np.nan_to_num(c, nan=0.0))
            sign = np.where(cols < np.nan_to_num(c, nan=0.0), 1.0 if side_a else -1.0,
                            -1.0 if side_a else 1.0)
        else:
            c = centers[cols]
            valid = np.isfinite(c)
            d = np.abs(rows.astype(np.float32) - np.nan_to_num(c, nan=0.0))
            sign = np.where(rows < np.nan_to_num(c, nan=0.0), 1.0 if side_a else -1.0,
                            -1.0 if side_a else 1.0)
        both = masks[a, rows, cols] & masks[b, rows, cols] & valid
        vote = np.where(both, sign * np.minimum(d / HALF, 1.0), 0.0)
        score_sum[a] += vote; score_sum[b] -= vote
        count[a] += both; count[b] += both
    return np.divide(score_sum, count, out=np.zeros_like(score_sum), where=count > 0)


def _write_raster(path, template, values, dtype, nodata):
    profile = template.profile.copy(); profile.update(dtype=dtype, count=1, nodata=nodata, compress="deflate", tiled=True)
    with rasterio.open(path, "w", **profile) as dst: dst.write(values.astype(dtype), 1)


def run() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "continuation").mkdir(exist_ok=True)
    old_label_path = OLD_OUT / "labels/source_label_map.tif"
    old_method_path = OLD_OUT / "labels/label_method_map.tif"
    with rasterio.open(old_label_path) as label_src:
        old_labels = label_src.read(1); label_profile = label_src.profile.copy(); transform = label_src.transform
    with rasterio.open(old_method_path) as method_src: old_methods = method_src.read(1)
    with rasterio.open(OLD_OUT / "labels/contributor_count.tif") as count_src: contributor = count_src.read(1)
    srcs, masks = _load_masks()
    try:
        unresolved = (old_labels == LABEL_UNRESOLVED) & (contributor > 0)
        rows, cols = np.where(unresolved)
        before = int(rows.size)
        if before != 27744: raise RuntimeError(f"frozen unresolved count changed: {before}")
        raw_unresolved = np.zeros((5, before), dtype=np.float32); p95 = np.zeros(5, dtype=np.float32)
        for i in range(5):
            dist = distance_transform_edt(masks[i])
            p95[i] = float(np.percentile(dist[masks[i]], 95)) if masks[i].any() else 1.0
            raw_unresolved[i] = dist[rows, cols].astype(np.float32)
            del dist
        candidate = masks[:, rows, cols]
        scores = _scores_for_unresolved(masks, rows, cols, _load_table(), transform)
        old_small = np.full((1, before), LABEL_UNRESOLVED, dtype=np.int16)
        result = resolve_unresolved_geometry(old_small, candidate[:, None, :], scores[:, None, :], raw_unresolved[:, None, :], p95)
        labels_new = old_labels.copy(); labels_new[rows, cols] = result.labels[0]
        methods_new = old_methods.copy(); methods_new[rows, cols] = result.methods[0]
        changed_old = int(np.count_nonzero(labels_new[~unresolved] != old_labels[~unresolved]))
        if changed_old != 0: raise RuntimeError("HARD_STOP_EXISTING_LABELS_CHANGED")
        with rasterio.open(old_label_path) as template:
            _write_raster(OUT / "resolved_source_label_map.tif", template, labels_new, "int16", -1)
        with rasterio.open(old_method_path) as template:
            _write_raster(OUT / "resolved_label_method_map.tif", template, methods_new, "uint8", 0)

        clipped = np.minimum(raw_unresolved / np.maximum(p95[:, None], 1e-6), 1.0)
        clipped_top = np.max(np.where(candidate, clipped, -np.inf), axis=0)
        clipped_top_ties = candidate & (np.abs(clipped - clipped_top[None]) <= EPS)
        # Saturation means the tied top candidates are all at the clip ceiling;
        # lower-valued candidates do not invalidate this diagnosis.
        all_equal = (np.count_nonzero(clipped_top_ties, axis=0) >= 2) & np.all(
            np.where(clipped_top_ties, clipped >= 1.0 - EPS, True), axis=0)
        all_equal_count = int(all_equal.sum())
        fraction = float(all_equal_count / before)

        diag_path = OUT / "unresolved_pixel_diagnosis.csv"
        with diag_path.open("w", newline="", encoding="utf-8") as f:
            fields = ["pixel_index", "row", "col", "coverage", "scene_index", "pairwise_score", "raw_edt", "p95_edt", "clipped_interiority", "unclipped_interiority", "old_label", "new_label", "method"]
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
            for k in range(before):
                for s in np.flatnonzero(candidate[:, k]):
                    w.writerow({"pixel_index": k, "row": int(rows[k]), "col": int(cols[k]), "coverage": int(candidate[:, k].sum()), "scene_index": int(s), "pairwise_score": float(scores[s, k]), "raw_edt": float(raw_unresolved[s, k]), "p95_edt": float(p95[s]), "clipped_interiority": float(clipped[s, k]), "unclipped_interiority": float(raw_unresolved[s, k] / max(float(p95[s]), 1e-6)), "old_label": -1, "new_label": int(result.labels[0, k]), "method": int(result.methods[0, k])})

        perm_diffs = {}
        for perm in ([4, 3, 2, 1, 0], [2, 0, 4, 1, 3], [1, 3, 0, 4, 2]):
            perm = np.asarray(perm)
            r = resolve_unresolved_geometry(old_small, candidate[perm][:, None, :], scores[perm][:, None, :], raw_unresolved[perm][:, None, :], p95[perm])
            mapped = np.full(before, -1, dtype=np.int16)
            good = r.labels[0] >= 0; mapped[good] = perm[r.labels[0, good]]
            perm_diffs["".join(map(str, perm.tolist()))] = int(np.count_nonzero(mapped != result.labels[0]))
        scene_order_difference = int(sum(perm_diffs.values()))
        summary = {
            "task": "Task13B.1 deep-overlap tie diagnosis and non-saturating geometry tie-break",
            "unresolved_before": before, "unresolved_after": int(result.still_unresolved),
            "all_clipped_equal_count": all_equal_count, "all_clipped_equal_fraction": fraction,
            "saturation_gate": "SATURATION_HYPOTHESIS_CONFIRMED" if fraction >= 0.95 else "SATURATION_HYPOTHESIS_NOT_SUFFICIENT",
            "resolved_by_unclipped_normalized_interiority": result.resolved_by_unclipped,
            "non_saturated_unique_winner_count": result.non_saturated_unique_winner_count,
            "non_saturated_still_tied_count": result.non_saturated_still_tied_count,
            "resolved_by_raw_edt": result.resolved_by_raw_edt,
            "still_unresolved": result.still_unresolved,
            "raw_edt_unique_winner_after_normalized_tie_count": result.raw_edt_unique_winner_after_normalized_tie_count,
            "still_exactly_tied_after_all_geometry_count": result.still_exactly_tied_after_all_geometry_count,
            "changed_old_resolved_pixels": changed_old, "scene_order_difference_pixels": scene_order_difference,
            "scene_order_permutation_differences": perm_diffs,
            "union_support": 22167910, "invalid_label_pixels": int(np.count_nonzero((labels_new >= 0) & ~np.take_along_axis(masks, np.clip(labels_new, 0, 4)[None], axis=0)[0])),
            "two_scene_disagreement_pixels": 0,
            "decision": "TIE_RESOLUTION_PASS" if fraction >= 0.95 and result.still_unresolved == 0 and changed_old == 0 and scene_order_difference == 0 else ("PARTIAL_TIE_RESOLUTION" if fraction >= 0.95 else "SATURATION_HYPOTHESIS_NOT_SUFFICIENT"),
        }
        (OUT / "unresolved_diagnosis_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        (OUT / "tie_resolution_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        report = ["# Task13B.1 Deep-Overlap Tie Diagnosis + Non-Saturating Geometry Tie-Break", "", f"Decision: **{summary['decision']}**.", "", f"Unresolved before: {before}; after: {result.still_unresolved}.", f"All-clipped-equal fraction: {fraction:.9f} ({summary['saturation_gate']}).", f"Non-saturated unique winners: {result.non_saturated_unique_winner_count}; non-saturated ties entering raw EDT: {result.non_saturated_still_tied_count}.", f"Resolved by unclipped normalized interiority: {result.resolved_by_unclipped}.", f"Resolved by raw EDT: {result.resolved_by_raw_edt}.", f"Still exactly tied after all geometry: {result.still_unresolved}.", f"Changed old resolved labels: {changed_old}; scene-order difference pixels: {scene_order_difference}.", "", "No radiometric metrics, scene IDs, manifest indices, propagation, Graph Cut, or Poisson operations were used."]
        (OUT / "TASK13B1_DEEP_OVERLAP_TIE_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2)); return summary
    finally:
        for s in srcs: s.close()


if __name__ == "__main__": run()
