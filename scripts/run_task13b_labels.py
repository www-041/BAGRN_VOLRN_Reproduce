"""Windowed frozen Task13B five-scene source labeling replay."""
from __future__ import annotations

import csv, json, math, tempfile
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from rasterio.windows import Window
from scipy.ndimage import distance_transform_edt

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/output/b9_five_scene_validation"
OUT = BASE / "multiscene_task13b"
BAGRNCACHE = BASE / "final_replay_v2_hardened/efficient_loftr_translation_l2_bagrn"
HALF = 64.0
EPS = 1e-6


def _load_table():
    with (OUT / "authoritative_pair_table.csv").open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _seam_centers(row, inverse, height, width):
    path = ROOT / "data/output/b9_five_scene_validation" / row["refined_seam_path"]
    rec = json.loads(path.read_text(encoding="utf-8"))
    coords = np.asarray(rec["features"][0]["geometry"]["coordinates"], dtype=float)
    px = np.asarray([inverse * (float(x), float(y)) for x, y in coords])
    cols = np.rint(px[:, 0] - 0.5).astype(int)
    rows = np.rint(px[:, 1] - 0.5).astype(int)
    orientation = row["orientation"]
    if orientation == "vertical":
        if rows.min() < 0 or rows.max() >= height or len(np.unique(rows)) != len(rows):
            raise RuntimeError(f"seam row span mismatch: {row['pair_id']}")
        centers = np.full(height, np.nan, dtype=np.float32); centers[rows] = cols
    else:
        if cols.min() < 0 or cols.max() >= width or len(np.unique(cols)) != len(cols):
            raise RuntimeError(f"seam column span mismatch: {row['pair_id']}")
        centers = np.full(width, np.nan, dtype=np.float32); centers[cols] = rows
    return orientation, centers, row.get("side_1_source", "") == "A"


def _resolve(scores, counts, masks, interiors):
    n, h, w = masks.shape
    masked = np.where(masks, scores, -np.inf)
    top = np.max(masked, axis=0)
    candidate = np.any(masks, axis=0)
    ties = masks & (np.abs(scores - top[None]) <= EPS)
    tie_count = np.count_nonzero(ties, axis=0)
    tie_mask = candidate & (tie_count > 1)
    best = np.argmax(masked, axis=0).astype(np.int16)
    labels = np.where(candidate & ~tie_mask, best, -1).astype(np.int16)
    second = np.full((h, w), -np.inf, dtype=np.float32)
    for i in range(n):
        second = np.maximum(second, np.where(masked[i] < top, masked[i], -np.inf))
    margin = np.full((h, w), np.inf, dtype=np.float32)
    finite_second = candidate & np.isfinite(second)
    margin[finite_second] = (top[finite_second] - second[finite_second]).astype(np.float32)
    imasked = np.where(ties, interiors, -np.inf)
    ibest = np.max(imasked, axis=0)
    ities = ties & (np.abs(interiors - ibest[None]) <= EPS)
    icount = np.count_nonzero(ities, axis=0)
    fallback = tie_mask & (icount == 1)
    labels = np.where(fallback, np.argmax(imasked, axis=0), labels).astype(np.int16)
    unresolved = tie_mask & (icount != 1)
    method = np.zeros((h, w), dtype=np.uint8)
    method[candidate & ~tie_mask] = 1
    method[fallback] = 2
    method[unresolved] = 0
    return labels, method, margin, candidate, tie_mask, fallback, unresolved


def run() -> dict:
    OUT.joinpath("labels").mkdir(parents=True, exist_ok=True)
    protocol = json.loads((OUT / "protocol.json").read_text(encoding="utf-8"))
    grid = protocol["dataset"]["canonical_grid"]
    height, width = int(grid["height"]), int(grid["width"])
    transform = Affine(*grid["transform"])
    paths = [BAGRNCACHE / "normalized_scenes" / s["path"] for s in json.loads((BAGRNCACHE / "normalized_scenes_manifest.json").read_text(encoding="utf-8"))["scenes"]]
    srcs = [rasterio.open(p) for p in paths]
    try:
        if any((s.width, s.height) != (width, height) for s in srcs): raise RuntimeError("scene grid mismatch")
        masks_full = []
        for s in srcs:
            masks_full.append(s.read_masks(1) > 0)
        masks_full = np.stack(masks_full)
        with tempfile.TemporaryDirectory(prefix="task13b_interiority_") as td:
            inter_paths = []
            for i, mask in enumerate(masks_full):
                dist = distance_transform_edt(mask)
                p95 = float(np.percentile(dist[mask], 95)) if mask.any() else 1.0
                scale = max(p95, 1.0)
                norm = np.where(mask, np.minimum(dist / scale, 1.0), 0).astype(np.float32)
                p = Path(td) / f"interiority_{i}.npy"; np.save(p, norm); inter_paths.append(p)
                del dist, norm
            interiors_full = [np.load(p) for p in inter_paths]
            table = _load_table(); inverse = ~transform; pairs = []
            for r in table:
                orient, centers, side_a = _seam_centers(r, inverse, height, width)
                pairs.append({"a": int(r["scene_i"]), "b": int(r["scene_j"]), "orient": orient,
                              "centers": centers, "side_a": side_a, "resolved": r["ownership_status"] == "PASS",
                              "pair_id": r["pair_id"]})
            meta = {"driver": "GTiff", "height": height, "width": width, "count": 1, "dtype": "int16",
                    "crs": protocol["dataset"]["crs"], "transform": transform, "nodata": -1, "compress": "deflate", "tiled": True}
            paths_out = {"labels": OUT / "labels/source_label_map.tif", "method": OUT / "labels/label_method_map.tif",
                         "margin": OUT / "labels/label_score_margin.tif", "count": OUT / "labels/contributor_count.tif"}
            mmeta = dict(meta); mmeta["dtype"] = "uint8"; mmeta["nodata"] = 0
            smeta = dict(meta); smeta["dtype"] = "float32"; smeta["nodata"] = np.nan
            writers = [rasterio.open(paths_out["labels"], "w", **meta), rasterio.open(paths_out["method"], "w", **mmeta),
                       rasterio.open(paths_out["margin"], "w", **smeta), rasterio.open(paths_out["count"], "w", **mmeta)]
            hist = np.zeros(6, dtype=np.int64); scene_counts = np.zeros(5, dtype=np.int64)
            stats = {"cycle_pixels": 0, "multiscene_pixels": 0, "ties": 0, "fallback": 0, "unresolved": 0,
                     "invalid_label_pixels": 0, "two_scene_disagreement_pixels": 0, "finite_union_pixels": 0}
            try:
                for r0 in range(0, height, 128):
                    h = min(128, height-r0); win = Window(0, r0, width, h)
                    masks = np.stack([s.read_masks(1, window=win) > 0 for s in srcs])
                    scores = np.zeros((5, h, width), dtype=np.float32); counts = np.zeros_like(scores)
                    edge_votes = {}
                    for p in pairs:
                        if not p["resolved"]: continue
                        a,b = p["a"],p["b"]; both = masks[a] & masks[b]
                        if p["orient"] == "vertical":
                            c = p["centers"][r0:r0+h]; valid = np.isfinite(c)
                            d = np.abs(np.arange(width, dtype=np.float32)[None,:] - np.nan_to_num(c, nan=0)[:,None])
                            sg = np.where(np.arange(width)[None,:] < np.nan_to_num(c, nan=0)[:,None], 1.0 if p["side_a"] else -1.0, -1.0 if p["side_a"] else 1.0)
                            vote = np.where(valid[:,None], sg*np.minimum(d/HALF,1.0), 0.0)
                        else:
                            c = p["centers"]; cc = np.arange(r0,r0+h)
                            valid = np.isfinite(c[None, :])
                            d = np.abs(cc[:,None].astype(np.float32) - np.nan_to_num(c, nan=0)[None,:])
                            sg = np.where(cc[:,None] < np.nan_to_num(c, nan=0)[None,:], 1.0 if p["side_a"] else -1.0, -1.0 if p["side_a"] else 1.0)
                            vote = np.where(valid, sg*np.minimum(d/HALF,1.0), 0.0)
                        vote = np.where(both, vote, 0.0).astype(np.float32)
                        # A seam-center pixel is still an available resolved
                        # comparison; its prescribed confidence is simply 0.
                        avail_shape = valid[:, None] if p["orient"] == "vertical" else valid
                        avail = both & np.broadcast_to(avail_shape, both.shape)
                        scores[a] += vote; scores[b] -= vote; counts[a] += avail; counts[b] += avail; edge_votes[(a,b)] = (vote, avail)
                    scores = np.divide(scores, counts, out=np.zeros_like(scores), where=counts>0)
                    interiors = np.stack([np.asarray(x[r0:r0+h], dtype=np.float32) for x in interiors_full])
                    labels, methods, margin, candidate, ties, fallback, unresolved = _resolve(scores, counts, masks, interiors)
                    coverage = masks.sum(axis=0).astype(np.uint8); hist += np.bincount(coverage.ravel(), minlength=6)
                    stats["multiscene_pixels"] += int(np.count_nonzero(coverage>=3)); stats["ties"] += int(ties.sum()); stats["fallback"] += int(fallback.sum()); stats["unresolved"] += int(unresolved.sum()); stats["finite_union_pixels"] += int(candidate.sum())
                    for i in range(5): scene_counts[i] += int(np.count_nonzero(labels==i))
                    stats["invalid_label_pixels"] += int(np.count_nonzero((labels>=0) & ~np.take_along_axis(masks, np.clip(labels,0,4)[None], axis=0)[0]))
                    cycle = np.zeros((h,width),bool)
                    for a in range(5):
                        for b in range(a+1,5):
                            for c in range(b+1,5):
                                if (a,b) not in edge_votes or (a,c) not in edge_votes or (b,c) not in edge_votes: continue
                                ab, av=edge_votes[(a,b)]; ac, acv=edge_votes[(a,c)]; bc, bcv=edge_votes[(b,c)]
                                support=masks[a]&masks[b]&masks[c]; cycle |= support & (((ab>0)&(bc>0)&(ac<0))|((ab<0)&(bc<0)&(ac>0)))
                    stats["cycle_pixels"] += int(cycle.sum())
                    for (a,b),(vote,avail) in edge_votes.items():
                        only=masks[a]&masks[b]&(coverage==2)&avail
                        stats["two_scene_disagreement_pixels"] += int(np.count_nonzero(only & (((vote>0)&(labels!=a))|((vote<0)&(labels!=b)))))
                    writers[0].write(labels.astype(np.int16),1,window=win); writers[1].write(methods,1,window=win); writers[2].write(margin,1,window=win); writers[3].write(coverage.astype(np.uint8),1,window=win)
            finally:
                for w in writers: w.close()
            stats["coverage_histogram"] = {str(i): int(hist[i]) for i in range(6)}; stats["label_pixel_counts"] = {str(i): int(scene_counts[i]) for i in range(5)}
            stats["tie_pixels_before_interiority"] = stats["ties"]
            stats["interiority_fallback_pixels"] = stats["fallback"]
            stats["unresolved_pixels"] = stats["unresolved"]
            stats["cycle_fraction"] = float(stats["cycle_pixels"]/stats["multiscene_pixels"]) if stats["multiscene_pixels"] else 0.0
            stats["invalid_pixels"] = int(hist[0]); stats["decision"] = "LABELING_PASS" if stats["unresolved"] == 0 and stats["two_scene_disagreement_pixels"] == 0 and stats["invalid_label_pixels"] == 0 else "HARD_STOP_UNRESOLVED_LABELS"
            (OUT/"labeling_summary.json").write_text(json.dumps({"task":"Task13B real five-scene labeling", "status":stats["decision"], **stats}, indent=2)+"\n",encoding="utf-8")
            del interiors_full
            print(json.dumps(stats, indent=2)); return stats
    finally:
        for s in srcs: s.close()

if __name__ == "__main__": run()
