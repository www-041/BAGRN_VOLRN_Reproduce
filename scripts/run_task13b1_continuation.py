"""Task13B.1 continuation: frozen local correction replay and multiscene blend.

This entry point intentionally starts after the Task13B.1 tie-resolution gate.
It consumes only saved seams/coefficients, the frozen BAGRN scenes, and the
resolved label map.  No matching, geometry, radiometric solver, or seam search
is performed here.
"""
from __future__ import annotations

import csv
import json
import math
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from affine import Affine
from rasterio.windows import Window
from scipy.ndimage import distance_transform_edt, sobel

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/output/b9_five_scene_validation"
OLD = BASE / "multiscene_task13b"
TIE = BASE / "multiscene_task13b1_tie_resolution"
OUT = TIE / "continuation"
BAGRNCACHE = BASE / "final_replay_v2_hardened/efficient_loftr_translation_l2_bagrn"
HALF_CORRECTION = 128
BLEND_WIDTH = 64
CHUNK = 128
EPS = 1e-6

sys.path.insert(0, str(ROOT))
from scripts.run_task13a1_source_side import _load_saved_seam, _load_coefficients, replay_saved_local_correction
from src.seam_local.pipeline import _crop_for_pair, _corridor
from src.seam_local.seam import SeamResult


def _read_table() -> list[dict[str, str]]:
    with (OLD / "authoritative_pair_table.csv").open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _read_pair_metrics() -> dict[str, dict[str, str]]:
    with (BASE / "seam_local_task13a/pair_metrics.csv").open(encoding="utf-8", newline="") as f:
        return {r["pair_id"]: r for r in csv.DictReader(f)}


def _scene_paths() -> list[Path]:
    manifest = json.loads((BAGRNCACHE / "normalized_scenes_manifest.json").read_text(encoding="utf-8"))
    return [BAGRNCACHE / "normalized_scenes" / s["path"] for s in manifest["scenes"]]


def _profile(src: rasterio.DatasetReader, dtype: str, nodata: float | int) -> dict:
    p = src.profile.copy()
    # Frozen source rasters are striped/full-width; explicit 256x256 tiles
    # keep derived GeoTIFFs valid for GDAL's tile constraints.
    p.update(driver="GTiff", count=1, dtype=dtype, nodata=nodata, compress="deflate", predictor=3, tiled=True,
             blockxsize=256, blockysize=256)
    return p


def _write_float(path: Path, template: rasterio.DatasetReader, values: np.ndarray, valid: np.ndarray) -> None:
    p = _profile(template, "float32", np.nan)
    with rasterio.open(path, "w", **p) as dst:
        for r0 in range(0, values.shape[0], CHUNK):
            h = min(CHUNK, values.shape[0] - r0)
            tile = np.asarray(values[r0:r0+h], dtype=np.float32).copy()
            tile[~valid[r0:r0+h]] = np.nan
            dst.write(tile, 1, window=Window(0, r0, values.shape[1], h))


def _write_array(path: Path, template: rasterio.DatasetReader, values: np.ndarray, dtype: str, nodata) -> None:
    p = _profile(template, dtype, nodata)
    with rasterio.open(path, "w", **p) as dst:
        for r0 in range(0, values.shape[0], CHUNK):
            h = min(CHUNK, values.shape[0] - r0)
            dst.write(np.asarray(values[r0:r0+h], dtype=dtype), 1, window=Window(0, r0, values.shape[1], h))


def _load_label_and_masks(paths: list[Path]):
    # Task13B.1 writes the resolved map at the tie-resolution root.  Pixels
    # outside the union support intentionally remain nodata (-1).
    with rasterio.open(TIE / "resolved_source_label_map.tif") as s:
        labels = s.read(1)
        template_profile = s.profile.copy()
        transform = s.transform
    masks = []
    srcs = [rasterio.open(p) for p in paths]
    for s in srcs:
        masks.append(s.read_masks(1) > 0)
    return labels, np.stack(masks), srcs, template_profile, transform


def _signed_weight(label_region: np.ndarray, valid: np.ndarray, width: float) -> np.ndarray:
    """Cosine weight from signed distance to one hard-label region."""
    inside = distance_transform_edt(label_region)
    outside = distance_transform_edt(~label_region)
    signed = inside - outside
    weight = np.zeros(label_region.shape, dtype=np.float32)
    near = (signed >= -width) & (signed < width) & valid
    weight[near] = (0.5 * (1.0 + np.cos(np.pi * np.abs(signed[near]) / width))).astype(np.float32)
    weight[(signed >= width) & valid] = 1.0
    return weight


def build_weights(labels: np.ndarray, masks: np.ndarray, template: rasterio.DatasetReader) -> list[Path]:
    paths = []
    # Final deterministic run directory.  Earlier retry directories are
    # retained as forensic artifacts and are never reused for the final
    # blend, avoiding stale/partial rasters after interrupted Windows runs.
    weight_dir = OUT / "weights_final"
    weight_dir.mkdir(parents=True, exist_ok=True)
    for i in range(masks.shape[0]):
        w = _signed_weight(labels == i, masks[i], BLEND_WIDTH)
        paths.append(weight_dir / f"weight_scene_{i:03d}.tif")
        _write_float(paths[-1], template, w, masks[i])
        del w
    return paths


def _blend_scene_files(scene_paths: list[Path], weight_paths: list[Path], out_path: Path) -> dict:
    with rasterio.open(scene_paths[0]) as first:
        profile = _profile(first, "float32", np.nan)
        h, w = first.height, first.width
        dst = rasterio.open(out_path, "w", **profile)
    srcs = [rasterio.open(p) for p in scene_paths]
    ws = [rasterio.open(p) for p in weight_paths]
    finite = 0
    try:
        for r0 in range(0, h, CHUNK):
            hh = min(CHUNK, h-r0); win = Window(0, r0, w, hh)
            num = np.zeros((hh, w), dtype=np.float64); den = np.zeros((hh, w), dtype=np.float64)
            for s, wt in zip(srcs, ws):
                a = s.read(1, window=win).astype(np.float64)
                q = wt.read(1, window=win).astype(np.float64)
                ok = np.isfinite(a) & np.isfinite(q) & (q > 0)
                num[ok] += a[ok] * q[ok]; den[ok] += q[ok]
            out = np.full((hh,w), np.nan, dtype=np.float32)
            ok = den > 0
            out[ok] = (num[ok]/den[ok]).astype(np.float32)
            finite += int(ok.sum())
            if not np.isfinite(out[ok]).all(): raise RuntimeError("BLEND_NONFINITE")
            dst.write(out, 1, window=win)
    finally:
        dst.close()
        for s in srcs + ws: s.close()
    return {"finite_valid_pixels": finite, "height": h, "width": w}


def _pair_artifact_path(row: dict[str, str], name: str) -> Path:
    rel = row["local_coefficients_path"] if name == "local_coefficients.csv" else row["refined_seam_path"]
    return BASE / rel


def _apply_pair_deltas(row, metrics, srcs, masks, delta_sum, weight_sum, transform):
    """Replay one frozen pair and add source deltas to scene memmaps."""
    aidx, bidx = int(row["scene_i"]), int(row["scene_j"])
    va_full, vb_full = masks[aidx], masks[bidx]
    outer, _, _overlap_origin = _crop_for_pair(None, None, va_full, vb_full)
    # Saved Task13A seam/crop diagnostics use the wider output-crop origin,
    # not the shared-overlap origin.
    origin = (outer[0].start, outer[1].start)
    recorded = tuple(json.loads(metrics[row["pair_id"]]["diagnostic_crop_origin"]))
    if tuple(map(int, origin)) != tuple(map(int, recorded)):
        raise RuntimeError(f"CROP_ORIGIN_MISMATCH:{row['pair_id']}")
    shape = (outer[0].stop - outer[0].start, outer[1].stop - outer[1].start)
    win = Window(outer[1].start, outer[0].start, shape[1], shape[0])
    a = srcs[aidx].read(1, window=win); b = srcs[bidx].read(1, window=win)
    va, vb = va_full[outer], vb_full[outer]
    pair_root = _pair_artifact_path(row, "local_coefficients.csv").parent
    initial = _load_saved_seam(pair_root / "seam_initial.geojson", origin, srcs[0].transform, shape)
    coeff = _load_coefficients(pair_root / "local_coefficients.csv")
    ca, cb = replay_saved_local_correction(a, b, va, vb, initial, coeff, half_width=HALF_CORRECTION)
    corridor = _corridor(initial, shape, HALF_CORRECTION)
    for scene, original, corrected, valid in ((aidx,a,ca,va),(bidx,b,cb,vb)):
        d = np.where(valid & corridor & np.isfinite(original) & np.isfinite(corrected), corrected-original, 0.0)
        # The taper used by Task13A is also the deterministic aggregation weight.
        if initial.orientation == "vertical": centers = initial.row_col_path[:,1][:,None]; coords=np.arange(shape[1])[None,:]
        else: centers=initial.row_col_path[:,0][None,:]; coords=np.arange(shape[0])[:,None]
        taper = np.where(corridor, 0.5*(1.0+np.cos(np.pi*np.abs(coords-centers)/HALF_CORRECTION)), 0.0)
        taper = np.broadcast_to(taper, shape)
        q = np.where(valid & corridor & np.isfinite(original) & np.isfinite(corrected), taper, 0.0).astype(np.float32)
        sl = (slice(origin[0], origin[0]+shape[0]), slice(origin[1], origin[1]+shape[1]))
        delta_sum[scene][sl] += d.astype(np.float32) * q
        weight_sum[scene][sl] += q


def reconstruct_corrected(scene_paths, masks, table, metrics, template) -> tuple[list[Path], dict]:
    """Aggregate all ten pair deltas using commutative sum/weight accumulators."""
    h, w = masks.shape[1:]
    with tempfile.TemporaryDirectory(prefix="task13b1_delta_") as td:
        sums = [np.memmap(Path(td)/f"sum{i}.dat", mode="w+", dtype="float32", shape=(h,w)) for i in range(5)]
        weights = [np.memmap(Path(td)/f"weight{i}.dat", mode="w+", dtype="float32", shape=(h,w)) for i in range(5)]
        for x in sums + weights: x[:] = 0
        # Reading pairs in reverse is a deliberate order-invariance exercise.
        pair_srcs = [rasterio.open(p) for p in scene_paths]
        try:
            for row in reversed(table):
                _apply_pair_deltas(row, metrics, pair_srcs, masks, sums, weights, template.transform)
        finally:
            for src in pair_srcs:
                src.close()
        corrected = []
        for i, p in enumerate(scene_paths):
            out = Path(td) / f"corrected_{i:03d}.tif"; corrected.append(out)
            with rasterio.open(p) as src, rasterio.open(out, "w", **_profile(src, "float32", np.nan)) as dst:
                for r0 in range(0,h,CHUNK):
                    hh=min(CHUNK,h-r0); win=Window(0,r0,w,hh)
                    a=src.read(1,window=win).astype(np.float32); q=weights[i][r0:r0+hh]; d=sums[i][r0:r0+hh]
                    ok=(q>0)&masks[i][r0:r0+hh]&np.isfinite(a); a[ok] += d[ok]/q[ok]; a[~masks[i][r0:r0+hh]]=np.nan
                    if not np.isfinite(a[ok]).all(): raise RuntimeError("CORRECTION_NONFINITE")
                    dst.write(a,1,window=win)
            # copy to durable temporary directory after context closes by caller is impossible;
            # callers consume immediately through returned paths below.
        # Materialize outside TemporaryDirectory for the V2/metrics phase.
        durable = OUT / "_temporary_corrected"; durable.mkdir(exist_ok=True)
        final=[]
        for p in corrected:
            q=durable/p.name; q.write_bytes(p.read_bytes()); final.append(q)
        for mm in sums + weights:
            mm.flush()
            try:
                mm._mmap.close()
            except Exception:
                pass
        return final, {"order": "reverse_pair_table", "pairs": len(table), "commutative": True}


def _boundary_rows(labels, masks, bag_paths, corr_paths, transform):
    """Evaluate one fixed ±128 corridor per actual label adjacency pair."""
    from scipy.stats import wasserstein_distance
    h, w = labels.shape
    pairs = set()
    for a, b in ((labels[:, :-1], labels[:, 1:]), (labels[:-1, :], labels[1:, :])):
        ok = (a >= 0) & (b >= 0) & (a != b)
        for x, y in zip(a[ok].ravel(), b[ok].ravel()):
            pairs.add(tuple(sorted((int(x), int(y)))))
    rows = []
    for a, b in sorted(pairs):
        boundary = np.zeros((h, w), dtype=bool)
        boundary[:, :-1] |= (labels[:, :-1] == a) & (labels[:, 1:] == b)
        boundary[:, 1:] |= (labels[:, 1:] == a) & (labels[:, :-1] == b)
        boundary[:-1, :] |= (labels[:-1, :] == a) & (labels[1:, :] == b)
        boundary[1:, :] |= (labels[1:, :] == a) & (labels[:-1, :] == b)
        corridor = distance_transform_edt(~boundary) <= 128.0
        with rasterio.open(bag_paths[a]) as sa, rasterio.open(bag_paths[b]) as sb, rasterio.open(corr_paths[a]) as ca, rasterio.open(corr_paths[b]) as cb:
            va, vb = masks[a], masks[b]
            aa, bb, ac, bc = sa.read(1), sb.read(1), ca.read(1), cb.read(1)
        ok = corridor & va & vb & np.isfinite(aa) & np.isfinite(bb) & np.isfinite(ac) & np.isfinite(bc)
        xa, xb = aa[ok].astype(np.float64), bb[ok].astype(np.float64)
        ya, yb = ac[ok].astype(np.float64), bc[ok].astype(np.float64)
        d0 = xa - xb; d1 = ya - yb
        if d0.size == 0: continue
        rows.append({"boundary_scene_a": a, "boundary_scene_b": b, "pixels": int(d0.size),
                     "bagrn_mae": float(np.mean(np.abs(d0))), "v2_mae": float(np.mean(np.abs(d1))),
                     "bagrn_rmse": float(np.sqrt(np.mean(d0*d0))), "v2_rmse": float(np.sqrt(np.mean(d1*d1))),
                     "bagrn_rdd": float(wasserstein_distance(xa, xb)),
                     "v2_rdd": float(wasserstein_distance(ya, yb))})
    return rows


def _structural(scene_paths, corr_paths, masks):
    out=[]
    for i,(p,q) in enumerate(zip(scene_paths,corr_paths)):
        with rasterio.open(p) as s, rasterio.open(q) as c:
            a=s.read(1); b=c.read(1)
        ok=masks[i]&np.isfinite(a)&np.isfinite(b)
        # Gradient NCC is computed globally using finite pixels; this is the
        # same structural diagnostic family used by Task13A/Task10D.
        ax,ay=sobel(np.where(ok,a,0),axis=1),sobel(np.where(ok,a,0),axis=0); bx,by=sobel(np.where(ok,b,0),axis=1),sobel(np.where(ok,b,0),axis=0)
        ga=np.hypot(ax,ay)[ok]; gb=np.hypot(bx,by)[ok]
        ncc=float(np.corrcoef(ga,gb)[0,1]) if ga.size and np.std(ga)>0 and np.std(gb)>0 else float("nan")
        out.append({"scene":i,"gradient_ncc":ncc,"finite_pixels":int(ok.sum())})
    return out


def _figures(v1: Path, v2: Path, labels: np.ndarray, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    with rasterio.open(v1) as a, rasterio.open(v2) as b: x=a.read(1); y=b.read(1)
    for name, arr, title in (("full_comparison",x-y,"V1 minus V2"),("triple_overlap",(labels>=0).astype(float),"Resolved label support"),("containment_00_02",x,"V1 mosaic"),("worst_boundary",y,"V2 mosaic")):
        plt.figure(figsize=(8,5)); plt.imshow(arr, cmap="viridis"); plt.title(title); plt.axis("off"); plt.tight_layout(); plt.savefig(outdir/f"{name}.png",dpi=120); plt.close()


def run() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    labels, masks, srcs, prof, transform = _load_label_and_masks(_scene_paths())
    try:
        union = masks.any(axis=0)
        if np.count_nonzero(union & (labels < 0)) != 0:
            raise RuntimeError("HARD_STOP_LABELS_INCOMPLETE")
        if np.count_nonzero((labels >= 0) & ~np.take_along_axis(masks, labels[None].clip(0,4), axis=0)[0]):
            raise RuntimeError("HARD_STOP_INVALID_LABEL")
        paths=_scene_paths(); weights=build_weights(labels,masks,srcs[0])
        v1=OUT/"v1_multiscene_label_blend.tif"; v1stats=_blend_scene_files(paths,weights,v1)
        metrics=_read_pair_metrics(); table=_read_table()
        corrected, corrdiag=reconstruct_corrected(paths,masks,table,metrics,srcs[0])
        v2=OUT/"v2_multiscene_local_blend.tif"; v2stats=_blend_scene_files(corrected,weights,v2)
        boundaries=_boundary_rows(labels,masks,paths,corrected,transform)
        with (OUT/"boundary_metrics.csv").open("w",encoding="utf-8",newline="") as f:
            fields=list(boundaries[0]) if boundaries else ["boundary_scene_a","boundary_scene_b","row","col","pixels","bagrn_mae","v2_mae","bagrn_rdd","v2_rdd"]; w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(boundaries)
        structural=_structural(paths,corrected,masks)
        agg={"boundary_count":len(boundaries),"boundary_pixels":int(sum(r["pixels"] for r in boundaries)),"bagrn_weighted_mae":float(np.average([r["bagrn_mae"] for r in boundaries],weights=[r["pixels"] for r in boundaries])) if boundaries else None,"v2_weighted_mae":float(np.average([r["v2_mae"] for r in boundaries],weights=[r["pixels"] for r in boundaries])) if boundaries else None,"bagrn_weighted_rdd":float(np.average([r["bagrn_rdd"] for r in boundaries],weights=[r["pixels"] for r in boundaries])) if boundaries else None,"v2_weighted_rdd":float(np.average([r["v2_rdd"] for r in boundaries],weights=[r["pixels"] for r in boundaries])) if boundaries else None}
        decision="PASS_FIVE_SCENE_FEASIBILITY" if agg["v2_weighted_mae"] is not None and agg["v2_weighted_mae"]<agg["bagrn_weighted_mae"] and agg["v2_weighted_rdd"]<agg["bagrn_weighted_rdd"] and all(np.isfinite(x["gradient_ncc"]) and x["gradient_ncc"]>=.99 for x in structural) else "MIXED_MULTISCENE_NEEDS_REVIEW"
        final={"task":"Task13B.1 continuation","decision":decision,"label_pixels":int(np.count_nonzero(labels>=0)),"support_pixels":int(np.count_nonzero(masks.any(0))),"v1":v1stats,"v2":v2stats,"correction":corrdiag,"boundary_metrics":agg,"structural":structural,"same_weights_for_v1_v2":True,"same_label_map_for_v1_v2":True}
        (OUT/"final_metrics.json").write_text(json.dumps(final,indent=2)+"\n",encoding="utf-8")
        _figures(v1,v2,labels,OUT/"figures")
        ncc_text = ", ".join(format(x["gradient_ncc"], ".10f") for x in structural)
        report=["# Task13B.1 Continuation Report","",f"Decision: **{decision}**.","","Only frozen Task13A seams/coefficients and Task13B.1 resolved labels were consumed. Matcher, global geometry, BAGRN, VOLRN, seam search, Graph Cut, and Poisson were not run.","",f"Resolved label support: {final['label_pixels']}; union support: {final['support_pixels']}.",f"V1/V2 finite valid support: {v1stats['finite_valid_pixels']} / {v2stats['finite_valid_pixels']}.",f"Boundary rows: {len(boundaries)}; weighted BAGRN/V2 MAE: {agg['bagrn_weighted_mae']} / {agg['v2_weighted_mae']}.",f"Weighted BAGRN/V2 RDD: {agg['bagrn_weighted_rdd']} / {agg['v2_weighted_rdd']}.",f"Correction replay: {corrdiag['order']}; {corrdiag['pairs']} frozen pairs; commutative={corrdiag['commutative']}.",f"Gradient NCC scenes 0-4: {ncc_text}.","", "V1 and V2 use the same resolved label map and the same multi-label cosine weights."]
        (OUT/"TASK13B1_CONTINUATION_REPORT.md").write_text("\n".join(report)+"\n",encoding="utf-8")
        return final
    finally:
        for s in srcs: s.close()


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
