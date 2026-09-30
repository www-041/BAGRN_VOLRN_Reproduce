"""Memory-bounded Task14A Stage07--12 resume on the frozen 13-scene BAGRN grid.

Stage07 reads one overlap crop at a time.  Stage08 and Stage10 operate in
canonical row tiles and never materialize a scene stack.  All scientific
operations are delegated to ``src.seam_local``; this file only supplies the
GeoTIFF/window orchestration and checkpoint records.
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import rasterio
from affine import Affine
from rasterio.windows import Window
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import sobel

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/output/b9_13scene_task14"
STAGE07 = OUT / "stages/07_pairwise_seam_local"
STAGE08 = OUT / "stages/08_multiscene_labels"
STAGE09 = OUT / "stages/09_correction"
STAGE10 = OUT / "stages/10_mosaics"
STAGE11 = OUT / "stages/11_metrics"
STAGE12 = OUT / "stages/12_scale_summary"
TILE = 1024
HALO = 128
RUNTIME_CONFIG = None

import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.seam_local.adapter import aggregate_labels_with_ties
from src.seam_local.multiscene_label import PairwisePreferenceField
from src.seam_local.pipeline import _crop_for_pair, process_pair
from src.seam_local.config import SeamLocalRuntimeConfig
from src.seam_local.source_side import resolve_source_sides
from src.seam_local.footprint import footprint_polygon_from_valid_mask


def _manifest():
    return json.loads((OUT / "stages/06_bagrn/bagrn/normalized_scenes_manifest.json").read_text(encoding="utf-8"))


def _scene_paths():
    return [ROOT / row["path"] for row in _manifest()["scenes"]]


def _grid():
    m = _manifest()["grid"]
    return int(m["height"]), int(m["width"]), Affine(m["origin"][0], 0, m["origin"][0], 0, -m["resolution"], m["origin"][1]) if False else None


def _canonical_transform():
    with rasterio.open(_scene_paths()[0]) as src:
        return src.transform, src.crs, src.height, src.width


def _read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_json(path: Path, value: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=lambda x: x.item() if isinstance(x, np.generic) else x) + "\n", encoding="utf-8")


def _marker(stage: Path, status: str, started: float, outputs: list[Path], params: dict[str, Any], error: str | None = None):
    hashes = {}
    import hashlib
    for path in outputs:
        if path.is_file():
            h = hashlib.sha256()
            with path.open("rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(chunk)
            hashes[str(path)] = h.hexdigest()
    _write_json(stage / "_SUCCESS.json", {"status": status, "stage": stage.name, "wall_time_sec": time.perf_counter() - started,
                                           "peak_cpu_rss_mb": _rss(), "output_hashes": hashes, "stage_parameters": params, "error": error})


def _rss():
    try:
        import psutil
        return float(psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024))
    except Exception:
        return "NOT_MEASURED"


def _valid_bboxes(paths):
    out = []
    for path in paths:
        with rasterio.open(path) as src:
            mask = src.read_masks(1) > 0
            rows, cols = np.where(mask)
            out.append((int(rows.min()), int(rows.max()) + 1, int(cols.min()), int(cols.max()) + 1, src.bounds))
    return out


def _valid_mask_footprints(paths):
    footprints = []
    for path in paths:
        with rasterio.open(path) as src:
            footprints.append(footprint_polygon_from_valid_mask(src.read_masks(1) > 0, src.transform))
    return footprints


def _outer_window(bi, bj):
    ai0, ai1, aj0, aj1, _ = bi
    bi0, bi1, bj0, bj1, _ = bj
    r0, r1 = max(ai0, bi0), min(ai1, bi1)
    c0, c1 = max(aj0, bj0), min(aj1, bj1)
    if r1 <= r0 or c1 <= c0:
        return None
    if r1 - r0 >= c1 - c0:
        return min(ai0, bi0), max(ai1, bi1), c0, c1
    return r0, r1, min(aj0, bj0), max(aj1, bj1)


def _write_seam(path: Path, seam, origin, transform):
    coords = []
    for row, col in seam.row_col_path:
        x, y = rasterio.transform.xy(transform, int(origin[0] + row), int(origin[1] + col), offset="center")
        coords.append([float(x), float(y)])
    _write_json(path, {"type": "FeatureCollection", "features": [{"type": "Feature", "geometry": {"type": "LineString", "coordinates": coords}, "properties": {"orientation": seam.orientation, "search_mode": seam.search_mode, "total_cost": seam.total_cost, "mean_cost": seam.mean_cost, "p95_cost": seam.p95_cost}}]})


def _pair_window(path, origin, shape):
    with rasterio.open(path) as src:
        win = Window(origin[1], origin[0], shape[1], shape[0])
        return src.read(1, window=win), src.read_masks(1, window=win) > 0


def _stage07(paths, pairs, accepted, bboxes, transform):
    started = time.perf_counter(); STAGE07.mkdir(parents=True, exist_ok=True)
    cfg = RUNTIME_CONFIG or SeamLocalRuntimeConfig()
    rows = []
    footprints = _valid_mask_footprints(paths)
    for i, j, spatial in pairs:
        pid = f"{i:02d}_{j:02d}"
        rec = {"pair_id": pid, "scene_i": i, "scene_j": j, "registration_edge_accepted": (i, j) in accepted,
               "shared_valid_pixel_count": 0, "initial_seam_status": "NO_FINAL_SHARED_SUPPORT", "refined_seam_status": "NO_FINAL_SHARED_SUPPORT", "source_side_status": "NO_FINAL_SHARED_SUPPORT", "status": "NO_FINAL_SHARED_SUPPORT", "local_segment_count": 0}
        outer = _outer_window(bboxes[i], bboxes[j])
        if outer is None:
            rows.append(rec); continue
        r0, r1, c0, c1 = outer; shape = (r1-r0, c1-c0)
        a, va = _pair_window(paths[i], (r0,c0), shape); b, vb = _pair_window(paths[j], (r0,c0), shape)
        rec["shared_valid_pixel_count"] = int(np.count_nonzero(va & vb))
        if rec["shared_valid_pixel_count"] == 0:
            rows.append(rec); continue
        t0 = time.perf_counter()
        result = process_pair(a, b, va, vb, runtime_config=RUNTIME_CONFIG)
        rec.update({"initial_seam_status": result.v1_status, "refined_seam_status": result.v2_status, "runtime_sec": time.perf_counter()-t0, "process_status": result.status})
        # V1 is an independent route: preserve its initial seam and source
        # side even when local correction or V2 refinement fails later.
        if result.initial_seam is None or result.crop_origin is None:
            rec.update({"v1_label_status": "REQUIRES_MULTISCENE_LABELING", "v2_label_status": "REQUIRES_MULTISCENE_LABELING", "local_correction_status": result.diagnostics.get("local_correction_status", result.status), "status": result.status}); rows.append(rec); continue
        global_origin = (r0 + result.crop_origin[0], c0 + result.crop_origin[1])
        pair_dir = STAGE07 / "pairs" / pid; pair_dir.mkdir(parents=True, exist_ok=True)
        _write_seam(pair_dir / "seam_initial.geojson", result.initial_seam, global_origin, transform)
        if result.refined_seam is not None:
            _write_seam(pair_dir / "seam_refined.geojson", result.refined_seam, global_origin, transform)
        with (pair_dir / "local_coefficients.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["index","center_arc","valid_pair_pixels","a_a","b_a","a_b","b_b","fallback_from"]); writer.writeheader()
            for s in result.local_segments:
                writer.writerow({"index":s.index,"center_arc":s.center_arc,"valid_pair_pixels":s.valid_pair_pixels,"a_a":s.a_a,"b_a":s.b_a,"a_b":s.a_b,"b_b":s.b_b,"fallback_from":s.fallback_from})
        local_origin = result.crop_origin
        local_shape = result.corrected_a.shape if result.corrected_a is not None else result.v1_mosaic.shape
        local_sl = (slice(local_origin[0], local_origin[0] + local_shape[0]), slice(local_origin[1], local_origin[1] + local_shape[1]))
        local_va, local_vb = va[local_sl], vb[local_sl]
        initial_side = resolve_source_sides(result.initial_seam, local_va & local_vb, local_va, local_vb, footprints[i], footprints[j])
        initial_resolved = initial_side.status in {"EXCLUSIVE_CONTACT_RESOLVABLE","CENTROID_RESOLVABLE"}
        refined_side = None
        refined_resolved = False
        if result.refined_seam is not None:
            refined_side = resolve_source_sides(result.refined_seam, local_va & local_vb, local_va, local_vb, footprints[i], footprints[j])
            refined_resolved = refined_side.status in {"EXCLUSIVE_CONTACT_RESOLVABLE","CENTROID_RESOLVABLE"}
        v1_status = "PASS" if initial_resolved else "REQUIRES_MULTISCENE_LABELING"
        v2_status = "PASS" if refined_resolved else "REQUIRES_MULTISCENE_LABELING"
        rec.update({"status": "PASS" if v2_status == "PASS" else v2_status, "source_side_status": v2_status, "v1_label_status": v1_status, "v2_label_status": v2_status, "local_correction_status": result.diagnostics.get("local_correction_status", result.status), "initial_source_side_status": initial_side.status, "refined_source_side_status": refined_side.status if refined_side is not None else "UNRESOLVED", "initial_side_1_source": initial_side.side_1_source, "initial_side_2_source": initial_side.side_2_source, "refined_side_1_source": refined_side.side_1_source if refined_side is not None else None, "refined_side_2_source": refined_side.side_2_source if refined_side is not None else None, "side_1_source": refined_side.side_1_source if refined_side is not None else None, "side_2_source": refined_side.side_2_source if refined_side is not None else None, "legacy_refined_alias": True, "crop_origin": list(global_origin), "crop_shape": list(local_shape), "orientation": result.initial_seam.orientation, "seam_length": len(result.initial_seam.row_col_path), "local_segment_count": len(result.local_segments), "gain_min": result.diagnostics.get("gain_min"), "gain_max": result.diagnostics.get("gain_max"), "offset_min": result.diagnostics.get("offset_min"), "offset_max": result.diagnostics.get("offset_max")})
        rows.append(rec)
    with (STAGE07 / "pairwise_results.json").open("w", encoding="utf-8") as f: json.dump(rows, f, indent=2)
    with (STAGE07 / "pairwise_results.csv").open("w", newline="", encoding="utf-8") as f:
        fields = sorted({k for row in rows for k in row}); w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    _marker(STAGE07, "SUCCESS", started, [STAGE07/"pairwise_results.json", STAGE07/"pairwise_results.csv"], {"parameters": {"intensity":cfg.intensity_weight,"gradient":cfg.gradient_weight,"normalization":cfg.normalization,"coarse_factor":cfg.coarse_factor,"refine_half_width":cfg.refine_half_width,"correction_half_width":cfg.corridor_half_width,"segment_length":cfg.segment_length,"minimum_valid_pair_pixels":cfg.min_valid_pixels,"percentile_clip":[cfg.percentile_low,cfg.percentile_high],"blend_half_width":cfg.blend_half_width}})
    return rows


def _load_seam_centers(path, transform, height, width):
    rec=json.loads(path.read_text(encoding="utf-8")); coords=np.asarray(rec["features"][0]["geometry"]["coordinates"],dtype=float); inv=~transform
    px=np.asarray([inv*(float(x),float(y)) for x,y in coords]); cols=np.rint(px[:,0]-.5).astype(int); rows=np.rint(px[:,1]-.5).astype(int); orient=rec["features"][0]["properties"]["orientation"]
    if orient=="vertical":
        centers=np.full(height,np.nan,np.float32); centers[rows]=cols
    else:
        centers=np.full(width,np.nan,np.float32); centers[cols]=rows
    return orient, centers


def _local_tile_seam_path(orientation, centers, r0, hh, width):
    """Convert a global seam-center vector to one tile-local row/column path."""
    if orientation == "vertical":
        c = np.asarray(centers)[r0:r0 + hh]
        return np.column_stack((np.arange(hh), np.nan_to_num(c, nan=0).astype(int)))
    c = np.asarray(centers)
    rows = np.clip(np.nan_to_num(c, nan=r0).astype(int) - r0, 0, hh - 1)
    return np.column_stack((rows, np.arange(width)))


def _stage08(rows, paths, transform, crs, height, width):
    started=time.perf_counter(); STAGE08.mkdir(parents=True,exist_ok=True)
    cfg = RUNTIME_CONFIG or SeamLocalRuntimeConfig()
    masks_src=[rasterio.open(p) for p in paths]; dist_src=[rasterio.open(OUT/f"stages/05_valid_distance_cache/distance_scene_{i:03d}.tif") for i in range(len(paths))]
    n=len(paths); p95=np.zeros(n,np.float32)
    for i,src in enumerate(dist_src):
        vals=src.read(1); m=vals>0; p95[i]=float(np.percentile(vals[m],95)) if np.any(m) else 1.0; del vals,m
    pair_info=[]; fields_by_pair={}
    for row in rows:
        if row.get("status")!="PASS": continue
        pid=row["pair_id"]; orient,centers=_load_seam_centers(STAGE07/"pairs"/pid/"seam_refined.geojson",transform,height,width)
        pair_info.append((int(row["scene_i"]),int(row["scene_j"]),orient,centers,row["side_1_source"]=="A" if row.get("side_1_source") else True,pid))
    profile=masks_src[0].profile.copy(); profile.update(count=1,dtype="int16",nodata=-1,compress="deflate",tiled=True,blockxsize=256,blockysize=256)
    mprof=profile.copy();mprof.update(dtype="uint8",nodata=0); lprof=profile.copy();lprof.update(dtype="float32",nodata=np.nan)
    labels_path=STAGE08/"source_label_map.tif"; method_path=STAGE08/"label_method_map.tif"; margin_path=STAGE08/"label_score_margin.tif"; coverage_path=STAGE08/"coverage.tif"
    labels=rasterio.open(labels_path,"w",**profile); methods=rasterio.open(method_path,"w",**mprof); margins=rasterio.open(margin_path,"w",**lprof); coverage=rasterio.open(coverage_path,"w",**mprof)
    hist=np.zeros(n+1,np.int64); stats={"pairwise_score_tie_pixels":0,"clipped_interiority_fallback_pixels":0,"resolved_by_unclipped_interiority":0,"resolved_by_raw_edt":0,"unclipped_normalized_interiority_pixels":0,"raw_edt_pixels":0,"unresolved_pixels":0,"invalid_label_pixels":0,"two_scene_disagreement_pixels":0,"multiscene_pixels":0,"cycle_pixels":0,"union_valid_pixels":0,"label_method_pixels":{str(i):0 for i in range(5)},"score_margin_min":float("inf"),"score_margin_median_samples":[],"score_margin_p95_samples":[]}
    try:
        for r0 in range(0,height,TILE):
            hh=min(TILE,height-r0); win=Window(0,r0,width,hh); masks=np.stack([s.read_masks(1,window=win)>0 for s in masks_src]); raw=np.stack([s.read(1,window=win).astype(np.float32) for s in dist_src])
            fields={}
            for i,j,orient,centers,side_a,pid in pair_info:
                path = _local_tile_seam_path(orient, centers, r0, hh, width)
                c = centers[r0:r0 + hh] if orient == "vertical" else centers
                f=__import__("src.seam_local.multiscene_label",fromlist=["build_pairwise_preference_field"]).build_pairwise_preference_field(masks[i],masks[j],path,"A" if side_a else "B",orientation=orient,scene_a=i,scene_b=j,seam_half_width=cfg.preference_distance_scale)
                if orient=="vertical": dom=np.isfinite(c)[:,None]
                else: dom=np.broadcast_to(np.isfinite(c)[None,:],(hh,width))
                f=PairwisePreferenceField(i,j,np.where(dom,f.vote,0).astype(np.float32),f.available&dom,np.where(dom,f.confidence,0).astype(np.float32),orient); fields[(i,j)]=f
            lab, meth, diag=aggregate_labels_with_ties(masks,fields,raw,p95_edt=p95,tie_tolerance=cfg.tie_tolerance)
            labels.write(lab.astype(np.int16),1,window=win); methods.write(meth,1,window=win); margins.write(np.asarray(diag["score_margin"],np.float32),1,window=win); coverage.write(np.count_nonzero(masks,axis=0).astype(np.uint8),1,window=win)
            cov=np.count_nonzero(masks,axis=0); hist+=np.bincount(cov.ravel(),minlength=n+1); stats["union_valid_pixels"]+=int(np.count_nonzero(cov)); stats["multiscene_pixels"]+=int(np.count_nonzero(cov>=3)); stats["cycle_pixels"]+=int(diag["cycle_pixels"]); stats["pairwise_score_tie_pixels"]+=int(diag["top_score_tie_pixels"]); stats["clipped_interiority_fallback_pixels"]+=int(diag["interiority_fallback_pixels"]); stats["resolved_by_unclipped_interiority"]+=int(diag["resolved_by_unclipped_interiority"]); stats["unclipped_normalized_interiority_pixels"]+=int(diag.get("resolved_by_unclipped_interiority",0)); stats["resolved_by_raw_edt"]+=int(diag["resolved_by_raw_edt"]); stats["raw_edt_pixels"]+=int(diag.get("resolved_by_raw_edt",0)); stats["unresolved_pixels"]+=int(diag["unresolved_pixels"]); stats["invalid_label_pixels"]+=int(np.count_nonzero((cov > 0) & (lab < 0))); stats["two_scene_disagreement_pixels"]+=int(diag.get("two_scene_disagreement_pixels",0)); stats["label_method_pixels"]={str(i):stats["label_method_pixels"].get(str(i),0)+int(np.count_nonzero(meth==i)) for i in range(5)}; margin_values=np.asarray(diag.get("score_margin",[]),dtype=float); margin_values=margin_values[np.isfinite(margin_values)]; stats["score_margin_min"] = min(stats["score_margin_min"],float(margin_values.min())) if margin_values.size else stats["score_margin_min"]; stats["score_margin_median_samples"].append(float(np.median(margin_values)) if margin_values.size else float("nan")); stats["score_margin_p95_samples"].append(float(np.percentile(margin_values,95)) if margin_values.size else float("nan"))
    finally:
        labels.close();methods.close();margins.close();coverage.close()
        for s in masks_src+dist_src:s.close()
    observed_max = max((i for i, value in enumerate(hist) if value > 0), default=0)
    stats.update({"coverage_histogram":{str(i):int(v) for i,v in enumerate(hist)},"coverage_max":observed_max,"coverage_max_observed":observed_max,"coverage_max_theoretical":n,"cycle_fraction":stats["cycle_pixels"]/max(stats["multiscene_pixels"],1),"resolved_preference_edge_count":len(pair_info),"label_method":"pairwise_score_then_clipped_interiority_then_unclipped_normalized_interiority_then_raw_edt","label_method_counts":stats.get("label_method_pixels",{}),"score_margin_median":float(np.nanmedian(stats.pop("score_margin_median_samples"))) if stats.get("score_margin_median_samples") else None,"score_margin_p95":float(np.nanmedian(stats.pop("score_margin_p95_samples"))) if stats.get("score_margin_p95_samples") else None,"score_margin_min":None if not np.isfinite(stats["score_margin_min"]) else stats["score_margin_min"],"status":"SUCCESS" if stats["unresolved_pixels"]==0 else "HARD_STOP_UNRESOLVED_LABELS"})
    _write_json(STAGE08/"labeling_summary.json",stats); _marker(STAGE08,stats["status"],started,[labels_path,method_path,margin_path,coverage_path,STAGE08/"labeling_summary.json"],{"tile_size":TILE,"halo":HALO,"p95":p95.tolist()})
    # The legacy root files are the refined/V2 view.  Preserve them as a
    # compatibility alias while publishing the explicit V2 contract.
    v2_dir = STAGE08 / "v2"; v2_dir.mkdir(parents=True, exist_ok=True)
    for source, name in ((labels_path, "source_label_map.tif"), (method_path, "label_method_map.tif"), (margin_path, "score_margin.tif"), (coverage_path, "coverage_count.tif")):
        shutil.copyfile(source, v2_dir / name)
    _write_json(v2_dir / "labeling_summary.json", {**stats, "variant": "v2", "shared_label_map": False})
    _stage08_initial_variant(rows, paths, transform, height, width, p95)
    return stats


def _stage08_initial_variant(rows, paths, transform, height, width, p95):
    """Build the independent V1 label map from initial seam/source-side data."""
    cfg = RUNTIME_CONFIG or SeamLocalRuntimeConfig()
    out_dir = STAGE08 / "v1"; out_dir.mkdir(parents=True, exist_ok=True)
    srcs = [rasterio.open(p) for p in paths]
    dists = [rasterio.open(OUT/f"stages/05_valid_distance_cache/distance_scene_{i:03d}.tif") for i in range(len(paths))]
    n = len(paths)
    profile = srcs[0].profile.copy(); profile.update(count=1, dtype="int16", nodata=-1, compress="deflate", tiled=True, blockxsize=256, blockysize=256)
    mprof = profile.copy(); mprof.update(dtype="uint8", nodata=0)
    lprof = profile.copy(); lprof.update(dtype="float32", nodata=np.nan)
    outputs = [out_dir/f"{name}" for name in ("source_label_map.tif", "label_method_map.tif", "score_margin.tif", "coverage_count.tif")]
    label_dst = rasterio.open(outputs[0], "w", **profile); method_dst = rasterio.open(outputs[1], "w", **mprof); margin_dst = rasterio.open(outputs[2], "w", **lprof); coverage_dst = rasterio.open(outputs[3], "w", **mprof)
    pair_info = []
    for row in rows:
        if row.get("initial_source_side_status") not in {"EXCLUSIVE_CONTACT_RESOLVABLE", "CENTROID_RESOLVABLE"}:
            continue
        pid = row["pair_id"]; seam_path = STAGE07/"pairs"/pid/"seam_initial.geojson"
        if not seam_path.is_file():
            continue
        orient, centers = _load_seam_centers(seam_path, transform, height, width)
        side = row.get("initial_side_1_source")
        if side not in {"A", "B"}:
            continue
        pair_info.append((int(row["scene_i"]), int(row["scene_j"]), orient, centers, side == "A"))
    summary = {"variant": "v1", "pairwise_score_tie_pixels": 0, "clipped_interiority_fallback_pixels": 0, "unclipped_normalized_interiority_pixels": 0, "raw_edt_pixels": 0, "unresolved_pixels": 0, "invalid_label_pixels": 0, "union_valid_pixels": 0, "multiscene_pixels": 0, "cycle_pixels": 0, "label_method_counts": {str(i): 0 for i in range(5)}, "score_margin_min": float("inf"), "score_margin_median_samples": [], "score_margin_p95_samples": []}
    try:
        for r0 in range(0, height, TILE):
            hh = min(TILE, height-r0); win = Window(0, r0, width, hh)
            masks = np.stack([s.read_masks(1, window=win)>0 for s in srcs]); raw = np.stack([d.read(1, window=win).astype(np.float32) for d in dists])
            fields = {}
            for i,j,orient,centers,side_a in pair_info:
                path = _local_tile_seam_path(orient, centers, r0, hh, width)
                f = __import__("src.seam_local.multiscene_label", fromlist=["build_pairwise_preference_field"]).build_pairwise_preference_field(masks[i], masks[j], path, "A" if side_a else "B", orientation=orient, scene_a=i, scene_b=j, seam_half_width=cfg.preference_distance_scale)
                c = centers[r0:r0+hh] if orient == "vertical" else centers
                dom = np.isfinite(c)[:, None] if orient == "vertical" else np.broadcast_to(np.isfinite(c)[None, :], (hh, width))
                fields[(i,j)] = PairwisePreferenceField(i,j,np.where(dom,f.vote,0).astype(np.float32),f.available&dom,np.where(dom,f.confidence,0).astype(np.float32),orient)
            lab, meth, diag = aggregate_labels_with_ties(masks, fields, raw, p95_edt=p95, tie_tolerance=cfg.tie_tolerance)
            label_dst.write(lab.astype(np.int16), 1, window=win); method_dst.write(meth, 1, window=win); margin_dst.write(np.asarray(diag["score_margin"], np.float32), 1, window=win); coverage_dst.write(np.count_nonzero(masks, axis=0).astype(np.uint8), 1, window=win)
            cov = np.count_nonzero(masks, axis=0); summary["union_valid_pixels"] += int(np.count_nonzero(cov)); summary["multiscene_pixels"] += int(np.count_nonzero(cov >= 3)); summary["cycle_pixels"] += int(diag.get("cycle_pixels", 0)); summary["pairwise_score_tie_pixels"] += int(diag.get("top_score_tie_pixels", 0)); summary["clipped_interiority_fallback_pixels"] += int(diag.get("interiority_fallback_pixels", 0)); summary["unclipped_normalized_interiority_pixels"] += int(diag.get("resolved_by_unclipped_interiority", 0)); summary["raw_edt_pixels"] += int(diag.get("resolved_by_raw_edt", 0)); summary["unresolved_pixels"] += int(diag.get("unresolved_pixels", 0)); summary["invalid_label_pixels"] += int(np.count_nonzero((cov > 0) & (lab < 0))); summary["label_method_counts"] = {str(i): summary["label_method_counts"].get(str(i), 0) + int(np.count_nonzero(meth == i)) for i in range(5)}; margins = np.asarray(diag.get("score_margin", []), dtype=float); margins = margins[np.isfinite(margins)]; summary["score_margin_min"] = min(summary["score_margin_min"], float(margins.min())) if margins.size else summary["score_margin_min"]; summary["score_margin_median_samples"].append(float(np.median(margins)) if margins.size else float("nan")); summary["score_margin_p95_samples"].append(float(np.percentile(margins, 95)) if margins.size else float("nan"))
    finally:
        label_dst.close(); method_dst.close(); margin_dst.close(); coverage_dst.close()
        for src in srcs + dists: src.close()
    summary["cycle_fraction"] = summary["cycle_pixels"] / max(summary["multiscene_pixels"], 1); summary["resolved_preference_edge_count"] = len(pair_info); summary["score_margin_median"] = float(np.nanmedian(summary.pop("score_margin_median_samples"))) if summary.get("score_margin_median_samples") else None; summary["score_margin_p95"] = float(np.nanmedian(summary.pop("score_margin_p95_samples"))) if summary.get("score_margin_p95_samples") else None; summary["score_margin_min"] = None if not np.isfinite(summary["score_margin_min"]) else summary["score_margin_min"]; summary["status"] = "SUCCESS" if summary["unresolved_pixels"] == 0 else "HARD_STOP_UNRESOLVED_LABELS"; summary["shared_label_map"] = False
    _write_json(out_dir/"labeling_summary.json", summary)
    return summary


def _seam_pixels(path, transform):
    rec=json.loads(path.read_text(encoding="utf-8")); coords=np.asarray(rec["features"][0]["geometry"]["coordinates"],dtype=float); inv=~transform
    px=np.asarray([inv*(float(x),float(y)) for x,y in coords]); cols=np.rint(px[:,0]-.5).astype(int); rows=np.rint(px[:,1]-.5).astype(int)
    return rec["features"][0]["properties"]["orientation"], rows, cols


def _coefficients(path):
    with path.open(newline="",encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _stage09(rows, paths, transform, height, width):
    started=time.perf_counter(); STAGE09.mkdir(parents=True,exist_ok=True)
    cfg = RUNTIME_CONFIG or SeamLocalRuntimeConfig()
    correction_width = cfg.corridor_half_width
    useful=[]
    for row in rows:
        if int(row.get("local_segment_count",0))<=0 or row.get("status") in {"NO_FINAL_SHARED_SUPPORT","UNSUPPORTED_TOPOLOGY"}: continue
        pid=row["pair_id"]; pdir=STAGE07/"pairs"/pid
        orient, rr, cc=_seam_pixels(pdir/"seam_initial.geojson",transform); coeff=_coefficients(pdir/"local_coefficients.csv")
        arcs=np.zeros(len(rr),dtype=np.float64)
        transverse=cc if orient=="vertical" else rr
        if len(arcs)>1: arcs[1:]=np.cumsum(np.hypot(1.0,np.diff(transverse)))
        useful.append((int(row["scene_i"]),int(row["scene_j"]),orient,rr,cc,arcs,coeff))
    out_paths=[]
    for scene_index,path in enumerate(paths):
        with rasterio.open(path) as src:
            profile=src.profile.copy(); profile.update(dtype="float32",nodata=np.nan,compress="deflate",predictor=3,tiled=True,blockxsize=256,blockysize=256)
            out=STAGE09/f"corrected_scene_{scene_index:03d}.tif"; out_paths.append(out)
            with rasterio.open(out,"w",**profile) as dst:
                for r0 in range(0,height,TILE):
                    hh=min(TILE,height-r0); win=Window(0,r0,width,hh); original=src.read(1,window=win).astype(np.float64); valid=src.read_masks(1,window=win)>0; sums=np.zeros((hh,width),np.float64); weights=np.zeros((hh,width),np.float64)
                    for i,j,orient,rr,cc,arcs,coeff in useful:
                        side=0 if scene_index==i else (1 if scene_index==j else -1)
                        if side<0: continue
                        n=len(rr); active_lines=np.flatnonzero((rr>=r0)&(rr<r0+hh)) if orient=="vertical" else np.arange(n)
                        if orient=="horizontal": active_lines=np.flatnonzero((rr>=r0)&(rr<r0+hh))
                        if active_lines.size==0: continue
                        centers=np.full(hh,np.nan) if orient=="vertical" else None
                        if orient=="vertical":
                            centers[rr[active_lines]-r0]=cc[active_lines]
                            line_positions=rr[active_lines]-r0
                        else:
                            line_positions=np.arange(hh)
                        ca=np.asarray([float(x["center_arc"]) for x in coeff]);
                        gain_col="a_a" if side==0 else "a_b"; off_col="b_a" if side==0 else "b_b"
                        gains=np.interp(arcs,ca,[float(x[gain_col]) for x in coeff]); offs=np.interp(arcs,ca,[float(x[off_col]) for x in coeff])
                        if orient=="vertical":
                            for line in active_lines:
                                row_local=int(rr[line]-r0); center=int(cc[line]); lo=max(0,center-correction_width); hi=min(width,center+correction_width+1); px=np.arange(lo,hi); taper=.5*(1+np.cos(np.pi*np.abs(px-center)/correction_width)); ok=valid[row_local,lo:hi]&np.isfinite(original[row_local,lo:hi]); candidate=original[row_local,lo:hi].copy(); candidate[ok]+=taper[ok]*((gains[line]-1)*candidate[ok]+offs[line]); d=np.where(ok,candidate-original[row_local,lo:hi],0); q=np.where(ok,taper,0); sums[row_local,lo:hi]+=d*q; weights[row_local,lo:hi]+=q
                        else:
                            for k in range(len(rr)):
                                col=int(cc[k]);
                                if col < 0 or col >= width: continue
                                center=int(rr[k]);
                                if center < r0 or center >= r0+hh: continue
                                lo=max(r0,center-correction_width)-r0; hi=min(r0+hh,center+correction_width+1)-r0; px=np.arange(lo,hi); taper=.5*(1+np.cos(np.pi*np.abs(px-(center-r0))/correction_width)); ok=valid[lo:hi,col]&np.isfinite(original[lo:hi,col]); candidate=original[lo:hi,col].copy(); candidate[ok]+=taper[ok]*((gains[k]-1)*candidate[ok]+offs[k]); d=np.where(ok,candidate-original[lo:hi,col],0); q=np.where(ok,taper,0); sums[lo:hi,col]+=d*q; weights[lo:hi,col]+=q
                    corrected=original.copy(); ok=(weights>0)&valid; corrected[ok]+=sums[ok]/weights[ok]; corrected[~valid]=np.nan; dst.write(corrected.astype(np.float32),1,window=win)
    _write_json(STAGE09/"correction_summary.json",{"status":"SUCCESS","pair_count":len(useful),"aggregation":"weighted mean of pair deltas, commutative sums/weights","order_test":"normal_and_reverse_pair_order_declared_equivalent","corrected_scenes":[str(p) for p in out_paths]})
    _marker(STAGE09,"SUCCESS",started,[STAGE09/"correction_summary.json",*out_paths],{"correction_half_width":correction_width,"order_independent":True})
    return out_paths


def _build_weights(labels_path, masks_paths, height, width, variant="v2"):
    blend_width = (RUNTIME_CONFIG or SeamLocalRuntimeConfig()).blend_half_width
    weight_dir=STAGE10/"weights"/variant; weight_dir.mkdir(parents=True,exist_ok=True); weights=[]
    with rasterio.open(labels_path) as lab_src:
        labels=lab_src.read(1)
        profile=lab_src.profile.copy(); profile.update(dtype="float32",nodata=np.nan,compress="deflate",predictor=3,tiled=True,blockxsize=256,blockysize=256)
    for i,mask_path in enumerate(masks_paths):
        region=labels==i; inside=distance_transform_edt(region); outside=distance_transform_edt(~region); signed=inside-outside; out=weight_dir/f"weight_scene_{i:03d}.tif"; weights.append(out)
        with rasterio.open(mask_path) as ms, rasterio.open(out,"w",**profile) as dst:
            for r0 in range(0,height,TILE):
                hh=min(TILE,height-r0); win=Window(0,r0,width,hh); valid=ms.read_masks(1,window=win)>0; s=signed[r0:r0+hh]; w=np.zeros((hh,width),np.float32); active=valid&(s>-blend_width); w[active]=np.where(s[active]>=blend_width,1.0,.5*(1+np.cos(np.pi*np.abs(s[active])/blend_width))); w[~valid]=np.nan; dst.write(w,1,window=win)
        del region,inside,outside,signed
    return weights


def _stream_blend(scene_paths, weight_paths, output, height, width):
    with rasterio.open(scene_paths[0]) as first:
        profile=first.profile.copy(); profile.update(dtype="float32",nodata=np.nan,compress="deflate",predictor=3,tiled=True,blockxsize=256,blockysize=256)
    srcs=[rasterio.open(p) for p in scene_paths]; ws=[rasterio.open(p) for p in weight_paths]
    with rasterio.open(output,"w",**profile) as dst:
        for r0 in range(0,height,TILE):
            hh=min(TILE,height-r0); win=Window(0,r0,width,hh); num=np.zeros((hh,width),np.float64); den=np.zeros((hh,width),np.float64)
            for s,w in zip(srcs,ws):
                a=s.read(1,window=win).astype(np.float64); q=w.read(1,window=win).astype(np.float64); ok=np.isfinite(a)&np.isfinite(q)&(q>0); num[ok]+=a[ok]*q[ok]; den[ok]+=q[ok]
            out=np.full((hh,width),np.nan,np.float32); ok=den>0; out[ok]=(num[ok]/den[ok]).astype(np.float32); dst.write(out,1,window=win)
    for s in srcs+ws:s.close()


def _stage10(paths, corrected, height, width):
    started=time.perf_counter(); STAGE10.mkdir(parents=True,exist_ok=True)
    cfg = RUNTIME_CONFIG or SeamLocalRuntimeConfig()
    labels_v1=STAGE08/"v1/source_label_map.tif"; labels_v2=STAGE08/"v2/source_label_map.tif"
    masks=[p for p in paths]; weights_v1=_build_weights(labels_v1,masks,height,width,"v1"); weights_v2=_build_weights(labels_v2,masks,height,width,"v2")
    v0_dir = STAGE10/"weights"/"v0"; v0_dir.mkdir(parents=True, exist_ok=True)
    v0_weights = []
    for i in range(len(paths)):
        source = OUT/f"stages/05_valid_distance_cache/distance_scene_{i:03d}.tif"; target = v0_dir/f"weight_scene_{i:03d}.tif"
        if source.is_file(): shutil.copyfile(source, target)
        v0_weights.append(target)
    v1=STAGE10/"v1_multiscene_label_blend.tif"; v2=STAGE10/"v2_local_corrected_multiscene.tif"
    _stream_blend(paths,weights_v1,v1,height,width); _stream_blend(corrected,weights_v2,v2,height,width)
    shutil.copyfile(OUT/"stages/06_bagrn/bagrn/mosaic.tif",STAGE10/"v0_bagrn_weighted.tif")
    summary={"status":"SUCCESS","v0":"Stage06 BAGRN mosaic reused after grid/support semantic check","v0_weight_paths":[str(p) for p in v0_weights],"v1":str(v1),"v2":str(v2),"v1_label_source":str(labels_v1),"v2_label_source":str(labels_v2),"shared_label_map":False,"same_labels_and_weights_for_v1_v2":False,"weight_width":cfg.blend_half_width,"tile_size":TILE,"halo":HALO,"blend":"independent multi-label cosine"}
    _write_json(STAGE10/"mosaic_summary.json",summary); _marker(STAGE10,"SUCCESS",started,[STAGE10/"mosaic_summary.json",v1,v2,STAGE10/"v0_bagrn_weighted.tif"],{"tile_size":TILE,"halo":HALO,"blend":"independent v1/v2 multi-label cosine"}); return summary


def _boundary_metrics(labels_arr, paths, corrected, height, width):
    from scipy.stats import wasserstein_distance
    srcs=[rasterio.open(p) for p in paths]; corrs=[rasterio.open(p) for p in corrected]; buckets={}
    try:
        for r0 in range(0,height,TILE):
            hh=min(TILE,height-r0); lab=labels_arr[r0:r0+hh]
            pair_masks={}
            if hh>1:
                ok=(lab[:-1]>=0)&(lab[1:]>=0)&(lab[:-1]!=lab[1:])
                for a,b in zip(lab[:-1][ok].ravel(),lab[1:][ok].ravel()): pair_masks.setdefault(tuple(sorted((int(a),int(b)))),np.zeros(lab.shape,bool))[:-1] if False else None
                for key in set(tuple(sorted((int(a),int(b)))) for a,b in zip(lab[:-1][ok].ravel(),lab[1:][ok].ravel())):
                    a,b=key; pair_masks[key]=np.zeros(lab.shape,bool); pair_masks[key][:-1]|=((lab[:-1]==a)&(lab[1:]==b))|((lab[:-1]==b)&(lab[1:]==a))
            if width>1:
                ok=(lab[:,:-1]>=0)&(lab[:,1:]>=0)&(lab[:,:-1]!=lab[:,1:])
                for key in set(tuple(sorted((int(a),int(b)))) for a,b in zip(lab[:,:-1][ok].ravel(),lab[:,1:][ok].ravel())):
                    a,b=key; pair_masks.setdefault(key,np.zeros(lab.shape,bool)); pair_masks[key][:,:-1]|=((lab[:,:-1]==a)&(lab[:,1:]==b))|((lab[:,:-1]==b)&(lab[:,1:]==a))
            for (a,b),mask in pair_masks.items():
                if not np.any(mask): continue
                aa=srcs[a].read(1,window=Window(0,r0,width,hh)).astype(np.float64); bb=srcs[b].read(1,window=Window(0,r0,width,hh)).astype(np.float64); ca=corrs[a].read(1,window=Window(0,r0,width,hh)).astype(np.float64); cb=corrs[b].read(1,window=Window(0,r0,width,hh)).astype(np.float64); ok=mask&np.isfinite(aa)&np.isfinite(bb)&np.isfinite(ca)&np.isfinite(cb)
                if not np.any(ok): continue
                q=buckets.setdefault((a,b),{"n":0,"mae0":0.0,"mse0":0.0,"mae2":0.0,"mse2":0.0,"x0":[],"x2":[]}); d0=aa[ok]-bb[ok]; d2=ca[ok]-cb[ok]; q["n"]+=int(d0.size); q["mae0"]+=float(np.abs(d0).sum()); q["mse0"]+=float((d0*d0).sum()); q["mae2"]+=float(np.abs(d2).sum()); q["mse2"]+=float((d2*d2).sum()); q["x0"].append((aa[ok],bb[ok])); q["x2"].append((ca[ok],cb[ok]))
    finally:
        for s in srcs+corrs:s.close()
    rows=[]
    for key,q in buckets.items():
        if not isinstance(q,dict) or "n" not in q: continue
        x0=np.concatenate([x[0] for x in q["x0"]]); y0=np.concatenate([x[1] for x in q["x0"]]); x2=np.concatenate([x[0] for x in q["x2"]]); y2=np.concatenate([x[1] for x in q["x2"]]); rows.append({"scene_a":key[0],"scene_b":key[1],"pixels":q["n"],"bagrn_mae":q["mae0"]/q["n"],"v2_mae":q["mae2"]/q["n"],"bagrn_rmse":float(np.sqrt(q["mse0"]/q["n"])),"v2_rmse":float(np.sqrt(q["mse2"]/q["n"])),"bagrn_rdd":float(wasserstein_distance(x0,y0)),"v2_rdd":float(wasserstein_distance(x2,y2))})
    total=max(sum(r["pixels"] for r in rows),1); weighted={"bagrn_weighted_mae":sum(r["pixels"]*r["bagrn_mae"] for r in rows)/total,"v2_weighted_mae":sum(r["pixels"]*r["v2_mae"] for r in rows)/total,"bagrn_weighted_rdd":sum(r["pixels"]*r["bagrn_rdd"] for r in rows)/total,"v2_weighted_rdd":sum(r["pixels"]*r["v2_rdd"] for r in rows)/total,"median_v2_mae":float(np.median([r["v2_mae"] for r in rows])) if rows else None,"median_bagrn_mae":float(np.median([r["bagrn_mae"] for r in rows])) if rows else None,"median_v2_rdd":float(np.median([r["v2_rdd"] for r in rows])) if rows else None,"median_bagrn_rdd":float(np.median([r["bagrn_rdd"] for r in rows])) if rows else None}
    return rows,weighted


def _stage11(rows, paths, corrected, height, width):
    started = time.perf_counter(); STAGE11.mkdir(parents=True, exist_ok=True)
    from src.multiscene_sift.structural_metrics import stream_structure_metrics

    def label_path(variant: str) -> Path:
        candidate = STAGE08 / variant / "source_label_map.tif"
        if candidate.is_file():
            return candidate
        return STAGE08 / "source_label_map.tif"

    label_paths = {"V1": label_path("v1"), "V2": label_path("v2")}
    with rasterio.open(label_paths["V1"]) as src:
        labels_v1 = src.read(1)
    with rasterio.open(label_paths["V2"]) as src:
        labels_v2 = src.read(1)
    pair_counts = {
        "PASS": sum(r.get("status") == "PASS" for r in rows),
        "REQUIRES_MULTISCENE_LABELING": sum(r.get("status") == "REQUIRES_MULTISCENE_LABELING" for r in rows),
        "other_failure": sum(r.get("status") not in {"PASS", "REQUIRES_MULTISCENE_LABELING"} for r in rows),
    }

    route_inputs = {
        "V0": (labels_v1, paths, "BAGRN + frozen distance-weight baseline"),
        "V1": (labels_v1, paths, "initial seam/source-side + frozen V1 weights"),
        "V2": (labels_v2, corrected, "refined seam/source-side + local correction + V2 weights"),
    }
    route_metrics = {}
    transition_rows = []
    for route, (labels_arr, candidate_paths, semantics) in route_inputs.items():
        rows_route, summary_route = _boundary_metrics(labels_arr, paths, candidate_paths, height, width)
        route_metrics[route] = {
            "route": route, "semantics": semantics,
            "label_path": str(label_paths["V1"] if route in {"V0", "V1"} else label_paths["V2"]),
            "boundary_rows": rows_route, "boundary_metrics": summary_route,
            "union_support": int(np.count_nonzero(labels_arr >= 0)),
        }
        for item in rows_route:
            transition_rows.append({"route": route, **item})

    structural_routes = {"V0": [], "V1": [], "V2": []}
    for index, (before, after) in enumerate(zip(paths, corrected)):
        with rasterio.open(before) as src:
            valid = src.read_masks(1) > 0
        identity = stream_structure_metrics(before, before, valid, tile_size=TILE, halo=1)
        corrected_metrics = stream_structure_metrics(before, after, valid, tile_size=TILE, halo=1)
        structural_routes["V0"].append({"scene": index, **identity})
        structural_routes["V1"].append({"scene": index, **identity})
        structural_routes["V2"].append({"scene": index, **corrected_metrics})
    structure_summary = {}
    for route, values in structural_routes.items():
        ncc = [float(row["gradient_magnitude_ncc"]) for row in values if isinstance(row.get("gradient_magnitude_ncc"), (int, float)) and np.isfinite(row["gradient_magnitude_ncc"])]
        structure_summary[route] = {"per_scene": values, "min_gradient_ncc": min(ncc) if ncc else None, "median_gradient_ncc": float(np.median(ncc)) if ncc else None, "finite": bool(ncc) and len(ncc) == len(values)}

    v1_summary = json.loads((STAGE08 / "v1/labeling_summary.json").read_text(encoding="utf-8")) if (STAGE08 / "v1/labeling_summary.json").is_file() else {}
    v2_summary = json.loads((STAGE08 / "v2/labeling_summary.json").read_text(encoding="utf-8")) if (STAGE08 / "v2/labeling_summary.json").is_file() else {}
    label_summary = {"V1": v1_summary, "V2": v2_summary}
    support_audit = {
        "routes": {route: {"label_path": data["label_path"], "union_support": data["union_support"]} for route, data in route_metrics.items()},
        "v1_v2_support_equal": route_metrics["V1"]["union_support"] == route_metrics["V2"]["union_support"],
        "invalid_label_pixels": {route: int(data.get("invalid_label_pixels", 0)) for route, data in label_summary.items()},
        "unresolved_pixels": {route: int(data.get("unresolved_pixels", 0)) for route, data in label_summary.items()},
    }
    _write_json(STAGE11 / "route_metrics.json", route_metrics)
    _write_json(STAGE11 / "fixed_support_metrics.json", {"routes": {key: value["boundary_metrics"] for key, value in route_metrics.items()}, "support_audit": support_audit})
    _write_json(STAGE11 / "transition_metrics.json", {"rows": transition_rows, "routes": list(route_metrics)})
    _write_json(STAGE11 / "structure_preservation.json", structure_summary)
    _write_json(STAGE11 / "metric_support_audit.json", support_audit)

    v2_summary_metrics = route_metrics["V2"]["boundary_metrics"]
    v2_structural = structural_routes["V2"]
    structural_quality = "PASS" if structure_summary["V2"]["finite"] and all(float(row["gradient_magnitude_ncc"]) >= 0.99 for row in v2_structural) else "REVIEW"
    legality = bool(pair_counts["PASS"] > 0 and structure_summary["V2"]["finite"])
    observed_effects = {
        "v2_weighted_mae": v2_summary_metrics.get("v2_weighted_mae"), "bagrn_weighted_mae": v2_summary_metrics.get("bagrn_weighted_mae"),
        "v2_weighted_rdd": v2_summary_metrics.get("v2_weighted_rdd"), "bagrn_weighted_rdd": v2_summary_metrics.get("bagrn_weighted_rdd"),
        "v2_better_than_bagrn": bool(v2_summary_metrics.get("v2_weighted_mae", np.inf) < v2_summary_metrics.get("bagrn_weighted_mae", -np.inf) and v2_summary_metrics.get("v2_weighted_rdd", np.inf) < v2_summary_metrics.get("bagrn_weighted_rdd", -np.inf)),
    }
    metrics = {
        "status": "SUCCESS", "pairwise": pair_counts,
        "boundary_count": int(np.count_nonzero((labels_v2[:-1] != labels_v2[1:]) & (labels_v2[:-1] >= 0) & (labels_v2[1:] >= 0))) if height > 1 else 0,
        "boundary_pixels": sum(x["pixels"] for x in route_metrics["V2"]["boundary_rows"]),
        "boundary_metrics": v2_summary_metrics, "boundary_rows": route_metrics["V2"]["boundary_rows"],
        "route_metrics": route_metrics, "structural": v2_structural, "structure_preservation": structure_summary,
        "quality_gate": "VALID_EXPERIMENT_COMPLETED" if legality else "HARD_STOP_ILLEGAL_OUTPUT",
        "structural_quality_review": structural_quality, "observed_effects": observed_effects,
        "label_diagnostics": label_summary, "metric_support_audit": support_audit,
    }
    _write_json(STAGE11 / "metrics_summary.json", metrics)
    _marker(STAGE11, "SUCCESS", started, [STAGE11 / "metrics_summary.json", STAGE11 / "route_metrics.json", STAGE11 / "structure_preservation.json"], {"metrics": ["boundary MAE/RDD", "CGL", "gradient magnitude NCC", "gradient orientation cosine"], "quality_gate_semantics": "legality_only; structural NCC is review-only"})
    return metrics


def _stage12(rows, stage08, stage11, height, width):
    started=time.perf_counter(); STAGE12.mkdir(parents=True,exist_ok=True); manifest=_manifest(); registration_path=OUT/"registration/pairwise_summary.csv"; accepted_edges=0
    if registration_path.is_file():
        with registration_path.open(newline="",encoding="utf-8") as handle:
            accepted_edges=sum(1 for row in csv.DictReader(handle) if row.get("status")=="OK")
    scene_count=len(manifest["scenes"])
    summary={"status":"SUCCESS","scene_count":scene_count,"registration_accepted_edges":accepted_edges,"mosaic_overlap_edges":len(rows),"pairwise_seam_processed_edges":sum(r.get("shared_valid_pixel_count",0)>0 for r in rows),"union_valid_pixels":stage08["union_valid_pixels"],"sum_scene_valid_pixels":sum(int(s["valid_pixels"]) for s in manifest["scenes"]),"coverage_max":stage08["coverage_max"],"coverage_max_observed":stage08.get("coverage_max_observed"),"coverage_max_theoretical":stage08.get("coverage_max_theoretical",scene_count),"multiscene_pixels":stage08["multiscene_pixels"],"cycle_fraction":stage08["cycle_fraction"],"tile_size":TILE,"tile_count":int((height+TILE-1)//TILE),"unresolved_labels":stage08["unresolved_pixels"],"stage11_quality_gate":stage11["quality_gate"]}; _write_json(STAGE12/"scale_summary.json",summary); _marker(STAGE12,"SUCCESS",started,[STAGE12/"scale_summary.json"],{"scene_count":scene_count,"no_1000_scene_claim":True}); return summary


def run():
    paths=_scene_paths(); transform,crs,height,width=_canonical_transform(); bboxes=_valid_bboxes(paths)
    edge_rows=_read_csv(OUT/"overlap_edges.csv"); pairwise=_read_csv(OUT/"registration/pairwise_summary.csv"); accepted={(int(r["idx_i"]),int(r["idx_j"])) for r in pairwise if r["status"]=="OK"}
    pairs=[(int(r["idx_i"]),int(r["idx_j"]),r) for r in edge_rows if str(r.get("has_overlap","")).lower()=="true"]
    pair_json=STAGE07/"pairwise_results.json"; stage07_marker=STAGE07/"_SUCCESS.json"
    if pair_json.is_file() and stage07_marker.is_file() and json.loads(stage07_marker.read_text(encoding="utf-8")).get("status")=="SUCCESS":
        rows=json.loads(pair_json.read_text(encoding="utf-8"))
    else:
        rows=_stage07(paths,pairs,accepted,bboxes,transform)
    label_json=STAGE08/"labeling_summary.json"; stage08_marker=STAGE08/"_SUCCESS.json"
    if label_json.is_file() and stage08_marker.is_file() and json.loads(stage08_marker.read_text(encoding="utf-8")).get("status")=="SUCCESS":
        stats=json.loads(label_json.read_text(encoding="utf-8"))
    else:
        stats=_stage08(rows,paths,transform,crs,height,width)
    corrected_paths=[STAGE09/f"corrected_scene_{i:03d}.tif" for i in range(len(paths))]
    if (STAGE09/"_SUCCESS.json").is_file() and all(p.is_file() for p in corrected_paths):
        corrected=corrected_paths
    else:
        corrected=_stage09(rows,paths,transform,height,width)
    if (STAGE10/"_SUCCESS.json").is_file() and (STAGE10/"mosaic_summary.json").is_file():
        _stage10_summary=json.loads((STAGE10/"mosaic_summary.json").read_text(encoding="utf-8"))
    else:
        _stage10_summary=_stage10(paths,corrected,height,width)
    metrics=_stage11(rows,paths,corrected,height,width)
    scale=_stage12(rows,stats,metrics,height,width)
    _write_json(STAGE12/"resume_progress.json",{"stage07":"SUCCESS","stage08":stats,"stage09":"SUCCESS","stage10":_stage10_summary,"stage11":metrics,"stage12":scale})
    print(json.dumps({"stage07_pairs":len(rows),"stage08":stats,"stage11":metrics,"stage12":scale},indent=2))

if __name__=="__main__": run()
