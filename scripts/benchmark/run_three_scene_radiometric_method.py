"""Run one fixed-grid three-scene radiometric method in isolation.

Methods: BAGRN-3, BAGRN+VOLRN-3, and the three pairwise LIRRN fits.
All inputs are frozen B9 GeoTIFFs; output directories are never overwritten.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import csv
import gc
import json
import os
from pathlib import Path
import sys
import threading
import time
import traceback

import numpy as np
import psutil
import rasterio
from affine import Affine
from rasterio.transform import array_bounds
from rasterio.windows import Window

ROOT = Path(__file__).resolve().parents[2]
PARENT = ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(PARENT))

from src.bagrn import bagrn_normalize
from src.volrn import volrn_normalize
from src.overlap import detect_multi_overlap
from src.benchmark_v1_lsstaa.scoring import score_pair
from src.benchmark_v1_unified.structure import compute_range_retention, compute_sobel_structure

FREEZE = ROOT.parent / "data/output/benchmark_v1_unified/freeze_20261013_v3"
OUTPUT_BASE = ROOT / "data/output/benchmark_v1_lsstaa/three_scene_method_comparison_20261015"
PAIR_IDS = ((0, 1, "00_01"), (0, 2, "00_02"), (1, 2, "01_02"))


class RssSampler:
    def __init__(self):
        self.stop = threading.Event()
        self.values = []
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self):
        process = psutil.Process(os.getpid())
        while not self.stop.wait(.1):
            try:
                rss = process.memory_info().rss
                for child in process.children(recursive=True):
                    try:
                        rss += child.memory_info().rss
                    except psutil.Error:
                        pass
                self.values.append(rss)
            except psutil.Error:
                break

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join(timeout=2)


def read_scene(index: int) -> dict:
    data_path = FREEZE / "aligned_inputs" / f"scene_{index:03d}_B9_input.tif"
    mask_path = FREEZE / "aligned_inputs" / f"scene_{index:03d}_input_mask.tif"
    r0, c0, r1, c1 = 10**9, 10**9, -1, -1
    with rasterio.open(mask_path) as mask:
        for row in range(0, mask.height, 512):
            for col in range(0, mask.width, 512):
                win = Window(col, row, min(512, mask.width-col), min(512, mask.height-row))
                tile = mask.read(1, window=win) != 0
                if tile.any():
                    rr, cc = np.where(tile)
                    r0, c0 = min(r0, row+int(rr.min())), min(c0, col+int(cc.min()))
                    r1, c1 = max(r1, row+int(rr.max())+1), max(c1, col+int(cc.max())+1)
        if r1 < 0:
            raise ValueError(f"EMPTY_FROZEN_MASK:{index}")
        window = Window(c0, r0, c1-c0, r1-r0)
        valid = mask.read(1, window=window) != 0
        transform = mask.window_transform(window)
        profile = mask.profile.copy()
    with rasterio.open(data_path) as data:
        if (data.width, data.height, data.transform, data.crs) != (profile["width"], profile["height"], mask.transform, mask.crs):
            raise ValueError(f"FROZEN_DATA_MASK_GRID_MISMATCH:{index}")
        values = data.read(1, window=window).astype(np.float32)
    values[~valid] = np.nan
    bounds = tuple(float(v) for v in array_bounds(values.shape[0], values.shape[1], transform))
    return {"index": index, "values": values[None, :, :], "mask": valid,
            "transform": transform, "bounds": (bounds[0], bounds[1], bounds[2], bounds[3]),
            "row_off": r0, "col_off": c0, "profile": profile,
            "data_path": data_path, "mask_path": mask_path,
            "valid_pixels": int(valid.sum())}


def overlap_records(scenes: list[dict]) -> list[dict]:
    records = detect_multi_overlap([s["bounds"] for s in scenes],
                                   [s["transform"] for s in scenes], min_pixels=100)
    found = {(r["idx_i"], r["idx_j"]) for r in records}
    expected = {(0, 1), (0, 2), (1, 2)}
    if found != expected:
        raise ValueError(f"THREE_SCENE_OVERLAP_GRAPH_MISMATCH:{found}")
    return records


def save_scene(scene: dict, result: np.ndarray, out: Path, name: str) -> tuple[Path, Path]:
    data_path = out / f"scene_{scene['index']:02d}_{name}_B9.tif"
    mask_path = out / f"scene_{scene['index']:02d}_{name}_mask.tif"
    if data_path.exists() or mask_path.exists():
        raise FileExistsError(data_path)
    full = scene["profile"].copy()
    full.update(dtype="float32", count=1, nodata=np.nan, compress="deflate", predictor=3)
    mprof = scene["profile"].copy()
    mprof.update(dtype="uint8", count=1, nodata=0, compress="deflate")
    cr0, cc0 = scene["row_off"], scene["col_off"]
    cr1, cc1 = cr0+result.shape[0], cc0+result.shape[1]
    with rasterio.open(data_path, "w", **full) as dst, rasterio.open(mask_path, "w", **mprof) as mds, \
         rasterio.open(scene["mask_path"]) as src_mask:
        for row in range(0, full["height"], 512):
            for col in range(0, full["width"], 512):
                win = Window(col, row, min(512, full["width"]-col), min(512, full["height"]-row))
                valid = src_mask.read(1, window=win) != 0
                h, w = int(win.height), int(win.width)
                tile = np.full((h, w), np.nan, dtype=np.float32)
                rlo, clo = max(row, cr0), max(col, cc0)
                rhi, chi = min(row+h, cr1), min(col+w, cc1)
                if rhi > rlo and chi > clo:
                    src = result[rlo-cr0:rhi-cr0, clo-cc0:chi-cc0]
                    tile[rlo-row:rhi-row, clo-col:chi-col] = src
                if np.any(valid & ~np.isfinite(tile)):
                    raise ValueError(f"MISSING_CORRECTED_VALID_PIXEL:{scene['index']}:{row}:{col}")
                dst.write(tile, 1, window=win)
                mds.write(valid.astype(np.uint8), 1, window=win)
    return data_path, mask_path


def pair_windows() -> dict[str, list[dict]]:
    rows = list(csv.DictReader((FREEZE / "fixed_windows.csv").open(encoding="utf-8-sig", newline="")))
    return {pair: [r for r in rows if r["pair_id"] == pair] for _, _, pair in PAIR_IDS}


def score_structure(raw_data: Path, raw_mask: Path, output_data: Path,
                    records: list[dict]) -> list[dict]:
    values = []
    with rasterio.open(raw_data) as raw, rasterio.open(raw_mask) as mask, rasterio.open(output_data) as corrected:
        for row in records:
            win = Window(int(row["col_off"]), int(row["row_off"]), int(row["width"]), int(row["height"]))
            valid = mask.read(1, window=win) != 0
            x, y = raw.read(1, window=win), corrected.read(1, window=win)
            values.append({"window_id": row["window_id"],
                           "sobel": asdict(compute_sobel_structure(x, y, valid)),
                           "dynamic_range": asdict(compute_range_retention(x, y, valid))})
    return values


def file_sha256(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run_global(method: str, out: Path, scenes: list[dict], overlaps: list[dict]) -> dict:
    arrays = [s["values"] for s in scenes]
    masks = [s["mask"] for s in scenes]
    transforms = [s["transform"] for s in scenes]
    bounds = [s["bounds"] for s in scenes]
    times = {}
    start = time.perf_counter()
    bagrn_result, theta_mu, theta_sigma = bagrn_normalize(
        arrays, [None]*3, overlaps, control_idx=0,
        # Frozen NoData masks are represented by NaN in the crop arrays.
    )
    times["bagrn_seconds"] = time.perf_counter()-start
    detail = {"theta_mu": theta_mu.tolist(), "theta_sigma": theta_sigma.tolist()}
    if method == "bagrn_volrn":
        start = time.perf_counter()
        final, coefficients, diagnostics = volrn_normalize(
            bagrn_result, transforms, bounds, [None]*3, block_size_pixels=400,
            lambda_param=.5, rho=1.0, max_iter=200, tol=1e-4,
            return_diagnostics=True, valid_masks=masks,
        )
        times["volrn_seconds"] = time.perf_counter()-start
        detail["volrn"] = {"all_converged": diagnostics.get("all_converged"),
                            "n_blocks": diagnostics.get("n_blocks"),
                            "n_pairs": diagnostics.get("n_pairs"),
                            "band_converged": diagnostics.get("band_converged"),
                            "band_iterations": diagnostics.get("band_iterations"),
                            "solver": diagnostics.get("band_solver_diagnostics"),
                            "block_coefficients_shape": list(coefficients.shape),
                            "settings": {"block_size_pixels": 400, "lambda": .5, "rho": 1.0,
                                         "max_iter": 200, "tol": 1e-4}}
    else:
        final = bagrn_result
    del arrays, bagrn_result
    gc.collect()
    paths = []
    write_start = time.perf_counter()
    for scene, result in zip(scenes, final):
        paths.append(save_scene(scene, result[0], out, method))
    times["write_seconds"] = time.perf_counter()-write_start
    del final
    gc.collect()
    return {"algorithm_elapsed_seconds": times["bagrn_seconds"] + times.get("volrn_seconds", 0.0),
            "stage_times": times, "parameters": detail, "outputs": [list(map(str, pair)) for pair in paths]}


def run_lirrn(out: Path, scenes: list[dict], windows: dict) -> dict:
    from src.benchmark_v1_lirrn.frozen_crop import RasterCrop
    from src.benchmark_v1_lirrn.official_port import (
        classify_three_levels, draw_shared_ids, select_lirrn_samples,
        fit_lirrn_linear, apply_lirrn_linear,
    )
    results, times = {}, {}
    for i, j, pair_id in PAIR_IDS:
        start = time.perf_counter()
        left, right = scenes[i], scenes[j]
        r0 = min(left["row_off"], right["row_off"])
        c0 = min(left["col_off"], right["col_off"])
        r1 = max(left["row_off"]+left["values"].shape[1], right["row_off"]+right["values"].shape[1])
        c1 = max(left["col_off"]+left["values"].shape[2], right["col_off"]+right["values"].shape[2])
        shape = (r1-r0, c1-c0)
        crops = []
        for scene in (left, right):
            value = np.zeros(shape, dtype=np.float32)
            valid = np.zeros(shape, dtype=bool)
            sr, sc = scene["row_off"]-r0, scene["col_off"]-c0
            h, w = scene["mask"].shape
            value[sr:sr+h, sc:sc+w] = scene["values"][0]
            valid[sr:sr+h, sc:sc+w] = scene["mask"]
            crops.append(RasterCrop(value, valid, Affine(14, 0, 0, 0, -14, 0), "EPSG:32650", scene["index"]))
        ref, sub = crops
        ref_labels, ref_otsu = classify_three_levels(ref.values, ref.valid)
        sub_labels, sub_otsu = classify_three_levels(sub.values, sub.valid)
        p_n = 1000
        classes = [min(ref_otsu["class_counts"][k], sub_otsu["class_counts"][k]) for k in range(3)]
        fallback = min(classes) < p_n
        if fallback:
            p_n = 200
        if min(classes) < p_n:
            raise ValueError(f"LIRRN_INSUFFICIENT_FULL_SCENE_STRATA:{pair_id}:{classes}")
        seed = 20261015 + i*10 + j
        ids = draw_shared_ids(p_n, seed)
        x, y, sample_audit = select_lirrn_samples(ref, sub, ref_labels, sub_labels, ids, p_n)
        model = fit_lirrn_linear(x, y)
        corrected_union = apply_lirrn_linear(sub, model)
        algorithm_seconds = time.perf_counter()-start
        # Store per-pair subject output with a distinct path and full canonical grid.
        pair_out = out / "lirrn_pairs" / pair_id
        pair_out.mkdir(parents=True, exist_ok=False)
        data_path = pair_out / f"scene_{j:02d}_LIRRN_PAIR_ONLY_B9.tif"
        mask_out = pair_out / f"scene_{j:02d}_valid_mask.tif"
        profile = right["profile"].copy(); profile.update(dtype="float32", count=1, nodata=np.nan, compress="deflate", predictor=3)
        mp = right["profile"].copy(); mp.update(dtype="uint8", count=1, nodata=0, compress="deflate")
        cr0, cc0 = right["row_off"], right["col_off"]
        cr1, cc1 = cr0+right["mask"].shape[0], cc0+right["mask"].shape[1]
        with rasterio.open(data_path, "w", **profile) as dst, rasterio.open(mask_out, "w", **mp) as md, rasterio.open(right["mask_path"]) as raw_mask, rasterio.open(right["data_path"]) as raw_data:
            for row in range(0, profile["height"], 512):
                for col in range(0, profile["width"], 512):
                    win = Window(col, row, min(512, profile["width"]-col), min(512, profile["height"]-row))
                    valid = raw_mask.read(1, window=win) != 0
                    tile = np.full((int(win.height), int(win.width)), np.nan, np.float32)
                    rlo, clo, rhi, chi = max(row, cr0), max(col, cc0), min(row+int(win.height), cr1), min(col+int(win.width), cc1)
                    if rhi > rlo and chi > clo:
                        src = raw_data.read(1, window=Window(clo, rlo, chi-clo, rhi-rlo))
                        sr, sc = rlo-r0, clo-c0
                        corrected = corrected_union[sr:sr+(rhi-rlo), sc:sc+(chi-clo)]
                        tile[rlo-row:rhi-row, clo-col:chi-col] = corrected
                    tile[~valid] = np.nan
                    if np.any(valid & ~np.isfinite(tile)):
                        raise ValueError(f"LIRRN_OUTPUT_HOLE:{pair_id}:{row}:{col}")
                    dst.write(tile, 1, window=win)
                    md.write(valid.astype(np.uint8), 1, window=win)
        write_seconds = time.perf_counter()-start-algorithm_seconds
        results[pair_id] = {"subject_scene_index": j, "reference_scene_index": i,
            "status": "SUCCESS_PAIR_ONLY_DIAGNOSTIC", "scale": model.scale, "offset": model.offset,
            "sample_count": model.sample_count, "fit_rmse": model.fit_rmse,
            "adjusted_r2": model.adjusted_r2, "rank": model.rank,
            "condition_number": model.condition_number, "requested_p_n": 1000,
            "actual_p_n": p_n, "fallback_to_200": fallback, "rng_seed": seed,
            "class_counts_min_pairwise": classes, "sampling_audit": sample_audit,
            "algorithm_elapsed_seconds": algorithm_seconds, "write_seconds": write_seconds, "output_data": str(data_path),
            "output_mask": str(mask_out)}
        del crops, ref_labels, sub_labels, x, y
        gc.collect()
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("bagrn", "bagrn_volrn", "lirrn_pairs"), required=True)
    parser.add_argument("--run-id", default="")
    args = parser.parse_args()
    out = OUTPUT_BASE / (args.method if not args.run_id else f"{args.method}_{args.run_id}")
    if out.exists():
        raise FileExistsError(f"refusing overwrite: {out}")
    out.mkdir(parents=True)
    status = {"method": args.method, "status": "RUNNING", "freeze_root": str(FREEZE),
              "scene_indices": [0, 1, 2], "formal_13scene_status": "NOT_RUN"}
    start = time.perf_counter()
    sampler = RssSampler()
    sampler.__enter__()
    try:
        lock_path = FREEZE / "freeze_lock.json"
        from src.benchmark_v1_unified.freeze import verify_lock
        lock = verify_lock(FREEZE)
        if not lock.valid:
            raise ValueError(f"FREEZE_LOCK_FAILED:{lock.errors}")
        manifest = json.loads((FREEZE / "code_manifest.json").read_text(encoding="utf-8"))
        for item in manifest["files"]:
            path = ROOT / item["path"]
            import hashlib
            h = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
            if h != item["sha256"]:
                raise ValueError(f"FROZEN_CODE_MANIFEST_MISMATCH:{item['path']}")
        windows = pair_windows()
        status["freeze_lock_entries"] = len(json.loads(lock_path.read_text(encoding="utf-8"))["files"])
        status["freeze_code_manifest_count"] = len(manifest["files"])
        load_start = time.perf_counter()
        scenes = [read_scene(i) for i in range(3)]
        overlaps = overlap_records(scenes)
        status["input_scenes"] = [{"scene_index": s["index"], "valid_pixels": s["valid_pixels"],
            "bbox_shape": list(s["mask"].shape), "bbox_origin": [s["row_off"], s["col_off"]]} for s in scenes]
        status["overlap_edges"] = [{"scene_i": o["idx_i"], "scene_j": o["idx_j"], "window_i": list(o["window_i"]),
                                    "window_j": list(o["window_j"]), "geometric_pixels": o["pixel_count"]} for o in overlaps]
        status["input_load_seconds"] = time.perf_counter()-load_start
        algorithm_start = time.perf_counter()
        if args.method == "lirrn_pairs":
            status["pair_results"] = run_lirrn(out, scenes, windows)
            method_pairs = {pair_id: (scenes[i], scenes[j], out/"lirrn_pairs"/pair_id/f"scene_{j:02d}_LIRRN_PAIR_ONLY_B9.tif", out/"lirrn_pairs"/pair_id/f"scene_{j:02d}_valid_mask.tif")
                            for i,j,pair_id in PAIR_IDS}
        else:
            method_result = run_global(args.method, out, scenes, overlaps)
            status["algorithm"] = method_result
            method_paths = [(Path(data), Path(mask)) for data,mask in method_result["outputs"]]
            corrected_data_paths = [p[0] for p in method_paths]
            method_pairs = {pair_id: (scenes[i], scenes[j], method_paths[i][0], method_paths[j][0], method_paths[i][1], method_paths[j][1])
                            for i,j,pair_id in PAIR_IDS}
        status["algorithm_elapsed_seconds"] = time.perf_counter()-algorithm_start
        if args.method == "lirrn_pairs":
            status["algorithm_elapsed_seconds"] = sum(x["algorithm_elapsed_seconds"] for x in status["pair_results"].values())
            status["algorithm_write_seconds"] = sum(x["write_seconds"] for x in status["pair_results"].values())
        else:
            status["algorithm_elapsed_seconds"] = method_result["algorithm_elapsed_seconds"]
            status["algorithm_write_seconds"] = method_result["stage_times"]["write_seconds"]
        status["algorithm_peak_rss_bytes"] = max(sampler.values, default=None)
        status["status"] = "ALGORITHM_COMPLETE"
        score_start = time.perf_counter()
        pair_scores = {}
        for i,j,pair_id in PAIR_IDS:
            records = windows[pair_id]
            if args.method == "lirrn_pairs":
                _, _, out_data, out_mask = method_pairs[pair_id]
                raw_score = score_pair((scenes[i]["data_path"], scenes[i]["mask_path"]),
                                       (scenes[j]["data_path"], scenes[j]["mask_path"]), records)
                corrected_score = score_pair((scenes[i]["data_path"], scenes[i]["mask_path"]),
                                             (out_data, out_mask), records)
                structure = score_structure(scenes[j]["data_path"], scenes[j]["mask_path"], out_data, records)
            else:
                si, sj, di, dj, mi, mj = method_pairs[pair_id]
                raw_score = score_pair((si["data_path"], si["mask_path"]), (sj["data_path"], sj["mask_path"]), records)
                corrected_score = score_pair((di, mi), (dj, mj), records)
                structure = None
            pair_scores[pair_id] = {"raw": raw_score, "corrected": corrected_score,
                                    "subject_structure": structure}
        if args.method != "lirrn_pairs":
            scene_structure = {}
            for k, scene in enumerate(scenes):
                relevant = [r for _, _, pair_id in PAIR_IDS
                            if k in tuple(map(int, pair_id.split("_")))
                            for r in windows[pair_id]]
                result_path = corrected_data_paths[k]
                scene_structure[str(k)] = score_structure(scene["data_path"], scene["mask_path"], result_path, relevant)
            status["scene_structure"] = scene_structure
        status["scoring_seconds"] = time.perf_counter()-score_start
        status["pair_scores"] = pair_scores
        if args.method == "lirrn_pairs":
            for pair_id, pair in status["pair_results"].items():
                pair["data_sha256"] = file_sha256(Path(pair["output_data"]))
                pair["mask_sha256"] = file_sha256(Path(pair["output_mask"]))
        else:
            status["algorithm"]["outputs_sha256"] = [
                {"data": file_sha256(Path(data)), "mask": file_sha256(Path(mask))}
                for data, mask in status["algorithm"]["outputs"]]
        status["status"] = "SUCCESS_DIAGNOSTIC"
    except Exception as exc:
        status["status"] = "FAILED_" + type(exc).__name__.upper()
        status["error"] = f"{type(exc).__name__}: {exc}"
        status["traceback"] = traceback.format_exc()
        print(status["error"], file=sys.stderr, flush=True)
    finally:
        sampler.__exit__()
        status["algorithm_total_wall_seconds"] = time.perf_counter()-start
        status["rss_peak_bytes"] = max(sampler.values, default=None)
        status["rss_sample_count"] = len(sampler.values)
        status["rss_status"] = "SAMPLED" if sampler.values else "NOT_MEASURED"
        status["os_hard_memory_limit"] = False
        (out/"method_status.json").write_text(json.dumps(status, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return 0 if status["status"] == "SUCCESS_DIAGNOSTIC" else 2


if __name__ == "__main__":
    raise SystemExit(main())
