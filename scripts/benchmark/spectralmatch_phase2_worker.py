from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image, ImageDraw
from rasterio.windows import Window

from src.benchmark_v1_spectralmatch_phase2.adapter import execute_fixed_grid_stages, prepare_masked_copy
from src.benchmark_v1_spectralmatch_phase2.freeze_audit import audit_frozen_inputs
from src.benchmark_v1_spectralmatch_phase2 import UPSTREAM_COMMIT, UPSTREAM_VERSION
from src.benchmark_v1_unified.radiometric import (
    compute_pair_direct_metrics,
    compute_pair_distribution_metrics,
)
from src.benchmark_v1_unified.structure import compute_sobel_structure, compute_range_retention


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_write(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def _read_support_samples(image_a, image_b, mask_a, mask_b, out_a=None, out_b=None):
    arrays = {key: [] for key in ("a", "b", "oa", "ob")}
    counts = {"input_support": 0, "complete_output_support": 0}
    with rasterio.open(image_a) as a, rasterio.open(image_b) as b, rasterio.open(mask_a) as ma, rasterio.open(mask_b) as mb:
        oa_ctx = rasterio.open(out_a) if out_a else None
        ob_ctx = rasterio.open(out_b) if out_b else None
        try:
            for row in range(0, a.height, 512):
                for col in range(0, a.width, 512):
                    w = Window(col, row, min(512, a.width-col), min(512, a.height-row))
                    va = (ma.read(1, window=w) > 0) & np.isfinite(a.read(1, window=w))
                    vb = (mb.read(1, window=w) > 0) & np.isfinite(b.read(1, window=w))
                    support = va & vb
                    if not support.any():
                        continue
                    xa, xb = a.read(1, window=w), b.read(1, window=w)
                    arrays["a"].append(xa[support].astype(np.float64, copy=False))
                    arrays["b"].append(xb[support].astype(np.float64, copy=False))
                    counts["input_support"] += int(support.sum())
                    if oa_ctx and ob_ctx:
                        ya, yb = oa_ctx.read(1, window=w), ob_ctx.read(1, window=w)
                        out_valid = np.isfinite(ya) & np.isfinite(yb)
                        if oa_ctx.nodata is not None and not math.isnan(oa_ctx.nodata):
                            out_valid &= (ya != oa_ctx.nodata) & (yb != ob_ctx.nodata)
                        common = support & out_valid
                        arrays["oa"].append(ya[common].astype(np.float64, copy=False))
                        arrays["ob"].append(yb[common].astype(np.float64, copy=False))
                        counts["complete_output_support"] += int(common.sum())
        finally:
            if oa_ctx:
                oa_ctx.close()
            if ob_ctx:
                ob_ctx.close()
    return {k: (np.concatenate(v) if v else np.empty(0, dtype=np.float64)) for k, v in arrays.items()}, counts


def _evaluate_pair(pre_a, pre_b, post_a, post_b, input_mask_a, input_mask_b):
    samples, counts = _read_support_samples(pre_a, pre_b, input_mask_a, input_mask_b, post_a, post_b)
    n, n_out = counts["input_support"], counts["complete_output_support"]
    ratio = n_out / n if n else 0.0
    status = "FULL_SUPPORT" if n_out == n else "PARTIAL_OUTPUT" if n_out else "NO_METHOD_OUTPUT_SUPPORT"
    if n_out:
        ones = np.ones(n_out, dtype=bool)
        direct = compute_pair_direct_metrics(samples["oa"], samples["ob"], ones)
        distribution = compute_pair_distribution_metrics(samples["oa"], samples["ob"], ones)
    else:
        direct = distribution = None
    input_direct = compute_pair_direct_metrics(samples["a"], samples["b"], np.ones(n, dtype=bool)) if n else None
    input_distribution = compute_pair_distribution_metrics(samples["a"], samples["b"], np.ones(n, dtype=bool)) if n else None
    return {
        "status": status,
        "input_support_pixels": n,
        "evaluated_pixels": n_out,
        "missing_pixels": n - n_out,
        "output_complete_rate": ratio,
        "evaluation_support": "frozen pair-mask intersection AND finite valid output; no mask regeneration",
        "input_consistency": {"direct": input_direct.__dict__ if input_direct else None,
                              "distribution": input_distribution.__dict__ if input_distribution else None},
        "corrected_consistency": {"direct": direct.__dict__ if direct else None,
                                  "distribution": distribution.__dict__ if distribution else None},
    }


def _mask_raster(source_path: Path, dest_path: Path):
    with rasterio.open(source_path) as src:
        profile = src.profile.copy()
        profile.update(driver="GTiff", count=1, dtype="uint8", nodata=0, compress="deflate", tiled=True,
                       blockxsize=512, blockysize=512)
        with rasterio.open(dest_path, "w", **profile) as dst:
            for row in range(0, src.height, 512):
                for col in range(0, src.width, 512):
                    w = Window(col, row, min(512, src.width-col), min(512, src.height-row))
                    valid = (src.dataset_mask(window=w) > 0) & np.isfinite(src.read(1, window=w))
                    dst.write(valid.astype(np.uint8), 1, window=w)
                    dst.write_mask(np.full(valid.shape, 255, dtype=np.uint8), window=w)


def _write_crop_mask(source_path: Path, dest_path: Path, crop: Window):
    with rasterio.open(source_path) as src:
        profile = src.profile.copy()
        profile.update(driver="GTiff", width=int(crop.width), height=int(crop.height),
                       transform=src.window_transform(crop), count=1, dtype="uint8", nodata=0,
                       compress="deflate", tiled=True, blockxsize=512, blockysize=512)
        with rasterio.open(dest_path, "w", **profile) as dst:
            for row in range(0, int(crop.height), 512):
                for col in range(0, int(crop.width), 512):
                    w = Window(col, row, min(512, int(crop.width)-col), min(512, int(crop.height)-row))
                    sw = Window(crop.col_off+col, crop.row_off+row, w.width, w.height)
                    values = (src.read(1, window=sw) > 0).astype(np.uint8)
                    dst.write(values, 1, window=w)
                    dst.write_mask(np.full(values.shape, 255, dtype=np.uint8), window=w)


def _coverage(input_masks, output, output_mask):
    counts = {"frozen_union": 0, "output_valid_inside_union": 0, "extra_output_outside_union": 0}
    with rasterio.open(input_masks[0]) as ma, rasterio.open(input_masks[1]) as mb, rasterio.open(output) as out, rasterio.open(output_mask) as om:
        if (ma.width, ma.height, ma.transform, ma.crs) != (mb.width, mb.height, mb.transform, mb.crs):
            return {"status": "INVALID_GRID", "counts": counts}
        if (out.width, out.height, out.transform, out.crs) != (om.width, om.height, om.transform, om.crs):
            return {"status": "INVALID_OUTPUT_MASK_GRID", "counts": counts}
        if out.crs != ma.crs or out.transform.b != 0 or out.transform.d != 0 or ma.transform.b != 0 or ma.transform.d != 0:
            return {"status": "UNSUPPORTED_GRID_TRANSFORM", "counts": counts}
        if not np.isclose(out.transform.a, ma.transform.a) or not np.isclose(out.transform.e, ma.transform.e):
            return {"status": "UNSUPPORTED_OUTPUT_RESOLUTION", "counts": counts}
        col_shift_f = (out.transform.c - ma.transform.c) / ma.transform.a
        row_shift_f = (out.transform.f - ma.transform.f) / ma.transform.e
        col_shift, row_shift = round(col_shift_f), round(row_shift_f)
        if not np.isclose(col_shift_f, col_shift, atol=1e-7) or not np.isclose(row_shift_f, row_shift, atol=1e-7):
            return {"status": "UNSUPPORTED_SUBPIXEL_GRID_SHIFT", "counts": counts}

        # Count the frozen union on its original grid, then map each output pixel
        # back by the exact integer-pixel affine offset (no resampling).
        for row in range(0, ma.height, 512):
            for col in range(0, ma.width, 512):
                w = Window(col, row, min(512, ma.width-col), min(512, ma.height-row))
                union = (ma.read(1, window=w) > 0) | (mb.read(1, window=w) > 0)
                counts["frozen_union"] += int(union.sum())
        for row in range(0, out.height, 512):
            for col in range(0, out.width, 512):
                w = Window(col, row, min(512, out.width-col), min(512, out.height-row))
                vals = out.read(1, window=w)
                valid = (om.read(1, window=w) > 0) & np.isfinite(vals)
                if out.nodata is not None and not math.isnan(out.nodata):
                    valid &= vals != out.nodata
                source_row, source_col = row + row_shift, col + col_shift
                h, width = int(w.height), int(w.width)
                sr0, sc0 = max(0, source_row), max(0, source_col)
                sr1, sc1 = min(ma.height, source_row + h), min(ma.width, source_col + width)
                if sr1 <= sr0 or sc1 <= sc0:
                    counts["extra_output_outside_union"] += int(valid.sum())
                    continue
                or0, oc0 = sr0 - source_row, sc0 - source_col
                or1, oc1 = or0 + sr1 - sr0, oc0 + sc1 - sc0
                source_window = Window(sc0, sr0, sc1-sc0, sr1-sr0)
                source_union = ((ma.read(1, window=source_window) > 0) |
                                (mb.read(1, window=source_window) > 0))
                tile_valid = valid[or0:or1, oc0:oc1]
                inside = tile_valid & source_union
                counts["output_valid_inside_union"] += int(inside.sum())
                counts["extra_output_outside_union"] += int(valid.sum() - inside.sum())
    total = counts["frozen_union"]
    got = counts["output_valid_inside_union"]
    counts["coverage"] = got / total if total else 0.0
    counts["missing_union_pixels"] = total - got
    counts["status"] = "PASS" if got == total and counts["extra_output_outside_union"] == 0 else "COVERAGE_GAP"
    return counts


def _preview(paths, output_path, window, title, *, stretch=None):
    x, y = [], []
    for path in paths:
        with rasterio.open(path) as src:
            vals = src.read(1, window=window)
            mask = (src.dataset_mask(window=window) > 0) & np.isfinite(vals)
            x.append(vals)
            y.extend(vals[mask].astype(np.float32).tolist())
    if not y:
        raise ValueError(f"no valid pixels for preview {title}")
    if stretch is None:
        lo, hi = np.quantile(np.asarray(y, dtype=np.float32), [0.02, 0.98]).tolist()
    else:
        lo, hi = float(stretch["p2"]), float(stretch["p98"])
    if hi <= lo:
        hi = lo + 1.0
    panels = []
    for arr in x:
        scaled = np.clip((arr - lo) / (hi - lo), 0, 1)
        rgb = np.repeat((scaled * 255).astype(np.uint8)[..., None], 3, axis=2)
        panels.append(Image.fromarray(rgb).resize((512, 512), Image.Resampling.NEAREST))
    canvas = Image.new("RGB", (1024, 550), "white")
    canvas.paste(panels[0], (0, 38)); canvas.paste(panels[1], (512, 38))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 10), f"{title} | fixed shared stretch p2-p98: {lo:.6g}..{hi:.6g}", fill="black")
    draw.text((8, 526), "left", fill="black"); draw.text((520, 526), "right", fill="black")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return {"path": str(output_path), "window": {"row_off": int(window.row_off), "col_off": int(window.col_off),
            "height": int(window.height), "width": int(window.width)}, "stretch": {"p2": lo, "p98": hi},
            "display_only": True}


def run(pair_id: str, freeze_root: Path, output_root: Path, *, crop_smoke=False):
    audit = audit_frozen_inputs(freeze_root)
    if audit["status"] != "PASS":
        raise RuntimeError(f"frozen input audit failed: {audit['errors']}")
    pair = audit["requested_pairs"][pair_id]
    scene_i, scene_j = int(pair_id[:2]), int(pair_id[3:5])
    run_dir = output_root / pair_id / ("crop_smoke" if crop_smoke else "full_pair")
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"refusing to reuse non-empty run directory: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    root = Path(freeze_root).resolve()
    aligned = root / "aligned_inputs"
    crop_window = None
    preview_row = None
    if crop_smoke:
        with (root / "fixed_windows.csv").open(encoding="utf-8-sig", newline="") as stream:
            pair_windows = [r for r in csv.DictReader(stream) if r["pair_id"] == pair_id]
        if not pair_windows:
            raise RuntimeError(f"no frozen B2 window to anchor crop for {pair_id}")
        preview_row = pair_windows[0]
        with rasterio.open(aligned / f"scene_{scene_i:03d}_B9_input.tif") as src:
            size = 1024
            row_center = int(preview_row["row_off"]) + int(preview_row["height"]) // 2
            col_center = int(preview_row["col_off"]) + int(preview_row["width"]) // 2
            row_off = min(max(0, row_center - size // 2), src.height - size)
            col_off = min(max(0, col_center - size // 2), src.width - size)
            crop_window = Window(col_off, row_off, size, size)
    masked_dir = run_dir / "spectralmatch_inputs"
    input_records = {}
    input_paths = []
    mask_paths = []
    for index, side in ((scene_i, "i"), (scene_j, "j")):
        source = aligned / f"scene_{index:03d}_B9_input.tif"
        mask = aligned / f"scene_{index:03d}_input_mask.tif"
        working = masked_dir / f"scene_{index:03d}_B9_frozen_mask_nan.tif"
        rec = prepare_masked_copy(source, mask, working, crop_window=crop_window)
        input_records[side] = rec
        input_paths.append(str(working))
        if crop_window is not None:
            crop_mask = masked_dir / f"scene_{index:03d}_input_mask_crop.tif"
            _write_crop_mask(mask, crop_mask, crop_window)
            mask_paths.append(str(crop_mask))
        else:
            mask_paths.append(str(mask))

    import spectralmatch
    package_root = Path(spectralmatch.__file__).resolve().parent
    optional_mask_module = package_root / "mask" / "mask.py"
    environment_metadata = {
        "python": platform.python_version(),
        "rasterio": rasterio.__version__,
        "gdal": rasterio.__gdal_version__,
        "proj": rasterio.__proj_version__,
        "torch_loaded": "torch" in sys.modules,
        "upstream_compatibility_patch": {
            "file": str(optional_mask_module),
            "sha256": sha256(optional_mask_module) if optional_mask_module.is_file() else None,
            "change": "lazy import optional omnicloudmask inside its mask function; radiometric stages do not load optional Torch model stack",
            "scope": "isolated temporary upstream checkout only; no radiometric math changed",
        },
    }
    stage_start = time.monotonic()
    stage_outputs = execute_fixed_grid_stages(
        spectralmatch, input_paths, run_dir / "stages",
        local_options={"number_of_blocks": 16} if crop_smoke else None,
    )
    stage_elapsed = time.monotonic() - stage_start

    scene_masks_dir = run_dir / "output_masks"
    scene_masks_dir.mkdir()
    corrected_masks = {}
    for side, path in zip(("i", "j"), stage_outputs["scene_outputs"]):
        mp = scene_masks_dir / f"corrected_{side}_mask.tif"
        _mask_raster(Path(path), mp)
        corrected_masks[side] = str(mp)
    mosaic_mask = scene_masks_dir / "mosaic_mask.tif"
    _mask_raster(Path(stage_outputs["mosaic"]), mosaic_mask)

    with rasterio.open(input_paths[0]) as a, rasterio.open(stage_outputs["scene_outputs"][0]) as b:
        grid_status = "PASS" if (a.width, a.height, a.transform, a.crs) == (b.width, b.height, b.transform, b.crs) else "INVALID_GRID"
        grid = {"crs": a.crs.to_string(), "transform": list(a.transform)[:6], "width": a.width, "height": a.height,
                "dtype_in": a.dtypes[0], "dtype_out": b.dtypes[0], "nodata_in": str(a.nodata), "nodata_out": str(b.nodata)}
    pair_eval = _evaluate_pair(input_paths[0], input_paths[1], stage_outputs["scene_outputs"][0],
                               stage_outputs["scene_outputs"][1], mask_paths[0], mask_paths[1])

    if preview_row is None:
        with (root / "fixed_windows.csv").open(encoding="utf-8-sig", newline="") as stream:
            windows = [r for r in csv.DictReader(stream) if r["pair_id"] == pair_id]
        if not windows:
            raise RuntimeError(f"no frozen fixed B2 preview window for {pair_id}")
        preview_row = windows[0]
        win = Window(int(preview_row["col_off"]), int(preview_row["row_off"]), int(preview_row["width"]), int(preview_row["height"]))
    else:
        win = Window(int(preview_row["col_off"])-int(crop_window.col_off),
                     int(preview_row["row_off"])-int(crop_window.row_off),
                     int(preview_row["width"]), int(preview_row["height"]))
    previews_dir = run_dir / "previews"
    input_preview = _preview([input_paths[0], input_paths[1]], previews_dir / "input_fixed_stretch.png",
                             win, f"{pair_id} frozen B9 input")
    shared_stretch = input_preview["stretch"]
    corrected_preview = _preview([stage_outputs["scene_outputs"][0], stage_outputs["scene_outputs"][1]],
                                  previews_dir / "corrected_fixed_stretch.png", win,
                                  f"{pair_id} Spectralmatch corrected", stretch=shared_stretch)
    scene_previews = []
    for side, input_path, corrected_path in zip(("i", "j"), input_paths, stage_outputs["scene_outputs"]):
        scene_previews.append(_preview([input_path, corrected_path], previews_dir / f"scene_{side}_before_after.png",
                                       win, f"{pair_id} scene {side}: input vs corrected", stretch=shared_stretch))
    previews = [input_preview, corrected_preview, *scene_previews]
    structure_range = {}
    for side, pre, post, mask_path in zip(("i", "j"), input_paths, stage_outputs["scene_outputs"], mask_paths):
        with rasterio.open(pre) as src, rasterio.open(post) as dst, rasterio.open(mask_path) as msk:
            x, y = src.read(1, window=win), dst.read(1, window=win)
            valid = (msk.read(1, window=win) > 0) & np.isfinite(x) & np.isfinite(y)
        structure_range[side] = {
            "window_id": preview_row["window_id"],
            "support_pixels": int(valid.sum()),
            "structure": compute_sobel_structure(x, y, valid).__dict__,
            "dynamic_range": compute_range_retention(x, y, valid).__dict__,
            "scope": "one frozen 128x128 B2 window; diagnostic only",
        }
    mosaic_cov = _coverage(mask_paths, stage_outputs["mosaic"], mosaic_mask)
    files = [Path(p) for p in stage_outputs["global_scene_outputs"] + stage_outputs["scene_outputs"] +
             stage_outputs["seam_clipped_scene_outputs"] + [stage_outputs["footprints"], stage_outputs["seamlines"], stage_outputs["mosaic"]] +
             list(corrected_masks.values()) + [str(mosaic_mask)]]
    hashes = {str(path.relative_to(run_dir)): sha256(path) for path in files if path.is_file()}
    result = {
        "status": "COMPLETED_DIAGNOSTIC_ONLY",
        "run_regime": "DETERMINISTIC_1024_CROP_SMOKE" if crop_smoke else "FULL_PAIR_SMOKE",
        "crop_definition": None if not crop_smoke else {
            "anchor_fixed_window_id": preview_row["window_id"],
            "window": {"row_off": int(crop_window.row_off), "col_off": int(crop_window.col_off),
                       "height": int(crop_window.height), "width": int(crop_window.width)},
            "rule": "1024x1024 canonical-grid crop centered on the first frozen B2 fixed window; no method-output-based selection",
            "local_block_count": 16,
        },
        "pair_id": pair_id,
        "pair_metadata": pair,
        "method_id": "spectralmatch",
        "upstream_version": UPSTREAM_VERSION,
        "upstream_commit": UPSTREAM_COMMIT,
        "environment": environment_metadata,
        "input_freeze_sha256": json.loads((root / "frozen_manifest.json").read_text(encoding="utf-8"))["input_freeze_sha256"],
        "geometry_freeze_sha256": json.loads((root / "frozen_manifest.json").read_text(encoding="utf-8"))["geometry_freeze_sha256"],
        "input_artifacts": input_records,
        "geometry_registration": "NOT_RUN_FROZEN_CANONICAL_GRID",
        "stage_outputs": stage_outputs,
        "stage_wall_seconds": stage_elapsed,
        "per_stage_wall_seconds": stage_outputs.get("stage_wall_seconds", {}),
        "grid_audit": {"status": grid_status, **grid},
        "radiometric_diagnostics": {"pair_overlap": pair_eval,
                                     "unit": "UNKNOWN; no reflectance interpretation"},
        "structure_and_range": structure_range,
        "mosaic_coverage": mosaic_cov,
        "seamline_source_metric": "NOT_AVAILABLE_IF_UPSTREAM_DOES_NOT_EXPORT_PROVENANCE_RASTER",
        "previews": previews,
        "stage_output_sha256": hashes,
        "timing": {"worker_total_s": time.monotonic() - stage_start, "stage_pipeline_s": stage_elapsed},
        "benchmark_scope": "two-pair compatibility smoke; not formal 13-scene ranking",
    }
    json_write(run_dir / "smoke_result.json", result)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", required=True, choices=["00_01", "00_10"])
    ap.add_argument("--freeze", required=True)
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--crop-smoke", action="store_true")
    args = ap.parse_args()
    try:
        report = run(args.pair, Path(args.freeze), Path(args.output_root), crop_smoke=args.crop_smoke)
        print(json.dumps({"status": report["status"], "pair_id": args.pair, "run_regime": report["run_regime"]}, ensure_ascii=False))
        return 0
    except Exception as exc:
        err = {"status": "METHOD_FAILED", "pair_id": args.pair, "error_type": type(exc).__name__,
               "error": str(exc), "traceback": traceback.format_exc(), "upstream_commit": UPSTREAM_COMMIT}
        out = Path(args.output_root) / args.pair
        out.mkdir(parents=True, exist_ok=True)
        json_write(out / "worker_failure.json", err)
        print(json.dumps(err, ensure_ascii=False), flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
