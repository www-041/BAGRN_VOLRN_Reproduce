"""Small two-scene runner; outputs are diagnostic-only and never formal rankings."""
from __future__ import annotations

from dataclasses import asdict
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import threading
import time

import numpy as np
import rasterio
from PIL import Image, ImageDraw
from rasterio.windows import Window

from src.benchmark_v1_unified.radiometric import (
    audit_pair_output_support, compute_pair_direct_metrics,
    compute_pair_distribution_metrics, metric_dict, pair_support,
)
from src.benchmark_v1_unified.structure import compute_range_retention, compute_sobel_structure
from . import METHOD_ID, UPSTREAM_COMMIT
from .frozen_crop import (CropSpec, RasterCrop, assert_same_crop_grid,
                          read_frozen_crop, select_pair_crop, verify_frozen_b9)
from .official_port import (apply_lirrn_linear, classify_three_levels, draw_shared_ids,
                            fit_lirrn_linear, select_lirrn_samples)

PROTOCOL_ID = "DZ01V_UNIFIED_A_B_V1_20261013"
NODATA = float(np.finfo(np.float32).min)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def write_corrected_geotiff(dst: Path, crop: RasterCrop, corrected: np.ndarray) -> dict:
    dst = Path(dst)
    out = np.asarray(corrected, dtype=np.float32)
    if out.shape != crop.values.shape or crop.valid.shape != out.shape:
        raise ValueError("OUTPUT_SHAPE_MISMATCH")
    if not np.isfinite(out[crop.valid]).all():
        raise ValueError("NONFINITE_VALID_OUTPUT_BEFORE_WRITE")
    if np.any(crop.valid & (out == np.float32(NODATA))):
        raise ValueError("NODATA_SENTINEL_COLLIDES_WITH_VALID_VALUE")
    dst.parent.mkdir(parents=True, exist_ok=True)
    profile = {"driver": "GTiff", "width": out.shape[1], "height": out.shape[0],
               "count": 1, "dtype": "float32", "crs": crop.crs,
               "transform": crop.transform, "nodata": NODATA,
               "compress": "deflate", "predictor": 3, "tiled": True,
               "blockxsize": 256, "blockysize": 256}
    disk_values = out.copy()
    disk_values[~crop.valid] = np.float32(NODATA)
    with rasterio.open(dst, "w", **profile) as ds:
        ds.write(disk_values, 1)
        ds.write_mask(crop.valid.astype(np.uint8) * 255)
    return {"path": str(dst), "sha256": _sha256(dst), "nodata": NODATA,
            "valid_pixels": int(crop.valid.sum()), "dtype": "float32"}


def score_pair_crop(ref: RasterCrop, sub: RasterCrop, corrected: np.ndarray,
                    output_valid: np.ndarray | None = None) -> dict:
    corrected = np.asarray(corrected, dtype=np.float64)
    output_valid = sub.valid if output_valid is None else np.asarray(output_valid, dtype=bool)
    support = pair_support(ref.valid, sub.valid)
    coverage = audit_pair_output_support(support, corrected, output_valid)
    before_direct = compute_pair_direct_metrics(ref.values, sub.values, support)
    before_distribution = compute_pair_distribution_metrics(ref.values, sub.values, support)
    if coverage.primary_eligible:
        after_direct = compute_pair_direct_metrics(ref.values, corrected, support)
        after_distribution = compute_pair_distribution_metrics(ref.values, corrected, support)
        status = "FULL_SUPPORT"
    else:
        after_direct = None
        after_distribution = None
        status = "PARTIAL_OUTPUT"
    structure = compute_sobel_structure(sub.values, corrected, sub.valid) if coverage.primary_eligible else None
    dynamic_range = metric_dict(compute_range_retention(sub.values, corrected, sub.valid)) if coverage.primary_eligible else None
    return {"status": status, "support_pixels": coverage.input_pixels,
            "output_support_pixels": coverage.output_pixels, "missing_pixels": coverage.missing_pixels,
            "coverage": coverage.coverage, "full_support": coverage.primary_eligible,
            "pixel_value_unit": "UNKNOWN_PIXEL_UNIT",
            "before": {"direct": metric_dict(before_direct), "distribution": metric_dict(before_distribution)},
            "after": None if after_direct is None else {
                "direct": metric_dict(after_direct), "distribution": metric_dict(after_distribution)},
            "structure_preservation_subject_to_corrected": None if structure is None else asdict(structure),
            "range_retention_subject_to_corrected": dynamic_range}


def _read_saved(path: Path, expected: RasterCrop) -> tuple[np.ndarray, np.ndarray]:
    with rasterio.open(path) as ds:
        if (ds.height, ds.width) != expected.values.shape or ds.crs.to_string() != expected.crs or ds.transform != expected.transform:
            raise ValueError("OUTPUT_GRID_MISMATCH_AFTER_REOPEN")
        if ds.dtypes[0] != "float32":
            raise ValueError("OUTPUT_DTYPE_MISMATCH_AFTER_REOPEN")
        values = ds.read(1).astype(np.float64)
        valid = ds.dataset_mask() > 0
    if not np.array_equal(valid, expected.valid):
        raise ValueError("OUTPUT_MASK_MISMATCH_AFTER_REOPEN")
    return values, valid


def _fixed_preview(dst: Path, ref: RasterCrop, sub: RasterCrop, corrected: np.ndarray) -> dict:
    valid = np.concatenate((ref.values[ref.valid], sub.values[sub.valid]))
    lo, hi = np.quantile(valid, [0.02, 0.98], method="linear")
    if not np.isfinite([lo, hi]).all() or hi <= lo:
        raise ValueError("PREVIEW_STRETCH_INVALID")

    def panel(values: np.ndarray, mask: np.ndarray) -> Image.Image:
        scaled = np.clip((values - lo) / (hi - lo), 0, 1)
        # Invalid pixels are drawn magenta below; keep them finite before the
        # display-only uint8 conversion (input data are never modified).
        scaled = np.where(mask, scaled, 0.0)
        gray = (scaled * 255).astype(np.uint8)
        rgb = np.repeat(gray[..., None], 3, axis=2)
        rgb[~mask] = np.array([255, 0, 255], dtype=np.uint8)
        return Image.fromarray(rgb, "RGB")

    before = Image.new("RGB", (2048, 1024), "black")
    after = Image.new("RGB", (2048, 1024), "black")
    before.paste(panel(ref.values, ref.valid), (0, 0)); before.paste(panel(sub.values, sub.valid), (1024, 0))
    after.paste(panel(ref.values, ref.valid), (0, 0)); after.paste(panel(corrected, sub.valid), (1024, 0))
    canvas = Image.new("RGB", (2048, 2048), "white")
    canvas.paste(before, (0, 0)); canvas.paste(after, (0, 1024))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), "BEFORE: reference | subject input; magenta = invalid", fill="yellow")
    draw.text((8, 1032), "AFTER: reference | corrected subject; shared P02/P98 stretch", fill="yellow")
    dst.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(dst)
    return {"path": str(dst), "sha256": _sha256(dst), "stretch_p02": float(lo), "stretch_p98": float(hi)}


class _RssSampler:
    def __init__(self, interval: float = 0.05):
        self.interval = interval
        self.peak = None
        self.samples = 0
        self.error = None
        self._stop = threading.Event()
        self._thread = None

    def _run(self):
        try:
            import psutil
            proc = psutil.Process(os.getpid())
            while not self._stop.is_set():
                total = 0
                for p in [proc, *proc.children(recursive=True)]:
                    try:
                        total += p.memory_info().rss
                    except psutil.Error:
                        continue
                self.peak = max(self.peak or 0, total)
                self.samples += 1
                self._stop.wait(self.interval)
        except Exception as exc:  # optional monitor: disclose failure, do not claim a limit
            self.error = f"{type(exc).__name__}:{exc}"

    def __enter__(self):
        try:
            import psutil  # noqa: F401
        except ImportError:
            self.error = "PSUTIL_UNAVAILABLE"
            return self
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def record(self):
        return {"peak_rss_bytes": self.peak, "rss_sample_count": self.samples,
                "rss_status": "SAMPLED" if self.samples else "NOT_MEASURED",
                "rss_error": self.error, "os_hard_memory_limit": False}


def _pair_scene_indices(pair_id: str) -> tuple[int, int]:
    if pair_id not in {"00_01", "00_10"}:
        raise ValueError(f"unsupported diagnostic pair: {pair_id}")
    return int(pair_id[:2]), int(pair_id[3:])


def run_pair(freeze_root: Path, pair_id: str, out_dir: Path,
             seed: int = 20261014, p_n: int = 1000) -> dict:
    start = time.perf_counter()
    freeze_root, out_dir = Path(freeze_root), Path(out_dir)
    pair_dir = out_dir / pair_id
    if pair_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing pair output: {pair_dir}")
    audit = verify_frozen_b9(freeze_root)
    if audit["status"] != "PASS":
        raise RuntimeError(f"FAILED_INPUT_OR_DEPENDENCY:freeze verification {audit['errors']}")
    ref_index, sub_index = _pair_scene_indices(pair_id)
    spec = select_pair_crop(freeze_root, pair_id, 1024)
    pair_dir.mkdir(parents=True, exist_ok=False)
    log_path = pair_dir / "execution.log"
    log_lines = [f"pair={pair_id} start_utc={time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}"]
    (pair_dir / "crop_spec.json").write_text(json.dumps(asdict(spec), indent=2)+"\n", encoding="utf-8")
    rss = _RssSampler()
    with rss:
        t_read = time.perf_counter()
        ref = read_frozen_crop(freeze_root, ref_index, spec)
        sub = read_frozen_crop(freeze_root, sub_index, spec)
        assert_same_crop_grid(ref, sub)
        read_seconds = time.perf_counter() - t_read
        t_algorithm = time.perf_counter()
        ref_labels, ref_otsu = classify_three_levels(ref.values, ref.valid)
        sub_labels, sub_otsu = classify_three_levels(sub.values, sub.valid)
        selected_pn = p_n
        fallback = False
        counts = [min(ref_otsu["class_counts"][k], sub_otsu["class_counts"][k]) for k in range(3)]
        if selected_pn == 1000 and min(counts) < 1000:
            selected_pn = 200
            fallback = True
            log_lines.append("P_N=1000 preflight insufficient; using sole declared P_N=200 fallback before fitting/scoring")
        if min(counts) < selected_pn:
            raise ValueError(f"FAILED_INSUFFICIENT_SAMPLES:counts={counts}:p_n={selected_pn}")
        ids = draw_shared_ids(selected_pn, seed)
        x_sub, y_ref, sample_audit = select_lirrn_samples(ref, sub, ref_labels, sub_labels, ids, selected_pn)
        model = fit_lirrn_linear(x_sub, y_ref)
        corrected = apply_lirrn_linear(sub, model)
        algorithm_seconds = time.perf_counter() - t_algorithm
        t_write = time.perf_counter()
        corrected_path = pair_dir / "subject_lirrn_corrected_B9.tif"
        write_info = write_corrected_geotiff(corrected_path, sub, corrected)
        mask_path = pair_dir / "subject_output_mask.tif"
        with rasterio.open(mask_path, "w", driver="GTiff", width=spec.width, height=spec.height,
                           count=1, dtype="uint8", crs=sub.crs, transform=sub.transform, nodata=0,
                           compress="deflate") as mask_ds:
            mask_ds.write(sub.valid.astype(np.uint8) * 255, 1)
        write_seconds = time.perf_counter() - t_write
        t_score = time.perf_counter()
        saved, saved_valid = _read_saved(corrected_path, sub)
        metrics = score_pair_crop(ref, sub, saved, saved_valid)
        if not metrics["full_support"]:
            raise RuntimeError("PARTIAL_OUTPUT:corrected GeoTIFF does not fully cover frozen intersection")
        # Local diagnostics are restricted to the first frozen 128x128 B2 anchor.
        with (freeze_root / "fixed_windows.csv").open(encoding="utf-8-sig", newline="") as stream:
            anchor = next(r for r in csv.DictReader(stream) if r["pair_id"] == pair_id)
        local_row = int(anchor["row_off"]) - spec.row_off
        local_col = int(anchor["col_off"]) - spec.col_off
        h, w = int(anchor["height"]), int(anchor["width"])
        sl = np.s_[local_row:local_row+h, local_col:local_col+w]
        local_support = pair_support(ref.valid[sl], sub.valid[sl])
        metrics["first_frozen_B2_window"] = {"window_id": anchor["window_id"],
            "support_pixels": int(local_support.sum()),
            "before_direct": metric_dict(compute_pair_direct_metrics(ref.values[sl], sub.values[sl], local_support)),
            "after_direct": metric_dict(compute_pair_direct_metrics(ref.values[sl], saved[sl], local_support)),
            "before_distribution": metric_dict(compute_pair_distribution_metrics(ref.values[sl], sub.values[sl], local_support)),
            "after_distribution": metric_dict(compute_pair_distribution_metrics(ref.values[sl], saved[sl], local_support))}
        scoring_seconds = time.perf_counter() - t_score
        preview = _fixed_preview(pair_dir / "before_after_fixed_stretch.png", ref, sub, saved)
    rss_result = rss.record()
    regression = asdict(model)
    regression.update({"requested_p_n": p_n, "actual_p_n": selected_pn,
                       "pn_200_predeclared_fallback": fallback, "rng_seed": seed,
                       "shared_ids_sha256": hashlib.sha256(ids.tobytes()).hexdigest(),
                       "ref_otsu": ref_otsu, "sub_otsu": sub_otsu,
                       "sampling_audit": sample_audit,
                       "sub_valid_pixels": int(sub.valid.sum()),
                       "ref_valid_pixels": int(ref.valid.sum())})
    input_hashes = {}
    for idx in (ref_index, sub_index):
        for suffix in ("B9_input.tif", "input_mask.tif"):
            path = freeze_root / "aligned_inputs" / f"scene_{idx:03d}_{suffix}"
            input_hashes[path.name] = _sha256(path)
    crop_details = {**asdict(spec), "window_transform": list(sub.transform)[:6], "crs": sub.crs,
        "reference_scene_index": ref_index, "subject_scene_index": sub_index,
        "reference_valid_pixels": int(ref.valid.sum()), "subject_valid_pixels": int(sub.valid.sum()),
        "common_valid_pixels": int(pair_support(ref.valid, sub.valid).sum()), "input_sha256": input_hashes,
        "freeze_lock_sha256": _sha256(freeze_root / "freeze_lock.json"),
        "source_input_scope": "frozen canonical aligned B9; no resampling"}
    _write_json(pair_dir / "crop_spec.json", crop_details)
    _write_json(pair_dir / "method_metadata.json", {
        "method_id": METHOD_ID,
        "method_kind": "PAIR_ONLY",
        "upstream_repository": "https://github.com/ArminMoghimi/LIRRN",
        "upstream_commit_pinned_by_plan": UPSTREAM_COMMIT,
        "official_source_urls": [
            "https://github.com/ArminMoghimi/LIRRN/blob/main/Function/LIRRN.m",
            "https://github.com/ArminMoghimi/LIRRN/blob/main/Function/LinearReg.m",
        ],
        "official_source_sha256": None,
        "official_source_sha256_status": "NOT_VERIFIED_NETWORK_FETCH_BLOCKED",
        "python_numeric_equivalence_to_matlab_verified": False,
        "port_notes": [
            "local SampleSelection behavior ported from Function/LIRRN.m",
            "three identical ComputeDistances max-distance selections retained",
            "OLS with intercept; no robust regression",
            "NumPy 256-bin three-class Otsu backend is deterministic but not claimed bit-identical to MATLAB",
        ],
        "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                    "rasterio": rasterio.__version__},
        "protocol_id": PROTOCOL_ID,
        "scope": "1024_CROP_DIAGNOSTIC_ONLY",
        "pixel_value_unit": "UNKNOWN_PIXEL_UNIT",
        "seed": seed,
        "requested_p_n": p_n,
    })
    _write_json(pair_dir / "regression_audit.json", regression)
    _write_json(pair_dir / "pair_metrics.json", {**metrics, "preview": preview,
        "output": write_info, "mask_path": str(mask_path), "mask_sha256": _sha256(mask_path),
        "timing": {"read_s": read_seconds, "algorithm_s": algorithm_seconds,
                   "write_s": write_seconds, "metrics_s": scoring_seconds,
                   "total_wall_s": time.perf_counter()-start, **rss_result}})
    log_lines.extend([f"status={metrics['status']}", f"samples={model.sample_count}",
                      f"scale={model.scale:.12g} offset={model.offset:.12g}",
                      f"support={metrics['support_pixels']} coverage={metrics['coverage']:.12g}",
                      f"corrected={corrected_path}", f"corrected_sha256={write_info['sha256']}"])
    log_path.write_text("\n".join(log_lines)+"\n", encoding="utf-8")
    result = {"protocol_id": PROTOCOL_ID, "method_id": METHOD_ID, "method_kind": "PAIR_ONLY",
        "benchmark_scope": "1024_CROP_DIAGNOSTIC_ONLY", "formal_ranking_eligible": False,
        "pair_id": pair_id, "status": "SUCCESS_DIAGNOSTIC", "reference_scene": f"scene_{ref_index:03d}",
        "subject_scene": f"scene_{sub_index:03d}", "upstream_commit": UPSTREAM_COMMIT,
        "matlab_numeric_equivalence_verified": False, "three_identical_max_sampling_calls": True,
        "actual_p_n": selected_pn, "result_path": str(pair_dir / "pair_metrics.json")}
    run_manifest_path = out_dir / "run_manifest.json"
    if run_manifest_path.exists():
        manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
        if manifest.get("protocol_id") != PROTOCOL_ID or manifest.get("method_id") != METHOD_ID:
            raise RuntimeError("OUTPUT_ROOT_CONTAINS_DIFFERENT_RUN")
    else:
        manifest = {"protocol_id": PROTOCOL_ID, "method_id": METHOD_ID,
                    "run_id": "lirrn_phase3_20261014", "scope": "PAIR_ONLY_DIAGNOSTIC",
                    "freeze_root": str(freeze_root), "pairs": []}
    manifest["pairs"] = [r for r in manifest["pairs"] if r.get("pair_id") != pair_id] + [result]
    _write_json(run_manifest_path, manifest)
    return {**result, "metrics": metrics, "regression": regression,
            "output_dir": str(pair_dir), "timing": {"read_s": read_seconds,
            "algorithm_s": algorithm_seconds, "write_s": write_seconds,
            "metrics_s": scoring_seconds, "total_wall_s": time.perf_counter()-start, **rss_result}}
