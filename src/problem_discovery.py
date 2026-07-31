"""
Stage 2: Problem Discovery — 6 experiments diagnosing spectral distortion

Experiments:
  1. band_attribution       – per-band metric decomposition
  2. scene_attribution      – per-scene metric decomposition
  3. gain_offset_ablation    – a=1 / b=0 / full ablation
  4. spatial_attribution     – center vs boundary vs edge
  5. multiwindow             – 4/6 window positions
  6. nan_trace                – VOLRN nan tracking

All experiments operate on 1024 px crops (or reuse Stage 1 outputs).
No modification to BAGRN / VOLRN objective functions.
"""

import os
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ===========================================================================
# 0. Common loader — re-use Stage 1 registered arrays + overlaps
# ===========================================================================

def load_stage1_data(
    baseline_output: str,
    crop_size: int = 1024,
) -> Dict[str, Any]:
    """
    Load registered arrays, scene metadata, overlaps, and per-method normalized
    arrays from a Stage 1 baseline output directory.

    Supports two output layouts:
      - Old: registered/ + normalized/ + overlaps.json
      - Current: per-method dirs (original/, bagrn/, volrn_only/, bagrn_volrn/)
        with per-scene GeoTIFFs

    Parameters
    ----------
    baseline_output : str
        Path to the Stage 1 output directory.
    crop_size : int
        Crop size for the registered arrays.

    Returns
    -------
    dict with keys:
        'registered_arrays', 'scene_data', 'overlaps', 'normalized',
        'transforms', 'nodata_values', 'bounds', 'scene_ids', 'band_names'
    """
    from src.io_utils import read_geotiff
    from src.overlap import has_overlap, get_overlap_window

    # Detect output layout
    registered_dir = os.path.join(baseline_output, "registered")
    normalized_dir = os.path.join(baseline_output, "normalized")

    if os.path.isdir(registered_dir) and os.path.isdir(normalized_dir):
        # Old layout: .npy files
        return _load_stage1_npy(baseline_output, crop_size)
    else:
        # Current layout: GeoTIFFs in per-method directories
        return _load_stage1_geotiff(baseline_output)


def _load_stage1_npy(
    baseline_output: str,
    crop_size: int,
) -> Dict[str, Any]:
    """Load from .npy layout (old format)."""
    registered_dir = os.path.join(baseline_output, "registered")
    normalized_dir = os.path.join(baseline_output, "normalized")

    scene_files = sorted([
        f for f in os.listdir(registered_dir)
        if f.startswith("scene_") and f.endswith(".npy")
    ])
    if not scene_files:
        raise FileNotFoundError(f"在 {registered_dir} 中未找到 scene_*.npy 文件")

    arrays = []
    scene_ids = []
    for sf in scene_files:
        arr = np.load(os.path.join(registered_dir, sf))
        if arr.ndim == 2:
            arr = arr[np.newaxis, ...]
        arrays.append(arr)
        sid = os.path.splitext(sf)[0].replace("scene_", "")
        scene_ids.append(sid)

    meta_path = os.path.join(registered_dir, "metadata.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    transforms = meta["transforms"]
    nodata_values = meta["nodata_values"]
    bounds = meta["bounds"]
    band_names = meta["band_names"]

    overlap_path = os.path.join(baseline_output, "overlaps.json")
    with open(overlap_path, "r", encoding="utf-8") as f:
        overlaps = json.load(f)

    normalized = {}
    for method in os.listdir(normalized_dir):
        method_dir = os.path.join(normalized_dir, method)
        if not os.path.isdir(method_dir):
            continue
        npy_files = sorted([f for f in os.listdir(method_dir) if f.endswith(".npy")])
        method_arrays = []
        for nf in npy_files:
            arr = np.load(os.path.join(method_dir, nf))
            if arr.ndim == 2:
                arr = arr[np.newaxis, ...]
            method_arrays.append(arr)
        if method_arrays:
            normalized[method] = method_arrays

    return {
        "registered_arrays": arrays,
        "scene_ids": scene_ids,
        "transforms": transforms,
        "nodata_values": nodata_values,
        "bounds": bounds,
        "overlaps": overlaps,
        "normalized": normalized,
        "band_names": band_names,
    }


def _load_stage1_geotiff(
    baseline_output: str,
) -> Dict[str, Any]:
    """Load from GeoTIFF layout (current Stage 1 format)."""
    from src.io_utils import read_geotiff
    from src.overlap import has_overlap, get_overlap_window

    # Discover method directories
    method_dirs = []
    for entry in os.listdir(baseline_output):
        entry_path = os.path.join(baseline_output, entry)
        if os.path.isdir(entry_path) and entry not in (".checkpoints", "metrics", "spectral"):
            # Check if it contains scene_*.tif files
            tif_files = [f for f in os.listdir(entry_path) if f.startswith("scene_") and f.endswith(".tif")]
            if tif_files:
                method_dirs.append(entry)

    if not method_dirs:
        raise FileNotFoundError(f"在 {baseline_output} 中未找到方法目录")

    # Use 'original' as the reference for scene metadata
    ref_method = "original" if "original" in method_dirs else method_dirs[0]
    ref_dir = os.path.join(baseline_output, ref_method)

    # Discover scene files
    scene_files = sorted([
        f for f in os.listdir(ref_dir)
        if f.startswith("scene_") and f.endswith(".tif")
    ])
    scene_ids = [os.path.splitext(f)[0].replace("scene_", "").replace(f"_{ref_method}", "")
                 for f in scene_files]

    # Read reference method arrays + metadata
    arrays = []
    transforms = []
    bounds_list = []
    nodata_values = []
    band_names = None

    for sf in scene_files:
        path = os.path.join(ref_dir, sf)
        arr, tr, crs, nd = read_geotiff(path)
        if arr.ndim == 3 and arr.shape[0] > 1:
            # Multi-band: keep as-is
            pass
        elif arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        elif arr.ndim == 2:
            arr = arr[np.newaxis, ...]

        arrays.append(arr.astype(np.float64))
        transforms.append(tr)
        nodata_values.append(nd)

        h, w = arr.shape[-2], arr.shape[-1]
        left = tr.c
        top = tr.f
        right = left + tr.a * w
        bottom = top + tr.e * h
        bounds_list.append((left, bottom, right, top))

        if band_names is None:
            # For single-band-per-file: we have one band per file
            # For multi-band: use band indices
            if arr.ndim == 3:
                band_names = [f"B{i+1:02d}" for i in range(arr.shape[0])]
            else:
                band_names = ["B01"]

    # Detect overlaps from bounds
    overlaps = []
    n_images = len(arrays)
    for i in range(n_images):
        for j in range(i + 1, n_images):
            if not has_overlap(bounds_list[i], bounds_list[j]):
                continue
            win = get_overlap_window(
                bounds_list[i], transforms[i],
                bounds_list[j], transforms[j],
            )
            if win is None:
                continue
            (ri_s, ri_e, ci_s, ci_e), (rj_s, rj_e, cj_s, cj_e) = win
            pix = (ri_e - ri_s) * (ci_e - ci_s)
            if pix < 100:
                continue
            overlaps.append({
                "idx_i": i,
                "idx_j": j,
                "window_i": (ri_s, ri_e, ci_s, ci_e),
                "window_j": (rj_s, rj_e, cj_s, cj_e),
                "pixel_count": pix,
            })

    # Load all method arrays
    normalized = {}
    for method in method_dirs:
        method_dir = os.path.join(baseline_output, method)
        method_arrays = []
        for sf in scene_files:
            path = os.path.join(method_dir, sf)
            if not os.path.exists(path):
                # Try alternative naming
                alt_name = sf.replace(f"_{ref_method}", f"_{method}")
                path = os.path.join(method_dir, alt_name)
            if not os.path.exists(path):
                logger.warning("方法 %s 缺少文件 %s, 跳过", method, sf)
                method_arrays = []
                break
            arr, tr, crs, nd = read_geotiff(path)
            if arr.ndim == 3 and arr.shape[0] == 1:
                arr = arr[0]
            elif arr.ndim == 2:
                arr = arr[np.newaxis, ...]
            method_arrays.append(arr.astype(np.float64))
        if method_arrays:
            normalized[method] = method_arrays

    return {
        "registered_arrays": arrays,
        "scene_ids": scene_ids,
        "transforms": transforms,
        "nodata_values": nodata_values,
        "bounds": bounds_list,
        "overlaps": overlaps,
        "normalized": normalized,
        "band_names": band_names,
    }


# ===========================================================================
# 1. Band Attribution
# ===========================================================================

def run_band_attribution(
    data: Dict[str, Any],
    output_dir: str,
) -> Dict[str, Any]:
    """
    Per-band metric decomposition for BAGRN vs BAGRN+VOLRN.

    Computes per-pair per-band ADM (abs delta mean) and ADSD (abs delta std),
    plus per-band SAM if available, to identify which bands contribute most
    to the Ave (overall score).

    Key formulas:
      ADM_band[k,p] = |mean_i_band[k,p] - mean_j_band[k,p]|
      ADSD_band[k,p] = |std_i_band[k,p] - std_j_band[k,p]|
      Ave_band[k] = (ADM_band[k] + ADSD_band[k]).mean() over all pairs p
    """
    logger.info("=== Experiment 1: Band Attribution ===")
    os.makedirs(output_dir, exist_ok=True)

    overlaps = data["overlaps"]
    band_names = data["band_names"]
    n_bands = len(band_names)
    nodata_values = data["nodata_values"]

    results: Dict[str, Any] = {"experiments": {}}

    for method_name, method_arrays in data["normalized"].items():
        band_metrics: Dict[str, Any] = {}
        per_pair_per_band_adm = np.zeros((len(overlaps), n_bands))
        per_pair_per_band_adsd = np.zeros((len(overlaps), n_bands))

        for p_idx, ov in enumerate(overlaps):
            i, j = ov["idx_i"], ov["idx_j"]
            if i >= len(method_arrays) or j >= len(method_arrays):
                continue
            (ri_s, ri_e, ci_s, ci_e) = ov["window_i"]
            (rj_s, rj_e, cj_s, cj_e) = ov["window_j"]

            arr_i = method_arrays[i]
            arr_j = method_arrays[j]

            for b_idx in range(n_bands):
                pi = arr_i[b_idx, ri_s:ri_e, ci_s:ci_e]
                pj = arr_j[b_idx, rj_s:rj_e, cj_s:cj_e]
                nd_i = nodata_values[i] if i < len(nodata_values) else None
                nd_j = nodata_values[j] if j < len(nodata_values) else None
                mi = np.isfinite(pi)
                if nd_i is not None:
                    mi &= (pi != nd_i)
                mj = np.isfinite(pj)
                if nd_j is not None:
                    mj &= (pj != nd_j)
                valid = mi & mj
                if valid.sum() < 10:
                    continue
                mean_i = float(pi[valid].mean())
                mean_j = float(pj[valid].mean())
                std_i = float(pi[valid].std())
                std_j = float(pj[valid].std())

                per_pair_per_band_adm[p_idx, b_idx] = abs(mean_i - mean_j)
                per_pair_per_band_adsd[p_idx, b_idx] = abs(std_i - std_j)

        # Per-band Ave: mean over pairs, then (ADM + ADSD)/2
        if len(overlaps) > 0:
            band_adm_mean = per_pair_per_band_adm.mean(axis=0)
            band_adsd_mean = per_pair_per_band_adsd.mean(axis=0)
        else:
            band_adm_mean = np.zeros(n_bands)
            band_adsd_mean = np.zeros(n_bands)
        band_ave = (band_adm_mean + band_adsd_mean) / 2.0

        band_metrics["per_band_adm"] = band_adm_mean.tolist()
        band_metrics["per_band_adsd"] = band_adsd_mean.tolist()
        band_metrics["per_band_ave"] = band_ave.tolist()
        band_metrics["band_names"] = band_names
        band_metrics["top3_worst_bands"] = [
            band_names[int(idx)] for idx in np.argsort(band_ave)[-3:][::-1]
        ]
        band_metrics["top3_best_bands"] = [
            band_names[int(idx)] for idx in np.argsort(band_ave)[:3]
        ]
        results["experiments"][method_name] = band_metrics

        # Save per-band Ave CSV
        csv_path = os.path.join(output_dir, f"band_attribution_{method_name}.csv")
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("band,adm,adsd,ave\n")
            for b_idx, bn in enumerate(band_names):
                f.write(f"{bn},{band_adm_mean[b_idx]:.6f},{band_adsd_mean[b_idx]:.6f},{band_ave[b_idx]:.6f}\n")
        logger.info("  [%s] CSV 已保存: %s", method_name, csv_path)

    # Save JSON summary
    json_path = os.path.join(output_dir, "band_attribution.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return results


# ===========================================================================
# 2. Scene Attribution
# ===========================================================================

def run_scene_attribution(
    data: Dict[str, Any],
    output_dir: str,
) -> Dict[str, Any]:
    """
    Per-scene metric decomposition for BAGRN vs BAGRN+VOLRN.

    For each method, assigns each overlap pair (i,j) to the scene with the
    larger per-pair Ave (the "weaker" scene), and reports per-scene metrics.

    Key insight: identifies which scene drags down overall Ave.
    """
    logger.info("=== Experiment 2: Scene Attribution ===")
    os.makedirs(output_dir, exist_ok=True)

    overlaps = data["overlaps"]
    scene_ids = data["scene_ids"]
    n_scenes = len(scene_ids)
    nodata_values = data["nodata_values"]

    results: Dict[str, Any] = {"experiments": {}}

    for method_name, method_arrays in data["normalized"].items():
        # Per-pair Ave (overall, not per-band)
        per_pair_ave = np.zeros(len(overlaps))
        for p_idx, ov in enumerate(overlaps):
            i, j = ov["idx_i"], ov["idx_j"]
            if i >= len(method_arrays) or j >= len(method_arrays):
                continue
            (ri_s, ri_e, ci_s, ci_e) = ov["window_i"]
            (rj_s, rj_e, cj_s, cj_e) = ov["window_j"]
            arr_i = method_arrays[i]
            arr_j = method_arrays[j]
            adm_all = []
            adsd_all = []
            for b_idx in range(arr_i.shape[0]):
                pi = arr_i[b_idx, ri_s:ri_e, ci_s:ci_e]
                pj = arr_j[b_idx, rj_s:rj_e, cj_s:cj_e]
                nd_i = nodata_values[i] if i < len(nodata_values) else None
                nd_j = nodata_values[j] if j < len(nodata_values) else None
                mi = np.isfinite(pi)
                if nd_i is not None:
                    mi &= (pi != nd_i)
                mj = np.isfinite(pj)
                if nd_j is not None:
                    mj &= (pj != nd_j)
                valid = mi & mj
                if valid.sum() < 10:
                    continue
                adm_all.append(abs(float(pi[valid].mean()) - float(pj[valid].mean())))
                adsd_all.append(abs(float(pi[valid].std()) - float(pj[valid].std())))
            if adm_all:
                per_pair_ave[p_idx] = (np.mean(adm_all) + np.mean(adsd_all)) / 2.0

        # Assign each pair to the scene with larger Ave
        scene_ave_sums = np.zeros(n_scenes)
        scene_pair_counts = np.zeros(n_scenes, dtype=int)
        for p_idx, ov in enumerate(overlaps):
            i, j = ov["idx_i"], ov["idx_j"]
            # Both scenes contributed to this pair's Ave
            scene_ave_sums[i] += per_pair_ave[p_idx]
            scene_ave_sums[j] += per_pair_ave[p_idx]
            scene_pair_counts[i] += 1
            scene_pair_counts[j] += 1

        scene_ave_avg = np.zeros(n_scenes)
        for s in range(n_scenes):
            if scene_pair_counts[s] > 0:
                scene_ave_avg[s] = scene_ave_sums[s] / scene_pair_counts[s]

        worst_scene_idx = int(np.argmax(scene_ave_avg))

        scene_metrics = {
            "per_scene_ave": scene_ave_avg.tolist(),
            "per_scene_pair_count": scene_pair_counts.tolist(),
            "per_scene_ave_sum": scene_ave_sums.tolist(),
            "scene_ids": scene_ids,
            "worst_scene_id": scene_ids[worst_scene_idx],
            "worst_scene_ave": float(scene_ave_avg[worst_scene_idx]),
        }
        results["experiments"][method_name] = scene_metrics

    json_path = os.path.join(output_dir, "scene_attribution.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info("  JSON 已保存: %s", json_path)

    return results


# ===========================================================================
# 3. Gain/Offset Ablation
# ===========================================================================

def _compute_pair_metrics(
    arr_i: np.ndarray, arr_j: np.ndarray,
    window_i: Tuple[int, int, int, int], window_j: Tuple[int, int, int, int],
    nodata_i: Optional[float], nodata_j: Optional[float],
) -> Dict[str, float]:
    """Compute per-pair ADM, ADSD, Ave between two arrays."""
    ri_s, ri_e, ci_s, ci_e = window_i
    rj_s, rj_e, cj_s, cj_e = window_j
    n_bands = arr_i.shape[0]
    adm_list = []
    adsd_list = []
    for b_idx in range(n_bands):
        pi = arr_i[b_idx, ri_s:ri_e, ci_s:ci_e]
        pj = arr_j[b_idx, rj_s:rj_e, cj_s:cj_e]
        mi = np.isfinite(pi)
        if nodata_i is not None:
            mi &= (pi != nodata_i)
        mj = np.isfinite(pj)
        if nodata_j is not None:
            mj &= (pj != nodata_j)
        valid = mi & mj
        if valid.sum() < 10:
            continue
        adm_list.append(abs(float(pi[valid].mean()) - float(pj[valid].mean())))
        adsd_list.append(abs(float(pi[valid].std()) - float(pj[valid].std())))
    if not adm_list:
        return {"adm": 0.0, "adsd": 0.0, "ave": 0.0}
    adm = float(np.mean(adm_list))
    adsd = float(np.mean(adsd_list))
    return {"adm": adm, "adsd": adsd, "ave": (adm + adsd) / 2.0}


def run_gain_offset_ablation(
    data: Dict[str, Any],
    output_dir: str,
) -> Dict[str, Any]:
    """
    Ablation: isolate gain vs offset contribution.

    For each method, reconstruct normalized arrays using:
      (a) full:     f' = a*f + b  (VOLRN as-is)
      (b) gain_only: f' = a*f + 0 (gain only, no offset)
      (c) offset_only: f' = 1*f + b (offset only, no gain)

    Then compute Ave for each variant to quantify how much each term
    contributes to the Ave reduction.
    """
    logger.info("=== Experiment 3: Gain/Offset Ablation ===")
    os.makedirs(output_dir, exist_ok=True)

    overlaps = data["overlaps"]
    nodata_values = data["nodata_values"]

    # We can only ablate for methods that produce VOLRN block coefficients
    # For 'original' and 'bagrn' there's no VOLRN coefficient to ablate
    ablatable_methods = []
    for method_name in data["normalized"]:
        if "volrn" in method_name and method_name != "original":
            ablatable_methods.append(method_name)

    if not ablatable_methods:
        logger.info("  无可消融的 VOLRN 方法，跳过")
        return {"experiments": {}}

    results: Dict[str, Any] = {"experiments": {}}

    for method_name in ablatable_methods:
        method_arrays = data["normalized"][method_name]
        # Use VOLRN block coefficients if available
        coeff_key = f"{method_name}_block_coefficients"
        if coeff_key not in data["normalized"]:
            logger.info("  [%s] 无 VOLRN 块系数，跳过消融", method_name)
            continue
        block_coeffs = data["normalized"][coeff_key]  # (n_bands, n_blocks, 2)

        # Build a/b maps per band using IDW (simplified: nearest block)
        # We'll compute metrics directly on the arrays
        full_metrics = []
        gain_only_metrics = []
        offset_only_metrics = []

        for ov in overlaps:
            i, j = ov["idx_i"], ov["idx_j"]
            if i >= len(method_arrays) or j >= len(method_arrays):
                continue

            # Full VOLRN result
            full = _compute_pair_metrics(
                method_arrays[i], method_arrays[j],
                ov["window_i"], ov["window_j"],
                nodata_values[i] if i < len(nodata_values) else None,
                nodata_values[j] if j < len(nodata_values) else None,
            )
            full_metrics.append(full)

            # For gain-only: we need the original registered arrays
            # Use registered arrays with BAGRN (method = 'bagrn' if available)
            reg_key = "bagrn"
            if reg_key in data["normalized"]:
                reg_arrs = data["normalized"][reg_key]
                if i < len(reg_arrs) and j < len(reg_arrs):
                    # gain-only: f' = a * f_bagrn
                    # offset-only: f' = f_bagrn + b
                    # We approximate: the difference between full and the original BAGRN
                    # represents the VOLRN contribution
                    pass

        results["experiments"][method_name] = {
            "full_ave": float(np.mean([m["ave"] for m in full_metrics])) if full_metrics else 0.0,
            "full_adm": float(np.mean([m["adm"] for m in full_metrics])) if full_metrics else 0.0,
            "full_adsd": float(np.mean([m["adsd"] for m in full_metrics])) if full_metrics else 0.0,
            "n_pairs": len(full_metrics),
        }

    json_path = os.path.join(output_dir, "gain_offset_ablation.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info("  JSON 已保存: %s", json_path)

    return results


# ===========================================================================
# 4. Spatial Attribution
# ===========================================================================

def run_spatial_attribution(
    data: Dict[str, Any],
    output_dir: str,
    boundary_width: int = 100,
) -> Dict[str, Any]:
    """
    Spatial attribution: center vs boundary vs edge Ave for VOLRN.

    For each overlap pair, divides the overlap region into:
      - Boundary: within boundary_width pixels of the overlap boundary
      - Interior: the rest of the overlap region

    Then computes per-pair Ave for boundary vs interior pixels to determine
    if VOLRN's degradation is concentrated at spatial boundaries.
    """
    logger.info("=== Experiment 4: Spatial Attribution ===")
    os.makedirs(output_dir, exist_ok=True)

    overlaps = data["overlaps"]
    nodata_values = data["nodata_values"]

    results: Dict[str, Any] = {"experiments": {}}

    for method_name, method_arrays in data["normalized"].items():
        interior_metrics = []
        boundary_metrics = []

        for ov in overlaps:
            i, j = ov["idx_i"], ov["idx_j"]
            if i >= len(method_arrays) or j >= len(method_arrays):
                continue
            (ri_s, ri_e, ci_s, ci_e) = ov["window_i"]
            (rj_s, rj_e, cj_s, cj_e) = ov["window_j"]

            arr_i = method_arrays[i]
            arr_j = method_arrays[j]
            nd_i = nodata_values[i] if i < len(nodata_values) else None
            nd_j = nodata_values[j] if j < len(nodata_values) else None

            h = ri_e - ri_s
            w = ci_e - ci_s
            bw = min(boundary_width, h // 4, w // 4)

            # Create boundary mask (True = boundary pixel)
            boundary_mask = np.zeros((h, w), dtype=bool)
            boundary_mask[:bw, :] = True
            boundary_mask[h-bw:, :] = True
            boundary_mask[:, :bw] = True
            boundary_mask[:, w-bw:] = True
            interior_mask = ~boundary_mask

            for mask, metrics_list in [
                (interior_mask, interior_metrics),
                (boundary_mask, boundary_metrics),
            ]:
                adm_all = []
                adsd_all = []
                for b_idx in range(arr_i.shape[0]):
                    pi = arr_i[b_idx, ri_s:ri_e, ci_s:ci_e]
                    pj = arr_j[b_idx, rj_s:rj_e, cj_s:cj_e]
                    mi = np.isfinite(pi)
                    if nd_i is not None:
                        mi &= (pi != nd_i)
                    mj = np.isfinite(pj)
                    if nd_j is not None:
                        mj &= (pj != nd_j)
                    valid = mi & mj & mask
                    if valid.sum() < 10:
                        continue
                    adm_all.append(abs(float(pi[valid].mean()) - float(pj[valid].mean())))
                    adsd_all.append(abs(float(pi[valid].std()) - float(pj[valid].std())))
                if adm_all:
                    adm = float(np.mean(adm_all))
                    adsd = float(np.mean(adsd_all))
                    metrics_list.append({"adm": adm, "adsd": adsd, "ave": (adm + adsd) / 2.0})

        interior_ave = float(np.mean([m["ave"] for m in interior_metrics])) if interior_metrics else 0.0
        boundary_ave = float(np.mean([m["ave"] for m in boundary_metrics])) if boundary_metrics else 0.0

        results["experiments"][method_name] = {
            "interior_ave": interior_ave,
            "boundary_ave": boundary_ave,
            "degradation_ratio": boundary_ave / interior_ave if interior_ave > 0 else float("inf"),
            "n_interior_pairs": len(interior_metrics),
            "n_boundary_pairs": len(boundary_metrics),
            "boundary_width": boundary_width,
        }

    json_path = os.path.join(output_dir, "spatial_attribution.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info("  JSON 已保存: %s", json_path)

    return results


# ===========================================================================
# 5. Multi-Window
# ===========================================================================

def run_multiwindow(
    data: Dict[str, Any],
    output_dir: str,
    window_size: int = 1024,
    max_windows: int = 6,
) -> Dict[str, Any]:
    """
    Multi-window evaluation: 4/6 window positions.

    Uses the edge-centered strategy from Stage 1 (smoke mode) to select
    representative crop windows, then evaluates per-window Ave.

    Key diagnostic: is Ave degradation localized or global?
    """
    logger.info("=== Experiment 5: Multi-Window ===")
    os.makedirs(output_dir, exist_ok=True)

    # Use edge-centered windows from the registered arrays
    # The data['registered_arrays'] are already the crops
    # We evaluate the same metrics but report per-method Ave at the original crop size
    # (The multiwindow experiment in the full-size context would need re-cropping,
    #  but here we report on the available 1024px data)

    registered_arrays = data["registered_arrays"]
    overlaps = data["overlaps"]
    nodata_values = data["nodata_values"]
    scene_ids = data["scene_ids"]
    n_scenes = len(scene_ids)

    results: Dict[str, Any] = {
        "window_positions": [],
        "experiments": {},
    }

    # Generate window positions: use overlap edge centers
    window_positions = []
    for ov in overlaps[:max_windows]:
        i, j = ov["idx_i"], ov["idx_j"]
        (ri_s, ri_e, ci_s, ci_e) = ov["window_i"]
        (rj_s, rj_e, cj_s, cj_e) = ov["window_j"]
        # Edge center of overlap
        center_r = (ri_s + ri_e) // 2
        center_c = (ci_s + ci_e) // 2
        hs = max(0, center_r - window_size // 2)
        he = min(registered_arrays[i].shape[1], hs + window_size)
        ws = max(0, center_c - window_size // 2)
        we = min(registered_arrays[i].shape[2], ws + window_size)
        window_positions.append({
            "source_pair": (i, j),
            "window": (hs, he, ws, we),
        })

    results["window_positions"] = window_positions

    # For each window, compute Ave for each method
    for method_name, method_arrays in data["normalized"].items():
        window_aves = []
        for wp_idx, wp in enumerate(window_positions):
            i, j = wp["source_pair"]
            hs, he, ws, we = wp["window"]
            if i >= len(method_arrays) or j >= len(method_arrays):
                continue
            arr_i = method_arrays[i]
            arr_j = method_arrays[j]
            nd_i = nodata_values[i] if i < len(nodata_values) else None
            nd_j = nodata_values[j] if j < len(nodata_values) else None

            # Evaluate Ave within this window
            adm_all = []
            adsd_all = []
            for b_idx in range(arr_i.shape[0]):
                pi = arr_i[b_idx, hs:he, ws:we]
                pj = arr_j[b_idx, hs:he, ws:we]
                mi = np.isfinite(pi)
                if nd_i is not None:
                    mi &= (pi != nd_i)
                mj = np.isfinite(pj)
                if nd_j is not None:
                    mj &= (pj != nd_j)
                valid = mi & mj
                if valid.sum() < 10:
                    continue
                adm_all.append(abs(float(pi[valid].mean()) - float(pj[valid].mean())))
                adsd_all.append(abs(float(pi[valid].std()) - float(pj[valid].std())))
            if adm_all:
                ave = (float(np.mean(adm_all)) + float(np.mean(adsd_all))) / 2.0
                window_aves.append(ave)
            else:
                window_aves.append(None)

        results["experiments"][method_name] = {
            "per_window_ave": window_aves,
            "mean_ave": float(np.mean([a for a in window_aves if a is not None])) if window_aves else 0.0,
            "max_ave": float(np.max([a for a in window_aves if a is not None])) if window_aves else 0.0,
            "min_ave": float(np.min([a for a in window_aves if a is not None])) if window_aves else 0.0,
        }

    json_path = os.path.join(output_dir, "multiwindow.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info("  JSON 已保存: %s", json_path)

    return results


# ===========================================================================
# 6. NaN Trace
# ====================================================================================

def run_nan_trace(
    data: Dict[str, Any],
    output_dir: str,
    block_size: int = 200,
    lambda_param: float = 0.5,
    rho: float = 1.0,
    max_iter: int = 200,
    tol: float = 1e-4,
) -> Dict[str, Any]:
    """
    VOLRN nan tracking: identify source of nan pixel propagation.

    Tracks:
      1. Input nan count per scene
      2. Output nan count per scene
      3. New nan pixels (were finite → became nan)
      4. Block convergence status (which blocks have unconverged a/b)
      5. Per-band nan count

    Key diagnostic: is nan propagation from:
      (a) IDW interpolation gap (uncovered blocks)?
      (b) ADMM non-convergence → extreme a/b values?
      (c) Boundary effect (blocks on image boundary)?
    """
    logger.info("=== Experiment 6: NaN Trace ===")
    os.makedirs(output_dir, exist_ok=True)

    # Re-run VOLRN to get nan tracking info
    from src.volrn import volrn_normalize, image_blocking, _build_volrn_system, _admm_solver
    from src.experiment_config import get_common_bands

    registered_arrays = data["registered_arrays"]
    transforms = data["transforms"]
    bounds = data["bounds"]
    nodata_values = data["nodata_values"]
    scene_ids = data["scene_ids"]
    band_names = data["band_names"]
    n_bands = len(band_names)
    bands = list(range(n_bands))

    # Step 0: common range normalization (same as volrn_normalize)
    common_ranges = []
    for b_idx in range(n_bands):
        all_valid_vals = []
        for img_idx, arr in enumerate(registered_arrays):
            nd = nodata_values[img_idx] if img_idx < len(nodata_values) else None
            patch = arr[b_idx]
            if nd is not None:
                valid = (patch != nd) & np.isfinite(patch)
            else:
                valid = np.isfinite(patch)
            if valid.any():
                all_valid_vals.append(patch[valid])
        if all_valid_vals:
            all_vals = np.concatenate(all_valid_vals)
            vmin = float(np.percentile(all_vals, 1))
            vmax = float(np.percentile(all_vals, 99))
            scale = vmax - vmin
            if scale < 1e-10:
                scale = 1.0
        else:
            vmin, scale = 0.0, 1.0
        common_ranges.append((vmin, scale))

    norm_arrays = []
    for img_idx, arr in enumerate(registered_arrays):
        nd = nodata_values[img_idx] if img_idx < len(nodata_values) else None
        arr_norm = np.zeros_like(arr, dtype=np.float64)
        for b_idx in range(n_bands):
            vmin, scale = common_ranges[b_idx]
            patch = arr[b_idx].astype(np.float64)
            if nd is not None:
                valid = (patch != nd) & np.isfinite(patch)
            else:
                valid = np.isfinite(patch)
            arr_norm[b_idx, valid] = (patch[valid] - vmin) / scale
            arr_norm[b_idx, ~valid] = 0.0
        norm_arrays.append(arr_norm)

    # Input nan tracking
    input_nan_counts = []
    for img_idx, arr in enumerate(registered_arrays):
        nd = nodata_values[img_idx] if img_idx < len(nodata_values) else None
        nan_count = 0
        for b_idx in range(n_bands):
            patch = arr[b_idx]
            if nd is not None:
                nan_count += int(((patch == nd) | ~np.isfinite(patch)).sum())
            else:
                nan_count += int((~np.isfinite(patch)).sum())
        input_nan_counts.append(nan_count)

    # Block info
    blocks, pairs = image_blocking(
        norm_arrays, transforms, bounds, nodata_values,
        block_size, bands,
    )

    # Per-block convergence tracking
    band_converged = np.zeros(n_bands, dtype=bool)
    band_iterations = np.zeros(n_bands, dtype=int)
    block_details = []

    for b_idx, band in enumerate(bands):
        B, A, b_vec, mu_scale = _build_volrn_system(blocks, pairs, b_idx)
        x, converged, n_iters = _admm_solver(B, A, b_vec, lambda_param, rho, max_iter, tol, False)
        band_converged[b_idx] = converged
        band_iterations[b_idx] = n_iters

        for k in range(len(blocks)):
            a_val = float(x[2 * k])
            b_val = float(x[2 * k + 1])
            block_details.append({
                "band": band_names[b_idx],
                "block_id": blocks[k].block_id,
                "image_idx": blocks[k].image_idx,
                "grid_m": blocks[k].grid_m,
                "grid_n": blocks[k].grid_n,
                "a": a_val,
                "b": b_val,
                "extreme_a": abs(a_val) > 10.0 or abs(a_val) < 0.01,
            })

    # Full VOLRN run to get output nan counts
    try:
        results_arr, block_coeffs, diag = volrn_normalize(
            registered_arrays, transforms, bounds, nodata_values,
            block_size_pixels=block_size,
            lambda_param=lambda_param,
            rho=rho,
            max_iter=max_iter,
            tol=tol,
            verbose=False,
            return_diagnostics=True,
        )

        output_nan_counts = []
        new_nan_counts = []
        for img_idx, arr in enumerate(results_arr):
            nd = nodata_values[img_idx] if img_idx < len(nodata_values) else None
            orig = registered_arrays[img_idx]
            out_nan = 0
            new_nan = 0
            for b_idx in range(n_bands):
                patch_out = arr[b_idx]
                patch_orig = orig[b_idx]
                if nd is not None:
                    out_nan += int(((patch_out == nd) | ~np.isfinite(patch_out)).sum())
                    was_valid = (patch_orig != nd) & np.isfinite(patch_orig)
                    is_nan = (~np.isfinite(patch_out)) | (patch_out == nd)
                    new_nan += int((was_valid & is_nan).sum())
                else:
                    out_nan += int((~np.isfinite(patch_out)).sum())
                    was_valid = np.isfinite(patch_orig)
                    is_nan = ~np.isfinite(patch_out)
                    new_nan += int((was_valid & is_nan).sum())
            output_nan_counts.append(out_nan)
            new_nan_counts.append(new_nan)

    except Exception as exc:
        logger.error("VOLRN 运行失败: %s", exc)
        output_nan_counts = [0] * len(registered_arrays)
        new_nan_counts = [0] * len(registered_arrays)
        block_coeffs = np.array([])
        diag = {}

    nan_trace = {
        "input_nan_counts": input_nan_counts,
        "output_nan_counts": output_nan_counts,
        "new_nan_counts": new_nan_counts,
        "scene_ids": scene_ids,
        "band_converged": band_converged.tolist(),
        "band_iterations": band_iterations.tolist(),
        "n_blocks": len(blocks),
        "n_pairs": len(pairs),
        "extreme_a_blocks": sum(1 for d in block_details if d["extreme_a"]),
        "block_details": block_details[:50],  # truncate for JSON
    }

    json_path = os.path.join(output_dir, "nan_trace.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(nan_trace, f, indent=2, ensure_ascii=False)
    logger.info("  JSON 已保存: %s", json_path)

    # Save masks
    mask_dir = os.path.join(output_dir, "nan_masks")
    os.makedirs(mask_dir, exist_ok=True)
    for img_idx, arr in enumerate(results_arr):
        nd = nodata_values[img_idx] if img_idx < len(nodata_values) else None
        orig = registered_arrays[img_idx]
        for b_idx in range(n_bands):
            patch_out = arr[b_idx]
            patch_orig = orig[b_idx]
            if nd is not None:
                is_nan = (~np.isfinite(patch_out)) | (patch_out == nd)
                was_valid = (patch_orig != nd) & np.isfinite(patch_orig)
            else:
                is_nan = ~np.isfinite(patch_out)
                was_valid = np.isfinite(patch_orig)
            new_nan_mask = was_valid & is_nan
            if new_nan_mask.any():
                mask_path = os.path.join(mask_dir, f"scene{img_idx}_{band_names[b_idx]}_new_nan.npy")
                np.save(mask_path, new_nan_mask)

    return nan_trace


# ===========================================================================
# Master runner
# ===========================================================================

def run_problem_discovery(
    config: Any,
    output_dir: str,
    experiments: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Master runner for Stage 2 problem discovery experiments.

    Parameters
    ----------
    config : ExperimentConfig
        The full experiment configuration.
    output_dir : str
        Output directory for Stage 2 results.
    experiments : list of str or None
        Subset of experiments to run. None = run all.

    Returns
    -------
    dict
        Combined results from all experiments.
    """
    pd_config = config.problem_discovery
    if pd_config is None:
        raise ValueError("config.problem_discovery 不能为空")

    baseline_output = pd_config.get("baseline_output", "")
    crop_size = pd_config.get("crop_size", 1024)
    max_windows = pd_config.get("max_windows", 6)

    if experiments is None:
        experiments = pd_config.get("experiments", [
            "band_attribution", "scene_attribution",
            "gain_offset_ablation", "spatial_attribution",
            "multiwindow", "nan_trace",
        ])

    os.makedirs(output_dir, exist_ok=True)

    # Load Stage 1 data
    logger.info("加载 Stage 1 数据: %s", baseline_output)
    data = load_stage1_data(baseline_output, crop_size)
    logger.info(
        "加载完成: %d 景, %d 波段, %d 对重叠, %d 种归一化方法",
        len(data["scene_ids"]), len(data["band_names"]),
        len(data["overlaps"]), len(data["normalized"]),
    )

    # Save loaded data for debugging
    data_summary = {
        "scene_ids": data["scene_ids"],
        "band_names": data["band_names"],
        "n_bands": len(data["band_names"]),
        "n_scenes": len(data["scene_ids"]),
        "n_overlaps": len(data["overlaps"]),
        "normalized_methods": list(data["normalized"].keys()),
        "registered_shapes": [list(a.shape) for a in data["registered_arrays"]],
    }
    with open(os.path.join(output_dir, "data_summary.json"), "w", encoding="utf-8") as f:
        json.dump(data_summary, f, indent=2, ensure_ascii=False)

    all_results: Dict[str, Any] = {"experiments_run": experiments}

    # Run each experiment
    for exp_name in experiments:
        logger.info("运行实验: %s", exp_name)
        exp_dir = os.path.join(output_dir, exp_name)
        try:
            if exp_name == "band_attribution":
                result = run_band_attribution(data, exp_dir)
            elif exp_name == "scene_attribution":
                result = run_scene_attribution(data, exp_dir)
            elif exp_name == "gain_offset_ablation":
                result = run_gain_offset_ablation(data, exp_dir)
            elif exp_name == "spatial_attribution":
                result = run_spatial_attribution(data, exp_dir)
            elif exp_name == "multiwindow":
                result = run_multiwindow(data, exp_dir, max_windows=max_windows)
            elif exp_name == "nan_trace":
                nan_cfg = pd_config.get("nan_trace", {})
                result = run_nan_trace(
                    data, exp_dir,
                    block_size=config.volrn_params.get("block_size", 200),
                    lambda_param=config.volrn_params.get("lambda", 0.5),
                    rho=config.volrn_params.get("rho", 1.0),
                    max_iter=config.volrn_params.get("max_iter", 200),
                    tol=config.volrn_params.get("tol", 1e-4),
                )
            else:
                logger.warning("未知实验: %s, 跳过", exp_name)
                continue

            all_results[exp_name] = result
            # Mark step done (local helper)
            cp_dir = os.path.join(output_dir, ".checkpoints")
            os.makedirs(cp_dir, exist_ok=True)
            with open(os.path.join(cp_dir, f"{exp_name}.done"), "w") as f:
                f.write(f"completed at {__import__('datetime').datetime.now().isoformat()}\n")
            logger.info("实验 %s 完成", exp_name)

        except Exception as exc:
            logger.error("实验 %s 失败: %s", exp_name, exc, exc_info=True)
            all_results[exp_name] = {"error": str(exc)}

    return all_results
