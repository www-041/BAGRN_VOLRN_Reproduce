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

Sensor: DZ01 VNIR (B01-B14, 410-860 nm). B15-B16 not used.
SWIR is a separate sensor (DZ01S, B01-B10, 1178-2468 nm) and is not
included in Stage 1 or Stage 2 experiments.
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
    sensor_id: str = "VNIR",
    band_wavelengths: Optional[Dict[str, Dict[str, float]]] = None,
    metadata_status: str = "from_mtl",
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

    # Build wavelength lookup
    if band_wavelengths is None:
        band_wavelengths = {}

    results: Dict[str, Any] = {"experiments": {}, "sensor_id": sensor_id}

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
        band_metrics["sensor_id"] = sensor_id
        band_metrics["top3_worst_bands"] = [
            band_names[int(idx)] for idx in np.argsort(band_ave)[-3:][::-1]
        ]
        band_metrics["top3_best_bands"] = [
            band_names[int(idx)] for idx in np.argsort(band_ave)[:3]
        ]
        results["experiments"][method_name] = band_metrics

        # Save per-band Ave CSV with sensor-qualified names
        csv_path = os.path.join(output_dir, f"band_attribution_{method_name}.csv")
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("sensor_id,band_name,qualified_band_name,wavelength_center_nm,adm,adsd,ave,metadata_status\n")
            for b_idx, bn in enumerate(band_names):
                wl = band_wavelengths.get(bn, {})
                center_wl = wl.get("center_nm", "")
                qualified = f"{sensor_id}_{bn}"
                f.write(
                    f"{sensor_id},{bn},{qualified},{center_wl},"
                    f"{band_adm_mean[b_idx]:.6f},{band_adsd_mean[b_idx]:.6f},{band_ave[b_idx]:.6f},"
                    f"{metadata_status}\n"
                )
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
    metadata_list: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    BAGRN gain/offset ablation experiment.

    Determines whether BAGRN's spectral changes come primarily from
    multiplicative gain (omega) or additive offset (upsilon).

    BAGRN formula (Eq. 10-11):
        mu_target  = mu_orig + theta_mu[band, scene]
        sigma_target = sigma_orig + theta_sigma[band, scene]
        omega  = sigma_target / sigma_orig       # gain
        upsilon = mu_target - omega * mu_orig    # offset
        output  = omega * input + upsilon        # moment matching

    Ablation variants:
        1. original            - registered, no normalization
        2. bagrn_gain_only     - omega * input (upsilon = 0)
        3. bagrn_offset_only   - input + upsilon (omega = 1)
        4. bagrn_full_reconstructed - omega * input + upsilon
        5. bagrn_volrn_reference    - loaded from Stage 1
    """
    logger.info("=== Experiment 3: BAGRN Gain/Offset Ablation ===")
    os.makedirs(output_dir, exist_ok=True)

    from src.gain_offset_ablation import run_gain_offset_ablation as _run_ablation

    # Load Stage 1 BAGRN and VOLRN outputs for validation/reference
    baseline_output = data.get("baseline_output", "")
    bagrn_arrays = data["normalized"].get("bagrn")
    volrn_arrays = data["normalized"].get("bagrn_volrn")

    # Run ablation
    result = _run_ablation(
        registered_arrays=data["registered_arrays"],
        nodata_values=data["nodata_values"],
        overlaps=data["overlaps"],
        band_names=data["band_names"],
        scene_ids=data["scene_ids"],
        bagrn_output_arrays=bagrn_arrays,
        volrn_output_arrays=volrn_arrays,
        output_dir=output_dir,
        control_idx=0,
    )

    return result["summary"]


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
# 7. Acquisition Attribution (diagnostic)
# ===========================================================================

def run_acquisition_attribution(
    data: Dict[str, Any],
    output_dir: str,
    scene_metadata: Optional[List[Dict[str, Any]]] = None,
    band_metadata: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Diagnostic experiment: correlate acquisition conditions with radiometric residuals.

    Merges scene_attribution and band_attribution results with per-scene
    acquisition parameters and per-band integration parameters.

    Parameters
    ----------
    data : dict
        Loaded Stage 1 data (from load_stage1_data).
    output_dir : str
        Output directory for results.
    scene_metadata : list of dict or None
        Per-scene MTL metadata (from parse_dz01_mtl).
    band_metadata : list of dict or None
        Per-band integration parameters (long-format table).

    Returns
    -------
    dict
        Experiment results.
    """
    logger.info("=== Experiment 7: Acquisition Attribution ===")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "metrics"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "figures"), exist_ok=True)

    overlaps = data["overlaps"]
    band_names = data["band_names"]
    n_bands = len(band_names)
    scene_ids = data["scene_ids"]
    n_scenes = len(scene_ids)
    nodata_values = data["nodata_values"]

    results: Dict[str, Any] = {"experiments": {}}

    # Build per-scene per-band Ave for bagrn_volrn
    method_name = "bagrn_volrn"
    if method_name not in data["normalized"]:
        # Fall back to bagrn
        method_name = "bagrn"
    if method_name not in data["normalized"]:
        logger.warning("No normalized method available for acquisition attribution")
        return results

    method_arrays = data["normalized"][method_name]

    # Per-pair per-band metrics
    per_pair_per_band_ave = np.zeros((len(overlaps), n_bands))
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
            adm = abs(float(pi[valid].mean()) - float(pj[valid].mean()))
            adsd = abs(float(pi[valid].std()) - float(pj[valid].std()))
            per_pair_per_band_adm[p_idx, b_idx] = adm
            per_pair_per_band_adsd[p_idx, b_idx] = adsd
            per_pair_per_band_ave[p_idx, b_idx] = (adm + adsd) / 2.0

    # Per-scene per-band Ave: average over pairs involving that scene
    scene_band_ave = np.zeros((n_scenes, n_bands))
    scene_band_counts = np.zeros(n_scenes, dtype=int)
    for p_idx, ov in enumerate(overlaps):
        i, j = ov["idx_i"], ov["idx_j"]
        scene_band_ave[i] += per_pair_per_band_ave[p_idx]
        scene_band_ave[j] += per_pair_per_band_ave[p_idx]
        scene_band_counts[i] += 1
        scene_band_counts[j] += 1

    for s in range(n_scenes):
        if scene_band_counts[s] > 0:
            scene_band_ave[s] /= scene_band_counts[s]

    # Build scene_band_attribution.csv
    scene_band_rows = []
    # Build lookup for acquisition params per scene
    acq_lookup = {}
    if scene_metadata:
        for meta in scene_metadata:
            sid = meta.get("date_acquired", "unknown")
            scene = meta.get("scene", {})
            acq_lookup[sid] = {
                "sun_elevation": scene.get("SUN_ELEVATION"),
                "sun_zenith": scene.get("SUN_ZENITH"),
                "sat_zenith": scene.get("SAT_ZENITH"),
                "sat_azimuth": scene.get("SAT_AZIMUTH"),
                "roll_angle": scene.get("ROLL_ANGLE"),
                "cloud_cover": scene.get("CLOUD_COVER"),
            }

    # Build lookup for integration params per (scene, band)
    int_lookup = {}
    if band_metadata:
        for row in band_metadata:
            key = (row.get("scene_id"), row.get("band_name"))
            int_lookup[key] = {
                "integration_time": row.get("integration_time"),
                "integration_level": row.get("integration_level"),
            }

    for s_idx, s_id in enumerate(scene_ids):
        for b_idx, b_name in enumerate(band_names):
            acq = acq_lookup.get(s_id, {})
            integ = int_lookup.get((s_id, b_name), {})
            wl_info = {}
            if scene_metadata:
                for meta in scene_metadata:
                    if meta.get("date_acquired") == s_id:
                        band_info = meta.get("bands", {}).get(b_name, {})
                        wl_info = {"wavelength_center_nm": band_info.get("wavelength_center_nm")}
                        break

            row = {
                "scene_id": s_id,
                "band_name": b_name,
                "wavelength_center_nm": wl_info.get("wavelength_center_nm"),
                "ave": float(scene_band_ave[s_idx, b_idx]),
                "adm": float(per_pair_per_band_adm[:, b_idx].mean()),
                "adsd": float(per_pair_per_band_adsd[:, b_idx].mean()),
                "integration_time": integ.get("integration_time"),
                "integration_level": integ.get("integration_level"),
                "sun_elevation": acq.get("sun_elevation"),
                "sun_zenith": acq.get("sun_zenith"),
                "sat_zenith": acq.get("sat_zenith"),
                "sat_azimuth": acq.get("sat_azimuth"),
                "roll_angle": acq.get("roll_angle"),
                "cloud_cover": acq.get("cloud_cover"),
            }
            scene_band_rows.append(row)

    # Save scene_band_attribution.csv
    csv_path = os.path.join(output_dir, "metrics", "scene_band_attribution.csv")
    if scene_band_rows:
        headers = list(scene_band_rows[0].keys())
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write(",".join(headers) + "\n")
            for row in scene_band_rows:
                vals = [str(row.get(h, "")) for h in headers]
                f.write(",".join(vals) + "\n")
        logger.info("  scene_band_attribution.csv saved: %s", csv_path)

    # Save acquisition_attribution.csv (per-scene summary)
    acq_csv_path = os.path.join(output_dir, "metrics", "acquisition_attribution.csv")
    scene_ave_per_method = {}
    for m_name, m_arrays in data["normalized"].items():
        scene_ave = np.zeros(n_scenes)
        scene_cnt = np.zeros(n_scenes, dtype=int)
        for p_idx, ov in enumerate(overlaps):
            i, j = ov["idx_i"], ov["idx_j"]
            if i >= len(m_arrays) or j >= len(m_arrays):
                continue
            (ri_s, ri_e, ci_s, ci_e) = ov["window_i"]
            (rj_s, rj_e, cj_s, cj_e) = ov["window_j"]
            adm_all = []
            for b_idx in range(n_bands):
                pi = m_arrays[i][b_idx, ri_s:ri_e, ci_s:ci_e]
                pj = m_arrays[j][b_idx, rj_s:rj_e, cj_s:cj_e]
                nd_i = nodata_values[i] if i < len(nodata_values) else None
                nd_j = nodata_values[j] if j < len(nodata_values) else None
                mi = np.isfinite(pi)
                if nd_i is not None:
                    mi &= (pi != nd_i)
                mj = np.isfinite(pj)
                if nd_j is not None:
                    mj &= (pj != nd_j)
                valid = mi & mj
                if valid.sum() >= 10:
                    adm_all.append(abs(float(pi[valid].mean()) - float(pj[valid].mean())))
            if adm_all:
                scene_ave[i] += np.mean(adm_all)
                scene_ave[j] += np.mean(adm_all)
                scene_cnt[i] += 1
                scene_cnt[j] += 1
        for s in range(n_scenes):
            if scene_cnt[s] > 0:
                scene_ave[s] /= scene_cnt[s]
        scene_ave_per_method[m_name] = scene_ave.tolist()

    acq_rows = []
    for s_idx, s_id in enumerate(scene_ids):
        acq = acq_lookup.get(s_id, {})
        row = {"scene_id": s_id}
        for m_name, aves in scene_ave_per_method.items():
            row[f"{m_name}_ave"] = aves[s_idx]
        row.update(acq)
        acq_rows.append(row)

    if acq_rows:
        headers = list(acq_rows[0].keys())
        with open(acq_csv_path, "w", encoding="utf-8") as f:
            f.write(",".join(headers) + "\n")
            for row in acq_rows:
                vals = [str(row.get(h, "")) for h in headers]
                f.write(",".join(vals) + "\n")
        logger.info("  acquisition_attribution.csv saved: %s", acq_csv_path)

    # Save JSON summary
    json_path = os.path.join(output_dir, "acquisition_attribution.json")
    summary = {
        "method": method_name,
        "scene_ids": scene_ids,
        "band_names": band_names,
        "scene_band_ave": scene_band_ave.tolist(),
        "per_pair_per_band_ave": per_pair_per_band_ave.tolist(),
        "n_scenes": n_scenes,
        "n_bands": n_bands,
        "n_overlaps": len(overlaps),
        "note": "descriptive analysis only because n_scenes=4; exploratory only",
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info("  JSON saved: %s", json_path)

    results["experiments"][method_name] = summary
    return results


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

    # ---- Sensor and metadata info ----
    sensor_config = getattr(config, 'sensor', None) or {}
    sensor_id = sensor_config.get("id", "VNIR") if isinstance(sensor_config, dict) else "VNIR"
    selected_bands = sensor_config.get("selected_bands", []) if isinstance(sensor_config, dict) else []
    excluded_bands = sensor_config.get("excluded_bands", []) if isinstance(sensor_config, dict) else []
    ref_meta_path = (sensor_config.get("reference_metadata", {}) or {}).get("path") if isinstance(sensor_config, dict) else None
    scene_meta_paths = sensor_config.get("scene_metadata_paths", {}) if isinstance(sensor_config, dict) else {}

    # ---- Parse reference metadata ----
    band_wavelengths: Dict[str, Dict[str, float]] = {}
    metadata_status = "no_mtl"
    if ref_meta_path and os.path.isfile(ref_meta_path):
        try:
            from src.dz01_metadata import parse_dz01_mtl, extract_band_metadata
            ref_meta = parse_dz01_mtl(ref_meta_path)
            band_wavelengths_raw = extract_band_metadata(ref_meta, selected_bands or None)
            for bn, info in band_wavelengths_raw.items():
                band_wavelengths[bn] = {
                    "center_nm": info.get("wavelength_center_nm"),
                    "min_nm": info.get("wavelength_min_nm"),
                    "max_nm": info.get("wavelength_max_nm"),
                }
            metadata_status = "from_reference_mtl"
            logger.info("已加载参考元数据: %s (sensor=%s)", ref_meta_path, ref_meta.get("sensor_id"))
        except Exception as exc:
            logger.warning("参考元数据加载失败: %s", exc)
            metadata_status = "mtl_load_failed"

    # ---- Parse per-scene metadata (all scenes) ----
    from src.dz01_metadata import (
        parse_dz01_mtl, extract_band_metadata, validate_cross_scene_band_metadata,
        build_scene_acquisition_table, analyze_acquisition_differences,
        build_band_acquisition_table, detect_integration_anomalies,
    )
    scene_metadata_list = []
    scene_metadata_status = {}
    for scene_cfg in config.scenes:
        sid = scene_cfg.get("id", "unknown")
        mtl_path = scene_meta_paths.get(sid)
        if mtl_path and os.path.isfile(mtl_path):
            try:
                meta = parse_dz01_mtl(mtl_path)
                scene_metadata_list.append(meta)
                scene_metadata_status[sid] = "complete"
                logger.info("已加载场景元数据: %s (sensor=%s)", sid, meta.get("sensor_id"))
            except Exception as exc:
                logger.warning("场景 %s 元数据加载失败: %s", sid, exc)
                scene_metadata_status[sid] = "load_failed"
        else:
            scene_metadata_status[sid] = "missing"
            logger.warning("场景 %s 无元数据路径或文件不存在: %s", sid, mtl_path)

    n_scenes_loaded = len(scene_metadata_list)
    n_scenes_expected = len(config.scenes)
    scene_specific_metadata_complete = (
        n_scenes_loaded == n_scenes_expected
        and all(v == "complete" for v in scene_metadata_status.values())
    )

    # ---- Cross-scene band consistency validation ----
    cross_scene_consistency = {}
    if scene_metadata_list:
        cross_scene_consistency = validate_cross_scene_band_metadata(
            scene_metadata_list, selected_bands or None
        )
        if cross_scene_consistency.get("warnings"):
            for w in cross_scene_consistency["warnings"]:
                logger.warning("Cross-scene: %s", w)
        else:
            logger.info("Cross-scene band consistency: PASS")

    # ---- Acquisition condition analysis ----
    acquisition_analysis = {}
    if scene_metadata_list:
        acquisition_analysis = analyze_acquisition_differences(scene_metadata_list)

    # ---- Band acquisition parameters (long-format) ----
    band_acq_table = []
    if scene_metadata_list:
        band_acq_table = build_band_acquisition_table(scene_metadata_list, selected_bands or None)

    # ---- Integration anomaly detection ----
    integration_anomalies = []
    if scene_metadata_list:
        integration_anomalies = detect_integration_anomalies(
            scene_metadata_list, selected_bands or None
        )
        if integration_anomalies:
            logger.info("Detected %d integration anomalies", len(integration_anomalies))
            for anom in integration_anomalies:
                logger.warning(
                    "  Anomaly: %s %s %s = %.4f (median=%.4f, ratio=%.4f)",
                    anom["scene_id"], anom["band_name"], anom["parameter"],
                    anom["value"], anom["cross_scene_median"], anom["ratio"],
                )

    # Load Stage 1 data
    logger.info("加载 Stage 1 数据: %s", baseline_output)
    data = load_stage1_data(baseline_output, crop_size)
    data["baseline_output"] = baseline_output
    logger.info(
        "加载完成: %d 景, %d 波段, %d 对重叠, %d 种归一化方法",
        len(data["scene_ids"]), len(data["band_names"]),
        len(data["overlaps"]), len(data["normalized"]),
    )

    # Save loaded data for debugging
    wavelength_range = {}
    if band_wavelengths:
        centers = [v["center_nm"] for v in band_wavelengths.values() if v.get("center_nm") is not None]
        if centers:
            wavelength_range = {"min": min(centers), "max": max(centers)}

    # Build cross-scene band consistency summary
    cross_scene_summary = {}
    if cross_scene_consistency:
        cross_scene_summary = {
            "consistent": cross_scene_consistency.get("consistent", False),
            "n_warnings": len(cross_scene_consistency.get("warnings", [])),
        }

    # Build acquisition summary
    acq_summary = {}
    if acquisition_analysis and "field_stats" in acquisition_analysis:
        for field, stats in acquisition_analysis["field_stats"].items():
            acq_summary[field] = {
                "min": stats.get("min"),
                "max": stats.get("max"),
                "median": stats.get("median"),
            }

    data_summary = {
        "scene_ids": data["scene_ids"],
        "band_names": data["band_names"],
        "n_bands": len(data["band_names"]),
        "n_scenes": len(data["scene_ids"]),
        "n_overlaps": len(data["overlaps"]),
        "normalized_methods": list(data["normalized"].keys()),
        "registered_shapes": [list(a.shape) for a in data["registered_arrays"]],
        "sensor_id": sensor_id,
        "sensor_full_band_count": 16 if sensor_id == "VNIR" else 10,
        "selected_band_count": len(selected_bands) if selected_bands else len(data["band_names"]),
        "selected_bands": selected_bands or data["band_names"],
        "excluded_bands": excluded_bands,
        "wavelength_range_nm": wavelength_range,
        "band_metadata_source": {
            "scene_date": "2025-12-08" if ref_meta_path and "20251208" in str(ref_meta_path) else "unknown",
            "sensor": sensor_id,
            "role": "reference_only",
        },
        "metadata_coverage": {
            "status": "complete" if scene_specific_metadata_complete else "incomplete",
            "n_scenes_expected": n_scenes_expected,
            "n_scenes_loaded": n_scenes_loaded,
            "sensors": [sensor_id],
            "scene_status": scene_metadata_status,
        },
        "band_definition_consistency": cross_scene_summary,
        "scene_specific_metadata_complete": scene_specific_metadata_complete,
        "acquisition_analysis": acq_summary,
        "integration_anomaly_count": len(integration_anomalies),
    }
    with open(os.path.join(output_dir, "data_summary.json"), "w", encoding="utf-8") as f:
        json.dump(data_summary, f, indent=2, ensure_ascii=False)

    all_results: Dict[str, Any] = {"experiments_run": experiments, "sensor_id": sensor_id}

    # Save metadata tables
    meta_dir = os.path.join(output_dir, "metadata")
    os.makedirs(meta_dir, exist_ok=True)

    # Save scene_acquisition_conditions.csv
    if scene_metadata_list:
        acq_table = build_scene_acquisition_table(scene_metadata_list)
        if acq_table:
            headers = list(acq_table[0].keys())
            with open(os.path.join(meta_dir, "scene_acquisition_conditions.csv"), "w", encoding="utf-8") as f:
                f.write(",".join(headers) + "\n")
                for row in acq_table:
                    vals = [str(row.get(h, "")) for h in headers]
                    f.write(",".join(vals) + "\n")

    # Save band_acquisition_parameters.csv (long format)
    if band_acq_table:
        headers = list(band_acq_table[0].keys())
        with open(os.path.join(meta_dir, "band_acquisition_parameters.csv"), "w", encoding="utf-8") as f:
            f.write(",".join(headers) + "\n")
            for row in band_acq_table:
                vals = [str(row.get(h, "")) for h in headers]
                f.write(",".join(vals) + "\n")

    # Save acquisition_parameter_anomalies.csv
    if integration_anomalies:
        headers = list(integration_anomalies[0].keys())
        with open(os.path.join(meta_dir, "acquisition_parameter_anomalies.csv"), "w", encoding="utf-8") as f:
            f.write(",".join(headers) + "\n")
            for row in integration_anomalies:
                vals = [str(row.get(h, "")) for h in headers]
                f.write(",".join(vals) + "\n")

    # Save cross_scene_band_consistency.json
    if cross_scene_consistency:
        with open(os.path.join(meta_dir, "cross_scene_band_consistency.json"), "w", encoding="utf-8") as f:
            json.dump(cross_scene_consistency, f, indent=2, ensure_ascii=False)

    # Save band_metadata_table.csv
    if scene_metadata_list:
        band_table_rows = []
        for meta in scene_metadata_list:
            sid = meta.get("date_acquired", "unknown")
            bands = meta.get("bands", {})
            for band_name in sorted(bands.keys()):
                band_info = bands[band_name]
                consistent = cross_scene_consistency.get("band_consistency", {}).get(band_name, {}).get("consistent", True)
                band_table_rows.append({
                    "scene_id": sid,
                    "date_acquired": meta.get("date_acquired"),
                    "sensor_id": meta.get("sensor_id"),
                    "band_name": band_name,
                    "qualified_band_name": f"{meta.get('sensor_id', 'VNIR')}_{band_name}",
                    "wavelength_min_nm": band_info.get("wavelength_min_nm"),
                    "wavelength_center_nm": band_info.get("wavelength_center_nm"),
                    "wavelength_max_nm": band_info.get("wavelength_max_nm"),
                    "data_type": band_info.get("data_type"),
                    "grid_cell_size_m": meta.get("resolution_vi"),
                    "processing_software_version": meta.get("scene", {}).get("PROCESSING_SOFTWARE_VERSION"),
                    "consistent_across_scenes": consistent,
                })
        if band_table_rows:
            headers = list(band_table_rows[0].keys())
            with open(os.path.join(meta_dir, "band_metadata_table.csv"), "w", encoding="utf-8") as f:
                f.write(",".join(headers) + "\n")
                for row in band_table_rows:
                    vals = [str(row.get(h, "")) for h in headers]
                    f.write(",".join(vals) + "\n")

    # Save scene_acquisition_differences.json
    if acquisition_analysis:
        with open(os.path.join(meta_dir, "scene_acquisition_differences.json"), "w", encoding="utf-8") as f:
            json.dump(acquisition_analysis, f, indent=2, ensure_ascii=False)

    # Run each experiment
    for exp_name in experiments:
        logger.info("运行实验: %s", exp_name)
        exp_dir = os.path.join(output_dir, exp_name)
        try:
            if exp_name == "band_attribution":
                result = run_band_attribution(
                    data, exp_dir,
                    sensor_id=sensor_id,
                    band_wavelengths=band_wavelengths,
                    metadata_status=metadata_status,
                )
            elif exp_name == "scene_attribution":
                result = run_scene_attribution(data, exp_dir)
            elif exp_name == "gain_offset_ablation":
                result = run_gain_offset_ablation(
                    data, exp_dir, metadata_list=scene_metadata_list,
                )
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
            elif exp_name == "acquisition_attribution":
                result = run_acquisition_attribution(
                    data, exp_dir,
                    scene_metadata=scene_metadata_list,
                    band_metadata=band_acq_table,
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
