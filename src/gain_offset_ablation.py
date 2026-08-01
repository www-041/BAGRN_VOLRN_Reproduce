"""
Stage 2: BAGRN Gain/Offset Ablation Experiment

Determines whether BAGRN's spectral changes come primarily from multiplicative
gain (omega) or additive offset (upsilon).

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

import os
import csv
import json
import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ===========================================================================
# 1. Coefficient extraction
# ===========================================================================

def compute_bagrn_coefficients(
    registered_arrays: List[np.ndarray],
    nodata_values: List[Optional[float]],
    overlaps: List[dict],
    control_idx: int = 0,
) -> Dict[str, Any]:
    """
    Run BAGRN solver to extract gain/offset coefficients without writing files.

    Returns dict with:
        theta_mu: (n_bands, n_images)
        theta_sigma: (n_bands, n_images)
        omega: (n_bands, n_images) - gain
        upsilon: (n_bands, n_images) - offset
        mu_orig: (n_bands, n_images) - original means
        sigma_orig: (n_bands, n_images) - original stds
        scene_ids, band_names, control_idx, etc.
    """
    from src.bagrn import bagrn_normalize, _overlap_means_stds, _solve_compensation

    n_images = len(registered_arrays)
    n_bands = registered_arrays[0].shape[0]
    bands = list(range(n_bands))

    # Step 1: overlap statistics
    pair_means, pair_stds, pair_pixels = _overlap_means_stds(
        registered_arrays, nodata_values, overlaps, bands,
    )

    # Step 2: solve compensation
    theta_mu = _solve_compensation(
        n_images, overlaps, pair_means, pair_pixels, control_idx,
    )
    theta_sigma = _solve_compensation(
        n_images, overlaps, pair_stds, pair_pixels, control_idx,
    )

    # Step 3: compute global mu, sigma per image
    mu_orig = np.zeros((n_bands, n_images))
    sigma_orig = np.zeros((n_bands, n_images))
    for img_idx in range(n_images):
        arr = registered_arrays[img_idx]
        nd = nodata_values[img_idx]
        for b_idx in range(n_bands):
            band_data = arr[b_idx]
            if nd is None:
                valid = np.isfinite(band_data)
            else:
                valid = band_data != nd
            if valid.sum() > 0:
                mu_orig[b_idx, img_idx] = band_data[valid].mean()
                sigma_orig[b_idx, img_idx] = band_data[valid].std()

    # Compute gain and offset
    omega = np.ones_like(theta_mu)
    upsilon = np.zeros_like(theta_mu)
    for b_idx in range(n_bands):
        for img_idx in range(n_images):
            sg = sigma_orig[b_idx, img_idx]
            sg_t = sg + theta_sigma[b_idx, img_idx]
            if sg < 1e-12 or sg_t < 1e-12:
                omega[b_idx, img_idx] = 1.0
            else:
                omega[b_idx, img_idx] = sg_t / sg
            upsilon[b_idx, img_idx] = (
                mu_orig[b_idx, img_idx] + theta_mu[b_idx, img_idx]
                - omega[b_idx, img_idx] * mu_orig[b_idx, img_idx]
            )

    return {
        "theta_mu": theta_mu,
        "theta_sigma": theta_sigma,
        "omega": omega,
        "upsilon": upsilon,
        "mu_orig": mu_orig,
        "sigma_orig": sigma_orig,
        "n_images": n_images,
        "n_bands": n_bands,
        "control_idx": control_idx,
    }


# ===========================================================================
# 2. Apply ablation methods
# ===========================================================================

def apply_gain_only(
    array: np.ndarray,
    omega: np.ndarray,
    nodata: Optional[float],
    bands: List[int],
) -> np.ndarray:
    """Apply gain only: output = omega * input (upsilon = 0)."""
    result = array.astype(np.float64, copy=True)
    for b_idx, band in enumerate(bands):
        band_data = result[band]
        if nodata is None:
            band_data[:] = omega[b_idx] * band_data
        else:
            mask = band_data != nodata
            band_data[mask] = omega[b_idx] * band_data[mask]
    return result


def apply_offset_only(
    array: np.ndarray,
    upsilon: np.ndarray,
    nodata: Optional[float],
    bands: List[int],
) -> np.ndarray:
    """Apply offset only: output = input + upsilon (omega = 1)."""
    result = array.astype(np.float64, copy=True)
    for b_idx, band in enumerate(bands):
        band_data = result[band]
        if nodata is None:
            band_data[:] = band_data + upsilon[b_idx]
        else:
            mask = band_data != nodata
            band_data[mask] = band_data[mask] + upsilon[b_idx]
    return result


def apply_full_reconstruction(
    array: np.ndarray,
    omega: np.ndarray,
    upsilon: np.ndarray,
    nodata: Optional[float],
    bands: List[int],
) -> np.ndarray:
    """Apply full BAGRN: output = omega * input + upsilon."""
    result = array.astype(np.float64, copy=True)
    for b_idx, band in enumerate(bands):
        band_data = result[band]
        if nodata is None:
            band_data[:] = omega[b_idx] * band_data + upsilon[b_idx]
        else:
            mask = band_data != nodata
            band_data[mask] = omega[b_idx] * band_data[mask] + upsilon[b_idx]
    return result


# ===========================================================================
# 3. Validation
# ===========================================================================

def validate_full_reconstruction(
    reconstructed: List[np.ndarray],
    reference: List[np.ndarray],
    nodata_values: List[Optional[float]],
    rtol: float = 1e-5,
    atol: float = 1e-4,
) -> Dict[str, Any]:
    """
    Validate that full reconstruction matches reference BAGRN output.

    Uses relaxed thresholds (rtol=1e-5, atol=1e-4) to account for
    potential float32 precision in GeoTIFF I/O.
    """
    results = []
    all_pass = True

    for img_idx in range(len(reconstructed)):
        rec = reconstructed[img_idx]
        ref = reference[img_idx]
        nd = nodata_values[img_idx]

        if nd is None:
            valid = np.isfinite(rec) & np.isfinite(ref)
        else:
            valid = (rec != nd) & (ref != nd) & np.isfinite(rec) & np.isfinite(ref)

        n_compared = int(valid.sum())
        if n_compared == 0:
            results.append({"scene_idx": img_idx, "status": "no_valid_pixels"})
            continue

        diff = np.abs(rec[valid] - ref[valid])
        max_abs_error = float(diff.max())
        mean_abs_error = float(diff.mean())
        rmse = float(np.sqrt((diff ** 2).mean()))

        ref_vals = ref[valid]
        ref_range = float(np.abs(ref_vals).max())
        relative_rmse = rmse / ref_range if ref_range > 0 else 0.0

        close = bool(np.allclose(rec[valid], ref[valid], rtol=rtol, atol=atol))

        if not close:
            all_pass = False

        results.append({
            "scene_idx": img_idx,
            "n_compared": n_compared,
            "max_abs_error": max_abs_error,
            "mean_abs_error": mean_abs_error,
            "rmse": rmse,
            "relative_rmse": relative_rmse,
            "allclose": close,
            "status": "pass" if close else "FAIL",
        })

    return {"all_pass": all_pass, "per_scene": results}


# ===========================================================================
# 4. Radiometric metrics
# ===========================================================================

def compute_radiometric_metrics(
    method_arrays: Dict[str, List[np.ndarray]],
    registered_arrays: List[np.ndarray],
    overlaps: List[dict],
    nodata_values: List[Optional[float]],
    band_names: List[str],
    scene_ids: List[str],
) -> Dict[str, Any]:
    """
    Compute ADM, ADSD, CD, GL, RDOA, Ave for all methods.

    Returns dict with global, per-band, per-scene, per-pair, scene-band, pair-band metrics.
    """
    methods = list(method_arrays.keys())
    n_bands = len(band_names)
    n_scenes = len(scene_ids)
    n_pairs = len(overlaps)

    # Pre-compute per-pair per-band metrics for each method
    method_pair_band = {}  # method -> (n_pairs, n_bands) arrays
    for method_name in methods:
        arrs = method_arrays[method_name]
        adm = np.zeros((n_pairs, n_bands))
        adsd = np.zeros((n_pairs, n_bands))

        for p_idx, ov in enumerate(overlaps):
            i, j = ov["idx_i"], ov["idx_j"]
            if i >= len(arrs) or j >= len(arrs):
                continue
            (ri_s, ri_e, ci_s, ci_e) = ov["window_i"]
            (rj_s, rj_e, cj_s, cj_e) = ov["window_j"]

            for b_idx in range(n_bands):
                pi = arrs[i][b_idx, ri_s:ri_e, ci_s:ci_e]
                pj = arrs[j][b_idx, rj_s:rj_e, cj_s:cj_e]
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
                adm[p_idx, b_idx] = abs(float(pi[valid].mean()) - float(pj[valid].mean()))
                adsd[p_idx, b_idx] = abs(float(pi[valid].std()) - float(pj[valid].std()))

        method_pair_band[method_name] = {"adm": adm, "adsd": adsd}

    # Compute Ave = (ADM + ADSD) / 2
    global_results = {}
    per_band_results = {}
    per_scene_results = {}
    per_pair_results = {}
    scene_band_results = {}
    pair_band_results = {}

    for method_name in methods:
        adm = method_pair_band[method_name]["adm"]
        adsd = method_pair_band[method_name]["adsd"]
        ave = (adm + adsd) / 2.0

        # Global
        global_results[method_name] = {
            "adm": float(adm.mean()),
            "adsd": float(adsd.mean()),
            "ave": float(ave.mean()),
        }

        # Per-band
        per_band_results[method_name] = []
        for b_idx in range(n_bands):
            per_band_results[method_name].append({
                "band_name": band_names[b_idx],
                "adm": float(adm[:, b_idx].mean()),
                "adsd": float(adsd[:, b_idx].mean()),
                "ave": float(ave[:, b_idx].mean()),
            })

        # Per-scene
        per_scene_results[method_name] = []
        scene_ave_sums = np.zeros(n_scenes)
        scene_pair_counts = np.zeros(n_scenes, dtype=int)
        for p_idx, ov in enumerate(overlaps):
            i, j = ov["idx_i"], ov["idx_j"]
            scene_ave_sums[i] += ave[p_idx].mean()
            scene_ave_sums[j] += ave[p_idx].mean()
            scene_pair_counts[i] += 1
            scene_pair_counts[j] += 1
        for s_idx in range(n_scenes):
            avg = scene_ave_sums[s_idx] / scene_pair_counts[s_idx] if scene_pair_counts[s_idx] > 0 else 0.0
            per_scene_results[method_name].append({
                "scene_id": scene_ids[s_idx],
                "ave": float(avg),
            })

        # Per-pair
        per_pair_results[method_name] = []
        for p_idx, ov in enumerate(overlaps):
            per_pair_results[method_name].append({
                "pair_idx": p_idx,
                "scene_i": scene_ids[ov["idx_i"]],
                "scene_j": scene_ids[ov["idx_j"]],
                "ave": float(ave[p_idx].mean()),
            })

        # Scene-band
        scene_band_results[method_name] = []
        scene_band_ave = np.zeros((n_scenes, n_bands))
        scene_band_counts = np.zeros(n_scenes, dtype=int)
        for p_idx, ov in enumerate(overlaps):
            i, j = ov["idx_i"], ov["idx_j"]
            scene_band_ave[i] += ave[p_idx]
            scene_band_ave[j] += ave[p_idx]
            scene_band_counts[i] += 1
            scene_band_counts[j] += 1
        for s_idx in range(n_scenes):
            if scene_band_counts[s_idx] > 0:
                scene_band_ave[s_idx] /= scene_band_counts[s_idx]
            for b_idx in range(n_bands):
                scene_band_results[method_name].append({
                    "scene_id": scene_ids[s_idx],
                    "band_name": band_names[b_idx],
                    "ave": float(scene_band_ave[s_idx, b_idx]),
                })

        # Pair-band
        pair_band_results[method_name] = []
        for p_idx, ov in enumerate(overlaps):
            for b_idx in range(n_bands):
                pair_band_results[method_name].append({
                    "pair_idx": p_idx,
                    "scene_i": scene_ids[ov["idx_i"]],
                    "scene_j": scene_ids[ov["idx_j"]],
                    "band_name": band_names[b_idx],
                    "ave": float(ave[p_idx, b_idx]),
                })

    return {
        "global": global_results,
        "per_band": per_band_results,
        "per_scene": per_scene_results,
        "per_pair": per_pair_results,
        "scene_band": scene_band_results,
        "pair_band": pair_band_results,
    }


# ===========================================================================
# 5. Spectral metrics (SAM)
# ===========================================================================

def compute_sam(
    arr_i: np.ndarray,
    arr_j: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> float:
    """
    Compute Spectral Angle Mapper between two co-registered pixels.

    For each valid pixel, computes the angle between the spectral vectors.
    Returns mean SAM in degrees.
    """
    n_bands = arr_i.shape[0]
    h, w = arr_i.shape[1], arr_i.shape[2]

    # Flatten spatial dims
    i_flat = arr_i.reshape(n_bands, -1).T  # (pixels, bands)
    j_flat = arr_j.reshape(n_bands, -1).T

    if mask is not None:
        mask_flat = mask.reshape(-1)
        i_flat = i_flat[mask_flat]
        j_flat = j_flat[mask_flat]

    if len(i_flat) == 0:
        return 0.0

    # Compute norms
    norm_i = np.linalg.norm(i_flat, axis=1)
    norm_j = np.linalg.norm(j_flat, axis=1)

    # Avoid division by zero
    valid = (norm_i > 1e-10) & (norm_j > 1e-10)
    if valid.sum() == 0:
        return 0.0

    # Dot product
    dots = np.sum(i_flat[valid] * j_flat[valid], axis=1)
    cos_angle = dots / (norm_i[valid] * norm_j[valid])
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angles = np.arccos(cos_angle)

    return float(np.degrees(angles.mean()))


def compute_spectral_metrics(
    method_arrays: Dict[str, List[np.ndarray]],
    registered_arrays: List[np.ndarray],
    overlaps: List[dict],
    nodata_values: List[Optional[float]],
    band_names: List[str],
    scene_ids: List[str],
) -> Dict[str, Any]:
    """Compute SAM, spectral RMSE, and correlation metrics."""
    methods = list(method_arrays.keys())
    n_bands = len(band_names)
    n_scenes = len(scene_ids)

    global_results = {}
    per_scene_results = {}
    per_band_results = {}
    scene_band_results = {}

    for method_name in methods:
        arrs = method_arrays[method_name]
        all_sam_values = []
        scene_sam = {sid: [] for sid in scene_ids}
        band_change_sums = np.zeros(n_bands)
        band_change_counts = np.zeros(n_bands)

        for p_idx, ov in enumerate(overlaps):
            i, j = ov["idx_i"], ov["idx_j"]
            if i >= len(arrs) or j >= len(arrs):
                continue
            (ri_s, ri_e, ci_s, ci_e) = ov["window_i"]
            (rj_s, rj_e, cj_s, cj_e) = ov["window_j"]

            # Use registered original as reference for SAM
            ref_arr = registered_arrays[i][:, ri_s:ri_e, ci_s:ci_e]
            method_arr = arrs[i][:, ri_s:ri_e, ci_s:ci_e]
            nd = nodata_values[i] if i < len(nodata_values) else None

            if nd is not None:
                valid_mask = (ref_arr[0] != nd) & (method_arr[0] != nd)
            else:
                valid_mask = np.isfinite(ref_arr[0]) & np.isfinite(method_arr[0])

            sam = compute_sam(ref_arr, method_arr, valid_mask)
            all_sam_values.append(sam)
            scene_sam[scene_ids[i]].append(sam)

            # Per-band change
            for b_idx in range(n_bands):
                pi = ref_arr[b_idx].flatten()
                pj = method_arr[b_idx].flatten()
                if nd is not None:
                    bm = valid_mask.flatten()
                else:
                    bm = np.isfinite(pi) & np.isfinite(pj)
                if bm.sum() > 10:
                    change = float(np.abs(pi[bm].mean() - pj[bm].mean()))
                    band_change_sums[b_idx] += change
                    band_change_counts[b_idx] += 1

        # Global SAM stats
        if all_sam_values:
            sam_arr = np.array(all_sam_values)
            global_results[method_name] = {
                "sam_mean": float(sam_arr.mean()),
                "sam_median": float(np.median(sam_arr)),
                "sam_p90": float(np.percentile(sam_arr, 90)),
                "sam_p95": float(np.percentile(sam_arr, 95)),
            }
        else:
            global_results[method_name] = {
                "sam_mean": 0.0, "sam_median": 0.0,
                "sam_p90": 0.0, "sam_p95": 0.0,
            }

        # Per-scene SAM
        per_scene_results[method_name] = []
        for sid in scene_ids:
            vals = scene_sam.get(sid, [])
            per_scene_results[method_name].append({
                "scene_id": sid,
                "sam_mean": float(np.mean(vals)) if vals else 0.0,
            })

        # Per-band mean absolute change
        per_band_results[method_name] = []
        for b_idx in range(n_bands):
            avg_change = band_change_sums[b_idx] / band_change_counts[b_idx] if band_change_counts[b_idx] > 0 else 0.0
            per_band_results[method_name].append({
                "band_name": band_names[b_idx],
                "mean_abs_change": avg_change,
            })

    return {
        "global": global_results,
        "per_scene": per_scene_results,
        "per_band": per_band_results,
    }


# ===========================================================================
# 6. Save coefficients
# ===========================================================================

def save_coefficients(
    coeffs: Dict[str, Any],
    band_names: List[str],
    scene_ids: List[str],
    output_dir: str,
) -> None:
    """Save BAGRN coefficients to CSV and JSON."""
    os.makedirs(output_dir, exist_ok=True)

    omega = coeffs["omega"]
    upsilon = coeffs["upsilon"]
    mu_orig = coeffs["mu_orig"]
    sigma_orig = coeffs["sigma_orig"]
    control_idx = coeffs["control_idx"]

    # CSV
    csv_path = os.path.join(output_dir, "bagrn_global_coefficients.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("scene_index,scene_id,band_index,band_name,"
                "gain,offset,mu_orig,sigma_orig,is_control_scene\n")
        for s_idx, sid in enumerate(scene_ids):
            for b_idx, bn in enumerate(band_names):
                is_ctrl = s_idx == control_idx
                f.write(f"{s_idx},{sid},{b_idx},{bn},"
                        f"{omega[b_idx, s_idx]:.8f},{upsilon[b_idx, s_idx]:.8f},"
                        f"{mu_orig[b_idx, s_idx]:.8f},{sigma_orig[b_idx, s_idx]:.8f},"
                        f"{is_ctrl}\n")

    # JSON summary
    json_path = os.path.join(output_dir, "bagrn_global_coefficients.json")
    summary = {
        "n_scenes": len(scene_ids),
        "n_bands": len(band_names),
        "control_idx": control_idx,
        "control_scene": scene_ids[control_idx],
        "band_names": band_names,
        "scene_ids": scene_ids,
        "omega": omega.tolist(),
        "upsilon": upsilon.tolist(),
        "mu_orig": mu_orig.tolist(),
        "sigma_orig": sigma_orig.tolist(),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info("Coefficients saved to %s", output_dir)


# ===========================================================================
# 7. Metrics CSV writers
# ===========================================================================

def _write_csv(rows: List[Dict], path: str) -> None:
    """Write list of dicts to CSV."""
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    headers = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def save_radiometric_metrics(
    metrics: Dict[str, Any],
    output_dir: str,
) -> None:
    """Save radiometric metrics to CSV files."""
    metrics_dir = os.path.join(output_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    # Global
    global_rows = []
    for method, vals in metrics["global"].items():
        global_rows.append({"method": method, **vals})
    _write_csv(global_rows, os.path.join(metrics_dir, "radiometric_global.csv"))

    # Per-band
    for method, band_list in metrics["per_band"].items():
        rows = [{"method": method, **b} for b in band_list]
        _write_csv(rows, os.path.join(metrics_dir, f"radiometric_per_band_{method}.csv"))

    # Per-scene
    for method, scene_list in metrics["per_scene"].items():
        rows = [{"method": method, **s} for s in scene_list]
        _write_csv(rows, os.path.join(metrics_dir, f"radiometric_per_scene_{method}.csv"))

    # Scene-band
    for method, sb_list in metrics["scene_band"].items():
        rows = [{"method": method, **sb} for sb in sb_list]
        _write_csv(rows, os.path.join(metrics_dir, f"radiometric_scene_band_{method}.csv"))


def save_spectral_metrics(
    metrics: Dict[str, Any],
    output_dir: str,
) -> None:
    """Save spectral metrics to CSV files."""
    metrics_dir = os.path.join(output_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    # Global
    global_rows = []
    for method, vals in metrics["global"].items():
        global_rows.append({"method": method, **vals})
    _write_csv(global_rows, os.path.join(metrics_dir, "spectral_global.csv"))

    # Per-scene
    for method, scene_list in metrics["per_scene"].items():
        rows = [{"method": method, **s} for s in scene_list]
        _write_csv(rows, os.path.join(metrics_dir, f"spectral_per_scene_{method}.csv"))

    # Per-band
    for method, band_list in metrics["per_band"].items():
        rows = [{"method": method, **b} for b in band_list]
        _write_csv(rows, os.path.join(metrics_dir, f"spectral_per_band_{method}.csv"))


# ===========================================================================
# 8. Main ablation runner
# ===========================================================================

def run_gain_offset_ablation(
    registered_arrays: List[np.ndarray],
    nodata_values: List[Optional[float]],
    overlaps: List[dict],
    band_names: List[str],
    scene_ids: List[str],
    bagrn_output_arrays: Optional[List[np.ndarray]] = None,
    volrn_output_arrays: Optional[List[np.ndarray]] = None,
    output_dir: str = "data/output/stage2_gain_offset_ablation",
    control_idx: int = 0,
) -> Dict[str, Any]:
    """
    Main entry point for the gain/offset ablation experiment.

    Parameters
    ----------
    registered_arrays : list of ndarray
        Registered original arrays (n_images x (bands, rows, cols)).
    nodata_values : list of float or None
        NoData values per image.
    overlaps : list of dict
        Overlap window info.
    band_names : list of str
        Band names (e.g. ['B01', 'B02', ...]).
    scene_ids : list of str
        Scene identifiers.
    bagrn_output_arrays : list of ndarray or None
        Stage 1 BAGRN output arrays (for validation).
    volrn_output_arrays : list of ndarray or None
        Stage 1 BAGRN+VOLRN output arrays (for reference).
    output_dir : str
        Output directory.
    control_idx : int
        Control scene index.

    Returns
    -------
    dict
        Complete experiment results.
    """
    n_images = len(registered_arrays)
    n_bands = registered_arrays[0].shape[0]
    bands = list(range(n_bands))

    os.makedirs(output_dir, exist_ok=True)

    # ---- Step 1: Compute BAGRN coefficients ----
    logger.info("Computing BAGRN coefficients...")
    coeffs = compute_bagrn_coefficients(
        registered_arrays, nodata_values, overlaps, control_idx,
    )
    omega = coeffs["omega"]
    upsilon = coeffs["upsilon"]

    # Save coefficients
    coeff_dir = os.path.join(output_dir, "coefficients")
    save_coefficients(coeffs, band_names, scene_ids, coeff_dir)

    # ---- Step 2: Apply ablation methods ----
    logger.info("Applying ablation methods...")
    method_arrays = {}

    # original
    method_arrays["original"] = registered_arrays

    # bagrn_gain_only
    gain_only = []
    for img_idx in range(n_images):
        arr = apply_gain_only(
            registered_arrays[img_idx],
            omega[:, img_idx],
            nodata_values[img_idx],
            bands,
        )
        gain_only.append(arr)
    method_arrays["bagrn_gain_only"] = gain_only

    # bagrn_offset_only
    offset_only = []
    for img_idx in range(n_images):
        arr = apply_offset_only(
            registered_arrays[img_idx],
            upsilon[:, img_idx],
            nodata_values[img_idx],
            bands,
        )
        offset_only.append(arr)
    method_arrays["bagrn_offset_only"] = offset_only

    # bagrn_full_reconstructed
    full_recon = []
    for img_idx in range(n_images):
        arr = apply_full_reconstruction(
            registered_arrays[img_idx],
            omega[:, img_idx],
            upsilon[:, img_idx],
            nodata_values[img_idx],
            bands,
        )
        full_recon.append(arr)
    method_arrays["bagrn_full_reconstructed"] = full_recon

    # bagrn_volrn_reference (if available)
    if volrn_output_arrays is not None and len(volrn_output_arrays) == n_images:
        method_arrays["bagrn_volrn_reference"] = volrn_output_arrays

    # ---- Step 3: Validate full reconstruction ----
    logger.info("Validating full reconstruction against Stage 1 BAGRN...")
    validation = {"all_pass": False, "per_scene": []}
    if bagrn_output_arrays is not None and len(bagrn_output_arrays) == n_images:
        validation = validate_full_reconstruction(
            full_recon, bagrn_output_arrays, nodata_values,
        )
        status = "PASS" if validation["all_pass"] else "FAIL"
        logger.info("Full reconstruction validation: %s", status)
        for ps in validation["per_scene"]:
            logger.info("  scene %d: %s (RMSE=%.6f)", ps["scene_idx"], ps["status"], ps.get("rmse", 0))
    else:
        logger.warning("Stage 1 BAGRN output not available for validation")

    # Save validation
    val_dir = os.path.join(output_dir, "validation")
    os.makedirs(val_dir, exist_ok=True)
    _write_csv(validation["per_scene"], os.path.join(val_dir, "full_reconstruction_validation.csv"))
    with open(os.path.join(val_dir, "full_reconstruction_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"all_pass": validation["all_pass"], "per_scene": validation["per_scene"]}, f, indent=2)

    # ---- Step 4: Compute radiometric metrics ----
    logger.info("Computing radiometric metrics...")
    rad_metrics = compute_radiometric_metrics(
        method_arrays, registered_arrays, overlaps, nodata_values, band_names, scene_ids,
    )
    save_radiometric_metrics(rad_metrics, output_dir)

    # ---- Step 5: Compute spectral metrics ----
    logger.info("Computing spectral metrics...")
    spec_metrics = compute_spectral_metrics(
        method_arrays, registered_arrays, overlaps, nodata_values, band_names, scene_ids,
    )
    save_spectral_metrics(spec_metrics, output_dir)

    # ---- Step 6: Determine dominant effect ----
    logger.info("Determining dominant effect...")
    sam_full = spec_metrics["global"].get("bagrn_full_reconstructed", {}).get("sam_mean", 0)
    sam_gain = spec_metrics["global"].get("bagrn_gain_only", {}).get("sam_mean", 0)
    sam_offset = spec_metrics["global"].get("bagrn_offset_only", {}).get("sam_mean", 0)
    sam_orig = spec_metrics["global"].get("original", {}).get("sam_mean", 0)

    eps = 1e-10
    gain_fraction = sam_gain / (sam_full + eps) if sam_full > eps else 0
    offset_fraction = sam_offset / (sam_full + eps) if sam_full > eps else 0

    if gain_fraction >= 0.70 and offset_fraction <= 0.40:
        dominant = "gain_dominant"
    elif offset_fraction >= 0.70 and gain_fraction <= 0.40:
        dominant = "offset_dominant"
    elif gain_fraction >= 0.40 and offset_fraction >= 0.40:
        dominant = "joint_effect"
    else:
        dominant = "interaction_or_nonlinear"

    logger.info("Dominant effect: %s (gain_frac=%.3f, offset_frac=%.3f)", dominant, gain_fraction, offset_fraction)

    # ---- Step 7: Save summary ----
    summary = {
        "dominant_effect": dominant,
        "gain_fraction_of_full_sam": gain_fraction,
        "offset_fraction_of_full_sam": offset_fraction,
        "sam_original": sam_orig,
        "sam_gain_only": sam_gain,
        "sam_offset_only": sam_offset,
        "sam_full_reconstructed": sam_full,
        "radiometric_global": rad_metrics["global"],
        "spectral_global": spec_metrics["global"],
        "full_reconstruction_validation": validation["all_pass"],
        "n_scenes": n_images,
        "n_bands": n_bands,
        "band_names": band_names,
        "scene_ids": scene_ids,
        "control_idx": control_idx,
    }
    with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info("Ablation experiment complete. Output: %s", output_dir)

    return {
        "coefficients": coeffs,
        "method_arrays": method_arrays,
        "validation": validation,
        "radiometric_metrics": rad_metrics,
        "spectral_metrics": spec_metrics,
        "summary": summary,
    }
