"""
评价指标模块

实现论文 Section 3.1 中的六项指标：
  - ADM   (Eq.36)  重叠区均值绝对差
  - ADSD  (Eq.37)  重叠区标准差绝对差
  - CD    (Eq.38)  颜色距离（直方图差异）
  - GL    (Eq.39)  梯度损失
  - RDOA  (Eq.40)  重叠区综合差异 = (ADM + ADSD + CD) / 3
  - Ave   (Eq.41)  综合指标 = (ADM + ADSD + CD + GL) / 4

所有指标均为数值越小越好。
"""

import numpy as np
from scipy.ndimage import sobel
from typing import List, Optional


# ---------------------------------------------------------------------------
# 辅助：提取重叠区像素（排除 nodata 和非有限值）
# ---------------------------------------------------------------------------

def _extract_overlap_pixels(
    array_i: np.ndarray,
    array_j: np.ndarray,
    window_i,
    window_j,
    nodata_i: Optional[float],
    nodata_j: Optional[float],
    band: int,
):
    """提取一对重叠影像在指定波段上的重叠区有效像素。"""
    r1s, r1e, c1s, c1e = window_i
    r2s, r2e, c2s, c2e = window_j
    pi = array_i[band, r1s:r1e, c1s:c1e].ravel()
    pj = array_j[band, r2s:r2e, c2s:c2e].ravel()

    mi = np.isfinite(pi)
    mj = np.isfinite(pj)
    if nodata_i is not None:
        mi &= (pi != nodata_i)
    if nodata_j is not None:
        mj &= (pj != nodata_j)

    valid = mi & mj
    return pi[valid], pj[valid]


# ---------------------------------------------------------------------------
# 辅助：检查某波段是否存在有效重叠像素
# ---------------------------------------------------------------------------

def _has_valid_overlap_pixels(
    arrays: List[np.ndarray],
    nodata_values: List[Optional[float]],
    overlaps: List[dict],
    band: int,
) -> bool:
    """检查给定波段在任意重叠对中是否有有效像素。"""
    for ov in overlaps:
        pi, _ = _extract_overlap_pixels(
            arrays[ov["idx_i"]], arrays[ov["idx_j"]],
            ov["window_i"], ov["window_j"],
            nodata_values[ov["idx_i"]], nodata_values[ov["idx_j"]],
            band,
        )
        if len(pi) > 0:
            return True
    return False


# ---------------------------------------------------------------------------
# 辅助：从两个像素数组计算 CD（统一的概率直方图公式）
# ---------------------------------------------------------------------------

def _compute_cd_from_pixels(pi: np.ndarray, pj: np.ndarray, n_bins: int = 256) -> float:
    """
    使用总变差距离 (Total Variation Distance) 计算两组像素的 CD。

    CD = 0.5 * sum(|p_i - p_j|)

    其中 p_i, p_j 为归一化概率直方图（sum = 1）。
    """
    if len(pi) < 2 or len(pj) < 2:
        return 0.0
    vmin = min(pi.min(), pj.min())
    vmax = max(pi.max(), pj.max())
    if vmax - vmin < 1e-12:
        return 0.0
    counts_i, _ = np.histogram(pi, bins=n_bins, range=(vmin, vmax))
    counts_j, _ = np.histogram(pj, bins=n_bins, range=(vmin, vmax))
    total_i = counts_i.sum()
    total_j = counts_j.sum()
    if total_i == 0 or total_j == 0:
        return 0.0
    p_i = counts_i.astype(np.float64) / total_i
    p_j = counts_j.astype(np.float64) / total_j
    return float(0.5 * np.sum(np.abs(p_i - p_j)))


# ---------------------------------------------------------------------------
# ADM — 重叠区均值绝对差 (Eq.36)
# ---------------------------------------------------------------------------

def compute_adm(
    arrays: List[np.ndarray],
    nodata_values: List[Optional[float]],
    overlaps: List[dict],
    bands: Optional[List[int]] = None,
) -> float:
    """
    计算 ADM (Absolute Difference of Mean)。
    按实际有效重叠对计数，跳过无有效像素的对。
    """
    if bands is None:
        bands = list(range(arrays[0].shape[0]))
    if len(overlaps) == 0:
        return 0.0

    adm_sum = 0.0
    for band in bands:
        total_diff = 0.0
        valid_pair_count = 0
        for ov in overlaps:
            pi, pj = _extract_overlap_pixels(
                arrays[ov["idx_i"]], arrays[ov["idx_j"]],
                ov["window_i"], ov["window_j"],
                nodata_values[ov["idx_i"]], nodata_values[ov["idx_j"]],
                band,
            )
            if len(pi) == 0 or len(pj) == 0:
                continue
            total_diff += abs(pi.mean() - pj.mean())
            valid_pair_count += 1
        if valid_pair_count > 0:
            adm_sum += total_diff / valid_pair_count
    return float(adm_sum / len(bands))


# ---------------------------------------------------------------------------
# ADSD — 重叠区标准差绝对差 (Eq.37)
# ---------------------------------------------------------------------------

def compute_adsd(
    arrays: List[np.ndarray],
    nodata_values: List[Optional[float]],
    overlaps: List[dict],
    bands: Optional[List[int]] = None,
) -> float:
    """
    计算 ADSD (Absolute Difference of Standard Deviation)。
    按实际有效重叠对计数。
    """
    if bands is None:
        bands = list(range(arrays[0].shape[0]))
    if len(overlaps) == 0:
        return 0.0

    adsd_sum = 0.0
    for band in bands:
        total_diff = 0.0
        valid_pair_count = 0
        for ov in overlaps:
            pi, pj = _extract_overlap_pixels(
                arrays[ov["idx_i"]], arrays[ov["idx_j"]],
                ov["window_i"], ov["window_j"],
                nodata_values[ov["idx_i"]], nodata_values[ov["idx_j"]],
                band,
            )
            if len(pi) == 0 or len(pj) == 0:
                continue
            total_diff += abs(pi.std() - pj.std())
            valid_pair_count += 1
        if valid_pair_count > 0:
            adsd_sum += total_diff / valid_pair_count
    return float(adsd_sum / len(bands))


# ---------------------------------------------------------------------------
# CD — 颜色距离 / 直方图距离 (Eq.38)
# ---------------------------------------------------------------------------

def compute_cd(
    arrays: List[np.ndarray],
    nodata_values: List[Optional[float]],
    overlaps: List[dict],
    bands: Optional[List[int]] = None,
    n_bins: int = 256,
) -> float:
    """
    计算 CD (Color Distance)。

    使用统一的概率直方图总变差距离公式：
    CD_band = sum_k( w_k * TV(h_i^k, h_j^k) )
    CD = mean over bands

    权重 w_k 基于各重叠对的有效像元数（非窗口理论面积）。
    """
    if bands is None:
        bands = list(range(arrays[0].shape[0]))
    if len(overlaps) == 0:
        return 0.0

    ref_band = bands[0]
    valid_counts = []
    for ov in overlaps:
        pi, _ = _extract_overlap_pixels(
            arrays[ov["idx_i"]], arrays[ov["idx_j"]],
            ov["window_i"], ov["window_j"],
            nodata_values[ov["idx_i"]], nodata_values[ov["idx_j"]],
            ref_band,
        )
        valid_counts.append(len(pi))
    valid_counts = np.array(valid_counts, dtype=np.float64)
    total_valid = valid_counts.sum()
    if total_valid == 0:
        return 0.0
    weights = valid_counts / total_valid

    cd_per_band = []
    for band in bands:
        cd_weighted = 0.0
        has_valid = False
        for k, ov in enumerate(overlaps):
            pi, pj = _extract_overlap_pixels(
                arrays[ov["idx_i"]], arrays[ov["idx_j"]],
                ov["window_i"], ov["window_j"],
                nodata_values[ov["idx_i"]], nodata_values[ov["idx_j"]],
                band,
            )
            if len(pi) < 2:
                continue
            cd_val = _compute_cd_from_pixels(pi, pj, n_bins)
            cd_weighted += weights[k] * cd_val
            has_valid = True
        if has_valid:
            cd_per_band.append(cd_weighted)

    if not cd_per_band:
        return 0.0
    return float(np.mean(cd_per_band))


# ---------------------------------------------------------------------------
# GL — 梯度损失 (Eq.39)
# ---------------------------------------------------------------------------

def _gradient_orientation_map(band_data: np.ndarray, valid_mask: np.ndarray = None) -> np.ndarray:
    """
    计算单波段梯度方向图。

    在计算 Sobel 梯度前，将无效像元（NoData / NaN / Inf）
    替换为 0，避免梯度在边界处被污染。
    """
    arr = band_data.astype(np.float64)
    if valid_mask is not None:
        arr = np.where(valid_mask, arr, 0.0)
    gx = sobel(arr, axis=1)
    gy = sobel(arr, axis=0)
    return np.arctan2(np.abs(gy), np.abs(gx))


def compute_gl(
    arrays_before: List[np.ndarray],
    arrays_after: List[np.ndarray],
    nodata_values: List[Optional[float]],
    bands: Optional[List[int]] = None,
) -> float:
    """
    计算 GL (Gradient Loss)。

    对每幅影像分别计算梯度方向差异，然后取均值。
    无效像元在 Sobel 之前替换为 0，避免梯度污染。
    """
    if bands is None:
        bands = list(range(arrays_before[0].shape[0]))
    n_images = len(arrays_before)
    if n_images == 0:
        return 0.0

    gl_sum = 0.0
    for band in bands:
        total_gl = 0.0
        for idx in range(n_images):
            before = arrays_before[idx][band]
            after = arrays_after[idx][band]
            nd = nodata_values[idx]

            valid_mask = np.isfinite(before) & np.isfinite(after)
            if nd is not None:
                valid_mask &= (before != nd) & (after != nd)

            n_valid = valid_mask.sum()
            if n_valid == 0:
                continue

            orient_before = _gradient_orientation_map(before, valid_mask)
            orient_after = _gradient_orientation_map(after, valid_mask)

            diff = np.abs(orient_before - orient_after)
            delta_g = diff[valid_mask].sum()
            total_gl += delta_g / n_valid
        gl_sum += total_gl / n_images
    return float(gl_sum / len(bands))


# ---------------------------------------------------------------------------
# RDOA 与 Ave (Eq.40-41)
# ---------------------------------------------------------------------------

def compute_rdoa(adm: float, adsd: float, cd: float) -> float:
    """RDOA = (ADM + ADSD + CD) / 3"""
    return (adm + adsd + cd) / 3.0


def compute_ave(adm: float, adsd: float, cd: float, gl: float) -> float:
    """Ave = (ADM + ADSD + CD + GL) / 4"""
    return (adm + adsd + cd + gl) / 4.0


# ---------------------------------------------------------------------------
# 一站式计算
# ---------------------------------------------------------------------------

def compute_all(
    arrays_before: List[np.ndarray],
    arrays_after: List[np.ndarray],
    nodata_values: List[Optional[float]],
    overlaps: List[dict],
    bands: Optional[List[int]] = None,
    n_bins: int = 256,
) -> dict:
    """
    一次性计算全部六项指标。
    """
    if bands is None:
        bands = list(range(arrays_before[0].shape[0]))

    adm = compute_adm(arrays_after, nodata_values, overlaps, bands)
    adsd = compute_adsd(arrays_after, nodata_values, overlaps, bands)
    cd = compute_cd(arrays_after, nodata_values, overlaps, bands, n_bins)
    gl = compute_gl(arrays_before, arrays_after, nodata_values, bands)
    rdoa = compute_rdoa(adm, adsd, cd)
    ave = compute_ave(adm, adsd, cd, gl)

    return {
        "adm": float(round(adm, 10)),
        "adsd": float(round(adsd, 10)),
        "cd": float(round(cd, 10)),
        "gl": float(round(gl, 10)),
        "rdoa": float(round(rdoa, 10)),
        "ave": float(round(ave, 10)),
    }


# ---------------------------------------------------------------------------
# 逐对计算
# ---------------------------------------------------------------------------

def compute_per_pair(
    arrays_after: List[np.ndarray],
    nodata_values: List[Optional[float]],
    overlaps: List[dict],
    bands: Optional[List[int]] = None,
    n_bins: int = 256,
    arrays_before: Optional[List[np.ndarray]] = None,
) -> List[dict]:
    """
    逐影像对计算 ADM / ADSD / CD / GL / RDOA / Ave + 逐波段分解。

    所有指标均调用核心函数 compute_adm / compute_adsd / compute_cd / compute_gl。
    per_band 使用实际波段号作为键（非循环索引）。

    当 arrays_before 未提供时，gl=None 且 ave=None。

    返回列表，每个元素：
    {'idx_i', 'idx_j', 'pixel_count', 'adm', 'adsd', 'cd', 'gl', 'rdoa', 'ave',
     'per_band': {band_num: {'adm', 'adsd', 'cd', 'gl', 'rdoa', 'ave', 'n_valid'}}}
    """
    if bands is None:
        bands = list(range(arrays_after[0].shape[0]))

    results = []
    for ov in overlaps:
        i, j = ov['idx_i'], ov['idx_j']
        single_ov = [ov]

        per_band = {}
        adm_vals, adsd_vals, cd_vals = [], [], []

        for band in bands:
            if not _has_valid_overlap_pixels(arrays_after, nodata_values, single_ov, band):
                per_band[band] = {
                    'adm': None, 'adsd': None, 'cd': None,
                    'gl': None, 'rdoa': None, 'ave': None,
                    'n_valid': 0,
                    'skipped_reason': 'no_valid_overlap_pixels',
                }
                continue

            pi, pj = _extract_overlap_pixels(
                arrays_after[i], arrays_after[j],
                ov['window_i'], ov['window_j'],
                nodata_values[i], nodata_values[j], band)

            adm = compute_adm(arrays_after, nodata_values, single_ov, [band])
            adsd = compute_adsd(arrays_after, nodata_values, single_ov, [band])
            cd = compute_cd(arrays_after, nodata_values, single_ov, [band], n_bins)
            rdoa = (adm + adsd + cd) / 3.0

            if arrays_before is not None:
                gl = compute_gl(
                    [arrays_before[i], arrays_before[j]],
                    [arrays_after[i], arrays_after[j]],
                    [nodata_values[i], nodata_values[j]],
                    [band])
                ave = compute_ave(adm, adsd, cd, gl)
            else:
                gl = None
                ave = None

            per_band[band] = {
                'adm': float(round(adm, 10)),
                'adsd': float(round(adsd, 10)),
                'cd': float(round(cd, 10)),
                'gl': float(round(gl, 10)) if gl is not None else None,
                'rdoa': float(round(rdoa, 10)),
                'ave': float(round(ave, 10)) if ave is not None else None,
                'n_valid': len(pi),
            }
            adm_vals.append(adm)
            adsd_vals.append(adsd)
            cd_vals.append(cd)

        if not adm_vals:
            continue

        agg_adm = float(np.mean(adm_vals))
        agg_adsd = float(np.mean(adsd_vals))
        agg_cd = float(np.mean(cd_vals))
        agg_rdoa = (agg_adm + agg_adsd + agg_cd) / 3.0

        if arrays_before is not None:
            gl_vals = []
            for idx in [i, j]:
                gl_val = compute_gl(
                    [arrays_before[idx]], [arrays_after[idx]],
                    [nodata_values[idx]], bands)
                gl_vals.append(gl_val)
            agg_gl = float(np.mean(gl_vals))
            agg_ave = compute_ave(agg_adm, agg_adsd, agg_cd, agg_gl)
        else:
            agg_gl = None
            agg_ave = None

        pixel_count = ov.get('pixel_count', sum(
            per_band[b]['n_valid'] for b in per_band
            if per_band[b].get('skipped_reason') is None))

        results.append({
            'idx_i': i,
            'idx_j': j,
            'pixel_count': pixel_count,
            'adm': float(round(agg_adm, 10)),
            'adsd': float(round(agg_adsd, 10)),
            'cd': float(round(agg_cd, 10)),
            'gl': float(round(agg_gl, 10)) if agg_gl is not None else None,
            'rdoa': float(round(agg_rdoa, 10)),
            'ave': float(round(agg_ave, 10)) if agg_ave is not None else None,
            'per_band': per_band,
        })

    return results


# ---------------------------------------------------------------------------
# 逐波段计算
# ---------------------------------------------------------------------------

def compute_per_band(
    arrays_after: List[np.ndarray],
    nodata_values: List[Optional[float]],
    overlaps: List[dict],
    bands: Optional[List[int]] = None,
    n_bins: int = 256,
    arrays_before: Optional[List[np.ndarray]] = None,
) -> List[dict]:
    """
    逐波段计算全部指标。

    调用核心函数 compute_adm / compute_adsd / compute_cd / compute_gl。
    当 bands 中某波段无有效重叠像素时，记录 skipped_reason，
    不将其纳入平均值计算。

    当 arrays_before 未提供时，gl=None 且 ave=None。

    返回列表，每个元素：
    {'band', 'adm', 'adsd', 'cd', 'gl', 'rdoa', 'ave'} 或
    {'band', ..., 'skipped_reason'} 当该波段被跳过时。
    """
    if bands is None:
        bands = list(range(arrays_after[0].shape[0]))

    results = []
    for band in bands:
        if not _has_valid_overlap_pixels(arrays_after, nodata_values, overlaps, band):
            results.append({
                'band': int(band),
                'adm': None, 'adsd': None, 'cd': None,
                'gl': None, 'rdoa': None, 'ave': None,
                'skipped_reason': 'no_valid_overlap_pixels',
            })
            continue

        adm = compute_adm(arrays_after, nodata_values, overlaps, [band])
        adsd = compute_adsd(arrays_after, nodata_values, overlaps, [band])
        cd = compute_cd(arrays_after, nodata_values, overlaps, [band], n_bins)
        rdoa = (adm + adsd + cd) / 3.0

        if arrays_before is not None:
            gl = compute_gl(arrays_before, arrays_after, nodata_values, [band])
            ave = compute_ave(adm, adsd, cd, gl)
        else:
            gl = None
            ave = None

        results.append({
            'band': int(band),
            'adm': float(round(adm, 10)),
            'adsd': float(round(adsd, 10)),
            'cd': float(round(cd, 10)),
            'gl': float(round(gl, 10)) if gl is not None else None,
            'rdoa': float(round(rdoa, 10)),
            'ave': float(round(ave, 10)) if ave is not None else None,
        })
    return results


# ---------------------------------------------------------------------------
# CSV 输出
# ---------------------------------------------------------------------------

def _csv_val(v):
    """将 None 转换为空字符串用于 CSV 输出。"""
    return '' if v is None else v


def save_metrics_csv(metrics: dict, output_dir: str,
                     bands: Optional[List[int]] = None,
                     overlaps: Optional[List[dict]] = None,
                     arrays_before: Optional[List[np.ndarray]] = None,
                     arrays_after: Optional[List[np.ndarray]] = None,
                     nodata_values: Optional[List[Optional[float]]] = None):
    """
    将指标保存为 CSV 文件。
    输出：
      - metrics_summary.csv（聚合）
      - metrics_per_band.csv（逐波段）
      - metrics_per_pair.csv（逐对）
      - metrics_per_pair_band.csv（逐对逐波段）

    CSV 中波段列使用实际波段号（非循环索引）。
    """
    import os
    import csv

    os.makedirs(output_dir, exist_ok=True)

    # summary
    summary_path = os.path.join(output_dir, 'metrics_summary.csv')
    with open(summary_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['metric', 'value', 'direction'])
        for key in ['adm', 'adsd', 'cd', 'gl', 'rdoa', 'ave']:
            val = metrics.get(key, float('nan'))
            w.writerow([key, val, 'lower_is_better'])

    # per_band
    if bands is not None and arrays_after is not None:
        per_band = compute_per_band(
            arrays_after, nodata_values, overlaps, bands,
            arrays_before=arrays_before)
        per_band_path = os.path.join(output_dir, 'metrics_per_band.csv')
        with open(per_band_path, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['band', 'adm', 'adsd', 'cd', 'gl', 'rdoa', 'ave',
                        'skipped_reason',
                        'adm_dir', 'adsd_dir', 'cd_dir', 'gl_dir', 'rdoa_dir', 'ave_dir'])
            for row in per_band:
                w.writerow([
                    row['band'],
                    _csv_val(row.get('adm')),
                    _csv_val(row.get('adsd')),
                    _csv_val(row.get('cd')),
                    _csv_val(row.get('gl')),
                    _csv_val(row.get('rdoa')),
                    _csv_val(row.get('ave')),
                    row.get('skipped_reason', ''),
                    'lower', 'lower', 'lower', 'lower', 'lower', 'lower'])

    # per_pair
    if overlaps is not None and arrays_after is not None:
        per_pair = compute_per_pair(
            arrays_after, nodata_values, overlaps, bands,
            arrays_before=arrays_before)
        per_pair_path = os.path.join(output_dir, 'metrics_per_pair.csv')
        with open(per_pair_path, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['idx_i', 'idx_j', 'pixel_count', 'adm', 'adsd', 'cd', 'rdoa', 'ave',
                        'adm_dir', 'adsd_dir', 'cd_dir', 'rdoa_dir', 'ave_dir'])
            for row in per_pair:
                w.writerow([row['idx_i'], row['idx_j'], row['pixel_count'],
                           row['adm'], row['adsd'], row['cd'], row['rdoa'],
                           _csv_val(row.get('ave')),
                           'lower', 'lower', 'lower', 'lower', 'lower'])

        # per_pair_band（波段列使用实际波段号）
        ppb_path = os.path.join(output_dir, 'metrics_per_pair_band.csv')
        with open(ppb_path, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['idx_i', 'idx_j', 'band', 'adm', 'adsd', 'cd', 'rdoa', 'ave',
                        'n_valid', 'skipped_reason'])
            for row in per_pair:
                for band_num, band_data in row.get('per_band', {}).items():
                    w.writerow([
                        row['idx_i'], row['idx_j'], band_num,
                        _csv_val(band_data.get('adm')),
                        _csv_val(band_data.get('adsd')),
                        _csv_val(band_data.get('cd')),
                        _csv_val(band_data.get('rdoa')),
                        _csv_val(band_data.get('ave')),
                        band_data.get('n_valid', 0),
                        band_data.get('skipped_reason', '')])
