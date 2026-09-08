"""Subpixel image co-registration using phase correlation (AROSICS core algorithm).

Implements frequency-domain phase correlation for subpixel shift estimation,
similar to the approach used in AROSICS (Scheffler et al. 2017).

Reference: Scheffler et al. "AROSICS: An Automated and Robust Open-Source
Image Co-Registration Software for Multi-Sensor Satellite Data"
Remote Sensing. 2017; 9(7):676.
"""
import logging
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
from scipy.ndimage import fourier_shift, shift as ndimage_shift
from scipy.signal import fftconvolve
import rasterio
from rasterio.transform import Affine
import os, glob, json

logger = logging.getLogger(__name__)


def structural_image(img, valid):
    """生成梯度结构影像，用于对辐射差异鲁棒的配准。

    对梯度幅值做归一化，突出山脊、河岸、道路等边缘特征。
    """
    from scipy.ndimage import sobel, gaussian_filter

    x = img.astype(np.float64)
    vals = x[valid]
    if len(vals) < 100:
        return x
    p2, p98 = np.percentile(vals, [2, 98])
    x = np.clip(x, p2, p98)
    x = (x - p2) / max(p98 - p2, 1e-8)
    x[~valid] = 0.0
    x = gaussian_filter(x, sigma=1.0)
    gx = sobel(x, axis=1)
    gy = sobel(x, axis=0)
    gradient = np.hypot(gx, gy)
    gradient[~valid] = 0.0
    return gradient


def phase_correlation(img_ref, img_target, valid_ref=None, valid_tgt=None):
    """Pure same-grid phase correlation.

    Computes subpixel shift between two 2D arrays on the same grid.
    No transforms, CRS, or reprojection - purely array-based.

    Parameters
    ----------
    img_ref, img_target : 2D ndarray
        Reference and target images on the same grid.
    valid_ref, valid_tgt : 2D bool ndarray or None
        Optional validity masks.

    Returns
    -------
    shift_y, shift_x : float
        Pixel shift to apply to target to align with ref.
    confidence : float
        NCC confidence after alignment.
    """
    from skimage.registration import phase_cross_correlation
    from scipy.ndimage import shift as ndimage_shift

    # Require 2D arrays
    if img_ref.ndim != 2 or img_target.ndim != 2:
        raise ValueError("phase_correlation requires 2D arrays")

    # Defensive crop to common minimum shape (one-pixel rounding tolerance)
    h = min(img_ref.shape[0], img_target.shape[0])
    w = min(img_ref.shape[1], img_target.shape[1])
    ref = img_ref[:h, :w].astype(np.float64)
    tgt = img_target[:h, :w].astype(np.float64)

    # Build finite + optional masks
    finite_ref = np.isfinite(ref)
    finite_tgt = np.isfinite(tgt)
    
    if valid_ref is not None:
        mask_ref = finite_ref & valid_ref[:h, :w]
    else:
        mask_ref = finite_ref
    
    if valid_tgt is not None:
        mask_tgt = finite_tgt & valid_tgt[:h, :w]
    else:
        mask_tgt = finite_tgt

    # Joint valid pixels
    joint = mask_ref & mask_tgt
    
    if joint.sum() < 100:
        return 0.0, 0.0, 0.0

    # Zero out invalid pixels for phase correlation
    ref_clean = np.where(joint, ref, 0.0)
    tgt_clean = np.where(joint, tgt, 0.0)

    # Phase correlation
    shift, error, diffphase = phase_cross_correlation(
        ref_clean, tgt_clean,
        upsample_factor=50,
        normalization=None,
        disambiguate=True,
    )

    shift_y, shift_x = float(shift[0]), float(shift[1])

    # Compute NCC confidence on post-shift joint valid pixels
    aligned = ndimage_shift(tgt_clean, [shift_y, shift_x], order=1,
                            mode='constant', cval=0, prefilter=False)

    # Shift the target validity mask
    aligned_mask = ndimage_shift(
        mask_tgt.astype(np.uint8),
        [shift_y, shift_x], order=0, mode='constant', cval=0, prefilter=False
    ).astype(bool)

    # Final valid region
    valid_after = mask_ref & aligned_mask & np.isfinite(aligned)

    if valid_after.sum() > 100:
        r = ref_clean[valid_after].ravel().astype(np.float64)
        a = aligned[valid_after].ravel().astype(np.float64)
        r = r - r.mean()
        a = a - a.mean()
        denom = np.linalg.norm(r) * np.linalg.norm(a) + 1e-12
        confidence = float(abs(np.dot(r, a) / denom))
    else:
        confidence = 0.0

    return shift_y, shift_x, confidence




def phase_correlation_from_overlap(
    arr_ref,
    tr_ref,
    arr_tgt,
    tr_tgt,
    nodata_ref=None,
    nodata_tgt=None,
):
    """Phase correlation using geographic overlap only.

    Extracts the geographic overlap region from both images,
    then calls phase_correlation on the overlap patches.

    Parameters
    ----------
    arr_ref, arr_tgt : 2D ndarray
        Reference and target images.
    tr_ref, tr_tgt : rasterio.Affine
        Affine transforms for both images.
    nodata_ref, nodata_tgt : float or None
        NoData values.

    Returns
    -------
    shift_y, shift_x, confidence : float
    """
    from rasterio.transform import array_bounds
    from src.overlap import get_overlap_window

    # Get bounds
    bounds_ref = array_bounds(arr_ref.shape[0], arr_ref.shape[1], tr_ref)
    bounds_tgt = array_bounds(arr_tgt.shape[0], arr_tgt.shape[1], tr_tgt)

    # Get overlap window
    overlap = get_overlap_window(bounds_ref, tr_ref, bounds_tgt, tr_tgt)
    if overlap is None:
        return 0.0, 0.0, 0.0

    (row_start_ref, row_end_ref, col_start_ref, col_end_ref),     (row_start_tgt, row_end_tgt, col_start_tgt, col_end_tgt) = overlap

    # Extract patches
    patch_ref = arr_ref[row_start_ref:row_end_ref, col_start_ref:col_end_ref]
    patch_tgt = arr_tgt[row_start_tgt:row_end_tgt, col_start_tgt:col_end_tgt]

    # Crop to same size (defensive for rounding)
    h = min(patch_ref.shape[0], patch_tgt.shape[0])
    w = min(patch_ref.shape[1], patch_tgt.shape[1])
    patch_ref = patch_ref[:h, :w]
    patch_tgt = patch_tgt[:h, :w]

    # Build masks
    valid_ref = np.isfinite(patch_ref)
    valid_tgt = np.isfinite(patch_tgt)
    if nodata_ref is not None:
        valid_ref &= (patch_ref != nodata_ref)
    if nodata_tgt is not None:
        valid_tgt &= (patch_tgt != nodata_tgt)

    # Call phase_correlation
    return phase_correlation(patch_ref, patch_tgt, valid_ref, valid_tgt)


def compute_shifts_from_overlap(arr_ref, tr_ref, arr_tgt, tr_tgt,
                                 nodata_ref=0, nodata_tgt=0,
                                 max_global_shift=40):
    """Compute subpixel shift from overlap area between two images.

    Uses block-based phase correlation on structural (gradient) images.

    Parameters
    ----------
    arr_ref, arr_tgt : 2D ndarray
    tr_ref, tr_tgt : Affine transform
    nodata_ref, nodata_tgt : float, separate NoData for each image

    Returns
    -------
    shift_y, shift_x : float, 像素位移（直接施加到 target 即可对齐到 ref）
    confidence : float, 配准后 NCC
    block_stats : dict
    """
    h_ref, w_ref = arr_ref.shape
    h_tgt, w_tgt = arr_tgt.shape

    def pix2geo(tr, r, c):
        return tr.c + c * tr.a, tr.f + r * tr.e

    def geo2pix(tr, x, y):
        r = round((y - tr.f) / tr.e)
        c = round((x - tr.c) / tr.a)
        return int(r), int(c)

    ref_left, ref_top = pix2geo(tr_ref, 0, 0)
    ref_right, ref_bottom = pix2geo(tr_ref, h_ref, w_ref)
    tgt_left, tgt_top = pix2geo(tr_tgt, 0, 0)
    tgt_right, tgt_bottom = pix2geo(tr_tgt, h_tgt, w_tgt)

    ov_left = max(ref_left, tgt_left)
    ov_right = min(ref_right, tgt_right)
    ov_top = min(ref_top, tgt_top)
    ov_bottom = max(ref_bottom, tgt_bottom)

    if ov_left >= ov_right or ov_top <= ov_bottom:
        return 0.0, 0.0, 0.0, {}

    r1_start, c1_start = geo2pix(tr_ref, ov_left, ov_top)
    r1_end, c1_end = geo2pix(tr_ref, ov_right, ov_bottom)
    r2_start, c2_start = geo2pix(tr_tgt, ov_left, ov_top)
    r2_end, c2_end = geo2pix(tr_tgt, ov_right, ov_bottom)

    r1_start = max(0, r1_start); r1_end = min(h_ref, r1_end)
    c1_start = max(0, c1_start); c1_end = min(w_ref, c1_end)
    r2_start = max(0, r2_start); r2_end = min(h_tgt, r2_end)
    c2_start = max(0, c2_start); c2_end = min(w_tgt, c2_end)

    patch_ref = arr_ref[r1_start:r1_end, c1_start:c1_end]
    patch_tgt = arr_tgt[r2_start:r2_end, c2_start:c2_end]

    if patch_ref.size < 100 or patch_tgt.size < 100:
        return 0.0, 0.0, 0.0, {}

    # 生成梯度结构影像（对辐射差异鲁棒）
    valid_ref_patch = np.isfinite(patch_ref) & (patch_ref != nodata_ref)
    valid_tgt_patch = np.isfinite(patch_tgt) & (patch_tgt != nodata_tgt)
    struct_ref = structural_image(patch_ref, valid_ref_patch)
    struct_tgt = structural_image(patch_tgt, valid_tgt_patch)

    # 自适应纹理阈值：全局标准差的 10%
    global_texture_std = float(np.std(struct_ref[valid_ref_patch])) if valid_ref_patch.sum() > 0 else 1.0
    texture_threshold = max(global_texture_std * 0.10, 1e-4)

    # Block-based estimation
    block_size = 512
    ph, pw = patch_ref.shape
    shifts_y = []
    shifts_x = []
    confs = []
    block_centers = []

    # 块筛选计数器
    n_total = 0
    n_low_valid = 0
    n_low_texture = 0
    n_low_conf = 0
    n_large_shift = 0
    n_accepted = 0

    for br in range(0, ph - block_size + 1, block_size // 2):
        for bc in range(0, pw - block_size + 1, block_size // 2):
            br2 = min(br + block_size, ph)
            bc2 = min(bc + block_size, pw)
            blk_ref = struct_ref[br:br2, bc:bc2]
            blk_tgt = struct_tgt[br:br2, bc:bc2]

            n_total += 1

            # 联合有效掩膜
            v_ref = valid_ref_patch[br:br2, bc:bc2]
            v_tgt = valid_tgt_patch[br:br2, bc:bc2]
            joint = v_ref & v_tgt

            if joint.sum() < block_size * block_size * 0.3:
                n_low_valid += 1
                continue

            # 低纹理块剔除（自适应阈值）
            texture_std = float(np.std(blk_ref[joint]))
            if texture_std < texture_threshold:
                n_low_texture += 1
                continue

            sy, sx, conf = phase_correlation(
                blk_ref, blk_tgt,
                valid_ref=v_ref, valid_tgt=v_tgt)

            if conf <= 0.5:
                n_low_conf += 1
                continue

            if abs(sy) >= max_global_shift or abs(sx) >= max_global_shift:
                n_large_shift += 1
                continue

            n_accepted += 1
            shifts_y.append(sy)
            shifts_x.append(sx)
            confs.append(conf)
            block_centers.append((r1_start + br + block_size // 2,
                                  c1_start + bc + block_size // 2))

    # 打印块筛选日志
    print(f"  Block screening: total={n_total}, low_valid={n_low_valid}, "
          f"low_texture={n_low_texture}, low_conf={n_low_conf}, "
          f"large_shift={n_large_shift}, accepted={n_accepted}")

    if len(shifts_y) < 3:
        # 回退到整个重叠区匹配
        sy, sx, conf = phase_correlation(
            struct_ref, struct_tgt,
            valid_ref=valid_ref_patch, valid_tgt=valid_tgt_patch)
        scr = {'total': n_total, 'low_valid': n_low_valid,
               'low_texture': n_low_texture, 'low_conf': n_low_conf,
               'large_shift': n_large_shift, 'accepted': n_accepted}
        if conf < 0.3:
            return 0.0, 0.0, 0.0, {
                'n_blocks_total': 0, 'n_blocks_inlier': 0,
                'available': False, 'failure_reason': f'insufficient blocks and low conf={conf:.3f}',
                'screening': scr}
        return sy, sx, conf, {
            'n_blocks_total': len(shifts_y), 'n_blocks_inlier': len(shifts_y),
            'available': True, 'fallback': 'whole_overlap', 'screening': scr}

    shifts_y = np.array(shifts_y)
    shifts_x = np.array(shifts_x)
    confs = np.array(confs)
    block_centers = np.array(block_centers)

    # 联合内点筛选（一次性对 y 和 x 同时筛选）
    med_y = np.median(shifts_y)
    med_x = np.median(shifts_x)
    mad_y = np.median(np.abs(shifts_y - med_y))
    mad_x = np.median(np.abs(shifts_x - med_x))

    inlier = ((np.abs(shifts_y - med_y) < max(3 * mad_y, 0.3)) &
              (np.abs(shifts_x - med_x) < max(3 * mad_x, 0.3)))

    if inlier.sum() > 0:
        w = confs[inlier]
        shift_y = float(np.average(shifts_y[inlier], weights=w))
        shift_x = float(np.average(shifts_x[inlier], weights=w))
        confidence = float(confs[inlier].mean())
        n_inlier = int(inlier.sum())
    else:
        shift_y, shift_x = float(med_y), float(med_x)
        confidence = 0.0  # Zero inliers = no reliable estimate
        n_inlier = 0

    # 块级残差统计（相对全局模型的残差，不是绝对位移量）
    residual_y = shifts_y[inlier] - shift_y if inlier.sum() > 0 else shifts_y - shift_y
    residual_x = shifts_x[inlier] - shift_x if inlier.sum() > 0 else shifts_x - shift_x
    residual_dist = np.hypot(residual_y, residual_x)

    block_stats = {
        'n_blocks_total': int(len(shifts_y)),
        'n_blocks_inlier': n_inlier,
        'inlier_ratio': float(n_inlier / len(shifts_y)) if len(shifts_y) > 0 else 0.0,
        'residual_median': float(np.median(residual_dist)) if len(residual_dist) > 0 else 0.0,
        'residual_rmse': float(np.sqrt(np.mean(residual_dist**2))) if len(residual_dist) > 0 else 0.0,
        'residual_p90': float(np.percentile(residual_dist, 90)) if len(residual_dist) > 0 else 0.0,
        'residual_p95': float(np.percentile(residual_dist, 95)) if len(residual_dist) > 0 else 0.0,
        'residual_max': float(np.max(residual_dist)) if len(residual_dist) > 0 else 0.0,
    }

    return shift_y, shift_x, confidence, block_stats


def collect_block_matches(arr_ref, tr_ref, arr_tgt, tr_tgt,
                          nodata_ref=0, nodata_tgt=0, block_size=512,
                          max_global_shift=40, confidence_threshold=0.5):
    """收集所有合格匹配块的控制点，用于后续仿射拟合。

    Parameters
    ----------
    arr_ref, arr_tgt : 2D ndarray
    tr_ref, tr_tgt : Affine transform
    nodata_ref, nodata_tgt : float

    Returns
    -------
    matches : list of dict
        每个元素包含 ref_x, ref_y, tgt_x, tgt_y, shift_dx, shift_dy, confidence
        ref = 参考影像块中心（像素坐标），tgt = 目标影像块中心（像素坐标）
    screening : dict
        块筛选统计
    """
    from scipy.ndimage import sobel, gaussian_filter

    h_ref, w_ref = arr_ref.shape
    h_tgt, w_tgt = arr_tgt.shape

    def pix2geo(tr, r, c):
        return tr.c + c * tr.a, tr.f + r * tr.e

    def geo2pix(tr, x, y):
        r = round((y - tr.f) / tr.e)
        c = round((x - tr.c) / tr.a)
        return int(r), int(c)

    ref_left, ref_top = pix2geo(tr_ref, 0, 0)
    ref_right, ref_bottom = pix2geo(tr_ref, h_ref, w_ref)
    tgt_left, tgt_top = pix2geo(tr_tgt, 0, 0)
    tgt_right, tgt_bottom = pix2geo(tr_tgt, h_tgt, w_tgt)

    ov_left = max(ref_left, tgt_left)
    ov_right = min(ref_right, tgt_right)
    ov_top = min(ref_top, tgt_top)
    ov_bottom = max(ref_bottom, tgt_bottom)

    if ov_left >= ov_right or ov_top <= ov_bottom:
        return [], {'total': 0, 'low_valid': 0, 'low_texture': 0,
                    'low_conf': 0, 'large_shift': 0, 'accepted': 0}

    r1_start, c1_start = geo2pix(tr_ref, ov_left, ov_top)
    r1_end, c1_end = geo2pix(tr_ref, ov_right, ov_bottom)
    r2_start, c2_start = geo2pix(tr_tgt, ov_left, ov_top)
    r2_end, c2_end = geo2pix(tr_tgt, ov_right, ov_bottom)

    r1_start = max(0, r1_start); r1_end = min(h_ref, r1_end)
    c1_start = max(0, c1_start); c1_end = min(w_ref, c1_end)
    r2_start = max(0, r2_start); r2_end = min(h_tgt, r2_end)
    c2_start = max(0, c2_start); c2_end = min(w_tgt, c2_end)

    patch_ref = arr_ref[r1_start:r1_end, c1_start:c1_end]
    patch_tgt = arr_tgt[r2_start:r2_end, c2_start:c2_end]

    if patch_ref.size < block_size * block_size:
        return [], {'total': 0, 'low_valid': 0, 'low_texture': 0,
                    'low_conf': 0, 'large_shift': 0, 'accepted': 0}

    valid_ref_patch = np.isfinite(patch_ref) & (patch_ref != nodata_ref)
    valid_tgt_patch = np.isfinite(patch_tgt) & (patch_tgt != nodata_tgt)
    struct_ref = structural_image(patch_ref, valid_ref_patch)
    struct_tgt = structural_image(patch_tgt, valid_tgt_patch)

    global_texture_std = float(np.std(struct_ref[valid_ref_patch])) if valid_ref_patch.sum() > 0 else 1.0
    texture_threshold = max(global_texture_std * 0.10, 1e-4)

    ph, pw = patch_ref.shape
    matches = []
    screening = {'total': 0, 'low_valid': 0, 'low_texture': 0,
                 'low_conf': 0, 'large_shift': 0, 'accepted': 0}

    for br in range(0, ph - block_size + 1, block_size // 2):
        for bc in range(0, pw - block_size + 1, block_size // 2):
            br2 = min(br + block_size, ph)
            bc2 = min(bc + block_size, pw)
            screening['total'] += 1

            v_ref = valid_ref_patch[br:br2, bc:bc2]
            v_tgt = valid_tgt_patch[br:br2, bc:bc2]
            joint = v_ref & v_tgt

            if joint.sum() < block_size * block_size * 0.3:
                screening['low_valid'] += 1
                continue

            blk_struct = struct_ref[br:br2, bc:bc2]
            if float(np.std(blk_struct[joint])) < texture_threshold:
                screening['low_texture'] += 1
                continue

            sy, sx, conf = phase_correlation(
                struct_ref[br:br2, bc:bc2], struct_tgt[br:br2, bc:bc2],
                valid_ref=v_ref, valid_tgt=v_tgt)

            if conf <= confidence_threshold:
                screening['low_conf'] += 1
                continue

            if abs(sy) >= max_global_shift or abs(sx) >= max_global_shift:
                screening['large_shift'] += 1
                continue

            screening['accepted'] += 1

            # 参考影像块中心（像素坐标）
            ref_x = c1_start + bc + block_size // 2
            ref_y = r1_start + br + block_size // 2

            # 目标影像块中心（像素坐标）
            tgt_x = c2_start + bc + block_size // 2
            tgt_y = r2_start + br + block_size // 2

            # shift 是需要施加到目标影像j的位移，使j对齐到i
            matches.append({
                'ref_x': float(ref_x),
                'ref_y': float(ref_y),
                'tgt_x': float(tgt_x),
                'tgt_y': float(tgt_y),
                'shift_dx': float(sx),
                'shift_dy': float(sy),
                'confidence': float(conf),
            })

    return matches, screening


def compute_model_residual_stats(matches, model_shift_y=0.0, model_shift_x=0.0,
                                  affine_model=None):
    """计算匹配点相对模型的残差统计。

    Parameters
    ----------
    matches : list of dict
    model_shift_y, model_shift_x : float, 平移模型参数
    affine_model : AffineTransform or None, 仿射模型

    Returns
    -------
    dict with median, rmse, p90, p95, max, n_points
    """
    if not matches:
        return {'median': 0.0, 'rmse': 0.0, 'p90': 0.0, 'p95': 0.0,
                'max': 0.0, 'n_points': 0}

    residuals = []
    for m in matches:
        src = np.array([[m['src_x'], m['src_y']]])
        if affine_model is not None:
            dst_pred = affine_model(src)[0]
            dst_actual = np.array([m['dst_x'], m['dst_y']])
        else:
            dst_pred = src[0] + np.array([model_shift_x, model_shift_y])
            dst_actual = np.array([m['dst_x'], m['dst_y']])
        residuals.append(np.linalg.norm(dst_actual - dst_pred))

    residuals = np.array(residuals)
    return {
        'median': float(np.median(residuals)),
        'rmse': float(np.sqrt(np.mean(residuals**2))),
        'p90': float(np.percentile(residuals, 90)),
        'p95': float(np.percentile(residuals, 95)),
        'max': float(np.max(residuals)),
        'n_points': len(residuals),
    }


def fit_affine_ransac(matches):
    """使用 RANSAC 拟合仿射变换模型。

    Parameters
    ----------
    matches : list of dict
        每个元素包含 src_x, src_y, dst_x, dst_y

    Returns
    -------
    model : AffineTransform or None
    inlier_mask : ndarray of bool
    stats : dict (scale, rotation, shear, translation, inlier_ratio, residual_stats)
    """
    from skimage.measure import ransac
    from skimage.transform import AffineTransform

    if len(matches) < 3:
        return None, np.zeros(len(matches), dtype=bool), {}

    src_points = np.array([[m['src_x'], m['src_y']] for m in matches])
    dst_points = np.array([[m['dst_x'], m['dst_y']] for m in matches])

    try:
        model, inlier_mask = ransac(
            (src_points, dst_points),
            AffineTransform,
            min_samples=3,
            residual_threshold=0.75,
            max_trials=2000,
        )
    except Exception:
        return None, np.zeros(len(matches), dtype=bool), {}

    n_inlier = int(inlier_mask.sum())
    inlier_ratio = n_inlier / len(matches) if len(matches) > 0 else 0.0

    # 解析仿射参数
    params = model.params  # 3x3 matrix
    scale_x = np.linalg.norm(params[0, :2])
    scale_y = np.linalg.norm(params[1, :2])
    rotation = np.degrees(np.arctan2(params[1, 0], params[0, 0]))
    shear = np.degrees(np.arctan2(params[0, 1], params[1, 1]))
    translation = (params[0, 2], params[1, 2])

    # 残差统计（仅内点）
    inlier_matches = [m for m, flag in zip(matches, inlier_mask) if flag]
    residual_stats = compute_model_residual_stats(inlier_matches, affine_model=model)

    stats = {
        'scale_x': float(scale_x),
        'scale_y': float(scale_y),
        'rotation_deg': float(rotation),
        'shear_deg': float(shear),
        'translation_x': float(translation[0]),
        'translation_y': float(translation[1]),
        'n_inlier': n_inlier,
        'inlier_ratio': float(inlier_ratio),
        'residual_stats': residual_stats,
    }

    return model, inlier_mask, stats


def validate_affine_model(stats, min_inlier_ratio=0.60, min_inlier_count=20):
    """判断仿射模型是否合理。

    Returns
    -------
    bool, str (valid, reason)
    """
    if stats.get('n_inlier', 0) < min_inlier_count:
        return False, f"内点数不足: {stats['n_inlier']} < {min_inlier_count}"

    if stats.get('inlier_ratio', 0) < min_inlier_ratio:
        return False, f"内点率不足: {stats['inlier_ratio']:.2%} < {min_inlier_ratio:.0%}"

    sx, sy = stats['scale_x'], stats['scale_y']
    if not (0.98 <= sx <= 1.02):
        return False, f"scale_x={sx:.4f} 超出 [0.98, 1.02]"
    if not (0.98 <= sy <= 1.02):
        return False, f"scale_y={sy:.4f} 超出 [0.98, 1.02]"

    rot = abs(stats['rotation_deg'])
    if rot > 1.0:
        return False, f"rotation={stats['rotation_deg']:.3f}度 超出 ±1度"

    sh = abs(stats['shear_deg'])
    if sh > 1.0:
        return False, f"shear={stats['shear_deg']:.3f}度 超出 ±1度"

    return True, "仿射模型参数合理"


def warp_affine_once(arr, model, tr_orig, crs, nodata, output_path):
    """对影像一次性施加仿射变换重采样。

    Parameters
    ----------
    arr : 2D or 3D ndarray (bands, rows, cols)
    model : AffineTransform, skimage 仿射模型
    tr_orig : Affine, 原始地理变换（保持不变）
    crs : str
    nodata : float
    output_path : str

    Returns
    -------
    warped : 3D ndarray (bands, rows, cols)
    """
    from skimage.transform import warp

    if arr.ndim == 2:
        arr = arr[np.newaxis, :, :]

    n_bands, h, w = arr.shape
    warped = np.full_like(arr, nodata, dtype=np.float64)

    for b in range(n_bands):
        band = arr[b].astype(np.float64)

        # 对每个波段施加仿射变换
        warped[b] = warp(
            band,
            model.inverse,
            order=1,
            mode='constant',
            cval=nodata,
            preserve_range=True,
        )

    # 有效掩膜单独用 order=0 重采样
    valid_mask = np.all(arr != nodata, axis=0).astype(np.uint8)
    warped_mask = warp(
        valid_mask,
        model.inverse,
        order=0,
        mode='constant',
        cval=0,
        preserve_range=True,
    ).astype(bool)

    # 恢复无效区 nodata
    warped[:, ~warped_mask] = nodata

    return warped


def build_local_residual_controls(matches, global_dx, global_dy,
                                   confidence_threshold=0.75, min_points=10):
    """从块匹配中构建局部残差控制点。

    Parameters
    ----------
    matches : list of dict
        每个元素包含 tgt_x, tgt_y, shift_dx, shift_dy, confidence
        （坐标在目标影像坐标系中）
    global_dx, global_dy : float, 全局平移模型参数
    confidence_threshold : float

    Returns
    -------
    dict with keys: points_xy, residual_dx, residual_dy, confidence, valid_mask, n_valid
    """
    if not matches:
        return {'points_xy': np.empty((0, 2)), 'residual_dx': np.array([]),
                'residual_dy': np.array([]), 'confidence': np.array([]),
                'valid_mask': np.array([], dtype=bool), 'n_valid': 0}

    src_x = np.array([m['tgt_x'] for m in matches])
    src_y = np.array([m['tgt_y'] for m in matches])
    shift_dx = np.array([m['shift_dx'] for m in matches])
    shift_dy = np.array([m['shift_dy'] for m in matches])
    conf = np.array([m['confidence'] for m in matches])

    # 相对全局平移的残差
    residual_dx = shift_dx - global_dx
    residual_dy = shift_dy - global_dy

    # 置信度筛选
    valid = conf >= confidence_threshold

    if valid.sum() < min_points:
        return {'points_xy': np.empty((0, 2)), 'residual_dx': np.array([]),
                'residual_dy': np.array([]), 'confidence': np.array([]),
                'valid_mask': np.array([], dtype=bool), 'n_valid': 0}

    # 联合MAD异常点筛选
    res_dx_v = residual_dx[valid]
    res_dy_v = residual_dy[valid]
    res_norm = np.hypot(res_dx_v, res_dy_v)

    med_norm = np.median(res_norm)
    mad_norm = np.median(np.abs(res_norm - med_norm))

    outlier_mask = res_norm > med_norm + 3 * mad_norm
    # 在 valid 内部去除 outlier
    valid_indices = np.where(valid)[0]
    for idx in valid_indices[outlier_mask]:
        valid[idx] = False

    points_xy = np.column_stack([src_x[valid], src_y[valid]])

    return {
        'points_xy': points_xy,
        'residual_dx': residual_dx[valid],
        'residual_dy': residual_dy[valid],
        'confidence': conf[valid],
        'valid_mask': valid,
        'n_valid': int(valid.sum()),
    }


def build_parent_based_local_controls(image_idx, parent_idx, pair_measurements,
                                       global_shifts, confidence_threshold=0.5,
                                       min_points=10):
    """基于父子关系构建局部RBF控制点。

    自动判断 pair 记录方向，确保控制点坐标始终在 image_idx 像素坐标系中。

    Parameters
    ----------
    image_idx : int, 待校正影像索引
    parent_idx : int, 父影像索引（已配准）
    pair_measurements : list of dict, 每个元素包含:
        idx_i, idx_j, shift_dx, shift_dy, confidence, n_blocks, rmse, matches
    global_shifts : ndarray, shape (n_images, 2)
    confidence_threshold : float

    Returns
    -------
    dict with keys: points_xy, residual_dx, residual_dy, confidence, valid_mask, n_valid
    """
    if not pair_measurements:
        return {'points_xy': np.empty((0, 2)), 'residual_dx': np.array([]),
                'residual_dy': np.array([]), 'confidence': np.array([]),
                'valid_mask': np.array([], dtype=bool), 'n_valid': 0}

    # 查找 image_idx 与 parent_idx 之间的匹配对
    pair = None
    flipped = False
    for pm in pair_measurements:
        if pm['idx_i'] == parent_idx and pm['idx_j'] == image_idx:
            # 父→子方向：shift 表示子影像需施加的位移
            pair = pm
            flipped = False
            break
        elif pm['idx_i'] == image_idx and pm['idx_j'] == parent_idx:
            # 子→父方向：需要翻转
            pair = pm
            flipped = True
            break

    if pair is None or not pair.get('available', True):
        return {'points_xy': np.empty((0, 2)), 'residual_dx': np.array([]),
                'residual_dy': np.array([]), 'confidence': np.array([]),
                'valid_mask': np.array([], dtype=bool), 'n_valid': 0}

    matches = pair.get('matches', [])
    if not matches:
        return {'points_xy': np.empty((0, 2)), 'residual_dx': np.array([]),
                'residual_dy': np.array([]), 'confidence': np.array([]),
                'valid_mask': np.array([], dtype=bool), 'n_valid': 0}

    # 提取坐标和位移，确保 points_xy 在 image_idx 像素坐标系中
    pts_x = []
    pts_y = []
    shift_dx_arr = []
    shift_dy_arr = []
    conf_arr = []

    for m in matches:
        if flipped:
            # 原始记录: idx_i=image_idx, idx_j=parent_idx
            # m['ref_x'], m['ref_y'] = image_idx 块中心
            # m['shift_dx'], m['shift_dy'] = 施加到 parent_idx 的位移
            # 翻转后: image_idx 需施加的位移 = -shift
            pts_x.append(m['ref_x'])
            pts_y.append(m['ref_y'])
            shift_dx_arr.append(-m['shift_dx'])
            shift_dy_arr.append(-m['shift_dy'])
        else:
            # 原始记录: idx_i=parent_idx, idx_j=image_idx
            # m['tgt_x'], m['tgt_y'] = image_idx 块中心
            # m['shift_dx'], m['shift_dy'] = 施加到 image_idx 的位移
            pts_x.append(m['tgt_x'])
            pts_y.append(m['tgt_y'])
            shift_dx_arr.append(m['shift_dx'])
            shift_dy_arr.append(m['shift_dy'])
        conf_arr.append(m.get('confidence', 0.5))

    pts_x = np.array(pts_x)
    pts_y = np.array(pts_y)
    shift_dx_arr = np.array(shift_dx_arr)
    shift_dy_arr = np.array(shift_dy_arr)
    conf_arr = np.array(conf_arr)

    # 网络全局位移差（父→子方向）
    parent_rel_dx = global_shifts[image_idx, 0] - global_shifts[parent_idx, 0]
    parent_rel_dy = global_shifts[image_idx, 1] - global_shifts[parent_idx, 1]

    # Post-global rematches already measure the residual displacement applied
    # to the target; initial pair matches still need the network difference.
    if pair.get('is_post_global_residual', False):
        residual_dx = shift_dx_arr
        residual_dy = shift_dy_arr
    else:
        residual_dx = shift_dx_arr - parent_rel_dx
        residual_dy = shift_dy_arr - parent_rel_dy

    # 置信度筛选
    valid = conf_arr >= confidence_threshold

    if valid.sum() < min_points:
        return {'points_xy': np.empty((0, 2)), 'residual_dx': np.array([]),
                'residual_dy': np.array([]), 'confidence': np.array([]),
                'valid_mask': np.array([], dtype=bool), 'n_valid': 0}

    # 联合MAD异常点筛选
    res_dx_v = residual_dx[valid]
    res_dy_v = residual_dy[valid]
    res_norm = np.hypot(res_dx_v, res_dy_v)

    med_norm = np.median(res_norm)
    mad_norm = np.median(np.abs(res_norm - med_norm))

    outlier_mask = res_norm > med_norm + 3 * mad_norm
    valid_indices = np.where(valid)[0]
    for idx in valid_indices[outlier_mask]:
        valid[idx] = False

    points_xy = np.column_stack([pts_x[valid], pts_y[valid]])

    return {
        'points_xy': points_xy,
        'residual_dx': residual_dx[valid],
        'residual_dy': residual_dy[valid],
        'confidence': conf_arr[valid],
        'valid_mask': valid,
        'n_valid': int(valid.sum()),
    }


def rematch_pair_on_registered(arr_ref, arr_tgt, tr_ref, tr_tgt,
                                nd_ref, nd_tgt,
                                max_residual_shift=5, block_size=512,
                                confidence_threshold=0.5):
    """对已全局配准的两景影像重新匹配，获取局部残余位移。

    两次完整匹配：第一次用 confidence_threshold，不足20块时第二次用 0.4。
    比较两次结果，选择有效块更多且RMSE没有明显恶化的。

    Returns
    -------
    dict with ... , used_confidence_threshold
    or None
    """
    # 第一次匹配
    matches1, screening1 = collect_block_matches(
        arr_ref, tr_ref, arr_tgt, tr_tgt, nd_ref, nd_tgt,
        block_size=block_size, max_global_shift=max_residual_shift,
        confidence_threshold=confidence_threshold)

    good1 = [m for m in matches1 if m['confidence'] >= confidence_threshold] if matches1 else []
    result1 = _build_rematch_result(good1, screening1, confidence_threshold) if len(good1) >= 5 else None

    # 如果0.5阈值下>=20块，直接返回
    if result1 is not None and result1['n_blocks'] >= 20:
        return result1

    # 第二次匹配：0.4
    if confidence_threshold > 0.4:
        matches2, screening2 = collect_block_matches(
            arr_ref, tr_ref, arr_tgt, tr_tgt, nd_ref, nd_tgt,
            block_size=block_size, max_global_shift=max_residual_shift,
            confidence_threshold=0.4)
        good2 = [m for m in matches2 if m['confidence'] >= 0.4] if matches2 else []
        result2 = _build_rematch_result(good2, screening2, 0.4) if len(good2) >= 5 else None
    else:
        result2 = None

    # 比较选择
    if result1 is None and result2 is None:
        return None
    if result1 is None:
        return result2
    if result2 is None:
        return result1

    # 选择有效块更多且RMSE没有明显恶化的
    rmse_ratio = result2['rmse'] / max(result1['rmse'], 1e-10)
    if result2['n_blocks'] > result1['n_blocks'] and rmse_ratio < 1.5:
        return result2
    return result1


def _build_rematch_result(matches, screening, used_confidence_threshold):
    """从匹配结果构建标准返回字典。"""
    confs = np.array([m['confidence'] for m in matches])
    dxs = np.array([m['shift_dx'] for m in matches])
    dys = np.array([m['shift_dy'] for m in matches])

    shift_dx = float(np.average(dxs, weights=confs))
    shift_dy = float(np.average(dys, weights=confs))

    res = np.hypot(dxs - shift_dx, dys - shift_dy)
    rmse = float(np.sqrt(np.mean(res**2)))

    return {
        'shift_dx': shift_dx, 'shift_dy': shift_dy,
        'confidence': float(confs.mean()),
        'matches': matches, 'n_blocks': len(matches),
        'rmse': rmse, 'screening': screening, 'available': True,
        'used_confidence_threshold': used_confidence_threshold,
    }


def refine_global_residual_shifts_from_original(original_arrays, global_shifts,
                                                transforms, nodatas,
                                                rematch_edges, params):
    """Refine global shifts from post-warp residual measurements.

    Each iteration warps the original arrays with the current global shifts,
    rematches the configured edges, and solves those residual edge shifts as a
    delta network. Temporary arrays are never used as inputs to later warps.
    """
    shifts = np.asarray(global_shifts, dtype=float).copy()
    params = params or {}
    max_iterations = int(params.get('global_refine_max_iterations', 2))
    stop_magnitude = float(params.get('global_refine_stop_magnitude', 0.15))
    max_correction = float(params.get('global_refine_max_correction', 5.0))
    block_size = int(params.get('global_refine_block_size', 384))
    confidence_threshold = float(params.get('global_confidence_threshold', 0.5))
    max_residual_shift = float(params.get('global_refine_max_residual_shift', max_correction))
    reference_idx = int(params.get('reference_idx', 0))

    history = []
    warnings = []

    for iteration in range(max_iterations):
        registered = []
        for idx, original in enumerate(original_arrays):
            dx, dy = shifts[idx]
            if abs(dx) < 1e-12 and abs(dy) < 1e-12:
                registered.append(np.asarray(original, dtype=np.float64).copy())
                continue
            h, w = original.shape[-2:]
            local_dx = np.zeros((h, w), dtype=np.float64)
            local_dy = np.zeros((h, w), dtype=np.float64)
            registered.append(warp_multiband_with_displacement_field(
                original, dx, dy, local_dx, local_dy, nodatas[idx]))

        delta_pairs = []
        edge_history = []
        for edge in rematch_edges:
            if isinstance(edge, dict):
                i, j = int(edge['idx_i']), int(edge['idx_j'])
            else:
                i, j = int(edge[0]), int(edge[1])
            result = rematch_pair_on_registered(
                registered[i], registered[j], transforms[i], transforms[j],
                nodatas[i], nodatas[j],
                max_residual_shift=max_residual_shift,
                block_size=block_size,
                confidence_threshold=confidence_threshold,
            )
            if not result or not result.get('available', True):
                warning = f'rematch unavailable for edge ({i}, {j})'
                warnings.append(warning)
                edge_history.append({'idx_i': i, 'idx_j': j,
                                     'available': False, 'warning': warning})
                continue

            delta_pairs.append({
                'idx_i': i, 'idx_j': j,
                'shift_dx': float(result['shift_dx']),
                'shift_dy': float(result['shift_dy']),
                'confidence': float(result.get('confidence', 0.0)),
                'n_blocks': int(result.get('n_blocks', 0)),
                'rmse': float(result.get('rmse', 1.0)),
                'p95': float(result.get('p95', result.get('rmse', 1.0))),
                'matches': result.get('matches', []),
            })
            edge_history.append({
                'idx_i': i, 'idx_j': j, 'available': True,
                'shift_dx': float(result['shift_dx']),
                'shift_dy': float(result['shift_dy']),
            })

        iteration_record = {'iteration': iteration + 1, 'edges': edge_history}
        if not delta_pairs:
            warning = f'no usable residual edges in iteration {iteration + 1}'
            warnings.append(warning)
            iteration_record.update({'accepted': False, 'warning': warning})
            history.append(iteration_record)
            break

        delta_result = multi_image_network_adjustment(
            delta_pairs, len(original_arrays), reference_idx)
        delta_shifts = np.asarray(delta_result['global_shifts'], dtype=float)
        magnitudes = np.hypot(delta_shifts[:, 0], delta_shifts[:, 1])
        non_reference = np.delete(magnitudes, reference_idx)
        correction_magnitude = float(np.max(non_reference)) if len(non_reference) else 0.0
        iteration_record.update({
            'delta_shifts': delta_shifts.tolist(),
            'correction_magnitude': correction_magnitude,
        })

        if np.any(magnitudes > max_correction):
            warning = (f'correction exceeds global_refine_max_correction '
                       f'in iteration {iteration + 1}')
            warnings.append(warning)
            iteration_record.update({'accepted': False, 'warning': warning})
            history.append(iteration_record)
            break

        if correction_magnitude < stop_magnitude:
            iteration_record.update({'accepted': False, 'stopped': True})
            history.append(iteration_record)
            break

        shifts += delta_shifts
        shifts[reference_idx, :] = 0.0
        iteration_record['accepted'] = True
        history.append(iteration_record)

    shifts[reference_idx, :] = 0.0
    return {'global_shifts': shifts, 'history': history, 'warnings': warnings}


def refine_global_residual_shifts(global_registered, global_shifts, arrays,
                                   transforms, nodatas, parent_map,
                                   max_iterations=2):
    """迭代修正影像[1]的整体残余位移。

    仅处理影像[1]相对[0]的修正。影像[3]必须等[1]完成局部RBF后再处理。
    每次从原始影像重新生成 global_registered，禁止连续重采样。

    Parameters
    ----------
    global_registered : dict
    global_shifts : ndarray (n_images, 2), 就地更新
    arrays : dict
    transforms : dict
    nodatas : dict
    parent_map : dict
    max_iterations : int

    Returns
    -------
    dict with n_iterations, corrections
    """
    from src.coregistration import compute_shifts_from_overlap, warp_with_displacement_field

    corrections_log = []

    for iteration in range(max_iterations):
        print(f"\n  Iteration {iteration + 1}/{max_iterations}:")

        any_correction = False
        iter_corrections = []

        # 仅处理影像[1]
        idx = 1
        res_y, res_x, conf, stats = compute_shifts_from_overlap(
            global_registered[0], transforms[0],
            global_registered[idx], transforms[idx],
            nodatas[0], nodatas[idx])

        if stats is None or not stats.get('available', True):
            print(f"    [{idx}]: verification unavailable, skipped")
            iter_corrections.append({'idx': idx, 'accepted': False, 'reason': 'unavailable'})
        else:
            n_blk = stats.get('n_blocks_inlier', 0)
            rmse = stats.get('residual_rmse', float('inf'))
            p95 = stats.get('residual_p95', float('inf'))
            mag = float(np.hypot(res_x, res_y))

            accept = (n_blk >= 3 and conf >= 0.45 and
                      rmse <= 0.75 and p95 <= 1.20 and
                      0.30 < mag <= 5.0)

            if accept:
                global_shifts[idx, 0] += res_x
                global_shifts[idx, 1] += res_y
                any_correction = True
                print(f"    [{idx}]: dx={res_x:.4f}, dy={res_y:.4f}, mag={mag:.4f} "
                      f"ACCEPTED (n={n_blk}, rmse={rmse:.3f}, p95={p95:.3f})")
            else:
                reason_parts = []
                if n_blk < 3: reason_parts.append(f"n={n_blk}<3")
                if conf < 0.45: reason_parts.append(f"conf={conf:.2f}<0.45")
                if rmse > 0.75: reason_parts.append(f"rmse={rmse:.3f}>0.75")
                if p95 > 1.20: reason_parts.append(f"p95={p95:.3f}>1.20")
                if mag <= 0.30: reason_parts.append(f"mag={mag:.4f}<=0.30")
                if mag > 5.0: reason_parts.append(f"mag={mag:.4f}>5.0")
                print(f"    [{idx}]: dx={res_x:.4f}, dy={res_y:.4f}, mag={mag:.4f} "
                      f"REJECTED ({', '.join(reason_parts)})")

            iter_corrections.append({
                'idx': idx, 'dx': float(res_x), 'dy': float(res_y),
                'magnitude': mag, 'accepted': accept,
            })

        corrections_log.append(iter_corrections)

        if not any_correction:
            print(f"  No corrections in iteration {iteration + 1}, stopping.")
            break

        # 从原始影像重新生成 global_registered
        if any_correction:
            print(f"  Regenerating global_registered from originals...")
            for i in range(len(arrays)):
                if i == 0:
                    global_registered[i] = arrays[i].astype(np.float64)
                    continue
                gdx = global_shifts[i, 0]
                gdy = global_shifts[i, 1]
                global_registered[i] = warp_with_displacement_field(
                    arrays[i], gdx, gdy,
                    np.zeros(arrays[i].shape), np.zeros(arrays[i].shape),
                    nodatas[i])

    # 安全钳制：参考影像[0]始终固定
    global_shifts[0, :] = 0.0

    return {'n_iterations': len(corrections_log), 'corrections': corrections_log}


def build_residual_controls_from_rematch(matches, confidence_threshold=0.5,
                                          min_points=10):
    """从重新匹配的结果构建残余控制点。

    匹配已在全局配准后完成，shift 已经是局部残余位移，
    不需要再减去 global_shifts。

    Parameters
    ----------
    matches : list of dict
    confidence_threshold : float
    min_points : int, 最少有效点数

    Returns
    -------
    dict with points_xy, residual_dx, residual_dy, confidence, n_valid
    """
    if not matches:
        return {'points_xy': np.empty((0, 2)), 'residual_dx': np.array([]),
                'residual_dy': np.array([]), 'confidence': np.array([]),
                'n_valid': 0}

    pts_x = np.array([m['tgt_x'] for m in matches])
    pts_y = np.array([m['tgt_y'] for m in matches])
    res_dx = np.array([m['shift_dx'] for m in matches])
    res_dy = np.array([m['shift_dy'] for m in matches])
    conf = np.array([m['confidence'] for m in matches])

    valid = conf >= confidence_threshold
    if valid.sum() < min_points:
        return {'points_xy': np.empty((0, 2)), 'residual_dx': np.array([]),
                'residual_dy': np.array([]), 'confidence': np.array([]),
                'n_valid': 0}

    # 联合MAD异常点剔除
    res_dx_v = res_dx[valid]
    res_dy_v = res_dy[valid]
    res_norm = np.hypot(res_dx_v, res_dy_v)
    med_norm = np.median(res_norm)
    mad_norm = np.median(np.abs(res_norm - med_norm))
    outlier = res_norm > med_norm + 3 * mad_norm
    valid_indices = np.where(valid)[0]
    for idx in valid_indices[outlier]:
        valid[idx] = False

    return {
        'points_xy': np.column_stack([pts_x[valid], pts_y[valid]]),
        'residual_dx': res_dx[valid],
        'residual_dy': res_dy[valid],
        'confidence': conf[valid],
        'n_valid': int(valid.sum()),
    }


def merge_multi_edge_controls(edge_controls, dedup_distance=50.0,
                               confidence_threshold=0.4, min_points=20):
    """合并多条边的局部控制点，用于多邻接影像约束的局部RBF。

    执行：邻近重复控制点合并 → confidence筛选 → 联合MAD异常点剔除。
    合并近邻点时保留置信度最高的控制点。

    Parameters
    ----------
    edge_controls : list of dict
        每个元素包含 points_xy (N,2), residual_dx (N,), residual_dy (N,), confidence (N,), n_valid (int)
        所有坐标必须在同一目标影像像素坐标系中
    dedup_distance : float, 合并近邻点的距离阈值（像素）
    confidence_threshold : float

    Returns
    -------
    dict with points_xy, residual_dx, residual_dy, confidence, n_valid
    """
    all_pts = []
    all_res_dx = []
    all_res_dy = []
    all_conf = []

    for ctrl in edge_controls:
        if ctrl['n_valid'] == 0:
            continue
        all_pts.append(ctrl['points_xy'])
        all_res_dx.append(ctrl['residual_dx'])
        all_res_dy.append(ctrl['residual_dy'])
        all_conf.append(ctrl['confidence'])

    if not all_pts:
        return {'points_xy': np.empty((0, 2)), 'residual_dx': np.array([]),
                'residual_dy': np.array([]), 'confidence': np.array([]),
                'n_valid': 0, 'n_before_filter': 0}

    pts = np.vstack(all_pts)
    res_dx = np.concatenate(all_res_dx)
    res_dy = np.concatenate(all_res_dy)
    conf = np.concatenate(all_conf)
    n_before = len(pts)

    # 邻近重复控制点合并：按置信度降序，保留最高置信度的点
    if len(pts) > 1:
        from scipy.spatial import KDTree
        tree = KDTree(pts)
        keep = np.ones(len(pts), dtype=bool)
        order = np.argsort(-conf)
        for i in order:
            if not keep[i]:
                continue
            nearby = tree.query_ball_point(pts[i], dedup_distance)
            for j in nearby:
                if j != i and keep[j] and conf[j] <= conf[i]:
                    keep[j] = False
        pts = pts[keep]
        res_dx = res_dx[keep]
        res_dy = res_dy[keep]
        conf = conf[keep]

    # confidence筛选
    valid = conf >= confidence_threshold
    if valid.sum() < min_points:
        return {'points_xy': np.empty((0, 2)), 'residual_dx': np.array([]),
                'residual_dy': np.array([]), 'confidence': np.array([]),
                'n_valid': 0, 'n_before_filter': n_before}

    # 联合MAD异常点剔除
    res_dx_v = res_dx[valid]
    res_dy_v = res_dy[valid]
    res_norm = np.hypot(res_dx_v, res_dy_v)
    med_norm = np.median(res_norm)
    mad_norm = np.median(np.abs(res_norm - med_norm))
    outlier = res_norm > med_norm + 3 * mad_norm
    valid_indices = np.where(valid)[0]
    for idx in valid_indices[outlier]:
        valid[idx] = False

    return {
        'points_xy': pts[valid],
        'residual_dx': res_dx[valid],
        'residual_dy': res_dy[valid],
        'confidence': conf[valid],
        'n_valid': int(valid.sum()),
        'n_before_filter': n_before,
    }


def balance_edge_controls(ctrl_list, max_total=60, grid_size=256,
                           min_per_edge=15, dedup_distance=50.0):
    """对每条边的控制点进行MAD剔除+空间均匀抽样。

    每条边最多保留 max_total 个控制点，
    每个 grid_size×grid_size 网格最多保留 2 个高置信度控制点，
    每条边至少保留 min_per_edge 个控制点（如果原始数量足够）。

    Parameters
    ----------
    ctrl_list : list of dict, 每个包含 points_xy, residual_dx, residual_dy, confidence, n_valid
    max_total : int, 每条边最多保留
    grid_size : int
    min_per_edge : int
    dedup_distance : float

    Returns
    -------
    list of dict, 平衡后的每条边控制点
    """
    balanced = []

    for ctrl in ctrl_list:
        if ctrl['n_valid'] == 0:
            balanced.append(ctrl)
            continue

        pts = ctrl['points_xy'].copy()
        res_dx = ctrl['residual_dx'].copy()
        res_dy = ctrl['residual_dy'].copy()
        conf = ctrl['confidence'].copy()

        # MAD异常点剔除
        res_norm = np.hypot(res_dx, res_dy)
        med_norm = np.median(res_norm)
        mad_norm = np.median(np.abs(res_norm - med_norm))
        keep = res_norm <= med_norm + 3 * mad_norm

        pts = pts[keep]
        res_dx = res_dx[keep]
        res_dy = res_dy[keep]
        conf = conf[keep]

        if len(pts) <= min_per_edge:
            balanced.append({
                'points_xy': pts, 'residual_dx': res_dx, 'residual_dy': res_dy,
                'confidence': conf, 'n_valid': len(pts),
            })
            continue

        # 空间均匀抽样：每个网格最多2个高置信度点
        if len(pts) > 0:
            x_min, y_min = pts.min(axis=0)
            x_max, y_max = pts.max(axis=0)
            x_range = max(x_max - x_min, 1e-10)
            y_range = max(y_max - y_min, 1e-10)

            x_bin = np.clip(((pts[:, 0] - x_min) / x_range * (x_range / grid_size)).astype(int),
                            0, int(x_range / grid_size))
            y_bin = np.clip(((pts[:, 1] - y_min) / y_range * (y_range / grid_size)).astype(int),
                            0, int(y_range / grid_size))
            grid_id = y_bin * 10000 + x_bin

            # 每个网格按置信度排序，保留前2个
            keep_grid = np.zeros(len(pts), dtype=bool)
            for gid in np.unique(grid_id):
                mask = grid_id == gid
                idxs = np.where(mask)[0]
                order = idxs[np.argsort(-conf[idxs])]
                for k in order[:2]:
                    keep_grid[k] = True

            pts = pts[keep_grid]
            res_dx = res_dx[keep_grid]
            res_dy = res_dy[keep_grid]
            conf = conf[keep_grid]

        # 如果超过max_total，按置信度降序保留
        if len(pts) > max_total:
            order = np.argsort(-conf)[:max_total]
            pts = pts[order]
            res_dx = res_dx[order]
            res_dy = res_dy[order]
            conf = conf[order]

        # 至少保留min_per_edge个（如果原始够多）
        n_keep = max(len(pts), min_per_edge) if ctrl['n_valid'] >= min_per_edge else len(pts)
        n_keep = min(n_keep, len(pts))

        balanced.append({
            'points_xy': pts[:n_keep], 'residual_dx': res_dx[:n_keep],
            'residual_dy': res_dy[:n_keep], 'confidence': conf[:n_keep],
            'n_valid': n_keep,
        })

    return balanced


def compute_hull_fade_mask(points_xy, h, w, buffer=128):
    """计算凸包 + 缓冲带衰减掩膜。

    - 凸包内部: 1.0
    - 凸包外 buffer 像素内: 线性衰减到 0.0
    - 更远区域: 0.0

    Parameters
    ----------
    points_xy : ndarray (N, 2)
    h, w : int, 输出影像尺寸
    buffer : int, 缓冲带宽度（像素）

    Returns
    -------
    mask : ndarray (h, w), float64
    """
    from scipy.spatial import Delaunay
    from scipy.ndimage import distance_transform_edt

    hull = Delaunay(points_xy)

    yy, xx = np.mgrid[0:h, 0:w]
    test_pts = np.column_stack([xx.ravel(), yy.ravel()])

    # 分块计算凸包内外
    in_hull = np.zeros(h * w, dtype=bool)
    chunk = 50000
    for ci in range(0, len(test_pts), chunk):
        cp = test_pts[ci:ci + chunk]
        in_hull[ci:ci + chunk] = hull.find_simplex(cp) >= 0
    in_hull = in_hull.reshape(h, w)

    # 凸包外像素到凸包边界的距离
    dist_outside = distance_transform_edt(~in_hull)

    # 衰减掩膜：凸包内=1，凸包外128px内线性衰减，更远=0
    mask = np.zeros((h, w), dtype=np.float64)
    mask[in_hull] = 1.0
    outside = ~in_hull
    mask[outside] = np.clip(
        1.0 - dist_outside[outside] / float(buffer),
        0.0, 1.0)

    return mask


def fit_local_rbf(points_xy, residual_dx, residual_dy, smoothing, neighbors=20):
    """使用RBF插值拟合局部残差位移场。

    Parameters
    ----------
    points_xy : (N, 2) array, 控制点坐标
    residual_dx, residual_dy : (N,) array, 残差位移
    smoothing : float, 平滑参数
    neighbors : int, RBF近邻数

    Returns
    -------
    rbf_dx, rbf_dy : RBFInterpolator objects
    coord_min, coord_max : 用于归一化的范围
    """
    from scipy.interpolate import RBFInterpolator

    x = points_xy[:, 0]
    y = points_xy[:, 1]
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    # 归一化坐标
    x_norm = (x - x_min) / max(x_max - x_min, 1e-10)
    y_norm = (y - y_min) / max(y_max - y_min, 1e-10)
    points_norm = np.column_stack([x_norm, y_norm])

    n_neighbors = min(neighbors, len(points_xy))

    rbf_dx = RBFInterpolator(
        points_norm, residual_dx,
        kernel="thin_plate_spline",
        smoothing=smoothing,
        neighbors=n_neighbors,
    )

    rbf_dy = RBFInterpolator(
        points_norm, residual_dy,
        kernel="thin_plate_spline",
        smoothing=smoothing,
        neighbors=n_neighbors,
    )

    return rbf_dx, rbf_dy, (x_min, y_min), (x_max, y_max)


def spatial_cross_validate(points_xy, residual_dx, residual_dy,
                           global_dx, global_dy, matches,
                           n_groups_x=4, n_groups_y=4):
    """空间分组交叉验证。

    将控制点按空间网格分组，使用GroupKFold评估模型。

    Parameters
    ----------
    points_xy : (N, 2)
    residual_dx, residual_dy : (N,)
    global_dx, global_dy : float
    matches : list of dict, 原始匹配（用于评估平移模型）
    n_groups_x, n_groups_y : int, 空间网格划分数

    Returns
    -------
    cv_results : dict
        每个模型的交叉验证统计
    group_labels : (N,) array, 空间网格编号
    """
    from sklearn.model_selection import GroupKFold

    if len(points_xy) < 20:
        return {}, np.array([])

    x = points_xy[:, 0]
    y = points_xy[:, 1]

    # 空间网格分组
    x_bin = np.clip(((x - x.min()) / max(x.max() - x.min(), 1e-10) * n_groups_x).astype(int),
                     0, n_groups_x - 1)
    y_bin = np.clip(((y - y.min()) / max(y.max() - y.min(), 1e-10) * n_groups_y).astype(int),
                     0, n_groups_y - 1)
    groups = y_bin * n_groups_x + x_bin

    n_splits = min(5, len(np.unique(groups)))
    if n_splits < 2:
        return {}, groups

    gkf = GroupKFold(n_splits=n_splits)

    models_to_test = ['translation', 'affine', 'rbf']
    cv_results = {m: {'median': [], 'rmse': [], 'p90': [], 'p95': [], 'max': []}
                  for m in models_to_test}

    for train_idx, test_idx in gkf.split(points_xy, groups=groups):
        # 训练集
        train_xy = points_xy[train_idx]
        train_dx = residual_dx[train_idx]
        train_dy = residual_dy[train_idx]

        # 测试集
        test_xy = points_xy[test_idx]
        test_res_dx = residual_dx[test_idx]
        test_res_dy = residual_dy[test_idx]

        # 模型1: 仅全局平移（预测残差=0）
        for m_name in ['translation']:
            pred_dx = np.zeros(len(test_idx))
            pred_dy = np.zeros(len(test_idx))
            err = np.hypot(test_res_dx - pred_dx, test_res_dy - pred_dy)
            cv_results[m_name]['median'].append(float(np.median(err)))
            cv_results[m_name]['rmse'].append(float(np.sqrt(np.mean(err**2))))
            cv_results[m_name]['p90'].append(float(np.percentile(err, 90)))
            cv_results[m_name]['p95'].append(float(np.percentile(err, 95)))
            cv_results[m_name]['max'].append(float(np.max(err)))

        # 模型2: 仿射
        if len(train_idx) >= 3:
            try:
                from skimage.measure import ransac
                from skimage.transform import AffineTransform
                model, _ = ransac(
                    (train_xy, train_xy + np.column_stack([train_dx, train_dy])),
                    AffineTransform, min_samples=3,
                    residual_threshold=1.0, max_trials=500)
                pred_all = model(test_xy)
                pred_dx = pred_all[:, 0] - test_xy[:, 0]
                pred_dy = pred_all[:, 1] - test_xy[:, 1]
                err = np.hypot(test_res_dx - pred_dx, test_res_dy - pred_dy)
                cv_results['affine']['median'].append(float(np.median(err)))
                cv_results['affine']['rmse'].append(float(np.sqrt(np.mean(err**2))))
                cv_results['affine']['p90'].append(float(np.percentile(err, 90)))
                cv_results['affine']['p95'].append(float(np.percentile(err, 95)))
                cv_results['affine']['max'].append(float(np.max(err)))
            except Exception:
                for k in cv_results['affine']:
                    cv_results['affine'][k].append(float('inf'))

        # 模型3: RBF
        if len(train_idx) >= 10:
            try:
                rbf_dx, rbf_dy, cmin, cmax = fit_local_rbf(
                    train_xy, train_dx, train_dy, smoothing=0.1, neighbors=min(20, len(train_idx)))
                tx = (test_xy[:, 0] - cmin[0]) / max(cmax[0] - cmin[0], 1e-10)
                ty = (test_xy[:, 1] - cmin[1]) / max(cmax[1] - cmin[1], 1e-10)
                test_norm = np.column_stack([tx, ty])
                pred_dx = rbf_dx(test_norm)
                pred_dy = rbf_dy(test_norm)
                err = np.hypot(test_res_dx - pred_dx, test_res_dy - pred_dy)
                cv_results['rbf']['median'].append(float(np.median(err)))
                cv_results['rbf']['rmse'].append(float(np.sqrt(np.mean(err**2))))
                cv_results['rbf']['p90'].append(float(np.percentile(err, 90)))
                cv_results['rbf']['p95'].append(float(np.percentile(err, 95)))
                cv_results['rbf']['max'].append(float(np.max(err)))
            except Exception:
                for k in cv_results['rbf']:
                    cv_results['rbf'][k].append(float('inf'))

    # 汇总：取各折均值
    summary = {}
    for m_name in models_to_test:
        summary[m_name] = {}
        for k in cv_results[m_name]:
            vals = [v for v in cv_results[m_name][k] if v != float('inf')]
            summary[m_name][k] = float(np.mean(vals)) if vals else float('inf')

    return summary, groups


def spatial_holdout_validation(
    arr_ref, tr_ref, arr_tgt, tr_tgt,
    nodata_ref, nodata_tgt,
    block_matches,
    n_grid_rows=4, n_grid_cols=4,
    n_folds=5,
    block_size=512, step=256,
    min_train_points=30,
    min_validation_blocks_per_fold=3,
    random_state=42,
):
    """严格空间分区留出验证。

    每折重新拟合全局平移和 RBF 模型，在留出区域内重新做相位相关残差估计。

    Parameters
    ----------
    arr_ref, arr_tgt : 2D ndarray
    tr_ref, tr_tgt : Affine
    nodata_ref, nodata_tgt : float
    block_matches : list of dict, src_x, src_y, shift_dx, shift_dy, confidence
    n_grid_rows, n_grid_cols : int, 空间网格划分
    n_folds : int, 最大折数
    block_size : int, 验证块大小
    step : int, 验证网格步长
    min_train_points : int
    min_validation_blocks_per_fold : int
    random_state : int

    Returns
    -------
    dict with available, folds, aggregate, coverage, failure_reason
    """
    from src.overlap import intersection_bounds
    from rasterio.transform import array_bounds, rowcol
    from sklearn.model_selection import GroupKFold

    rng = np.random.RandomState(random_state)

    if len(block_matches) < min_train_points:
        return {'available': False, 'failure_reason': f'Too few matches: {len(block_matches)}',
                'n_folds_requested': n_folds, 'n_folds_valid': 0,
                'n_validation_blocks_total': 0, 'covered_cells': 0,
                'total_cells': 0, 'coverage_ratio': 0, 'folds': [], 'aggregate': None}

    src_x = np.array([m['src_x'] for m in block_matches])
    src_y = np.array([m['src_y'] for m in block_matches])
    shift_dx = np.array([m['shift_dx'] for m in block_matches])
    shift_dy = np.array([m['shift_dy'] for m in block_matches])
    conf = np.array([m['confidence'] for m in block_matches])

    # 空间网格分组
    x_bin = np.clip(((src_x - src_x.min()) / max(src_x.max() - src_x.min(), 1e-10) * n_grid_cols).astype(int),
                     0, n_grid_cols - 1)
    y_bin = np.clip(((src_y - src_y.min()) / max(src_y.max() - src_y.min(), 1e-10) * n_grid_rows).astype(int),
                     0, n_grid_rows - 1)
    groups = y_bin * n_grid_cols + x_bin

    n_unique_groups = len(np.unique(groups))
    actual_n_folds = min(n_folds, n_unique_groups)
    if actual_n_folds < 3:
        return {'available': False, 'failure_reason': f'Only {n_unique_groups} unique spatial groups',
                'n_folds_requested': n_folds, 'n_folds_valid': 0,
                'n_validation_blocks_total': 0, 'covered_cells': 0,
                'total_cells': n_grid_rows * n_grid_cols, 'coverage_ratio': 0,
                'folds': [], 'aggregate': None}

    gkf = GroupKFold(n_splits=actual_n_folds)

    # 重叠区域像素范围
    h_ref, w_ref = arr_ref.shape
    h_tgt, w_tgt = arr_tgt.shape
    bounds_ref = array_bounds(h_ref, w_ref, tr_ref)
    bounds_tgt = array_bounds(h_tgt, w_tgt, tr_tgt)
    overlap = intersection_bounds(bounds_ref, bounds_tgt)
    if overlap is None:
        return {'available': False, 'failure_reason': 'No overlap',
                'n_folds_requested': n_folds, 'n_folds_valid': 0,
                'n_validation_blocks_total': 0, 'covered_cells': 0,
                'total_cells': n_grid_rows * n_grid_cols, 'coverage_ratio': 0,
                'folds': [], 'aggregate': None}

    r0, c0 = rowcol(tr_ref, overlap[0], overlap[3])
    r1, c1 = rowcol(tr_ref, overlap[2], overlap[1])
    r0, r1 = max(0, r0), min(h_ref, r1)
    c0, c1 = max(0, c0), min(w_ref, c1)

    valid_ref = np.isfinite(arr_ref) & (arr_ref != nodata_ref)
    valid_tgt = np.isfinite(arr_tgt) & (arr_tgt != nodata_tgt)
    struct_ref = structural_image(arr_ref, valid_ref)
    struct_tgt = structural_image(arr_tgt, valid_tgt)

    folds = []
    all_val_mags = []
    all_val_confs = []
    covered_cells = set()
    total_cells = n_grid_rows * n_grid_cols
    confidence_thresholds = [0.50, 0.40, 0.30]

    for fold_id, (train_idx, val_group_idx) in enumerate(gkf.split(
            np.zeros(len(block_matches)), groups=groups)):

        fold_info = {
            'fold_id': fold_id,
            'held_out_group_ids': sorted(set(groups[val_group_idx])),
            'n_train_points': len(train_idx),
            'n_validation_candidates': 0,
            'n_validation_accepted': 0,
            'confidence_threshold_used': None,
            'rbf_smoothing': None,
            'global_dx': None, 'global_dy': None,
            'median': None, 'rmse': None, 'p90': None, 'p95': None, 'max': None,
            'mean_confidence': None,
            'valid': False, 'failure_reason': None,
        }

        if len(train_idx) < min_train_points:
            fold_info['failure_reason'] = f'Too few train points: {len(train_idx)}'
            folds.append(fold_info)
            continue

        # 1. 训练集全局平移（加权平均）
        train_conf = conf[train_idx]
        train_sdx = shift_dx[train_idx]
        train_sdy = shift_dy[train_idx]
        w = train_conf / max(train_conf.sum(), 1e-10)
        fold_gdx = float(np.sum(train_sdx * w))
        fold_gdy = float(np.sum(train_sdy * w))
        fold_info['global_dx'] = fold_gdx
        fold_info['global_dy'] = fold_gdy

        # 2. 训练残差 + MAD筛选
        train_res_dx = train_sdx - fold_gdx
        train_res_dy = train_sdy - fold_gdy
        res_norm = np.hypot(train_res_dx, train_res_dy)
        med_n = np.median(res_norm)
        mad_n = np.median(np.abs(res_norm - med_n))
        outlier = res_norm > med_n + 3 * mad_n
        keep = ~outlier
        train_xy_clean = np.column_stack([src_x[train_idx][keep], src_y[train_idx][keep]])
        train_rdx_clean = train_res_dx[keep]
        train_rdy_clean = train_res_dy[keep]
        train_conf_clean = train_conf[keep]

        if len(train_xy_clean) < min_train_points:
            fold_info['failure_reason'] = f'After MAD: {len(train_xy_clean)} points'
            folds.append(fold_info)
            continue

        # 3. RBF smoothing 选择（在训练集上）
        best_sm = 0.1
        best_p95 = float('inf')
        for sm in [0.001, 0.01, 0.05, 0.1, 0.5, 1.0]:
            try:
                rbf_dx_f, rbf_dy_f, cmin_f, cmax_f = fit_local_rbf(
                    train_xy_clean, train_rdx_clean, train_rdy_clean,
                    smoothing=sm, neighbors=min(20, len(train_xy_clean)))
                tx = (train_xy_clean[:, 0] - cmin_f[0]) / max(cmax_f[0] - cmin_f[0], 1e-10)
                ty = (train_xy_clean[:, 1] - cmin_f[1]) / max(cmax_f[1] - cmin_f[1], 1e-10)
                pts_n = np.column_stack([tx, ty])
                pred_dx = rbf_dx_f(pts_n)
                pred_dy = rbf_dy_f(pts_n)
                err = np.hypot(train_rdx_clean - pred_dx, train_rdy_clean - pred_dy)
                p95 = float(np.percentile(err, 95))
                if p95 < best_p95:
                    best_p95 = p95
                    best_sm = sm
            except Exception:
                continue
        fold_info['rbf_smoothing'] = best_sm

        # 4. 用最佳smoothing拟合最终RBF
        try:
            rbf_dx_f, rbf_dy_f, cmin_f, cmax_f = fit_local_rbf(
                train_xy_clean, train_rdx_clean, train_rdy_clean,
                smoothing=best_sm, neighbors=min(20, len(train_xy_clean)))
        except Exception as e:
            fold_info['failure_reason'] = f'RBF fitting failed: {e}'
            folds.append(fold_info)
            continue

        # 5. 构建位移场并重采样目标影像
        local_dx_f = np.zeros((h_tgt, w_tgt), dtype=np.float64)
        local_dy_f = np.zeros((h_tgt, w_tgt), dtype=np.float64)
        try:
            from scipy.spatial import Delaunay
            hull = Delaunay(train_xy_clean)
            chunk_size = 50000
            yy, xx = np.mgrid[0:h_tgt, 0:w_tgt]
            test_pts = np.column_stack([xx.ravel(), yy.ravel()])
            for ci in range(0, len(test_pts), chunk_size):
                cp = test_pts[ci:ci+chunk_size]
                in_h = hull.find_simplex(cp) >= 0
                tx = (cp[:, 0] - cmin_f[0]) / max(cmax_f[0] - cmin_f[0], 1e-10)
                ty = (cp[:, 1] - cmin_f[1]) / max(cmax_f[1] - cmin_f[1], 1e-10)
                pts_n = np.column_stack([tx, ty])
                cdx = np.clip(rbf_dx_f(pts_n), -2.5, 2.5)
                cdy = np.clip(rbf_dy_f(pts_n), -2.5, 2.5)
                cdx[~in_h] = 0
                cdy[~in_h] = 0
                local_dx_f.ravel()[ci:ci+chunk_size] = cdx
                local_dy_f.ravel()[ci:ci+chunk_size] = cdy
        except Exception:
            pass

        from scipy.ndimage import shift as ndimage_shift
        arr_tgt_coreg = arr_tgt.astype(np.float64)
        total_dx_f = fold_gdx + local_dx_f
        total_dy_f = fold_gdy + local_dy_f
        from scipy.ndimage import map_coordinates
        yy2, xx2 = np.mgrid[0:h_tgt, 0:w_tgt]
        src_x2 = np.clip(xx2 - total_dx_f, 0, w_tgt - 1)
        src_y2 = np.clip(yy2 - total_dy_f, 0, h_tgt - 1)
        arr_tgt_coreg = map_coordinates(
            arr_tgt.astype(np.float64),
            [src_y2.ravel(), src_x2.ravel()],
            order=1, mode='constant', cval=nodata_tgt
        ).reshape(h_tgt, w_tgt)
        struct_tgt_coreg = structural_image(arr_tgt_coreg, np.isfinite(arr_tgt_coreg) & (arr_tgt_coreg != nodata_tgt))

        # 6. 留出区域内验证块
        val_group_ids = set(groups[val_group_idx])
        val_mask = np.isin(groups, list(val_group_ids))
        val_x = src_x[val_mask]
        val_y = src_y[val_mask]
        val_cx = val_x
        val_cy = val_y

        grid_rows = list(range(r0 + step // 2, r1 - block_size + 1, step))
        grid_cols = list(range(c0 + step // 2, c1 - block_size + 1, step))

        candidates = []
        for br in grid_rows:
            for bc in grid_cols:
                cx, cy = bc + block_size // 2, br + block_size // 2
                if len(val_x) > 0:
                    dists = np.hypot(val_cx - cx, val_cy - cy)
                    if np.min(dists) > block_size * 1.5:
                        continue
                candidates.append((br, bc, cx, cy))

        fold_info['n_validation_candidates'] = len(candidates)

        accepted_blocks = []
        used_threshold = None

        for thr in confidence_thresholds:
            accepted_blocks = []
            for br, bc, cx, cy in candidates:
                br2, bc2 = br + block_size, bc + block_size
                if br2 > h_ref or bc2 > w_ref:
                    continue
                v_ref = valid_ref[br:br2, bc:bc2]
                v_tgt = valid_tgt_coreg if 'valid_tgt_coreg' in dir() else valid_tgt
                if br2 > h_tgt or bc2 > w_tgt:
                    continue
                v_t = np.isfinite(arr_tgt_coreg[br:br2, bc:bc2]) & (arr_tgt_coreg[br:br2, bc:bc2] != nodata_tgt)
                joint = v_ref[:br2-br, :bc2-bc] & v_t[:br2-br, :bc2-bc]
                if joint.sum() < block_size * block_size * 0.3:
                    continue

                blk_ref = struct_ref[br:br2, bc:bc2]
                blk_tgt = struct_tgt_coreg[br:br2, bc:bc2]
                tex_std = float(np.std(blk_ref[blk_ref > 0])) if np.any(blk_ref > 0) else 0
                if tex_std < 1e-4:
                    continue

                sy, sx, cf = phase_correlation(blk_ref, blk_tgt, valid_ref=v_ref[:br2-br, :bc2-bc], valid_tgt=v_t[:br2-br, :bc2-bc])
                if cf < thr or abs(sy) >= 5 or abs(sx) >= 5:
                    continue
                mag = float(np.hypot(sx, sy))
                accepted_blocks.append({
                    'center_x': cx, 'center_y': cy,
                    'residual_dy': float(sy), 'residual_dx': float(sx),
                    'residual_magnitude': mag, 'confidence': float(cf),
                    'valid_ratio': float(joint.sum() / (block_size * block_size)),
                    'texture_std': tex_std,
                })

            if len(accepted_blocks) >= min_validation_blocks_per_fold:
                used_threshold = thr
                break

        fold_info['confidence_threshold_used'] = used_threshold
        fold_info['n_validation_accepted'] = len(accepted_blocks)

        if len(accepted_blocks) < min_validation_blocks_per_fold:
            fold_info['failure_reason'] = f'Only {len(accepted_blocks)} validation blocks'
            folds.append(fold_info)
            continue

        for b in accepted_blocks:
            gr = (b['center_y'] - r0) // step
            gc = (b['center_x'] - c0) // step
            covered_cells.add((min(gr, n_grid_rows - 1), min(gc, n_grid_cols - 1)))

        mags = np.array([b['residual_magnitude'] for b in accepted_blocks])
        confs = np.array([b['confidence'] for b in accepted_blocks])

        fold_info['median'] = float(np.median(mags))
        fold_info['rmse'] = float(np.sqrt(np.mean(mags**2)))
        fold_info['p90'] = float(np.percentile(mags, 90))
        fold_info['p95'] = float(np.percentile(mags, 95))
        fold_info['max'] = float(np.max(mags))
        fold_info['mean_confidence'] = float(np.mean(confs))
        fold_info['valid'] = True

        all_val_mags.extend(mags.tolist())
        all_val_confs.extend(confs.tolist())
        folds.append(fold_info)

    valid_folds = [f for f in folds if f['valid']]
    n_valid_folds = len(valid_folds)
    n_total_val_blocks = sum(f['n_validation_accepted'] for f in valid_folds)

    aggregate = None
    if all_val_mags:
        mags_arr = np.array(all_val_mags)
        confs_arr = np.array(all_val_confs)
        aggregate = {
            'median': float(np.median(mags_arr)),
            'rmse': float(np.sqrt(np.mean(mags_arr**2))),
            'p90': float(np.percentile(mags_arr, 90)),
            'p95': float(np.percentile(mags_arr, 95)),
            'max': float(np.max(mags_arr)),
            'mean_confidence': float(np.mean(confs_arr)),
        }

    coverage_ratio = len(covered_cells) / max(total_cells, 1)

    failure_reason = None
    if n_valid_folds < 3:
        failure_reason = f'Only {n_valid_folds} valid folds (need >= 3)'
    elif n_total_val_blocks < 15:
        failure_reason = f'Only {n_total_val_blocks} validation blocks (need >= 15)'
    elif coverage_ratio < 0.40:
        failure_reason = f'Coverage {coverage_ratio:.1%} < 40%'

    return {
        'available': n_valid_folds >= 3 and n_total_val_blocks >= 15 and coverage_ratio >= 0.40,
        'n_folds_requested': n_folds,
        'n_folds_valid': n_valid_folds,
        'n_validation_blocks_total': n_total_val_blocks,
        'covered_cells': len(covered_cells),
        'total_cells': total_cells,
        'coverage_ratio': coverage_ratio,
        'folds': folds,
        'aggregate': aggregate,
        'failure_reason': failure_reason,
    }


def analyze_displacement_spikes(
    local_dx_field, local_dy_field,
    valid_overlap_mask, control_hull_mask,
    nodata_boundary_mask, output_dir,
    top_n=200,
):
    """位移场梯度尖峰定位与诊断。

    Parameters
    ----------
    local_dx_field, local_dy_field : 2D ndarray
    valid_overlap_mask : 2D bool, 有效重叠区
    control_hull_mask : 2D bool, 控制点凸包内部
    nodata_boundary_mask : 2D bool, NoData边界区域
    output_dir : str
    top_n : int

    Returns
    -------
    dict with field_stats, gradient_stats, spike info, quality_warning
    """
    from scipy.ndimage import label, distance_transform_edt

    h, w = local_dx_field.shape

    def _stats(arr, valid=None):
        v = arr[valid] if valid is not None else arr[arr != 0]
        if len(v) == 0:
            return {'min': 0, 'median': 0, 'mean': 0, 'std': 0, 'p90': 0, 'p95': 0, 'p99': 0, 'max': 0}
        return {
            'min': float(np.min(v)), 'median': float(np.median(v)),
            'mean': float(np.mean(v)), 'std': float(np.std(v)),
            'p90': float(np.percentile(v, 90)), 'p95': float(np.percentile(v, 95)),
            'p99': float(np.percentile(v, 99)), 'max': float(np.max(v)),
        }

    mag = np.hypot(local_dx_field, local_dy_field)
    field_stats = {
        'local_dx': _stats(local_dx_field, valid_overlap_mask),
        'local_dy': _stats(local_dy_field, valid_overlap_mask),
        'magnitude': _stats(mag, valid_overlap_mask),
    }

    grad_dx_y, grad_dx_x = np.gradient(local_dx_field)
    grad_dy_y, grad_dy_x = np.gradient(local_dy_field)
    grad_mag = np.sqrt(grad_dx_x**2 + grad_dx_y**2 + grad_dy_x**2 + grad_dy_y**2)

    grad_valid = grad_mag[valid_overlap_mask]
    gradient_stats = _stats(grad_mag, valid_overlap_mask)

    # Spike threshold
    grad_median = gradient_stats['median']
    grad_mad = float(np.median(np.abs(grad_valid - grad_median))) if len(grad_valid) > 0 else 0
    threshold_robust = grad_median + 6 * grad_mad
    threshold_p99 = gradient_stats['p99']
    spike_threshold = max(threshold_robust, threshold_p99, 0.05)

    spike_mask = valid_overlap_mask & (grad_mag >= spike_threshold)

    # 连通域
    labeled, n_regions = label(spike_mask)
    region_sizes = np.bincount(labeled.ravel())[1:] if n_regions > 0 else np.array([])

    # 距离图
    edge_mask = np.zeros((h, w), dtype=bool)
    edge_mask[0, :] = True; edge_mask[-1, :] = True
    edge_mask[:, 0] = True; edge_mask[:, -1] = True
    dist_to_edge = distance_transform_edt(~edge_mask)

    hull_dist = distance_transform_edt(control_hull_mask)
    hull_boundary = control_hull_mask & (hull_dist <= 8)
    dist_to_hull_bnd = distance_transform_edt(~hull_boundary)

    nodata_bnd = nodata_boundary_mask
    dist_to_nodata = distance_transform_edt(~nodata_bnd) if nodata_bnd.any() else np.full((h, w), float('inf'))

    # 分类尖峰
    spike_pixels = np.argwhere(spike_mask)
    internal_count = 0
    hull_count = 0
    nodata_count = 0
    edge_count = 0

    for r, c in spike_pixels:
        if dist_to_nodata[r, c] <= 8:
            nodata_count += 1
        elif dist_to_hull_bnd[r, c] <= 16:
            hull_count += 1
        elif dist_to_edge[r, c] <= 8:
            edge_count += 1
        else:
            internal_count += 1

    # Max spike
    max_idx = np.nanargmax(np.where(valid_overlap_mask, grad_mag, np.nan))
    max_r, max_c = np.unravel_index(max_idx, grad_mag.shape)
    max_info = {
        'row': int(max_r), 'col': int(max_c),
        'gradient': float(grad_mag[max_r, max_c]),
        'local_dx': float(local_dx_field[max_r, max_c]),
        'local_dy': float(local_dy_field[max_r, max_c]),
        'magnitude': float(mag[max_r, max_c]),
        'inside_control_hull': bool(control_hull_mask[max_r, max_c]),
        'distance_to_hull_boundary': float(dist_to_hull_bnd[max_r, max_c]),
        'distance_to_nodata_boundary': float(dist_to_nodata[max_r, max_c]),
        'distance_to_image_edge': float(dist_to_edge[max_r, max_c]),
    }

    # Quality warnings
    warning_reasons = []
    if internal_count > 0:
        warning_reasons.append('Internal displacement spike detected. RBF may be overfitted.')
    if hull_count > 0:
        warning_reasons.append('Hull boundary transition spike. Consider decay buffer.')
    if nodata_count > 0:
        warning_reasons.append('NoData boundary spike.')

    # Region details
    regions = []
    for rid in range(1, n_regions + 1):
        rr, rc = np.where(labeled == rid)
        region_mask = labeled == rid
        region_grad = grad_mag * region_mask
        regions.append({
            'region_id': int(rid),
            'pixel_count': int(len(rr)),
            'bbox': [int(rr.min()), int(rc.min()), int(rr.max()), int(rc.max())],
            'max_gradient': float(region_grad.max()),
            'mean_gradient': float(region_grad[region_mask].mean()),
            'centroid_row': float(rr.mean()),
            'centroid_col': float(rc.mean()),
            'min_distance_to_hull': float(dist_to_hull_bnd[rr, rc].min()),
            'min_distance_to_nodata': float(dist_to_nodata[rr, rc].min()) if nodata_bnd.any() else float('inf'),
            'min_distance_to_edge': float(dist_to_edge[rr, rc].min()),
        })

    return {
        'field_stats': field_stats,
        'gradient_stats': gradient_stats,
        'spike_threshold': float(spike_threshold),
        'n_spike_pixels': int(spike_mask.sum()),
        'spike_region_count': n_regions,
        'internal_spike_count': internal_count,
        'hull_boundary_spike_count': hull_count,
        'nodata_boundary_spike_count': nodata_count,
        'image_edge_spike_count': edge_count,
        'max_spike': max_info,
        'spike_regions': regions,
        'quality_warning': len(warning_reasons) > 0,
        'warning_reasons': warning_reasons,
    }
    """收集独立验证块（偏移训练网格）。

    Parameters
    ----------
    arr_ref, arr_coreg : 2D ndarray
    tr_ref, tr_coreg : Affine transform
    nodata_ref, nodata_coreg : float
    train_matches : list of dict, 训练匹配点
    block_size : int
    step : int, 验证块步长（偏移 step//2）
    debug : bool, if True print diagnostic info

    Returns
    -------
    list of dict: src_x, src_y, shift_dx, shift_dy, confidence
    """
    from src.coregistration import structural_image, phase_correlation
    from src.overlap import intersection_bounds

    h_ref, w_ref = arr_ref.shape
    h_coreg, w_coreg = arr_coreg.shape

    valid_ref = np.isfinite(arr_ref) & (arr_ref != nodata_ref)
    valid_coreg = np.isfinite(arr_coreg) & (arr_coreg != nodata_coreg)

    struct_ref = structural_image(arr_ref, valid_ref)
    struct_coreg = structural_image(arr_coreg, valid_coreg)

    # 确定重叠区域的像素范围（在参考图像坐标系中）
    from rasterio.transform import array_bounds
    bounds_ref = array_bounds(h_ref, w_ref, tr_ref)   # (left, bottom, right, top)
    bounds_coreg = array_bounds(h_coreg, w_coreg, tr_coreg)
    overlap = intersection_bounds(bounds_ref, bounds_coreg)
    if overlap is None:
        if debug:
            print("    [val-debug] No overlap found")
        return []

    # overlap = (left, bottom, right, top) 地理坐标，转为参考图像的行列范围
    from rasterio.transform import rowcol
    r0, c0 = rowcol(tr_ref, overlap[0], overlap[3])   # left, top → row0, col0
    r1, c1 = rowcol(tr_ref, overlap[2], overlap[1])   # right, bottom → row1, col1
    r0 = max(0, r0)
    c0 = max(0, c0)
    r1 = min(h_ref, r1)
    c1 = min(w_ref, c1)

    if debug:
        print(f"    [val-debug] Overlap pixel region: rows [{r0},{r1}], cols [{c0},{c1}]")
        print(f"    [val-debug] Image shapes: ref={arr_ref.shape}, coreg={arr_coreg.shape}")

    # 训练块中心集合（用于排除距离太近的验证块）
    train_xy = np.array([[m['src_x'], m['src_y']] for m in train_matches])
    train_excl_radius = block_size * 0.35  # 排除半径 = 35% block_size

    offset = step // 2

    # 自适应 block_size：如果重叠区域太小，缩小 block_size
    overlap_h = r1 - r0
    overlap_w = c1 - c0
    effective_block = block_size
    if overlap_h < block_size * 2 or overlap_w < block_size * 2:
        effective_block = max(64, min(overlap_h, overlap_w) // 3)
        if debug:
            print(f"    [val-debug] Small overlap, reducing block_size from {block_size} to {effective_block}")

    # 在重叠区域内搜索验证块
    grid_start_r = r0 + offset
    grid_start_c = c0 + offset
    grid_end_r = r1 - effective_block + 1
    grid_end_c = c1 - effective_block + 1

    val_matches = []
    n_nodata_skip = 0
    n_texture_skip = 0
    n_conf_skip = 0
    n_train_skip = 0

    for br in range(grid_start_r, grid_end_r, step):
        for bc in range(grid_start_c, grid_end_c, step):
            br2 = br + effective_block
            bc2 = bc + effective_block

            cx = bc + effective_block // 2
            cy = br + effective_block // 2

            # 排除距训练点太近的验证块
            if len(train_xy) > 0:
                dists = np.sqrt((train_xy[:, 0] - cx)**2 + (train_xy[:, 1] - cy)**2)
                if np.min(dists) < train_excl_radius:
                    n_train_skip += 1
                    continue

            v_ref = valid_ref[br:br2, bc:bc2]
            v_coreg = valid_coreg[br:br2, bc:bc2]
            joint = v_ref & v_coreg

            if joint.sum() < effective_block * effective_block * 0.3:
                n_nodata_skip += 1
                continue

            blk_ref = struct_ref[br:br2, bc:bc2]
            blk_coreg = struct_coreg[br:br2, bc:bc2]

            sy, sx, conf = phase_correlation(
                blk_ref, blk_coreg, valid_ref=v_ref, valid_tgt=v_coreg)

            if conf <= 0.5 or abs(sy) >= 10 or abs(sx) >= 10:
                n_conf_skip += 1
                continue

            val_matches.append({
                'src_x': float(cx),
                'src_y': float(cy),
                'shift_dx': float(sx),
                'shift_dy': float(sy),
                'confidence': float(conf),
            })

    if debug:
        print(f"    [val-debug] grid=({grid_start_r}:{grid_end_r},{grid_start_c}:{grid_end_c}) "
              f"step={step} block={effective_block}")
        print(f"    [val-debug] candidates checked, results: {len(val_matches)} accepted, "
              f"nodata_skip={n_nodata_skip}, conf_skip={n_conf_skip}, train_skip={n_train_skip}")

    return val_matches


def evaluate_model_on_matches(matches, model_name, global_dx=0, global_dy=0,
                               rbf_dx=None, rbf_dy=None, coord_range=None,
                               affine_model=None):
    """在一组匹配点上评估模型残差。

    Returns
    -------
    dict with median, rmse, p90, p95, max, n_points
    """
    if not matches:
        return {'median': 0, 'rmse': 0, 'p90': 0, 'p95': 0, 'max': 0, 'n_points': 0}

    errors = []
    for m in matches:
        src = np.array([[m['src_x'], m['src_y']]])

        if model_name == 'translation':
            pred_dx = global_dx
            pred_dy = global_dy
        elif model_name == 'affine' and affine_model is not None:
            dst = affine_model(src)[0]
            pred_dx = dst[0] - m['src_x']
            pred_dy = dst[1] - m['src_y']
        elif model_name == 'rbf_local' and rbf_dx is not None and coord_range is not None:
            x_min, y_min = coord_range[0], coord_range[1]
            x_max, y_max = coord_range[2], coord_range[3]
            tx = (m['src_x'] - x_min) / max(x_max - x_min, 1e-10)
            ty = (m['src_y'] - y_min) / max(y_max - y_min, 1e-10)
            pred_dx = float(rbf_dx(np.array([[tx, ty]]))[0]) + global_dx
            pred_dy = float(rbf_dy(np.array([[tx, ty]]))[0]) + global_dy
        else:
            pred_dx = global_dx
            pred_dy = global_dy

        err = np.sqrt((m['shift_dx'] - pred_dx)**2 + (m['shift_dy'] - pred_dy)**2)
        errors.append(err)

    errors = np.array(errors)
    return {
        'median': float(np.median(errors)),
        'rmse': float(np.sqrt(np.mean(errors**2))),
        'p90': float(np.percentile(errors, 90)),
        'p95': float(np.percentile(errors, 95)),
        'max': float(np.max(errors)),
        'n_points': len(errors),
    }


def warp_with_displacement_field(arr, global_dx, global_dy,
                                  local_dx_field, local_dy_field, nodata):
    """对目标影像一次性施加全局平移+局部残差位移场。

    Parameters
    ----------
    arr : 2D ndarray
    global_dx, global_dy : float
    local_dx_field, local_dy_field : 2D ndarray, 与 arr 同尺寸
    nodata : float

    Returns
    -------
    warped : 2D ndarray (float64)
    """
    from scipy.ndimage import map_coordinates

    h, w = arr.shape
    yy, xx = np.mgrid[0:h, 0:w]

    total_dx = global_dx + local_dx_field
    total_dy = global_dy + local_dy_field

    src_x = xx - total_dx
    src_y = yy - total_dy

    # 不裁剪，让 map_coordinates mode="constant" cval=nodata 处理越界

    cval = nodata if nodata is not None else 0.0
    warped = map_coordinates(
        arr.astype(np.float64),
        [src_y.ravel(), src_x.ravel()],
        order=1, mode='constant', cval=cval
    ).reshape(h, w)

    # 有效掩膜
    valid_mask = np.isfinite(arr)
    if nodata is not None:
        valid_mask &= (arr != nodata)
    warped_mask = map_coordinates(
        valid_mask.astype(np.uint8),
        [src_y.ravel(), src_x.ravel()],
        order=0, mode='constant', cval=0
    ).reshape(h, w).astype(bool)

    warped[~warped_mask] = cval
    return warped


def warp_multiband_with_displacement_field(
    arr_3d, global_dx, global_dy,
    local_dx_field, local_dy_field, nodata,
    interpolation='bilinear'):
    """对多波段影像一次性施加全局平移+局部残差位移场。

    所有波段使用完全相同的位移场，每波段只做一次最终重采样。

    Parameters
    ----------
    arr_3d : 3D ndarray (bands, rows, cols)
    global_dx, global_dy : float
    local_dx_field, local_dy_field : 2D ndarray, 与单波段同尺寸
    nodata : float
    interpolation : str, 'bilinear' for continuous, 'nearest' for categorical

    Returns
    -------
    warped : 3D ndarray (bands, rows, cols), float64
    """
    from scipy.ndimage import map_coordinates

    assert local_dx_field.ndim == 2, f"local_dx_field must be 2D (rows, cols), got shape {local_dx_field.shape}"
    assert local_dy_field.ndim == 2, f"local_dy_field must be 2D (rows, cols), got shape {local_dy_field.shape}"
    assert local_dx_field.shape == local_dy_field.shape, f"dx/dy shape mismatch: {local_dx_field.shape} vs {local_dy_field.shape}"

    was_2d = arr_3d.ndim == 2
    if was_2d:
        arr_3d = arr_3d[np.newaxis, ...]
    n_bands, h, w = arr_3d.shape
    yy, xx = np.mgrid[0:h, 0:w]

    total_dx = global_dx + local_dx_field
    total_dy = global_dy + local_dy_field

    src_x = xx - total_dx
    src_y = yy - total_dy

    order = 1 if interpolation == 'bilinear' else 0

    cval = nodata if nodata is not None else 0.0
    warped = np.full_like(arr_3d, cval, dtype=np.float64)

    for b in range(n_bands):
        band = arr_3d[b].astype(np.float64)
        warped[b] = map_coordinates(
            band, [src_y.ravel(), src_x.ravel()],
            order=order, mode='constant', cval=cval
        ).reshape(h, w)

    # 有效掩膜（所有波段均有效）
    valid_mask = np.all(np.isfinite(arr_3d), axis=0)
    if nodata is not None:
        valid_mask &= np.all(arr_3d != nodata, axis=0)
    warped_mask = map_coordinates(
        valid_mask.astype(np.uint8),
        [src_y.ravel(), src_x.ravel()],
        order=0, mode='constant', cval=0
    ).reshape(h, w).astype(bool)

    warped[:, ~warped_mask] = cval
    return warped[0] if was_2d else warped


def compute_local_shift_field(arr_ref, tr_ref, arr_tgt, tr_tgt, nodata=0,
                               block_size=512, confidence_threshold=0.5):
    """Compute spatially varying (local) shift field between two images.

    Divides the overlap into blocks and estimates a shift for each block
    using phase correlation. Returns a grid of shifts that can be used
    for polynomial warping.

    Parameters
    ----------
    arr_ref, arr_tgt : 2D ndarray
    tr_ref, tr_tgt : Affine transform
    nodata : float
    block_size : int, size of each block for shift estimation
    confidence_threshold : float, minimum confidence for an accepted control

    Returns
    -------
    shift_grid_y : 2D ndarray, local y-shift at each grid point
    shift_grid_x : 2D ndarray, local x-shift at each grid point
    grid_rows : 1D ndarray, row indices of grid points in target image
    grid_cols : 1D ndarray, col indices of grid points in target image
    """
    h_ref, w_ref = arr_ref.shape
    h_tgt, w_tgt = arr_tgt.shape

    def pix2geo(tr, r, c):
        return tr.c + c * tr.a, tr.f + r * tr.e

    def geo2pix(tr, x, y):
        return int((y - tr.f) / tr.e), int((x - tr.c) / tr.a)

    ref_left, ref_top = pix2geo(tr_ref, 0, 0)
    ref_right, ref_bottom = pix2geo(tr_ref, h_ref, w_ref)
    tgt_left, tgt_top = pix2geo(tr_tgt, 0, 0)
    tgt_right, tgt_bottom = pix2geo(tr_tgt, h_tgt, w_tgt)

    ov_left = max(ref_left, tgt_left)
    ov_right = min(ref_right, tgt_right)
    ov_top = min(ref_top, tgt_top)
    ov_bottom = max(ref_bottom, tgt_bottom)

    if ov_left >= ov_right or ov_top <= ov_bottom:
        return None, None, None, None

    r1_start, c1_start = geo2pix(tr_ref, ov_left, ov_top)
    r1_end, c1_end = geo2pix(tr_ref, ov_right, ov_bottom)
    r2_start, c2_start = geo2pix(tr_tgt, ov_left, ov_top)
    r2_end, c2_end = geo2pix(tr_tgt, ov_right, ov_bottom)

    r1_start = max(0, r1_start); r1_end = min(h_ref, r1_end)
    c1_start = max(0, c1_start); c1_end = min(w_ref, c1_end)
    r2_start = max(0, r2_start); r2_end = min(h_tgt, r2_end)
    c2_start = max(0, c2_start); c2_end = min(w_tgt, c2_end)

    patch_ref = arr_ref[r1_start:r1_end, c1_start:c1_end]
    patch_tgt = arr_tgt[r2_start:r2_end, c2_start:c2_end]

    if patch_ref.size < block_size * block_size:
        return None, None, None, None

    ph, pw = patch_ref.shape
    step = block_size // 2

    grid_ys = []
    grid_xs = []
    grid_shifts_y = []
    grid_shifts_x = []
    grid_confs = []

    for br in range(0, ph - block_size + 1, step):
        for bc in range(0, pw - block_size + 1, step):
            br2 = min(br + block_size, ph)
            bc2 = min(bc + block_size, pw)
            blk_ref = patch_ref[br:br2, bc:bc2]
            blk_tgt = patch_tgt[br:br2, bc:bc2]

            sy, sx, conf = phase_correlation(blk_ref, blk_tgt)
            if conf >= confidence_threshold and abs(sy) < 10 and abs(sx) < 10:
                # Grid point = center of block in target image coords
                gy = r2_start + br + block_size // 2
                gx = c2_start + bc + block_size // 2
                grid_ys.append(gy)
                grid_xs.append(gx)
                grid_shifts_y.append(sy)
                grid_shifts_x.append(sx)
                grid_confs.append(conf)

    if len(grid_ys) < 4:
        return None, None, None, None

    grid_ys = np.array(grid_ys)
    grid_xs = np.array(grid_xs)
    grid_shifts_y = np.array(grid_shifts_y)
    grid_shifts_x = np.array(grid_shifts_x)
    grid_confs = np.array(grid_confs)

    # Outlier rejection: remove points > 3 MAD from median
    for arr in [grid_shifts_y, grid_shifts_x]:
        med = np.median(arr)
        mad = np.median(np.abs(arr - med))
        if mad > 0:
            mask = np.abs(arr - med) < 3 * mad
            grid_ys = grid_ys[mask]
            grid_xs = grid_xs[mask]
            grid_shifts_y = grid_shifts_y[mask]
            grid_shifts_x = grid_shifts_x[mask]
            grid_confs = grid_confs[mask]

    return grid_shifts_y, grid_shifts_x, grid_ys, grid_xs


def warp_by_shift_field(arr, grid_shifts_y, grid_shifts_x, grid_ys, grid_xs):
    """Warp an image using a local shift field via bilinear interpolation.

    For each pixel in the output, computes its corresponding position in the
    source image by interpolating the shift field.

    Parameters
    ----------
    arr : 2D ndarray, source image
    grid_shifts_y, grid_shifts_x : 1D ndarray, shift values at grid points
    grid_ys, grid_xs : 1D ndarray, grid point coordinates in source image

    Returns
    -------
    warped : 2D ndarray, warped image (same shape as arr)
    """
    from scipy.interpolate import griddata

    h, w = arr.shape

    # Create regular output grid
    yy, xx = np.mgrid[0:h, 0:w]

    # Interpolate shift field to full resolution
    points = np.column_stack([grid_ys, grid_xs])
    shift_y_full = griddata(points, grid_shifts_y, (yy, xx), method='cubic', fill_value=0)
    shift_x_full = griddata(points, grid_shifts_x, (yy, xx), method='cubic', fill_value=0)

    # Source coordinates (where to sample from)
    src_y = yy - shift_y_full
    src_x = xx - shift_x_full

    # NOTE: do NOT clip src_x/src_y — let map_coordinates handle out-of-bounds via mode='constant'

    # Bilinear interpolation
    from scipy.ndimage import map_coordinates
    warped = map_coordinates(arr.astype(np.float64),
                            [src_y.ravel(), src_x.ravel()],
                            order=1, mode='constant', cval=0)
    warped = warped.reshape(h, w)

    return warped


def apply_shift(arr, shift_y, shift_x, output_path, transform, crs, nodata=0,
                resampling='linear'):
    """Apply subpixel shift to an image and save as GeoTIFF.

    The shift is applied by modifying the geotransform (header shift),
    consistent with AROSICS global co-registration approach.
    For large shifts (>0.5 pixel), also applies resampling.

    Parameters
    ----------
    arr : 2D or 3D ndarray
    shift_y, shift_x : float, shift in pixel units
    output_path : str
    transform : Affine, original geotransform
    crs : CRS
    nodata : float
    """
    # Apply shift to geotransform (header-level correction)
    new_transform = Affine(
        transform.a, transform.b, transform.c + shift_x * transform.a,
        transform.d, transform.e, transform.f + shift_y * transform.e
    )

    # If shift is significant, also resample the array
    if abs(shift_y) > 0.1 or abs(shift_x) > 0.1:
        if arr.ndim == 2:
            arr = ndimage_shift(arr, [shift_y, shift_x], order=1, mode='constant',
                                cval=nodata)
        elif arr.ndim == 3:
            arr = ndimage_shift(arr, [0, shift_y, shift_x], order=1, mode='constant',
                                cval=nodata)
        # Reset transform since we've resampled
        new_transform = transform

    # Write output
    if arr.ndim == 2:
        arr = arr[np.newaxis, :, :]
        band_count = 1
    else:
        band_count = arr.shape[0]

    profile = {
        'driver': 'GTiff',
        'dtype': arr.dtype.name,
        'width': arr.shape[2],
        'height': arr.shape[1],
        'count': band_count,
        'crs': crs,
        'transform': new_transform,
        'nodata': nodata,
    }

    with rasterio.open(output_path, 'w', **profile) as dst:
        for b in range(band_count):
            dst.write(arr[b], b + 1)

    return output_path


def coregister_pair(ref_path, target_path, output_path, nodata=0):
    """Co-register target image to reference using phase correlation.

    Parameters
    ----------
    ref_path : str, path to reference GeoTIFF
    target_path : str, path to target GeoTIFF to be co-registered
    output_path : str, path for output co-registered GeoTIFF
    nodata : float

    Returns
    -------
    dict with shift_y, shift_x, confidence
    """
    with rasterio.open(ref_path) as src:
        arr_ref = src.read(1) if src.count == 1 else src.read()
        tr_ref = src.transform
        crs_ref = src.crs
        if arr_ref.ndim == 3:
            arr_ref = arr_ref[0]  # Use first band for shift estimation

    with rasterio.open(target_path) as src:
        arr_tgt = src.read(1) if src.count == 1 else src.read()
        tr_tgt = src.transform
        crs_tgt = src.crs
        count = src.count
        if arr_tgt.ndim == 3:
            # Read all bands for output
            arr_tgt_all = src.read()
        else:
            arr_tgt_all = arr_tgt[np.newaxis, :, :]

    # Compute shift
    shift_y, shift_x, conf, block_stats = compute_shifts_from_overlap(
        arr_ref, tr_ref, arr_tgt, tr_tgt, nodata, nodata)

    print(f"  Shift: dy={shift_y:.4f}, dx={shift_x:.4f}, confidence={conf:.4f}")

    # Apply shift to all bands
    if abs(shift_y) > 0.01 or abs(shift_x) > 0.01:
        for b in range(arr_tgt_all.shape[0]):
            arr_tgt_all[b] = ndimage_shift(
                arr_tgt_all[b], [shift_y, shift_x], order=1,
                mode='constant', cval=nodata)

    # Write with original transform (shift applied via resampling)
    profile = {
        'driver': 'GTiff',
        'dtype': arr_tgt_all.dtype.name,
        'width': arr_tgt_all.shape[2],
        'height': arr_tgt_all.shape[1],
        'count': count,
        'crs': crs_tgt,
        'transform': tr_tgt,
        'nodata': nodata,
    }

    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(arr_tgt_all)

    return {'shift_y': shift_y, 'shift_x': shift_x, 'confidence': conf}


def validate_registration_independent_grid(
    arr_ref, tr_ref, arr_registered, tr_registered,
    nodata_ref, nodata_tgt,
    training_points_xy,
    block_size=256, step=256,
    offset_row=128, offset_col=128,
    min_distance_from_training=256,
    confidence_threshold=0.5,
    max_residual_shift=10,
    min_accepted=20,
):
    """严格独立偏移网格验证。

    在配准后的影像上重新执行相位相关，验证块与训练控制点保持最小距离。
    不复用任何训练阶段缓存的 shift。

    Parameters
    ----------
    arr_ref : 2D ndarray, 参考影像
    tr_ref : Affine, 参考影像变换
    arr_registered : 2D ndarray, 配准后的目标影像
    tr_registered : Affine, 配准后目标影像变换
    nodata_ref : float
    nodata_tgt : float
    training_points_xy : ndarray of shape (N, 2), 训练控制点 (x, y)
    block_size : int, 验证块大小
    step : int, 验证网格步长
    offset_row : int, 验证网格行偏移
    offset_col : int, 验证网格列偏移
    min_distance_from_training : float, 最小训练距离（px）
    confidence_threshold : float, 验证块最小置信度
    max_residual_shift : float, 验证块最大允许残余位移（px）
    min_accepted : int, 最少接受验证块数

    Returns
    -------
    dict with keys:
        blocks : list of dict, 每个验证块的详细信息
        stats : dict, 统计指标
        coverage : dict, 空间覆盖信息
        failure_reason : str or None
    """
    from src.overlap import intersection_bounds
    from rasterio.transform import array_bounds, rowcol

    h_ref, w_ref = arr_ref.shape
    h_reg, w_reg = arr_registered.shape

    valid_ref = np.isfinite(arr_ref) & (arr_ref != nodata_ref)
    valid_reg = np.isfinite(arr_registered) & (arr_registered != nodata_tgt)

    struct_ref = structural_image(arr_ref, valid_ref)
    struct_reg = structural_image(arr_registered, valid_reg)

    bounds_ref = array_bounds(h_ref, w_ref, tr_ref)
    bounds_reg = array_bounds(h_reg, w_reg, tr_registered)
    overlap = intersection_bounds(bounds_ref, bounds_reg)
    if overlap is None:
        return {'blocks': [], 'stats': None, 'coverage': None,
                'failure_reason': 'No geographic overlap between images'}

    r0, c0 = rowcol(tr_ref, overlap[0], overlap[3])
    r1, c1 = rowcol(tr_ref, overlap[2], overlap[1])
    r0, r1 = max(0, r0), min(h_ref, r1)
    c0, c1 = max(0, c0), min(w_ref, c1)
    overlap_h, overlap_w = r1 - r0, c1 - c0

    if overlap_h < block_size or overlap_w < block_size:
        return {'blocks': [], 'stats': None, 'coverage': None,
                'failure_reason': f'Overlap too small ({overlap_h}x{overlap_w}) for block_size={block_size}'}

    train_xy = np.array(training_points_xy) if len(training_points_xy) > 0 else np.empty((0, 2))

    actual_min_dist = min_distance_from_training

    blocks = []
    n_total = 0
    n_nodata = 0
    n_texture = 0
    n_conf = 0
    n_near_train = 0
    n_shift_limit = 0

    grid_rows = list(range(r0 + offset_row, r1 - block_size + 1, step))
    grid_cols = list(range(c0 + offset_col, c1 - block_size + 1, step))

    for br in grid_rows:
        for bc in grid_cols:
            br2, bc2 = br + block_size, bc + block_size
            cx, cy = bc + block_size // 2, br + block_size // 2
            n_total += 1

            nearest_dist = float('inf')
            if len(train_xy) > 0:
                dists = np.hypot(train_xy[:, 0] - cx, train_xy[:, 1] - cy)
                nearest_dist = float(np.min(dists))

            if nearest_dist < actual_min_dist:
                n_near_train += 1
                blocks.append({
                    'validation_row': br, 'validation_col': bc,
                    'center_x': cx, 'center_y': cy,
                    'residual_dy': None, 'residual_dx': None,
                    'residual_magnitude': None, 'confidence': None,
                    'valid_ratio': None, 'texture_std': None,
                    'nearest_training_distance': nearest_dist,
                    'accepted': False, 'reject_reason': 'near_training',
                })
                continue

            v_ref = valid_ref[br:br2, bc:bc2]
            v_reg = valid_reg[br:br2, bc:bc2]
            joint = v_ref & v_reg
            valid_ratio = joint.sum() / (block_size * block_size)

            if valid_ratio < 0.3:
                n_nodata += 1
                blocks.append({
                    'validation_row': br, 'validation_col': bc,
                    'center_x': cx, 'center_y': cy,
                    'residual_dy': None, 'residual_dx': None,
                    'residual_magnitude': None, 'confidence': None,
                    'valid_ratio': valid_ratio, 'texture_std': None,
                    'nearest_training_distance': nearest_dist,
                    'accepted': False, 'reject_reason': 'low_valid_ratio',
                })
                continue

            blk_ref = struct_ref[br:br2, bc:bc2]
            blk_reg = struct_reg[br:br2, bc:bc2]
            texture_std = float(np.std(blk_ref[blk_ref > 0])) if np.any(blk_ref > 0) else 0.0

            if texture_std < 1e-4:
                n_texture += 1
                blocks.append({
                    'validation_row': br, 'validation_col': bc,
                    'center_x': cx, 'center_y': cy,
                    'residual_dy': None, 'residual_dx': None,
                    'residual_magnitude': None, 'confidence': None,
                    'valid_ratio': valid_ratio, 'texture_std': texture_std,
                    'nearest_training_distance': nearest_dist,
                    'accepted': False, 'reject_reason': 'low_texture',
                })
                continue

            sy, sx, conf = phase_correlation(
                blk_ref, blk_reg, valid_ref=v_ref, valid_tgt=v_reg)

            if conf < confidence_threshold:
                n_conf += 1
                blocks.append({
                    'validation_row': br, 'validation_col': bc,
                    'center_x': cx, 'center_y': cy,
                    'residual_dy': float(sy), 'residual_dx': float(sx),
                    'residual_magnitude': float(np.hypot(sx, sy)),
                    'confidence': float(conf),
                    'valid_ratio': valid_ratio, 'texture_std': texture_std,
                    'nearest_training_distance': nearest_dist,
                    'accepted': False, 'reject_reason': 'low_confidence',
                })
                continue

            if abs(sy) >= max_residual_shift or abs(sx) >= max_residual_shift:
                n_shift_limit += 1
                blocks.append({
                    'validation_row': br, 'validation_col': bc,
                    'center_x': cx, 'center_y': cy,
                    'residual_dy': float(sy), 'residual_dx': float(sx),
                    'residual_magnitude': float(np.hypot(sx, sy)),
                    'confidence': float(conf),
                    'valid_ratio': valid_ratio, 'texture_std': texture_std,
                    'nearest_training_distance': nearest_dist,
                    'accepted': False, 'reject_reason': 'large_shift',
                })
                continue

            blocks.append({
                'validation_row': br, 'validation_col': bc,
                'center_x': cx, 'center_y': cy,
                'residual_dy': float(sy), 'residual_dx': float(sx),
                'residual_magnitude': float(np.hypot(sx, sy)),
                'confidence': float(conf),
                'valid_ratio': valid_ratio, 'texture_std': texture_std,
                'nearest_training_distance': nearest_dist,
                'accepted': True, 'reject_reason': None,
            })

    accepted = [b for b in blocks if b['accepted']]

    mags = np.array([b['residual_magnitude'] for b in accepted]) if accepted else np.array([])
    confs = np.array([b['confidence'] for b in accepted]) if accepted else np.array([])

    stats = None
    if len(mags) > 0:
        stats = {
            'n_total': n_total,
            'n_accepted': len(accepted),
            'n_rejected_near_training': n_near_train,
            'n_rejected_nodata': n_nodata,
            'n_rejected_texture': n_texture,
            'n_rejected_confidence': n_conf,
            'n_rejected_shift_limit': n_shift_limit,
            'median': float(np.median(mags)),
            'mean': float(np.mean(mags)),
            'rmse': float(np.sqrt(np.mean(mags**2))),
            'p90': float(np.percentile(mags, 90)),
            'p95': float(np.percentile(mags, 95)),
            'max': float(np.max(mags)),
            'mean_confidence': float(np.mean(confs)),
            'min_distance_used': actual_min_dist,
        }

    grid_n_rows = max(1, (overlap_h - block_size) // step + 1)
    grid_n_cols = max(1, (overlap_w - block_size) // step + 1)
    total_cells = grid_n_rows * grid_n_cols
    covered = set()
    for b in accepted:
        gr = (b['validation_row'] - r0) // step
        gc = (b['validation_col'] - c0) // step
        covered.add((min(gr, grid_n_rows - 1), min(gc, grid_n_cols - 1)))
    coverage = {
        'covered_cells': len(covered),
        'total_cells': total_cells,
        'coverage_ratio': len(covered) / max(total_cells, 1),
    }

    failure_reason = None
    if stats is None:
        failure_reason = f'No accepted blocks (total={n_total}, near_train={n_near_train}, nodata={n_nodata}, texture={n_texture}, conf={n_conf})'
    elif stats['n_accepted'] < min_accepted:
        failure_reason = f'Only {stats["n_accepted"]} accepted blocks (need {min_accepted})'

    return {
        'blocks': blocks,
        'stats': stats,
        'coverage': coverage,
        'failure_reason': failure_reason,
    }


def aggregate_final_validation_quality(
    validation_results, params=None, required_edges=None,
):
    """Aggregate residuals from independent validation on final arrays."""
    params = params or {}
    validation_results = list(validation_results or [])
    required_edges = list(required_edges or [])
    residuals = []
    confidences = []
    summary_stats = []
    unavailable_edges = []

    if required_edges:
        validation_by_edge = {}
        for validation in validation_results:
            if not isinstance(validation, dict):
                continue
            idx_i = validation.get("idx_i")
            idx_j = validation.get("idx_j")
            if idx_i is None or idx_j is None:
                continue
            validation_by_edge[tuple(sorted((idx_i, idx_j)))] = validation

        for idx_i, idx_j in required_edges:
            validation = validation_by_edge.get(tuple(sorted((idx_i, idx_j))))
            if validation is None:
                unavailable_edges.append((idx_i, idx_j))
                continue
            has_accepted_block = any(
                block.get("accepted")
                and block.get("residual_magnitude") is not None
                and block.get("confidence") is not None
                for block in (validation.get("blocks") or [])
                if isinstance(block, dict)
            )
            stats = validation.get("stats") or {}
            try:
                has_usable_stats = (
                    int(stats.get("n_accepted", 0)) > 0
                    and all(np.isfinite(float(stats[name])) for name in (
                        "median", "rmse", "p95", "mean_confidence",
                    ))
                )
            except (TypeError, ValueError):
                has_usable_stats = False
            if not (has_accepted_block or has_usable_stats):
                unavailable_edges.append((idx_i, idx_j))

    for validation in validation_results:
        if not isinstance(validation, dict):
            continue
        blocks = validation.get('blocks') or []
        block_residuals = []
        block_confidences = []
        for block in blocks:
            if not block.get('accepted'):
                continue
            residual = block.get('residual_magnitude')
            confidence = block.get('confidence')
            if residual is None or confidence is None:
                continue
            residual = float(residual)
            confidence = float(confidence)
            if np.isfinite(residual) and np.isfinite(confidence):
                block_residuals.append(residual)
                block_confidences.append(confidence)
        if block_residuals:
            residuals.extend(block_residuals)
            confidences.extend(block_confidences)
            continue

        stats = validation.get('stats')
        if not stats:
            continue
        try:
            n_blocks = int(stats['n_accepted'])
            median = float(stats['median'])
            rmse = float(stats['rmse'])
            p95 = float(stats['p95'])
            mean_confidence = float(stats['mean_confidence'])
        except (KeyError, TypeError, ValueError):
            continue
        if n_blocks > 0 and all(np.isfinite(v) for v in (
                median, rmse, p95, mean_confidence)):
            summary_stats.append((n_blocks, median, rmse, p95, mean_confidence))

    if residuals:
        residuals = np.asarray(residuals, dtype=float)
        confidences = np.asarray(confidences, dtype=float)
        stats = {
            'median': float(np.median(residuals)),
            'rmse': float(np.sqrt(np.mean(residuals ** 2))),
            'p95': float(np.percentile(residuals, 95)),
            'mean_confidence': float(np.mean(confidences)),
            'n_accepted': int(len(residuals)),
        }
    elif summary_stats:
        counts = np.asarray([item[0] for item in summary_stats], dtype=float)
        total = int(counts.sum())
        if len(summary_stats) == 1:
            _, median, rmse, p95, mean_confidence = summary_stats[0]
        else:
            # Per-edge summaries cannot recover pooled quantiles. Use the
            # maximum edge quantile as a conservative upper bound instead.
            median = max(item[1] for item in summary_stats)
            rmse = float(np.sqrt(np.average(
                [item[2] ** 2 for item in summary_stats], weights=counts)))
            p95 = max(item[3] for item in summary_stats)
            mean_confidence = float(np.average(
                [item[4] for item in summary_stats], weights=counts))
        stats = {
            'median': float(median),
            'rmse': float(rmse),
            'p95': float(p95),
            'mean_confidence': float(mean_confidence),
            'n_accepted': total,
        }
    else:
        return {
            'quality': 'fail',
            'n_blocks': 0,
            'mean_confidence': 0.0,
            'median': float('inf'),
            'rmse': float('inf'),
            'p95': float('inf'),
            'failure_reason': 'No accepted independent validation blocks',
            'n_validation_results': len(validation_results),
            'unavailable_edges': unavailable_edges,
        }

    quality = classify_registration_quality({
        'status': 'pass',
        'confidence': stats['mean_confidence'],
        'residual_median': stats['median'],
        'residual_rmse': stats['rmse'],
        'residual_p95': stats['p95'],
        'n_inliers': stats['n_accepted'],
    }, params)
    result = {
        'quality': quality,
        'n_blocks': stats['n_accepted'],
        'mean_confidence': stats['mean_confidence'],
        'median': stats['median'],
        'rmse': stats['rmse'],
        'p95': stats['p95'],
        'n_validation_results': len(validation_results),
    }
    if unavailable_edges:
        result['quality'] = 'fail'
        result['unavailable_edges'] = unavailable_edges
        result['failure_reason'] = 'Unavailable required validation edges'
    return result


def compute_displacement_field_stats(dx_field, dy_field):
    """计算位移场统计信息。

    Parameters
    ----------
    dx_field, dy_field : 2D ndarray

    Returns
    -------
    dict with dx, dy, magnitude statistics
    """
    mag = np.hypot(dx_field, dy_field)

    def _stats(arr):
        valid = arr[arr != 0]
        if len(valid) == 0:
            return {'min': 0, 'median': 0, 'p95': 0, 'max': 0, 'std': 0}
        return {
            'min': float(np.min(valid)),
            'median': float(np.median(valid)),
            'p95': float(np.percentile(valid, 95)),
            'max': float(np.max(valid)),
            'std': float(np.std(valid)),
        }

    grad_dx_y, grad_dx_x = np.gradient(dx_field)
    grad_dy_y, grad_dy_x = np.gradient(dy_field)
    grad_mag = np.hypot(grad_dx_x, grad_dx_y) + np.hypot(grad_dy_x, grad_dy_y)
    grad_valid = grad_mag[grad_mag > 0]

    n_nan = int(np.sum(np.isnan(dx_field) | np.isnan(dy_field)))
    n_inf = int(np.sum(np.isinf(dx_field) | np.isinf(dy_field)))
    clip_limit = 2.5
    n_clipped_dx = int(np.sum(np.abs(dx_field) >= clip_limit - 1e-6))
    n_clipped_dy = int(np.sum(np.abs(dy_field) >= clip_limit - 1e-6))

    return {
        'local_dx': _stats(dx_field),
        'local_dy': _stats(dy_field),
        'magnitude': _stats(mag),
        'gradient': {
            'median': float(np.median(grad_valid)) if len(grad_valid) > 0 else 0,
            'p95': float(np.percentile(grad_valid, 95)) if len(grad_valid) > 0 else 0,
            'max': float(np.max(grad_valid)) if len(grad_valid) > 0 else 0,
        },
        'quality_flags': {
            'n_nan': n_nan,
            'n_inf': n_inf,
            'n_clipped_dx': n_clipped_dx,
            'n_clipped_dy': n_clipped_dy,
        },
    }


def generate_joint_stretched_rgb(arr_ref, arr_target, nodata_ref, nodata_tgt,
                                  low_pct=2, high_pct=98):
    """生成联合2%–98%拉伸的红绿叠加RGB。

    Parameters
    ----------
    arr_ref, arr_target : 2D ndarray
    nodata_ref, nodata_tgt : float
    low_pct, high_pct : float, 分位数百分比

    Returns
    -------
    rgb : 3D ndarray (3, H, W), float64, 0-255 range
    overlap_mask : 2D bool ndarray
    """
    h = min(arr_ref.shape[0], arr_target.shape[0])
    w = min(arr_ref.shape[1], arr_target.shape[1])

    ref = arr_ref[:h, :w].astype(np.float64)
    tgt = arr_target[:h, :w].astype(np.float64)

    v_ref = np.isfinite(ref) & (ref != nodata_ref)
    v_tgt = np.isfinite(tgt) & (tgt != nodata_tgt)
    overlap = v_ref & v_tgt

    if overlap.sum() < 100:
        rgb = np.zeros((3, h, w), dtype=np.float64)
        return rgb, overlap

    joint_vals = np.concatenate([ref[overlap], tgt[overlap]])
    p_low, p_high = np.percentile(joint_vals, [low_pct, high_pct])
    scale = max(p_high - p_low, 1e-8)

    ref_norm = np.clip((ref - p_low) / scale, 0, 1)
    tgt_norm = np.clip((tgt - p_low) / scale, 0, 1)

    rgb = np.zeros((3, h, w), dtype=np.float64)
    rgb[0] = ref_norm * 255
    rgb[1] = tgt_norm * 255
    rgb[:, ~overlap] = 0

    return rgb, overlap


def generate_hard_seam_mosaic(arr_ref, arr_target, tr_ref, tr_target,
                               nodata_ref, nodata_tgt, seam_axis='auto'):
    """生成硬切换接缝镶嵌图（无羽化、无距离权重平均）。

    Parameters
    ----------
    arr_ref, arr_target : 2D ndarray
    tr_ref, tr_target : Affine
    nodata_ref, nodata_tgt : float
    seam_axis : str, 'auto', 'row', or 'col'

    Returns
    -------
    mosaic : 2D ndarray (float64)
    seam_pos : int, 接缝位置
    tr_out : Affine, 输出变换
    """
    from src.overlap import intersection_bounds
    from rasterio.transform import array_bounds, rowcol

    h_ref, w_ref = arr_ref.shape
    h_tgt, w_tgt = arr_target.shape

    bounds_ref = array_bounds(h_ref, w_ref, tr_ref)
    bounds_tgt = array_bounds(h_tgt, w_tgt, tr_target)
    overlap = intersection_bounds(bounds_ref, bounds_tgt)
    if overlap is None:
        return None, 0, tr_ref

    r0, c0 = rowcol(tr_ref, overlap[0], overlap[3])
    r1, c1 = rowcol(tr_ref, overlap[2], overlap[1])
    r0, r1 = max(0, r0), min(h_ref, r1)
    c0, c1 = max(0, c0), min(w_ref, c1)

    if seam_axis == 'auto':
        overlap_h, overlap_w = r1 - r0, c1 - c0
        seam_axis = 'col' if overlap_w >= overlap_h else 'row'

    if seam_axis == 'col':
        seam_pos = (c0 + c1) // 2
    else:
        seam_pos = (r0 + r1) // 2

    h_out = max(h_ref, h_tgt)
    w_out = max(w_ref, w_tgt)
    mosaic = np.full((h_out, w_out), np.nan, dtype=np.float64)

    ref_valid = np.isfinite(arr_ref) & (arr_ref != nodata_ref)
    mosaic[:h_ref, :w_ref] = np.where(ref_valid, arr_ref, np.nan)

    tgt_valid = np.isfinite(arr_target) & (arr_target != nodata_tgt)
    if seam_axis == 'col':
        for r in range(h_tgt):
            for c in range(w_tgt):
                if tgt_valid[r, c] and c >= seam_pos:
                    mosaic[r, c] = arr_target[r, c]
                elif tgt_valid[r, c] and c < seam_pos and not np.isfinite(mosaic[r, c]):
                    mosaic[r, c] = arr_target[r, c]
    else:
        for r in range(h_tgt):
            for c in range(w_tgt):
                if tgt_valid[r, c] and r >= seam_pos:
                    mosaic[r, c] = arr_target[r, c]
                elif tgt_valid[r, c] and r < seam_pos and not np.isfinite(mosaic[r, c]):
                    mosaic[r, c] = arr_target[r, c]

    return mosaic, seam_pos, tr_ref


def coregister_sequence(image_paths, reference_idx=0, output_dir=None, nodata=0):
    """Co-register a sequence of images to a common reference.

    Parameters
    ----------
    image_paths : list of str, sorted list of GeoTIFF paths
    reference_idx : int, index of reference image
    output_dir : str, directory for co-registered outputs
    nodata : float

    Returns
    -------
    list of str, paths to co-registered images
    list of dict, shift information for each pair
    """
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(image_paths[0]), 'coregistered')
    os.makedirs(output_dir, exist_ok=True)

    ref_path = image_paths[reference_idx]
    results = []
    output_paths = []

    # Reference image is copied as-is
    ref_out = os.path.join(output_dir, os.path.basename(ref_path))
    if not os.path.exists(ref_out):
        import shutil
        shutil.copy2(ref_path, ref_out)
    output_paths.append(ref_out)

    for i, path in enumerate(image_paths):
        if i == reference_idx:
            results.append({'shift_y': 0, 'shift_x': 0, 'confidence': 1.0})
            continue

        basename = os.path.basename(path)
        out_path = os.path.join(output_dir, basename)
        print(f"Co-registering {basename} -> reference...")

        info = coregister_pair(ref_path, path, out_path, nodata)
        results.append(info)
        output_paths.append(out_path)

    return output_paths, results


def multi_image_network_adjustment(image_pairs, n_images, reference_idx=0):
    """多景全局网络平差：最小二乘求解每景相对于参考影像的全局位移。

    Parameters
    ----------
    image_pairs : list of dict
        每个元素包含: idx_i, idx_j, shift_dx, shift_dy, confidence, n_blocks, rmse
    n_images : int
    reference_idx : int

    Returns
    -------
    dict with global_shifts, pair_results, loop_errors, n_edges, is_tree
    """
    n_pairs = len(image_pairs)
    if n_pairs == 0:
        return {'global_shifts': np.zeros((n_images, 2)),
                'pair_results': [], 'loop_errors': [],
                'n_edges': 0, 'is_tree': True}

    # 检查连通性
    from collections import defaultdict, deque
    adj = defaultdict(set)
    for p in image_pairs:
        adj[p['idx_i']].add(p['idx_j'])
        adj[p['idx_j']].add(p['idx_i'])
    visited = set([reference_idx])
    queue = deque([reference_idx])
    while queue:
        node = queue.popleft()
        for nb in adj[node]:
            if nb not in visited:
                visited.add(nb)
                queue.append(nb)
    if len(visited) < n_images:
        logger.warning("网络不连通: %d/%d 景从参考景可达", len(visited), n_images)

    is_tree = (n_pairs == n_images - 1)

    # 构建最小二乘方程组
    n_vars = n_images * 2
    A_rows = []
    b_rows = []
    weights = []

    for pair in image_pairs:
        i, j = pair['idx_i'], pair['idx_j']
        dx = pair['shift_dx']
        dy = pair['shift_dy']
        # 权重 = confidence * n_blocks / max(rmse, 0.1)
        rmse = max(pair.get('rmse', 1.0), 0.1)
        w = pair.get('confidence', 0.5) * pair.get('n_blocks', 1) / rmse

        row_x = np.zeros(n_vars)
        row_x[i * 2] = -1
        row_x[j * 2] = 1
        A_rows.append(row_x)
        b_rows.append(dx)
        weights.append(w)

        row_y = np.zeros(n_vars)
        row_y[i * 2 + 1] = -1
        row_y[j * 2 + 1] = 1
        A_rows.append(row_y)
        b_rows.append(dy)
        weights.append(w)

    A = np.array(A_rows)
    b = np.array(b_rows)
    W = np.diag(weights)

    # 固定参考影像
    for offset in [0, 1]:
        row = np.zeros(n_vars)
        row[reference_idx * 2 + offset] = 1.0
        A = np.vstack([A, row])
        b = np.append(b, 0.0)
        W_ext = np.zeros((W.shape[0] + 1, W.shape[1] + 1))
        W_ext[:W.shape[0], :W.shape[1]] = W
        W_ext[-1, -1] = 1000.0
        W = W_ext

    AtW = A.T @ W
    AtWA = AtW @ A
    AtWb = AtW @ b

    try:
        x, _, _, _ = np.linalg.lstsq(AtWA, AtWb, rcond=None)
    except np.linalg.LinAlgError:
        x = np.zeros(n_vars)

    global_shifts = x.reshape(n_images, 2)
    global_shifts[reference_idx] = [0.0, 0.0]

    # 计算每对残差
    pair_results = []
    for pair in image_pairs:
        i, j = pair['idx_i'], pair['idx_j']
        pred_dx = global_shifts[j, 0] - global_shifts[i, 0]
        pred_dy = global_shifts[j, 1] - global_shifts[i, 1]
        err_dx = pair['shift_dx'] - pred_dx
        err_dy = pair['shift_dy'] - pred_dy
        pair_results.append({
            'idx_i': i, 'idx_j': j,
            'measured_dx': pair['shift_dx'], 'measured_dy': pair['shift_dy'],
            'predicted_dx': float(pred_dx), 'predicted_dy': float(pred_dy),
            'residual_dx': float(err_dx), 'residual_dy': float(err_dy),
            'residual_magnitude': float(np.hypot(err_dx, err_dy)),
            'confidence': pair.get('confidence', 0),
            'n_blocks': pair.get('n_blocks', 0),
        })

    # 闭环误差检测（使用原始实测位移累计）
    loop_errors = _detect_loop_errors(image_pairs, n_images)

    return {
        'global_shifts': global_shifts,
        'pair_results': pair_results,
        'loop_errors': loop_errors,
        'n_edges': n_pairs,
        'is_tree': is_tree,
    }


def _detect_loop_errors(image_pairs, n_images):
    """闭环误差检测：使用原始实测dx/dy累计，不用global_shifts。"""
    from collections import defaultdict

    # 构建邻接表（使用实测位移）
    adj = defaultdict(list)
    for pair in image_pairs:
        i, j = pair['idx_i'], pair['idx_j']
        dx = pair['shift_dx']
        dy = pair['shift_dy']
        adj[i].append((j, dx, dy))
        adj[j].append((i, -dx, -dy))

    loop_errors = []
    # 找所有简单环（3-4节点）
    for start in range(n_images):
        for length in [3, 4]:
            _dfs_loops(adj, start, length, [start], set([start]), loop_errors)

    # 去重
    seen = set()
    unique = []
    for le in loop_errors:
        key = tuple(sorted(le['nodes']))
        if key not in seen:
            seen.add(key)
            unique.append(le)
    return unique


def _dfs_loops(adj, start, target_len, path, visited, results):
    if len(path) == target_len:
        for neighbor, dx, dy in adj.get(path[-1], []):
            if neighbor == start:
                cum_dx, cum_dy = 0.0, 0.0
                nodes = path[:]
                for k in range(len(nodes)):
                    for nb, ddx, ddy in adj[nodes[k]]:
                        if nb == nodes[(k + 1) % len(nodes)]:
                            cum_dx += ddx
                            cum_dy += ddy
                            break
                results.append({
                    'nodes': nodes + [start],
                    'cumulative_dx': float(cum_dx),
                    'cumulative_dy': float(cum_dy),
                    'error_magnitude': float(np.hypot(cum_dx, cum_dy)),
                })
        return

    for neighbor, dx, dy in adj.get(path[-1], []):
        if neighbor not in visited:
            visited.add(neighbor)
            _dfs_loops(adj, start, target_len, path + [neighbor], visited, results)
            visited.discard(neighbor)


def collect_pair_matches(arr_i, tr_i, arr_j, tr_j, nd_i, nd_j,
                          block_size=512, confidence_threshold=0.5,
                          max_global_shift=40):
    """收集两景之间的块匹配，返回全局平移估计和控制点。

    Parameters
    ----------
    arr_i, arr_j : 2D ndarray
    tr_i, tr_j : Affine
    nd_i, nd_j : float
    block_size : int
    confidence_threshold : float
    max_global_shift : int

    Returns
    -------
    dict with shift_dy, shift_dx, confidence, matches, n_blocks, rmse, screening
    or None if no overlap
    """
    matches, screening = collect_block_matches(
        arr_i, tr_i, arr_j, tr_j, nd_i, nd_j,
        max_global_shift=max_global_shift)
    if not matches:
        return None

    shift_dy, shift_dx, conf, block_stats = compute_shifts_from_overlap(
        arr_i, tr_i, arr_j, tr_j, nd_i, nd_j,
        max_global_shift=max_global_shift)

    if block_stats and not block_stats.get('available', True):
        return {
            'shift_dy': float(shift_dy), 'shift_dx': float(shift_dx),
            'confidence': float(conf), 'matches': matches,
            'n_blocks': len(matches), 'rmse': float('inf'),
            'screening': screening, 'available': False,
            'failure_reason': block_stats.get('failure_reason', 'unknown'),
        }

    # 块级残差（使用 ref_x,ref_y 坐标）
    residuals = []
    for m in matches:
        pred_dx = m['ref_x'] + shift_dx
        pred_dy = m['ref_y'] + shift_dy
        # 实际对应点在ref坐标系中的位置
        actual_x = m['ref_x'] + m['shift_dx']
        actual_y = m['ref_y'] + m['shift_dy']
        err = np.hypot(actual_x - pred_dx, actual_y - pred_dy)
        residuals.append(err)
    residuals = np.array(residuals) if residuals else np.array([0.0])
    rmse = float(np.sqrt(np.mean(residuals**2)))

    return {
        'shift_dy': float(shift_dy),
        'shift_dx': float(shift_dx),
        'confidence': float(conf),
        'matches': matches,
        'n_blocks': len(matches),
        'rmse': rmse,
        'screening': screening,
        'available': True,
    }


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 3:
        print("Usage: python coregistration.py <reference.tif> <target.tif> [output.tif]")
        sys.exit(1)

    ref = sys.argv[1]
    tgt = sys.argv[2]
    out = sys.argv[3] if len(sys.argv) > 3 else tgt.replace('.tif', '_coreg.tif')

    info = coregister_pair(ref, tgt, out)
    print(f"\nResult: {info}")


# ---------------------------------------------------------------------------
# Robust 2-D shift estimation
# ---------------------------------------------------------------------------

def robust_shift_estimate(
    shifts_x: np.ndarray,
    shifts_y: np.ndarray,
    confidences: np.ndarray,
    params: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """
    Robust 2-D translation estimation using MAD-based filtering.
    
    Filters outliers using median absolute deviation, then computes
    weighted mean of inliers.
    """
    if params is None:
        params = {}
    
    mad_scale = params.get("robust_mad_scale", 3.0)
    residual_floor = params.get("robust_residual_floor", 0.75)
    min_inliers = params.get("robust_min_inliers", 5)
    min_inlier_ratio = params.get("robust_min_inlier_ratio", 0.35)
    
    n = len(shifts_x)
    if n == 0:
        return {"shift_x": 0.0, "shift_y": 0.0, "confidence": 0.0, "n_inliers": 0, "status": "fail"}
    
    shifts_x = np.asarray(shifts_x, dtype=np.float64)
    shifts_y = np.asarray(shifts_y, dtype=np.float64)
    confidences = np.asarray(confidences, dtype=np.float64)
    
    # Filter by confidence threshold
    conf_threshold = params.get("global_confidence_threshold", 0.50)
    conf_mask = confidences >= conf_threshold
    
    if conf_mask.sum() < min_inliers:
        return {"shift_x": 0.0, "shift_y": 0.0, "confidence": 0.0, "n_inliers": int(conf_mask.sum()), "status": "fail"}
    
    shifts_x_conf = shifts_x[conf_mask]
    shifts_y_conf = shifts_y[conf_mask]
    confs_conf = confidences[conf_mask]
    
    # Iterative MAD-based outlier rejection
    for iteration in range(5):
        median_x = np.median(shifts_x_conf)
        median_y = np.median(shifts_y_conf)
        
        residuals = np.sqrt((shifts_x_conf - median_x)**2 + (shifts_y_conf - median_y)**2)
        mad = np.median(residuals)
        if mad < 1e-10:
            break
        
        threshold = max(mad_scale * mad, residual_floor)
        inlier_mask = residuals <= threshold
        
        if inlier_mask.sum() < min_inliers:
            break
        
        shifts_x_conf = shifts_x_conf[inlier_mask]
        shifts_y_conf = shifts_y_conf[inlier_mask]
        confs_conf = confs_conf[inlier_mask]
    
    n_inliers = len(shifts_x_conf)
    inlier_ratio = n_inliers / n
    
    if n_inliers < min_inliers or inlier_ratio < min_inlier_ratio:
        return {"shift_x": 0.0, "shift_y": 0.0, "confidence": 0.0, "n_inliers": n_inliers, 
                "inlier_ratio": inlier_ratio, "status": "fail"}
    
    # Weighted mean of inliers
    weights = confs_conf / confs_conf.sum()
    shift_x = float(np.sum(weights * shifts_x_conf))
    shift_y = float(np.sum(weights * shifts_y_conf))
    mean_conf = float(np.mean(confs_conf))
    
    # Compute residual statistics
    residuals_final = np.sqrt((shifts_x_conf - shift_x)**2 + (shifts_y_conf - shift_y)**2)
    
    return {
        "shift_x": shift_x,
        "shift_y": shift_y,
        "confidence": mean_conf,
        "n_inliers": n_inliers,
        "inlier_ratio": inlier_ratio,
        "residual_median": float(np.median(residuals_final)),
        "residual_rmse": float(np.sqrt(np.mean(residuals_final**2))),
        "residual_p95": float(np.percentile(residuals_final, 95)),
        "status": "pass",
    }


def build_robust_pair_measurement(matches, params=None):
    """Build a robust pair shift measurement from block match samples."""
    params = params or {}
    matches = list(matches or [])
    n_total = len(matches)
    screening = params.get("screening", {})
    defaults = {
        "robust_mad_scale": 3.0,
        "robust_residual_floor": 0.75,
        "robust_min_inliers": 5,
        "robust_min_inlier_ratio": 0.35,
    }
    mad_scale = params.get("robust_mad_scale", defaults["robust_mad_scale"])
    residual_floor = params.get("robust_residual_floor", defaults["robust_residual_floor"])
    min_inliers = params.get("robust_min_inliers", defaults["robust_min_inliers"])
    min_ratio = params.get("robust_min_inlier_ratio", defaults["robust_min_inlier_ratio"])

    def failure(reason, n_inlier=0, ratio=0.0):
        return {
            "shift_dx": 0.0, "shift_dy": 0.0, "confidence": 0.0,
            "n_blocks_total": n_total, "n_blocks_inlier": n_inlier,
            "inlier_ratio": ratio, "residual_median": float("inf"),
            "rmse": float("inf"), "p95": float("inf"), "status": "fail",
            "reason": reason, "matches": matches, "screening": screening,
        }

    if not matches:
        return failure("no block matches")

    try:
        dx = np.asarray([m["shift_dx"] for m in matches], dtype=np.float64)
        dy = np.asarray([m["shift_dy"] for m in matches], dtype=np.float64)
        confidence = np.asarray([m["confidence"] for m in matches], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as exc:
        return failure(f"invalid block match: {exc}")

    conf_mask = np.isfinite(dx) & np.isfinite(dy) & np.isfinite(confidence)
    conf_mask &= confidence >= params.get("global_confidence_threshold", 0.50)
    n_conf = int(conf_mask.sum())
    if n_conf < min_inliers:
        return failure(f"too few confident blocks: {n_conf} < {min_inliers}", n_conf, n_conf / n_total)

    # Reject using one joint Euclidean residual for the 2-D shift vector.
    center = np.array([np.median(dx[conf_mask]), np.median(dy[conf_mask])])
    residuals = np.hypot(dx - center[0], dy - center[1])
    mad = float(np.median(residuals[conf_mask]))
    threshold = max(mad_scale * mad, residual_floor)
    inlier_mask = conf_mask & (residuals <= threshold)
    n_inlier = int(inlier_mask.sum())
    ratio = n_inlier / n_total if n_total else 0.0
    if n_inlier < min_inliers or ratio < min_ratio:
        return failure(
            f"too few robust inliers: {n_inlier} < {min_inliers} or ratio {ratio:.3f} < {min_ratio:.3f}",
            n_inlier, ratio)

    weights = confidence[inlier_mask]
    shift_dx = float(np.average(dx[inlier_mask], weights=weights))
    shift_dy = float(np.average(dy[inlier_mask], weights=weights))
    final_residuals = np.hypot(dx[inlier_mask] - shift_dx, dy[inlier_mask] - shift_dy)
    inlier_matches = [m for m, keep in zip(matches, inlier_mask) if keep]
    return {
        "shift_dx": shift_dx, "shift_dy": shift_dy,
        "confidence": float(np.average(weights)),
        "n_blocks_total": n_total, "n_blocks_inlier": n_inlier,
        "inlier_ratio": ratio,
        "residual_median": float(np.median(final_residuals)),
        "rmse": float(np.sqrt(np.mean(final_residuals ** 2))),
        "p95": float(np.percentile(final_residuals, 95)),
        "status": "pass", "matches": inlier_matches, "screening": screening,
    }


def classify_registration_quality(
    result: Dict[str, Any],
    params: Dict[str, Any],
) -> str:
    """
    Classify registration quality as 'pass', 'warn', or 'fail'.
    """
    if result.get("status") == "fail":
        return "fail"
    
    mean_conf = result.get("confidence", 0.0)
    median = result.get("residual_median", 999.0)
    rmse = result.get("residual_rmse", 999.0)
    p95 = result.get("residual_p95", 999.0)
    n_inliers = result.get("n_inliers", 0)
    
    # Check PASS thresholds
    pass_conf = params.get("pass_min_mean_confidence", 0.50)
    pass_median = params.get("pass_max_median", 0.35)
    pass_rmse = params.get("pass_max_rmse", 0.60)
    pass_p95 = params.get("pass_max_p95", 1.00)
    min_blocks = params.get("final_min_blocks", 5)
    
    if (mean_conf >= pass_conf and median <= pass_median and 
        rmse <= pass_rmse and p95 <= pass_p95 and n_inliers >= min_blocks):
        return "pass"
    
    # Check WARN thresholds
    warn_conf = params.get("warn_min_mean_confidence", 0.45)
    warn_median = params.get("warn_max_median", 0.50)
    warn_rmse = params.get("warn_max_rmse", 0.75)
    warn_p95 = params.get("warn_max_p95", 1.25)
    
    if (mean_conf >= warn_conf and median <= warn_median and 
        rmse <= warn_rmse and p95 <= warn_p95 and n_inliers >= min_blocks):
        return "warn"
    
    return "fail"
