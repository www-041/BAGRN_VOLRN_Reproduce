"""
BAGRN-VOLRN 命令行入口

用法:
    python -m src.main \\
        --input_dir data/input \\
        --output_dir data/output \\
        --control_image 0 \\
        --block_size 400 \\
        --lambda_param 0.5 \\
        --mode bagrn_volrn

支持模式:
    bagrn        仅执行全局归一化
    volrn        仅执行局部归一化（跳过全局）
    bagrn_volrn  执行完整两阶段流程
"""

import argparse
import glob
import json
import os
import re
import sys
import time
from typing import List, Tuple, Optional

import numpy as np
import rasterio
from tqdm import tqdm

from src.io_utils import read_geotiff, write_geotiff
from src.overlap import get_overlap_window, overlap_pixel_count, intersection_bounds
from src.bagrn import bagrn_normalize
from src.volrn import volrn_normalize
from src.mosaic import create_mosaic
from src.metrics import compute_all as compute_metrics
from src.comparison import run_comparison, METHODS as COMP_METHODS


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def _find_tiff_files(input_dir: str) -> List[str]:
    """递归查找 input_dir 下所有 .tif/.tiff 文件。"""
    patterns = ["**/*.tif", "**/*.tiff", "**/*.TIF", "**/*.TIFF",
                "*.tif", "*.tiff", "*.TIF", "*.TIFF"]
    files = []
    for pat in patterns:
        files.extend(glob.glob(os.path.join(input_dir, pat), recursive=True))
    seen = set()
    unique = []
    for f in files:
        norm = os.path.normcase(os.path.realpath(f))
        if norm not in seen:
            seen.add(norm)
            unique.append(f)
    return sorted(unique)


def _find_image_groups(input_dir: str) -> List[Tuple[str, List[str]]]:
    """
    按子目录分组查找，将每个包含 .tif 文件的子目录视为一个逻辑影像的多波段。

    仅当文件分布在不同深度子目录中时才分组；
    若所有 .tif 文件都直接位于 input_dir 下（扁平结构），返回空列表，
    让上游回退到传统方式（每个文件视为独立影像）。

    处理形如如下结构的真实遥感数据:
      data/input/20251114.../DZ01S_.../B01.TIF
      data/input/20251114.../DZ01S_.../B02.TIF
      ...

    返回 [(group_name, [band_file_paths]), ...]，
    每组内的文件按文件名中的数字排序。
    """
    all_tifs = _find_tiff_files(input_dir)
    if not all_tifs:
        return []

    input_dir_abs = os.path.realpath(input_dir)

    # 判断是否为扁平结构：所有文件都在 input_dir 的直接子目录中
    parents = set(os.path.dirname(f) for f in all_tifs)
    if len(parents) == 1 and list(parents)[0] == input_dir_abs:
        return []

    # 按父目录分组
    groups: dict = {}
    for f in all_tifs:
        parent = os.path.dirname(f)
        if parent not in groups:
            groups[parent] = []
        groups[parent].append(f)

    result = []
    for parent, files in groups.items():
        group_name = os.path.basename(parent)

        def _band_key(x):
            nums = re.findall(r'(\d+)', os.path.basename(x))
            return int(nums[-1]) if nums else 0

        files_sorted = sorted(files, key=_band_key)
        result.append((group_name, files_sorted))

    return sorted(result, key=lambda x: x[0])


def _compute_overlap_stats(array_i, array_j, win_i, win_j, nodata_i, nodata_j, bands):
    """
    计算一对重叠影像在重叠区的 mean 和 std，用于日志记录。

    返回：(mu_i, mu_j, sigma_i, sigma_j) 每个的形状为 (n_bands,)。
    """
    n_bands = len(bands)
    mu_i = np.zeros(n_bands)
    mu_j = np.zeros(n_bands)
    sg_i = np.zeros(n_bands)
    sg_j = np.zeros(n_bands)
    r1s, r1e, c1s, c1e = win_i
    r2s, r2e, c2s, c2e = win_j
    for b_idx, band in enumerate(bands):
        pi = array_i[band, r1s:r1e, c1s:c1e]
        pj = array_j[band, r2s:r2e, c2s:c2e]
        mi = np.isfinite(pi) if nodata_i is None else (pi != nodata_i)
        mj = np.isfinite(pj) if nodata_j is None else (pj != nodata_j)
        # 两幅影像分辨率可能不同（如 S=30m, V=14m），
        # 重叠窗口像素数不同，分开计算统计量
        if mi.sum() > 0:
            mu_i[b_idx] = float(pi[mi].mean())
            sg_i[b_idx] = float(pi[mi].std())
        if mj.sum() > 0:
            mu_j[b_idx] = float(pj[mj].mean())
            sg_j[b_idx] = float(pj[mj].std())
    return mu_i, mu_j, sg_i, sg_j


# ---------------------------------------------------------------------------
# 日志写入
# ---------------------------------------------------------------------------

def write_log(log_path: str, info: dict):
    """将运行日志写入 JSON 文件。"""
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2, default=str)


# ---------------------------------------------------------------------------
# 调试裁剪
# ---------------------------------------------------------------------------

def _debug_crop_arrays(arrays, transforms, bounds_list, nodata_list, filenames, crop_size):
    """
    调试模式下，将每幅影像裁剪至重叠区域附近的 crop_size × crop_size 窗口。

    1. 找所有影像的公共地理交集
    2. 若不存在公共交集，找第一对有重叠的影像对，以其交集为基准
    3. 在基准地理范围中取 crop_size × crop_size 的方形窗口
    4. 将窗口反向投影到每幅影像的像素坐标，裁剪数组并更新 transform
    """
    n = len(arrays)
    if n < 2:
        # 只有一幅影像，取中心裁剪
        arr = arrays[0]
        tr = transforms[0]
        rows, cols = arr.shape[1], arr.shape[2]
        r_s = max(0, (rows - crop_size) // 2)
        c_s = max(0, (cols - crop_size) // 2)
        r_e = min(rows, r_s + crop_size)
        c_e = min(cols, c_s + crop_size)
        arrays[0] = arr[:, r_s:r_e, c_s:c_e]
        transforms[0] = rasterio.Affine(tr.a, tr.b, tr.c + c_s * tr.a,
                                         tr.d, tr.e, tr.f + r_s * tr.e)
        bounds_list[0] = (transforms[0].c,
                          transforms[0].f + transforms[0].e * (r_e - r_s),
                          transforms[0].c + transforms[0].a * (c_e - c_s),
                          transforms[0].f)
        print(f"  [调试] 裁剪 {filenames[0]} 至中心 {crop_size}×{crop_size}")
        return arrays, transforms, bounds_list

    # 先找所有影像的公共交集
    common = intersection_bounds(bounds_list[0], bounds_list[1])
    if common is None:
        print("  [调试] 前两幅影像无公共交集，尝试其他组合...")
        for i in range(n):
            for j in range(i + 1, n):
                common = intersection_bounds(bounds_list[i], bounds_list[j])
                if common is not None:
                    break
            if common is not None:
                break
    if common is not None:
        for k in range(2, n):
            inter = intersection_bounds(common, bounds_list[k])
            if inter is not None:
                common = inter
    else:
        print("  [调试] 所有影像对均无重叠，使用各影像中心裁剪")
        for idx in range(n):
            arr = arrays[idx]
            tr = transforms[idx]
            rows, cols = arr.shape[1], arr.shape[2]
            r_s = max(0, (rows - crop_size) // 2)
            c_s = max(0, (cols - crop_size) // 2)
            r_e = min(rows, r_s + crop_size)
            c_e = min(cols, c_s + crop_size)
            arrays[idx] = arr[:, r_s:r_e, c_s:c_e]
            transforms[idx] = rasterio.Affine(tr.a, tr.b, tr.c + c_s * tr.a,
                                              tr.d, tr.e, tr.f + r_s * tr.e)
            bounds_list[idx] = (transforms[idx].c,
                                transforms[idx].f + transforms[idx].e * (r_e - r_s),
                                transforms[idx].c + transforms[idx].a * (c_e - c_s),
                                transforms[idx].f)
            print(f"  [调试] 裁剪 {filenames[idx]} 至中心 {crop_size}×{crop_size}")
        return arrays, transforms, bounds_list

    common_l, common_b, common_r, common_t = common
    cw = common_r - common_l
    ch = common_t - common_b

    # 在公共区域中取 crop_size 像素对应地理尺寸
    # 用第一幅影像的分辨率估算
    res_x = abs(transforms[0].a)
    res_y = abs(transforms[0].e)
    geo_size_x = crop_size * res_x
    geo_size_y = crop_size * res_y

    if geo_size_x > cw or geo_size_y > ch:
        print(f"  [调试] crop_size({crop_size}) 大于重叠区，使用重叠区本身")
        geo_size_x = cw
        geo_size_y = ch

    # 在公共重叠区中居中取方形窗口
    crop_left   = common_l + (cw - geo_size_x) / 2
    crop_bottom = common_b + (ch - geo_size_y) / 2
    crop_right  = crop_left + geo_size_x
    crop_top    = crop_bottom + geo_size_y

    print(f"  [调试] 裁剪地理范围: ({crop_left:.1f}, {crop_bottom:.1f}, {crop_right:.1f}, {crop_top:.1f})")

    for idx in range(n):
        arr = arrays[idx]
        tr = transforms[idx]
        rows, cols = arr.shape[1], arr.shape[2]
        # 将裁剪范围投影到该影像的像素坐标
        c_min, r_min = ~tr * (crop_left, crop_top)
        c_max, r_max = ~tr * (crop_right, crop_bottom)
        r_s = max(0, int(round(r_min)))
        r_e = min(rows, int(round(r_max)))
        c_s = max(0, int(round(c_min)))
        c_e = min(cols, int(round(c_max)))
        if r_e <= r_s or c_e <= c_s:
            print(f"  [调试] 警告: {filenames[idx]} 在裁剪区无有效像素，使用中心裁剪")
            r_s = max(0, (rows - crop_size) // 2)
            r_e = min(rows, r_s + crop_size)
            c_s = max(0, (cols - crop_size) // 2)
            c_e = min(cols, c_s + crop_size)
        arrays[idx] = arr[:, r_s:r_e, c_s:c_e]
        transforms[idx] = rasterio.Affine(tr.a, tr.b, tr.c + c_s * tr.a,
                                          tr.d, tr.e, tr.f + r_s * tr.e)
        bounds_list[idx] = (transforms[idx].c,
                            transforms[idx].f + transforms[idx].e * (r_e - r_s),
                            transforms[idx].c + transforms[idx].a * (c_e - c_s),
                            transforms[idx].f)
        print(f"  [调试] 裁剪 {filenames[idx]} 至 {arrays[idx].shape[1]}×{arrays[idx].shape[2]}")

    return arrays, transforms, bounds_list


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def run_pipeline(args):
    """执行 BAGRN-VOLRN 完整流程。"""
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Step 0: 查找并读取输入影像 ----
    # 先尝试按子目录分组（处理单波段文件分属不同目录的真实数据）
    image_groups = _find_image_groups(args.input_dir)

    if not image_groups:
        # 回退到传统方式：每个文件视为一个独立影像（可能多波段）
        tiff_paths = _find_tiff_files(args.input_dir)
        if not tiff_paths:
            print(f"错误: 在 {args.input_dir} 中未找到 .tif/.tiff 文件")
            sys.exit(1)
        print(f"找到 {len(tiff_paths)} 幅影像:")
        for p in tiff_paths:
            print(f"  {os.path.basename(p)}")
        image_groups = [(os.path.basename(p), [p]) for p in tiff_paths]
    else:
        print(f"找到 {len(image_groups)} 组影像:")
        for gname, flist in image_groups:
            print(f"  {gname}: {len(flist)} 波段")

    # 传感器过滤
    if hasattr(args, 'sensor_filter') and args.sensor_filter:
        sensor_f = args.sensor_filter.upper()
        image_groups = [(g, f) for g, f in image_groups if sensor_f in g.upper()]
        print(f"  --sensor_filter={args.sensor_filter}，筛选后 {len(image_groups)} 组影像:")
        for gname, flist in image_groups:
            print(f"    {gname}: {len(flist)} 波段")

    # 限制最大影像数
    if args.max_images and args.max_images < len(image_groups):
        print(f"  --max_images={args.max_images}，仅处理前 {args.max_images} 幅")
        image_groups = image_groups[:args.max_images]

    arrays = []
    transforms = []
    crs_list = []
    nodata_list = []
    bounds_list = []
    filenames = []

    for group_name, file_list in tqdm(image_groups, desc="读取影像"):
        band_records = []  # (shape, array, transform, crs, nodata)
        for fpath in file_list:
            arr, tr, crs, nd = read_geotiff(fpath)
            for b_idx in range(arr.shape[0]):
                sh = arr[b_idx].shape
                band_records.append((sh, arr[b_idx], tr, crs, nd))
        if not band_records:
            continue
        # 取最常见的 shape（以排除 PAN 等不同分辨率的波段）
        shapes_counter = {}
        for sh, _, _, _, _ in band_records:
            shapes_counter[sh] = shapes_counter.get(sh, 0) + 1
        majority_shape = max(shapes_counter, key=shapes_counter.get)
        filtered = [(arr, tr, crs, nd) for sh, arr, tr, crs, nd in band_records
                    if sh == majority_shape]
        if len(filtered) < 1:
            print(f"  跳过 {group_name}: 有效波段不足")
            continue
        # 堆叠为 (n_bands, rows, cols)，使用第一幅有效波段的元数据
        stacked = np.stack([f[0] for f in filtered], axis=0)
        arrays.append(stacked)
        transforms.append(filtered[0][1])
        crs_list.append(filtered[0][2])
        nodata_list.append(filtered[0][3])
        rows = stacked.shape[1]
        cols = stacked.shape[2]
        bounds_list.append((filtered[0][1].c,
                            filtered[0][1].f + filtered[0][1].e * rows,
                            filtered[0][1].c + filtered[0][1].a * cols,
                            filtered[0][1].f))
        filenames.append(group_name)

    n_images = len(arrays)
    if n_images == 0:
        print("错误: 未能读取任何有效影像")
        sys.exit(1)

    n_bands = arrays[0].shape[0]
    bands = list(range(n_bands))

    print(f"")
    print(f"========================================")
    print(f"  影像总数: {n_images}")
    print(f"  波段数:   {n_bands}")
    for idx in range(n_images):
        sh = arrays[idx].shape
        print(f"    [{idx}] {filenames[idx]}  {sh[1]}×{sh[2]}×{sh[0]}  nodata={nodata_list[idx]}")
    print(f"========================================")
    print(f"")

    # ---- Step 0.5: 调试裁剪 ----
    if args.debug and args.debug_crop_size:
        print("调试模式: 裁剪影像至重叠区域...")
        t0 = time.time()
        arrays, transforms, bounds_list = _debug_crop_arrays(
            arrays, transforms, bounds_list, nodata_list, filenames,
            args.debug_crop_size,
        )
        print(f"  裁剪完成，耗时 {time.time() - t0:.2f}s")
        print(f"  裁剪后尺寸:")
        for idx in range(n_images):
            sh = arrays[idx].shape
            print(f"    [{idx}] {filenames[idx]}  {sh[1]}×{sh[2]}")
        print(f"")

    # ---- Step 1: 计算所有重叠对 ----
    print("检测重叠关系...")
    overlaps = []
    overlap_details = []
    no_overlap_pairs = []

    for i in range(n_images):
        for j in range(i + 1, n_images):
            windows = get_overlap_window(
                bounds_list[i], transforms[i],
                bounds_list[j], transforms[j],
            )
            if windows is None:
                no_overlap_pairs.append((filenames[i], filenames[j]))
                continue
            win_i, win_j = windows
            n_pixels = (win_i[1] - win_i[0]) * (win_i[3] - win_i[2])
            if n_pixels == 0:
                no_overlap_pairs.append((filenames[i], filenames[j]))
                continue

            mu_i, mu_j, sg_i, sg_j = _compute_overlap_stats(
                arrays[i], arrays[j], win_i, win_j,
                nodata_list[i], nodata_list[j], bands,
            )

            overlaps.append({
                "idx_i": i,
                "idx_j": j,
                "window_i": win_i,
                "window_j": win_j,
            })
            overlap_details.append({
                "image_i": filenames[i],
                "image_j": filenames[j],
                "pixel_count": n_pixels,
                "window_i": list(win_i),
                "window_j": list(win_j),
                "mu_i": mu_i.tolist(),
                "mu_j": mu_j.tolist(),
                "sigma_i": sg_i.tolist(),
                "sigma_j": sg_j.tolist(),
            })

    if no_overlap_pairs:
        print("  以下影像对无重叠:")
        for ni, nj in no_overlap_pairs:
            print(f"    {ni} 与 {nj} 无重叠")

    if not overlaps:
        print("错误: 未检测到任何重叠的影像对，无法进行辐射归一化")
        sys.exit(1)

    print(f"  共 {len(overlaps)} 对重叠影像")

    # ---- Eval: baseline ----
    metrics_data = {}
    if args.eval or args.compare:
        print("\n计算基准指标 (baseline)...")
        t0 = time.time()
        metrics_data["baseline"] = _eval_and_print("baseline", arrays, arrays, nodata_list, overlaps, bands)
        print(f"  基准指标耗时 {time.time() - t0:.2f}s\n")

    # ---- Eval: 对比方法 ----
    if args.compare:
        comp_names = [s.strip() for s in args.compare.split(",") if s.strip() in COMP_METHODS]
        if not comp_names:
            print(f"  未找到有效对比方法，可用: {list(COMP_METHODS.keys())}")
        for cname in comp_names:
            print(f"  运行对比方法: {cname}...")
            t0 = time.time()
            try:
                comp_result = run_comparison(cname, arrays, nodata_list, overlaps, control_idx=args.control_image)
                metrics_data[cname] = _eval_and_print(cname, comp_result, arrays, nodata_list, overlaps, bands)
                print(f"    {cname} 耗时 {time.time() - t0:.2f}s")
            except Exception as e:
                print(f"    {cname} 执行失败: {type(e).__name__}: {e}")

    # ---- Step 2: BAGRN ----
    normalized = [arr.astype(np.float64, copy=True) for arr in arrays]
    theta_mu = None
    theta_sigma = None

    if args.mode in ("bagrn", "bagrn_volrn"):
        print("执行 BAGRN 全局辐射归一化...")
        t0 = time.time()
        normalized, theta_mu, theta_sigma = bagrn_normalize(
            arrays, nodata_list, overlaps,
            control_idx=args.control_image,
        )
        t_bagrn = time.time() - t0
        print(f"  BAGRN 完成，耗时 {t_bagrn:.2f}s")
        if args.eval or args.compare:
            t0 = time.time()
            metrics_data["bagrn"] = _eval_and_print("BAGRN", normalized, arrays, nodata_list, overlaps, bands)
            print(f"  BAGRN 指标耗时 {time.time() - t0:.2f}s")
    else:
        t_bagrn = 0.0

    # ---- Step 3: VOLRN ----
    volrn_coeffs = None
    volrn_error = None

    if args.mode in ("volrn", "bagrn_volrn"):
        print("执行 VOLRN 局部辐射归一化...")
        t0 = time.time()
        try:
            normalized, volrn_coeffs = volrn_normalize(
                normalized, transforms, bounds_list, nodata_list,
                block_size_pixels=args.block_size,
                lambda_param=args.lambda_param,
                rho=args.rho,
                max_iter=args.max_iter,
                tol=args.tol,
                verbose=args.verbose,
            )
            t_volrn = time.time() - t0
            print(f"  VOLRN 完成，耗时 {t_volrn:.2f}s")
            if args.eval or args.compare:
                t0 = time.time()
                metrics_data["volrn"] = _eval_and_print("VOLRN", normalized, arrays, nodata_list, overlaps, bands)
                print(f"  VOLRN 指标耗时 {time.time() - t0:.2f}s")
        except Exception as e:
            t_volrn = time.time() - t0
            volrn_error = f"{type(e).__name__}: {e}"
            print(f"  VOLRN 执行失败: {volrn_error}")
            print(f"  已回退至 BAGRN 结果")
    else:
        t_volrn = 0.0

    if len(metrics_data) > 1:
        _print_metrics_table(metrics_data)

    # ---- Step 4: 写出归一化后的 GeoTIFF ----
    print("写出归一化结果...")
    output_paths = []
    for idx, arr in tqdm(enumerate(normalized), desc="写出", total=len(normalized)):
        out_name = f"{filenames[idx]}_normalized.tif"
        out_path = os.path.join(args.output_dir, out_name)
        write_geotiff(
            out_path, arr,
            transforms[idx], crs_list[idx],
            nodata=nodata_list[idx],
        )
        output_paths.append(out_path)

    # ---- Step 5: 镶嵌 ----
    mosaic_path = None
    if args.mosaic:
        print("生成镶嵌图...")
        t0 = time.time()
        try:
            mosaic_path = create_mosaic(
                normalized, transforms, crs_list[0], nodata_list,
                args.mosaic,
                resolution=args.mosaic_resolution,
            )
            print(f"  镶嵌完成，耗时 {time.time() - t0:.2f}s: {mosaic_path}")
        except Exception as e:
            print(f"  镶嵌失败: {type(e).__name__}: {e}")

    # ---- Step 6: 写入日志 ----
    log_info = {
        "command": " ".join(sys.argv),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "parameters": {
            "input_dir": args.input_dir,
            "output_dir": args.output_dir,
            "control_image": args.control_image,
            "block_size_pixels": args.block_size,
            "lambda": args.lambda_param,
            "rho": args.rho,
            "max_iter": args.max_iter,
            "tol": args.tol,
            "mode": args.mode,
            "max_images": args.max_images,
            "debug": args.debug,
            "debug_crop_size": args.debug_crop_size,
            "mosaic": args.mosaic,
            "mosaic_resolution": args.mosaic_resolution,
            "eval": args.eval,
            "compare": args.compare,
        },
        "input_files": filenames,
        "n_images": n_images,
        "n_bands": n_bands,
        "n_overlap_pairs": len(overlaps),
        "no_overlap_pairs": [f"{a} 与 {b}" for a, b in no_overlap_pairs],
        "overlap_details": overlap_details,
        "bagrn": {
            "executed": args.mode in ("bagrn", "bagrn_volrn"),
            "time_seconds": round(t_bagrn, 2),
        },
        "volrn": {
            "executed": args.mode in ("volrn", "bagrn_volrn"),
            "time_seconds": round(t_volrn, 2),
            "n_blocks": volrn_coeffs.shape[0] if volrn_coeffs is not None and (hasattr(volrn_coeffs, 'size') and volrn_coeffs.size > 0) else 0,
        },
        "output_files": output_paths,
        "mosaic": mosaic_path,
        "metrics": metrics_data if args.eval else None,
    }

    if volrn_error is not None:
        log_info["volrn"]["error"] = volrn_error

    # BAGRN 系数（每个影像每个波段的 θ_μ、θ_σ）
    if theta_mu is not None and theta_mu.size > 0:
        log_info["bagrn"]["theta_mu"] = {
            filenames[i]: theta_mu[:, i].tolist() for i in range(n_images)
        }
    if theta_sigma is not None and theta_sigma.size > 0:
        log_info["bagrn"]["theta_sigma"] = {
            filenames[i]: theta_sigma[:, i].tolist() for i in range(n_images)
        }

    # 写入 JSON 日志
    log_path = os.path.join(args.output_dir, "run_log.json")
    write_log(log_path, log_info)

    # 写入 TXT 日志
    _write_run_log_txt(args.output_dir, log_info)

    print(f"日志已保存至 {log_path}")
    print("完成!")


def _eval_and_print(label: str, arrays, raw_arrays, nodata_list, overlaps, bands):
    """计算并打印评价指标。"""
    from src.metrics import compute_adm, compute_adsd, compute_cd, compute_gl, compute_rdoa, compute_ave
    adm = compute_adm(arrays, nodata_list, overlaps, bands)
    adsd = compute_adsd(arrays, nodata_list, overlaps, bands)
    cd = compute_cd(arrays, nodata_list, overlaps, bands)
    gl = compute_gl(raw_arrays, arrays, nodata_list, bands)
    rdoa = compute_rdoa(adm, adsd, cd)
    ave = compute_ave(adm, adsd, cd, gl)
    print(f"  [{label}] ADM={adm:.6f}  ADSD={adsd:.6f}  CD={cd:.6f}  GL={gl:.6f}  RDOA={rdoa:.6f}  Ave={ave:.6f}")
    return {"adm": adm, "adsd": adsd, "cd": cd, "gl": gl, "rdoa": rdoa, "ave": ave}


def _print_metrics_table(metrics_dict: dict):
    """打印评价指标对比表（支持任意列数）。"""
    labels = list(metrics_dict.keys())
    col_width = 16
    header_fmt = f"{{:<20}}" + f"{{:<{col_width}}}" * len(labels)
    row_fmt = f"{{:<20}}" + f"{{:<{col_width}}}" * len(labels)
    print()
    print("=" * (20 + col_width * len(labels)))
    print(header_fmt.format("指标", *labels))
    print("-" * (20 + col_width * len(labels)))
    for key in ["adm", "adsd", "cd", "gl", "rdoa", "ave"]:
        vals = [f"{metrics_dict[st][key]:.6f}" for st in labels]
        print(row_fmt.format(key.upper(), *vals))
    print("=" * (20 + col_width * len(labels)))
    print(f"  {'→ 指标下降说明归一化有效':>50}")
    print()


def _write_run_log_txt(output_dir: str, info: dict):
    """将运行日志写入纯文本文件 run_log.txt。"""
    lines = [
        "=" * 60,
        "BAGRN-VOLRN 辐射归一化 运行日志",
        "=" * 60,
        f"命令: {info.get('command', '')}",
        f"时间: {info.get('timestamp', '')}",
        "",
        "--- 参数 ---",
    ]
    for k, v in info.get("parameters", {}).items():
        lines.append(f"  {k}: {v}")

    lines.extend([
        "",
        f"--- 输入 ---",
        f"  影像数: {info.get('n_images', 0)}",
        f"  波段数: {info.get('n_bands', 0)}",
        f"  文件:",
    ])
    for f in info.get("input_files", []):
        lines.append(f"    {f}")

    lines.extend([
        "",
        f"--- 重叠关系 ---",
        f"  重叠对数: {info.get('n_overlap_pairs', 0)}",
    ])
    for pair in info.get("no_overlap_pairs", []):
        lines.append(f"  无重叠: {pair}")

    for od in info.get("overlap_details", []):
        lines.append(f"  {od['image_i']} <-> {od['image_j']}: {od['pixel_count']} 像素")
        lines.append(f"    窗口 i: {od['window_i']}")
        lines.append(f"    窗口 j: {od['window_j']}")
        lines.append(f"    mu_i: {od['mu_i']}, mu_j: {od['mu_j']}")

    lines.extend([
        "",
        f"--- BAGRN ---",
        f"  执行: {info.get('bagrn', {}).get('executed', False)}",
        f"  耗时: {info.get('bagrn', {}).get('time_seconds', 0)}s",
    ])
    if "theta_mu" in info.get("bagrn", {}):
        lines.append(f"  θ_μ:")
        for img, vals in info["bagrn"]["theta_mu"].items():
            lines.append(f"    {img}: {vals}")
    if "theta_sigma" in info.get("bagrn", {}):
        lines.append(f"  θ_σ:")
        for img, vals in info["bagrn"]["theta_sigma"].items():
            lines.append(f"    {img}: {vals}")

    lines.extend([
        "",
        f"--- VOLRN ---",
        f"  执行: {info.get('volrn', {}).get('executed', False)}",
        f"  耗时: {info.get('volrn', {}).get('time_seconds', 0)}s",
        f"  Block 数: {info.get('volrn', {}).get('n_blocks', 0)}",
    ])
    if "error" in info.get("volrn", {}):
        lines.append(f"  错误: {info['volrn']['error']}")

    lines.extend([
        "",
        f"--- 输出 ---",
    ])
    for p in info.get("output_files", []):
        lines.append(f"  {p}")

    lines.append("=" * 60)

    txt_path = os.path.join(output_dir, "run_log.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="BAGRN-VOLRN 辐射归一化 — 论文复现",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            "  python -m src.main --input_dir data/input --output_dir data/output\n"
            "  python -m src.main --input_dir data/input --output_dir data/output --mode bagrn\n"
            "  python -m src.main --input_dir data/input --output_dir data/output --block_size 200 --lambda_param 0.7\n"
        ),
    )
    parser.add_argument(
        "--input_dir", type=str, required=True,
        help="输入 GeoTIFF 影像所在目录",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="输出目录",
    )
    parser.add_argument(
        "--control_image", type=int, default=0,
        help="控制影像索引（默认为 0，补偿值为 0）",
    )
    parser.add_argument(
        "--block_size", type=int, default=400,
        help="VOLRN 分块大小（像素），论文推荐: 30m→200, 16m→400, 1m→600",
    )
    parser.add_argument(
        "--lambda", "--lambda_param", dest="lambda_param", type=float, default=0.5,
        help="VOLRN 正则化参数 λ（论文推荐 0.5）",
    )
    parser.add_argument(
        "--rho", type=float, default=1.0,
        help="ADMM 惩罚参数 ρ",
    )
    parser.add_argument(
        "--max_iter", type=int, default=200,
        help="ADMM 最大迭代次数",
    )
    parser.add_argument(
        "--tol", type=float, default=1e-4,
        help="ADMM 收敛容差",
    )
    parser.add_argument(
        "--mode", type=str, default="bagrn_volrn",
        choices=["bagrn", "volrn", "bagrn_volrn"],
        help="运行模式: bagrn（仅全局）, volrn（仅局部）, bagrn_volrn（两阶段，默认）",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="打印 ADMM 收敛信息",
    )
    parser.add_argument(
        "--max_images", type=int, default=None,
        help="最多读取 N 幅影像（调试用）",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="启用调试模式（裁剪影像、打印详细信息）",
    )
    parser.add_argument(
        "--debug_crop_size", type=int, default=0,
        help="调试模式下每幅影像裁剪至 N×N 像素（如 1024）",
    )
    parser.add_argument(
        "--mosaic", type=str, default=None,
        help="是否生成镶嵌图。指定输出路径（如 data/output/mosaic.tif），默认为 None 不生成",
    )
    parser.add_argument(
        "--mosaic_resolution", type=float, default=None,
        help="镶嵌图分辨率（地理单位/像素）。默认为 None（自动选用最精细分辨率）",
    )
    parser.add_argument(
        "--eval", action="store_true",
        help="输出归一化前后评价指标对比（ADM/ADSD/CD/GL/RDOA/Ave）",
    )
    parser.add_argument(
        "--compare", type=str, default=None,
        help="对比方法名称（逗号分隔），如 histogram_matching,moment_matching,wallis",
    )
    parser.add_argument(
        "--sensor_filter", type=str, default=None,
        help="仅处理包含指定字符串的影像组（如 DZ01S 或 DZ01V），用于传感器对比实验",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args)
