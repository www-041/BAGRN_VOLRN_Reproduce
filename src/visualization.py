"""
可视化模块

生成 RGB 和假彩色快视图，用于 BAGRN-VOLRN 项目的可视化输出。
支持多影像联合拉伸、接缝线放大、方法对比裁剪等功能。
"""

import os
import numpy as np
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple

from src.io_utils import read_geotiff


# ---------------------------------------------------------------------------
# 联合拉伸参数计算
# ---------------------------------------------------------------------------

def compute_joint_stretch_params(
    arrays_list: List[np.ndarray],
    band_indices: Tuple[int, int, int],
    low_pct: float = 2.0,
    high_pct: float = 98.0,
) -> dict:
    """
    从多幅影像中计算联合拉伸参数。

    遍历所有影像的指定波段，取全局低百分位和高百分位作为拉伸范围，
    保证同一组影像使用一致的拉伸，便于视觉对比。

    参数
    ----------
    arrays_list : list of np.ndarray
        影像列表，每幅形状为 (bands, rows, cols)。
    band_indices : tuple of int
        用于 RGB 显示的三个波段索引 (R, G, B)。
    low_pct : float
        低百分位（默认 2%）。
    high_pct : float
        高百分位（默认 98%）。

    返回
    -------
    dict
        包含 vmin, vmax, band_indices 的拉伸参数字典。
    """
    all_valid_pixels = []

    for arr in arrays_list:
        for b in band_indices:
            if b >= arr.shape[0]:
                continue
            band_data = arr[b].ravel()
            valid = np.isfinite(band_data)
            all_valid_pixels.append(band_data[valid])

    if not all_valid_pixels:
        return {"vmin": 0.0, "vmax": 1.0, "band_indices": band_indices}

    concatenated = np.concatenate(all_valid_pixels)
    vmin = float(np.percentile(concatenated, low_pct))
    vmax = float(np.percentile(concatenated, high_pct))

    if vmax - vmin < 1e-12:
        vmax = vmin + 1.0

    return {"vmin": vmin, "vmax": vmax, "band_indices": band_indices}


# ---------------------------------------------------------------------------
# 应用拉伸生成 RGB 数组
# ---------------------------------------------------------------------------

def apply_stretch_to_rgb(
    array: np.ndarray,
    band_indices: Tuple[int, int, int],
    stretch_params: dict,
    nodata: Optional[float] = None,
) -> np.ndarray:
    """
    将单波段拉伸参数应用到影像，生成 uint8 RGB 数组。

    处理 NoData：无效像素（非有限值或显式指定的 nodata 值）设为黑色 (0,0,0)。
    线性拉伸至 [0, 255]。

    参数
    ----------
    array : np.ndarray
        形状为 (bands, rows, cols) 的影像数组。
    band_indices : tuple of int
        RGB 三通道对应的波段索引。
    stretch_params : dict
        由 compute_joint_stretch_params 返回的拉伸参数。
    nodata : float or None
        显式 NoData 值。仅当提供时才将等于此值的像素视为无效。
        不再默认将 0 视为 NoData。

    返回
    -------
    np.ndarray
        形状为 (3, H, W) 的 uint8 RGB 数组，值域 [0, 255]。
    """
    vmin = stretch_params["vmin"]
    vmax = stretch_params["vmax"]
    bands_out = []

    for b in band_indices:
        if b >= array.shape[0]:
            bands_out.append(np.zeros(array.shape[1:], dtype=np.uint8))
            continue

        band_data = array[b].astype(np.float64)

        valid_mask = np.isfinite(band_data)
        if nodata is not None:
            valid_mask &= (band_data != nodata)

        stretched = np.clip((band_data - vmin) / (vmax - vmin) * 255.0, 0, 255)
        stretched = np.round(stretched).astype(np.uint8)

        stretched[~valid_mask] = 0
        bands_out.append(stretched)

    rgb = np.stack(bands_out, axis=0)
    return rgb


# ---------------------------------------------------------------------------
# 保存 RGB 数组为 PNG
# ---------------------------------------------------------------------------

def generate_quicklook(
    rgb_array: np.ndarray,
    output_path: str,
    title: Optional[str] = None,
) -> str:
    """
    将 RGB 数组保存为 PNG 快视图。

    参数
    ----------
    rgb_array : np.ndarray
        形状为 (3, H, W) 的 uint8 RGB 数组。
    output_path : str
        输出 PNG 文件路径。
    title : str or None
        可选标题，显示在图片上方。

    返回
    -------
    str
        保存的 PNG 文件路径。
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # (3, H, W) -> (H, W, 3)
    img = np.moveaxis(rgb_array, 0, -1)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(img)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# 批量生成快视图（自然彩色 + 假彩色）
# ---------------------------------------------------------------------------

def generate_rgb_quicklooks(
    arrays_dict: Dict[int, np.ndarray],
    transforms: list,
    nodata_values: list,
    rgb_bands: Tuple[int, int, int] = (2, 1, 0),
    false_color_bands: Tuple[int, int, int] = (3, 2, 1),
    output_dir: str = ".",
    prefix: str = "",
    method_name: str = "",
    shared_rgb_stretch: Optional[dict] = None,
    shared_fc_stretch: Optional[dict] = None,
) -> List[str]:
    """
    为一组影像同时生成自然彩色和假彩色快视图。

    使用联合拉伸保证同一方法下所有影像的色彩一致。
    若提供 shared_rgb_stretch / shared_fc_stretch，则使用外部传入的拉伸参数，
    而不是在本函数内重新计算，确保跨方法色彩一致。

    参数
    ----------
    arrays_dict : dict
        影像索引到 3D 数组的映射 {idx: (bands, rows, cols)}。
    transforms : list
        仿射变换列表（未直接使用，保留接口一致性）。
    nodata_values : list
        NoData 值列表（未直接使用，保留接口一致性）。
    rgb_bands : tuple of int
        自然彩色波段索引 (R, G, B)。
    false_color_bands : tuple of int
        假彩色波段索引 (R, G, B)。
    output_dir : str
        输出目录。
    prefix : str
        文件名前缀。
    method_name : str
        方法名称（如 "baseline"、"BAGRN" 等），用于标题和文件名。
    shared_rgb_stretch : dict or None
        外部提供的自然彩色拉伸参数。若为 None 则在函数内计算。
    shared_fc_stretch : dict or None
        外部提供的假彩色拉伸参数。若为 None 则在函数内计算。

    返回
    -------
    list of str
        所有输出 PNG 文件路径。
    """
    os.makedirs(output_dir, exist_ok=True)
    output_paths = []

    all_arrays = list(arrays_dict.values())

    if shared_rgb_stretch is not None:
        rgb_stretch = shared_rgb_stretch
    else:
        rgb_stretch = compute_joint_stretch_params(all_arrays, rgb_bands)

    if shared_fc_stretch is not None:
        fc_stretch = shared_fc_stretch
    else:
        fc_stretch = compute_joint_stretch_params(all_arrays, false_color_bands)

    for idx, arr in arrays_dict.items():
        rgb = apply_stretch_to_rgb(arr, rgb_bands, rgb_stretch)
        fname = f"{prefix}_{method_name}_img{idx}_rgb.png" if prefix else f"{method_name}_img{idx}_rgb.png"
        path = os.path.join(output_dir, fname)
        generate_quicklook(rgb, path, title=f"{method_name} Image {idx} (RGB)")
        output_paths.append(path)

        fc = apply_stretch_to_rgb(arr, false_color_bands, fc_stretch)
        fname_fc = f"{prefix}_{method_name}_img{idx}_falsecolor.png" if prefix else f"{method_name}_img{idx}_falsecolor.png"
        path_fc = os.path.join(output_dir, fname_fc)
        generate_quicklook(fc, path_fc, title=f"{method_name} Image {idx} (False Color)")
        output_paths.append(path_fc)

    return output_paths


# ---------------------------------------------------------------------------
# 镶嵌图快视图
# ---------------------------------------------------------------------------

def generate_mosaic_quicklook(
    mosaic_path: str,
    rgb_bands: Tuple[int, int, int] = (2, 1, 0),
    false_color_bands: Tuple[int, int, int] = (3, 2, 1),
    output_dir: str = ".",
    prefix: str = "",
) -> List[str]:
    """
    为镶嵌结果生成 RGB 和假彩色快视图。

    参数
    ----------
    mosaic_path : str
        镶嵌 GeoTIFF 文件路径。
    rgb_bands : tuple of int
        自然彩色波段索引。
    false_color_bands : tuple of int
        假彩色波段索引。
    output_dir : str
        输出目录。
    prefix : str
        文件名前缀。

    返回
    -------
    list of str
        输出 PNG 文件路径列表。
    """
    arr, _, _, _ = read_geotiff(mosaic_path)
    basename = os.path.splitext(os.path.basename(mosaic_path))[0]
    output_paths = []

    # 自然彩色
    rgb_stretch = compute_joint_stretch_params([arr], rgb_bands)
    rgb = apply_stretch_to_rgb(arr, rgb_bands, rgb_stretch)
    rgb_name = f"{prefix}_{basename}_rgb.png" if prefix else f"{basename}_rgb.png"
    rgb_path = os.path.join(output_dir, rgb_name)
    generate_quicklook(rgb, rgb_path, title=f"Mosaic {basename} (RGB)")
    output_paths.append(rgb_path)

    # 假彩色
    fc_stretch = compute_joint_stretch_params([arr], false_color_bands)
    fc = apply_stretch_to_rgb(arr, false_color_bands, fc_stretch)
    fc_name = f"{prefix}_{basename}_falsecolor.png" if prefix else f"{basename}_falsecolor.png"
    fc_path = os.path.join(output_dir, fc_name)
    generate_quicklook(fc, fc_path, title=f"Mosaic {basename} (False Color)")
    output_paths.append(fc_path)

    return output_paths


# ---------------------------------------------------------------------------
# 方法对比裁剪
# ---------------------------------------------------------------------------

def generate_comparison_crop(
    arrays_dict: Dict[str, np.ndarray],
    transforms: list,
    nodata_values: list,
    rgb_bands: Tuple[int, int, int],
    stretch_params: dict,
    crop_geo_bounds: Tuple[float, float, float, float],
    output_path: str,
) -> str:
    """
    生成多方法对比的裁剪图。

    将多幅影像的同一地理区域裁剪出来，横向排列对比显示。
    使用统一的拉伸参数保证色彩一致。

    参数
    ----------
    arrays_dict : dict
        方法名称到 3D 数组的映射 {"method_name": (bands, rows, cols)}。
    transforms : list
        仿射变换列表（与 arrays_dict 中影像一一对应）。
    nodata_values : list
        NoData 值列表。
    rgb_bands : tuple of int
        显示波段索引。
    stretch_params : dict
        联合拉伸参数。
    crop_geo_bounds : tuple of float
        裁剪地理范围 (left, bottom, right, top)。
    output_path : str
        输出 PNG 文件路径。

    返回
    -------
    str
        保存的 PNG 文件路径。
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    crop_left, crop_bottom, crop_right, crop_top = crop_geo_bounds
    methods = list(arrays_dict.keys())
    n_methods = len(methods)

    fig, axes = plt.subplots(1, n_methods, figsize=(6 * n_methods, 6))
    if n_methods == 1:
        axes = [axes]

    for ax, method_name in zip(axes, methods):
        arr = arrays_dict[method_name]
        tr = transforms[method_name] if isinstance(transforms, dict) else transforms[methods.index(method_name)]

        # 地理范围 -> 像素范围
        c_min, r_min = ~tr * (crop_left, crop_top)
        c_max, r_max = ~tr * (crop_right, crop_bottom)
        r_s = max(0, int(round(r_min)))
        r_e = min(arr.shape[1], int(round(r_max)))
        c_s = max(0, int(round(c_min)))
        c_e = min(arr.shape[2], int(round(c_max)))

        if r_e <= r_s or c_e <= c_s:
            ax.set_title(f"{method_name}\n(no data in crop)")
            ax.axis("off")
            continue

        cropped = arr[:, r_s:r_e, c_s:c_e]
        rgb = apply_stretch_to_rgb(cropped, rgb_bands, stretch_params)
        img = np.moveaxis(rgb, 0, -1)
        ax.imshow(img)
        ax.set_title(method_name, fontsize=11)
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# 接缝线放大图
# ---------------------------------------------------------------------------

def generate_seam_crop(
    mosaic_array: np.ndarray,
    transform: rasterio.Affine,
    seam_location: Tuple[int, int],
    output_path: str,
    width: int = 200,
) -> str:
    """
    生成接缝线区域的放大视图。

    参数
    ----------
    mosaic_array : np.ndarray
        镶嵌影像数组，形状为 (bands, rows, cols)。
    transform : rasterio.Affine
        仿射变换。
    seam_location : tuple of int
        接缝位置 (row, col)，作为裁剪中心。
    output_path : str
        输出 PNG 文件路径。
    width : int
        裁剪半宽度（像素），默认 200。

    返回
    -------
    str
        保存的 PNG 文件路径。
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    center_r, center_c = seam_location
    bands, rows, cols = mosaic_array.shape

    r_s = max(0, center_r - width)
    r_e = min(rows, center_r + width)
    c_s = max(0, center_c - width)
    c_e = min(cols, center_c + width)

    cropped = mosaic_array[:, r_s:r_e, c_s:c_e]

    # 使用前三个波段作为 RGB
    n_bands = min(3, cropped.shape[0])
    rgb_bands = tuple(range(n_bands))
    stretch = compute_joint_stretch_params([cropped], rgb_bands)
    rgb = apply_stretch_to_rgb(cropped, rgb_bands, stretch)
    img = np.moveaxis(rgb, 0, -1)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(img)

    # 在中心位置画一条垂直红色线标示接缝
    local_center_r = center_r - r_s
    ax.axvline(x=width, color="red", linewidth=1.5, linestyle="--", alpha=0.8)
    ax.set_title(f"Seam Area ({r_s}:{r_e}, {c_s}:{c_e})", fontsize=10)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    return output_path
