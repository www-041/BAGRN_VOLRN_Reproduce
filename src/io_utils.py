"""
GeoTIFF 读写工具模块

提供读取和写入 GeoTIFF 文件的基础函数，
支持单波段和多波段影像，保留地理参考信息。
"""

import numpy as np
import rasterio
from rasterio.profiles import DefaultGTiffProfile
from typing import Tuple, Optional


def read_geotiff(path: str) -> Tuple[np.ndarray, rasterio.Affine, str, Optional[float]]:
    """
    读取 GeoTIFF 影像，返回像素数组、仿射变换、CRS 和 NoData 值。

    参数
    ----------
    path : str
        GeoTIFF 文件路径。

    返回
    -------
    array : np.ndarray
        形状为 (bands, rows, cols) 的像素数组。
    transform : rasterio.Affine
        仿射变换（地理坐标 → 像素坐标）。
    crs : str
        坐标参考系（EPSG 字符串）。
    nodata : float or None
        NoData 值（若存在则返回，否则返回 None）。
    """
    with rasterio.open(path) as src:
        array = src.read()  # 读出所有波段，形状 (bands, rows, cols)
        transform = src.transform
        crs = src.crs.to_string() if src.crs else ""
        nodata = src.nodata
    return array, transform, crs, nodata


def read_geotiff_band(path: str, band_index: int = 1) -> Tuple[np.ndarray, rasterio.Affine, str, Optional[float]]:
    """
    读取 GeoTIFF 的指定波段。

    参数
    ----------
    path : str
        GeoTIFF 文件路径。
    band_index : int
        波段编号，从 1 开始（rasterio 惯例）。

    返回
    -------
    array : np.ndarray
        形状为 (rows, cols) 的单波段像素数组。
    transform : rasterio.Affine
        仿射变换。
    crs : str
        坐标参考系。
    nodata : float or None
        NoData 值。
    """
    with rasterio.open(path) as src:
        array = src.read(band_index)  # 形状 (rows, cols)
        transform = src.transform
        crs = src.crs.to_string() if src.crs else ""
        nodata = src.nodata
    return array, transform, crs, nodata


def write_geotiff(
    path: str,
    array: np.ndarray,
    transform: rasterio.Affine,
    crs: str,
    nodata: Optional[float] = None,
    dtype: Optional[str] = None,
) -> str:
    """
    将像素数组写出为 GeoTIFF 文件。

    参数
    ----------
    path : str
        输出文件路径。
    array : np.ndarray
        形状为 (rows, cols) 或 (bands, rows, cols) 的像素数组。
    transform : rasterio.Affine
        仿射变换。
    crs : str
        坐标参考系（如 "EPSG:4326"）。
    nodata : float or None
        NoData 值。
    dtype : str or None
        输出数据类型（如 "float32"）。若为 None 则自动从 array 推断。

    返回
    -------
    path : str
        写入成功的文件路径。
    """
    if array.ndim == 2:
        bands = 1
        rows, cols = array.shape
        array_out = array[np.newaxis, :, :]  # 增加波段维度
    elif array.ndim == 3:
        bands, rows, cols = array.shape
        array_out = array
    else:
        raise ValueError(f"array 维度必须为 2 或 3，实际为 {array.ndim}")

    if dtype is None:
        dtype = array_out.dtype.name

    profile = DefaultGTiffProfile()
    profile.update(
        dtype=dtype,
        count=bands,
        height=rows,
        width=cols,
        transform=transform,
        crs=crs,
        compress="lzw",
    )
    if nodata is not None:
        profile.update(nodata=nodata)

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array_out)

    return path


def create_synthetic_geotiff(
    path: str,
    rows: int,
    cols: int,
    transform: rasterio.Affine,
    crs: str = "EPSG:4326",
    bands: int = 3,
    nodata: Optional[float] = None,
    dtype: str = "float32",
    fill_value: float = 0.0,
) -> str:
    """
    创建用于测试的合成 GeoTIFF 影像。

    参数
    ----------
    path : str
        输出文件路径。
    rows, cols : int
        像素行数和列数。
    transform : rasterio.Affine
        仿射变换。
    crs : str
        坐标参考系。
    bands : int
        波段数。
    nodata : float or None
        NoData 值。
    dtype : str
        数据类型。
    fill_value : float
        填充像素值。

    返回
    -------
    path : str
        写入成功的文件路径。
    """
    array = np.full((bands, rows, cols), fill_value, dtype=dtype)
    return write_geotiff(path, array, transform, crs, nodata=nodata)
