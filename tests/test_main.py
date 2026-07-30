"""
测试命令行主入口。

创建合成 GeoTIFF → 执行 src.main → 检查输出文件和日志。
"""

import os
import sys
import json
import tempfile
import numpy as np
import rasterio
import pytest

from src.main import parse_args, run_pipeline, _find_tiff_files, _find_image_groups


def _prepare_synthetic_data(tmpdir: str) -> str:
    """
    在 tmpdir/input 下创建 3 张合成 GeoTIFF，模拟条带排列：

    img_0: 6×6, 均值 ~100, 范围 (0,0)-(6,6)
    img_1: 6×6, 均值 ~130, 范围 (3,0)-(9,6) → 与 0 重叠
    img_2: 6×6, 均值 ~85,  范围 (6,0)-(12,6) → 与 1 重叠
    """
    input_dir = os.path.join(tmpdir, "input")
    output_dir = os.path.join(tmpdir, "output")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    np.random.seed(42)
    rows = cols = 6
    res = 1.0
    crs = "EPSG:4326"

    specs = [
        ("img_0.tif", (0, 0), 100.0, 10.0),
        ("img_1.tif", (3, 0), 130.0, 18.0),
        ("img_2.tif", (6, 0), 85.0, 12.0),
    ]

    for fname, (x0, y0), mean, std in specs:
        arr = mean + std * np.random.randn(3, rows, cols)
        arr = arr.astype(np.float32)
        tr = rasterio.Affine(res, 0, x0, 0, -res, y0 + rows)
        profile = {
            "driver": "GTiff",
            "dtype": "float32",
            "count": 3,
            "height": rows,
            "width": cols,
            "transform": tr,
            "crs": crs,
            "compress": "lzw",
        }
        path = os.path.join(input_dir, fname)
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(arr)
    return tmpdir


class TestFindTiffFiles:
    def test_finds_tifs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _prepare_synthetic_data(tmpdir)
            files = _find_tiff_files(os.path.join(tmpdir, "input"))
            assert len(files) == 3
            for f in files:
                assert f.endswith(".tif")


class TestPipelineEndToEnd:
    """端到端流水线测试"""

    @pytest.fixture
    def data_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _prepare_synthetic_data(tmpdir)
            yield tmpdir

    def test_bagrn_only_mode(self, data_dir):
        """--mode bagrn 应正常执行并生成输出文件"""
        input_dir = os.path.join(data_dir, "input")
        output_dir = os.path.join(data_dir, "output")
        args = parse_args([
            "--input_dir", input_dir,
            "--output_dir", output_dir,
            "--control_image", "0",
            "--block_size", "3",
            "--mode", "bagrn",
        ])
        run_pipeline(args)

        # 检查输出文件
        out_files = [f for f in os.listdir(output_dir) if f.endswith(".tif")]
        assert len(out_files) == 3
        # 检查日志
        log_path = os.path.join(output_dir, "run_log.json")
        assert os.path.exists(log_path)
        txt_path = os.path.join(output_dir, "run_log.txt")
        assert os.path.exists(txt_path)

        with open(log_path, "r", encoding="utf-8") as f:
            log = json.load(f)
        assert log["bagrn"]["executed"] is True
        assert log["volrn"]["executed"] is False

    def test_volrn_only_mode(self, data_dir):
        """--mode volrn（跳过 BAGRN，直接对原图做 VOLRN）"""
        input_dir = os.path.join(data_dir, "input")
        output_dir = os.path.join(data_dir, "output")
        args = parse_args([
            "--input_dir", input_dir,
            "--output_dir", output_dir,
            "--control_image", "0",
            "--block_size", "3",
            "--mode", "volrn",
        ])
        run_pipeline(args)

        log_path = os.path.join(output_dir, "run_log.json")
        with open(log_path, "r", encoding="utf-8") as f:
            log = json.load(f)
        assert log["bagrn"]["executed"] is False
        assert log["volrn"]["executed"] is True

    def test_full_pipeline(self, data_dir):
        """--mode bagrn_volrn 完整两阶段"""
        input_dir = os.path.join(data_dir, "input")
        output_dir = os.path.join(data_dir, "output")
        args = parse_args([
            "--input_dir", input_dir,
            "--output_dir", output_dir,
            "--control_image", "0",
            "--block_size", "3",
            "--lambda_param", "0.5",
            "--mode", "bagrn_volrn",
        ])
        run_pipeline(args)

        # 检查输出
        out_files = sorted(f for f in os.listdir(output_dir) if f.endswith(".tif"))
        assert len(out_files) == 3
        # 验证输出 GeoTIFF 可读
        for fname in out_files:
            path = os.path.join(output_dir, fname)
            with rasterio.open(path) as src:
                arr = src.read()
                assert arr.shape[0] == 3  # 3 bands preserved

        # 检查日志内容
        log_path = os.path.join(output_dir, "run_log.json")
        with open(log_path, "r", encoding="utf-8") as f:
            log = json.load(f)
        assert log["bagrn"]["executed"] is True
        assert log["volrn"]["executed"] is True
        assert log["n_images"] == 3
        assert log["n_overlap_pairs"] >= 1
        assert "theta_mu" in log["bagrn"]
        assert "theta_sigma" in log["bagrn"]
        # 检查 overlap_details
        assert len(log["overlap_details"]) == log["n_overlap_pairs"]

    def test_output_shape_preserved(self, data_dir):
        """输出影像的形状与输入一致"""
        input_dir = os.path.join(data_dir, "input")
        output_dir = os.path.join(data_dir, "output")
        args = parse_args([
            "--input_dir", input_dir,
            "--output_dir", output_dir,
            "--control_image", "0",
            "--block_size", "3",
            "--mode", "bagrn_volrn",
            "--max_iter", "30",
        ])
        run_pipeline(args)

        for fname in os.listdir(output_dir):
            if not fname.endswith(".tif"):
                continue
            with rasterio.open(os.path.join(output_dir, fname)) as src:
                assert src.count == 3
                assert src.height == 6
                assert src.width == 6
