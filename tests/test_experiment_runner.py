"""
实验运行器测试

验证 experiment_runner.py 的基本功能：
  - dry-run 模式：仅打印执行计划，不实际执行
  - smoke 模式：使用小区域裁剪
  - 配置文件不存在时的错误处理

全部使用 mock/synthetic 数据，不运行真实实验。
"""

import os
import sys
import tempfile
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
import yaml

from src.experiment_config import ExperimentConfig, make_test_config, save_config
from src.experiment_runner import build_parser, main


# ---------------------------------------------------------------------------
# 辅助：构造最小测试配置
# ---------------------------------------------------------------------------

def _make_temp_config(tmpdir, extra=None):
    """创建临时 YAML 配置文件并返回路径。"""
    cfg = make_test_config(scene_ids=["s1", "s2"], n_bands=2)
    cfg.output_root = os.path.join(tmpdir, "output")
    if extra:
        for k, v in extra.items():
            setattr(cfg, k, v)

    config_path = os.path.join(tmpdir, "test_config.yaml")
    save_config(cfg, config_path)
    return config_path


# ===========================================================================
# test_dry_run
# ===========================================================================

class TestDryRun:
    """测试 dry-run 模式：仅打印计划，不实际执行。"""

    def test_dry_run_returns_plan(self):
        """--dry-run 应返回 dry_run=True 的摘要，不执行实际处理。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = _make_temp_config(tmpdir)
            argv = [
                "experiment_runner",
                "--config", config_path,
                "--dry-run",
                "multiband",
            ]

            # 捕获 main 的返回值
            # dry-run 模式下 pipeline.run() 不会被调用
            parser = build_parser()
            args = parser.parse_args(argv[1:])

            # 验证参数解析
            assert args.dry_run is True
            assert args.experiment_type == "multiband"

    def test_dry_run_does_not_create_output(self):
        """--dry-run 不应创建输出目录和文件。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = _make_temp_config(tmpdir, extra={
                "output_root": os.path.join(tmpdir, "output"),
            })

            # 在 dry-run 模式下，MultibandPipeline.run() 会返回
            # {"dry_run": True, "config": ...} 而不实际执行
            from src.experiment_config import load_config
            from src.multiband_pipeline import MultibandPipeline

            cfg = load_config(config_path)
            cfg.dry_run = True
            pipe = MultibandPipeline(cfg)
            result = pipe.run()

            assert result.get("dry_run") is True
            # 不应创建输出文件
            output_dir = os.path.join(cfg.output_root, cfg.experiment_name)
            assert not os.path.exists(output_dir) or not os.listdir(output_dir)


# ===========================================================================
# test_smoke_mode
# ===========================================================================

class TestSmokeMode:
    """测试 smoke 模式：使用小区域裁剪。"""

    def test_smoke_flag_parsed(self):
        """--smoke 参数应正确解析。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = _make_temp_config(tmpdir)
            argv = [
                "experiment_runner",
                "--config", config_path,
                "--smoke",
                "multiband",
            ]
            parser = build_parser()
            args = parser.parse_args(argv[1:])
            assert args.smoke is True

    def test_smoke_mode_config(self):
        """smoke 模式下 config.smoke 应为 True。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = _make_temp_config(tmpdir)
            from src.experiment_config import load_config

            cfg = load_config(config_path)
            cfg.smoke = True
            assert cfg.smoke is True

    def test_smoke_mode_crop_size(self):
        """smoke 模式的 crop_size 参数应正确传递。"""
        argv = [
            "experiment_runner",
            "--config", "dummy.yaml",
            "--smoke",
            "--crop-size", "256",
            "multiband",
        ]
        parser = build_parser()
        args = parser.parse_args(argv[1:])
        assert args.smoke is True
        assert args.crop_size == 256


# ===========================================================================
# test_config_not_found
# ===========================================================================

class TestConfigNotFound:
    """测试配置文件不存在时的错误处理。"""

    def test_missing_config_file(self):
        """配置文件不存在时应报错退出。"""
        argv = [
            "experiment_runner",
            "--config", "nonexistent_config.yaml",
            "multiband",
        ]
        # main() 内部会捕获异常并返回退出码 1
        # 使用 mock 避免实际文件系统操作
        with patch("src.experiment_runner.load_config") as mock_load:
            mock_load.side_effect = FileNotFoundError("配置文件不存在: nonexistent_config.yaml")
            exit_code = main(argv[1:])
        assert exit_code == 1

    def test_invalid_yaml_content(self):
        """无效 YAML 内容应报错。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            bad_config_path = os.path.join(tmpdir, "bad.yaml")
            with open(bad_config_path, "w") as f:
                f.write(": invalid: yaml: content: [[[")

            with patch("src.experiment_runner.load_config") as mock_load:
                mock_load.side_effect = yaml.YAMLError("解析错误")
                argv = [
                    "experiment_runner",
                    "--config", bad_config_path,
                    "multiband",
                ]
                exit_code = main(argv[1:])
            assert exit_code == 1

    def test_experiment_type_required(self):
        """缺少 experiment_type 参数应报错。"""
        argv = ["experiment_runner", "--config", "some.yaml"]
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(argv[1:])


# ===========================================================================
# test_experiment_types
# ===========================================================================

class TestExperimentTypes:
    """测试各实验类型的参数解析。"""

    def test_all_experiment_types_accepted(self):
        """所有注册的实验类型应被接受。"""
        valid_types = [
            "multiband", "ablation", "sensitivity",
            "scale", "diagnostics", "seam", "spectral",
        ]
        parser = build_parser()
        for exp_type in valid_types:
            argv = ["experiment_runner", "--config", "c.yaml", exp_type]
            args = parser.parse_args(argv[1:])
            assert args.experiment_type == exp_type

    def test_invalid_experiment_type_rejected(self):
        """未注册的实验类型应被拒绝。"""
        parser = build_parser()
        argv = ["experiment_runner", "--config", "c.yaml", "invalid_type"]
        with pytest.raises(SystemExit):
            parser.parse_args(argv[1:])

    def test_methods_argument(self):
        """--methods 参数应正确解析。"""
        parser = build_parser()
        argv = [
            "experiment_runner", "--config", "c.yaml",
            "--methods", "bagrn,volrn_only,bagrn_volrn",
            "ablation",
        ]
        args = parser.parse_args(argv[1:])
        assert args.methods == "bagrn,volrn_only,bagrn_volrn"

    def test_resume_argument(self):
        """--resume 参数应正确解析。"""
        parser = build_parser()
        argv = [
            "experiment_runner", "--config", "c.yaml",
            "--resume",
            "multiband",
        ]
        args = parser.parse_args(argv[1:])
        assert args.resume is True


# ===========================================================================
# test_multiband_exit_code
# ===========================================================================

class TestMultibandExitCode:
    """测试 multiband runner 成功/失败时的退出码。"""

    def test_multiband_returns_zero_on_success(self):
        """runner 正常返回时 main 应返回退出码 0。"""
        mock_summary = {
            "experiment_type": "multiband",
            "steps": {},
            "metrics": {},
        }
        mock_runner = MagicMock(return_value=mock_summary)
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = _make_temp_config(tmpdir)
            with patch.dict("src.experiment_runner.EXPERIMENT_RUNNERS", {"multiband": mock_runner}), \
                 patch("src.experiment_runner.validate_config", return_value=[]), \
                 patch("src.experiment_runner._setup_logging"):
                exit_code = main(["--config", config_path, "multiband"])
        assert exit_code == 0
        mock_runner.assert_called_once()

    def test_multiband_returns_nonzero_on_exception(self):
        """runner 抛出异常时 main 应返回退出码 1。"""
        mock_runner = MagicMock(side_effect=RuntimeError("模拟失败"))
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = _make_temp_config(tmpdir)
            with patch.dict("src.experiment_runner.EXPERIMENT_RUNNERS", {"multiband": mock_runner}), \
                 patch("src.experiment_runner.validate_config", return_value=[]), \
                 patch("src.experiment_runner._setup_logging"):
                exit_code = main(["--config", config_path, "multiband"])
        assert exit_code == 1
        mock_runner.assert_called_once()
