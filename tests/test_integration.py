"""
集成测试

验证：
  - 所有 src/ 模块可正常导入（跳过缺少可选依赖的模块）
  - experiment_runner.main() 退出码正确
  - run_multiband 在 mock I/O 下可正常完成
"""

import importlib
import os
import sys
import tempfile
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.experiment_config import make_test_config, save_config


ALL_MODULES = [
    "src.main",
    "src.io_utils",
    "src.gen_report_figures",
    "src.experiment_runner",
    "src.experiment_config",
    "src.diagnostics",
    "src.coregistration",
    "src.comparison",
    "src.bagrn",
    "src.seam_metrics",
    "src.overlap",
    "src.multiband_pipeline",
    "src.mosaic",
    "src.metrics",
    "src.visualization",
    "src.spectral_metrics",
    "src.volrn",
]


def _module_importable(module_name):
    """Check if a module can be imported (ignoring missing optional deps)."""
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False


# ===========================================================================
# test_imports_all_modules
# ===========================================================================

class TestImportsAllModules:
    """导入 src/ 下所有模块，验证无 ImportError。"""

    @pytest.mark.parametrize("module_name", ALL_MODULES,
        ids=[m.split(".")[-1] for m in ALL_MODULES])
    def test_imports_all_modules(self, module_name):
        """每个 src/ 子模块应可成功导入。"""
        try:
            mod = importlib.import_module(module_name)
        except ImportError as e:
            pytest.skip(f"缺少可选依赖: {e}")
        assert mod is not None, f"{module_name} 导入后为 None"


# ===========================================================================
# test_main_returns_exit_code
# ===========================================================================

class TestMainReturnsExitCode:
    """验证 experiment_runner.main() 返回正确的退出码。"""

    def test_main_returns_exit_code(self):
        """mock runner 成功返回时 main() 返回 0，异常时返回 1。"""
        from src.experiment_runner import main

        mock_summary = {
            "experiment_type": "multiband",
            "steps": {},
            "metrics": {},
        }
        mock_runner = MagicMock(return_value=mock_summary)

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = make_test_config(scene_ids=["s1", "s2"], n_bands=2)
            cfg.output_root = os.path.join(tmpdir, "output")
            config_path = os.path.join(tmpdir, "test_config.yaml")
            save_config(cfg, config_path)

            with patch.dict("src.experiment_runner.EXPERIMENT_RUNNERS", {"multiband": mock_runner}), \
                 patch("src.experiment_runner.validate_config", return_value=[]), \
                 patch("src.experiment_runner._setup_logging"):
                rc = main(["--config", config_path, "multiband"])
            assert rc == 0

            mock_runner_fail = MagicMock(side_effect=RuntimeError("fail"))
            with patch.dict("src.experiment_runner.EXPERIMENT_RUNNERS", {"multiband": mock_runner_fail}), \
                 patch("src.experiment_runner.validate_config", return_value=[]), \
                 patch("src.experiment_runner._setup_logging"):
                rc = main(["--config", config_path, "multiband"])
            assert rc == 1


# ===========================================================================
# test_run_multiband_runs_smoke
# ===========================================================================

class TestRunMultibandRunsSmoke:
    """验证 run_multiband 在 mock I/O 下可正常完成。"""

    def test_run_multiband_runs_smoke(self):
        """mock MultibandPipeline.run()，验证 run_multiband 返回 summary。"""
        from src.experiment_runner import run_multiband
        import argparse

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = make_test_config(scene_ids=["s1", "s2"], n_bands=2)
            cfg.output_root = tmpdir
            cfg.experiment_name = "smoke_test"

            args = argparse.Namespace(
                methods=None,
                resume=False,
                dry_run=False,
                smoke=False,
                crop_size=512,
                output_root=None,
            )

            mock_results = {
                "normalized": {
                    "original": [np.zeros((2, 10, 10)), np.zeros((2, 10, 10))],
                },
                "metrics": {
                    "original": {
                        "adm": 0.0, "adsd": 0.0, "cd": 0.0,
                        "gl": 0.0, "rdoa": 0.0, "ave": 0.0,
                    },
                },
                "scene_data": {
                    "transforms": [],
                    "nodata_values": [None, None],
                    "band_names": ["B01", "B02"],
                },
            }

            mock_viz = MagicMock()
            with patch("src.multiband_pipeline.MultibandPipeline") as MockPipe, \
                 patch.dict("sys.modules", {"src.visualization": mock_viz}):
                MockPipe.return_value.run.return_value = mock_results
                summary = run_multiband(cfg, args)

            assert isinstance(summary, dict)
            assert summary["experiment_type"] == "multiband"
