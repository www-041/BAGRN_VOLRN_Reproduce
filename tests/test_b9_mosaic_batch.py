import importlib
import subprocess
import sys
from pathlib import Path

from .test_b9_weighted_mosaic import _synthetic_inputs


batch = importlib.import_module("scripts.run_b9_weighted_mosaic_batch")
runner = importlib.import_module("scripts.run_b9_weighted_mosaic")


def test_batch_output_validation_requires_exact_canonical_grid(tmp_path):
    config, transforms, grid = _synthetic_inputs(tmp_path)
    output_dir = tmp_path / "run"
    runner.run_weighted_mosaic(config, transforms.parent, grid, output_dir)

    assert batch.validate_run_outputs(output_dir, grid, expected_mosaic_dtype="uint16") == {
        "status": "PASS",
        "valid_pixels": 18,
        "mosaic_dtype": "uint16",
    }


def test_batch_script_can_be_invoked_directly():
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(repo_root / "scripts" / "run_b9_weighted_mosaic_batch.py"), "--help"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--source-config" in result.stdout
