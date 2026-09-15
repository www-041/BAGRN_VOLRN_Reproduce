import shutil
import tempfile
from pathlib import Path

import numpy as np
from rasterio.transform import Affine

from scripts import diagnose_two_image_registration as diagnostic_script


def test_default_cli_parameters():
    args = diagnostic_script.parse_args(["--band", "B14"])

    assert args.band == "B14"
    assert args.block_size == 512
    assert args.confidence_threshold == 0.5
    assert args.max_shift == 40.0


def test_main_writes_diagnostic_json_and_csv_with_monkeypatched_reader(monkeypatch):
    yy, xx = np.mgrid[0:512, 0:512]
    arr = np.sin(xx / 15.0) + np.cos(yy / 19.0)
    transform = Affine(14.0, 0.0, 1000.0, 0.0, -14.0, 5000.0)
    repo_root = Path(__file__).resolve().parents[1]
    output_parent = repo_root / "data" / "output"
    output_dir = Path(tempfile.mkdtemp(
        prefix="registration-diagnostics-test-", dir=output_parent))

    def fake_read(path):
        return arr.copy(), transform, "EPSG:4326", 0.0

    monkeypatch.setattr(diagnostic_script, "read_geotiff", fake_read)
    real_makedirs = diagnostic_script.os.makedirs

    def fake_makedirs(path, exist_ok=False):
        if str(path).endswith("chips"):
            return None
        return real_makedirs(path, exist_ok=exist_ok)

    monkeypatch.setattr(diagnostic_script.os, "makedirs", fake_makedirs)
    monkeypatch.setattr(
        diagnostic_script, "save_common_grid_chips", lambda *args, **kwargs: 0)
    try:
        result = diagnostic_script.main([
            "--band", "B14",
            "--ref", "ref.tif",
            "--tgt", "tgt.tif",
            "--output", str(output_dir),
        ])

        assert result == 0
        for name in (
            "grid_metadata.json",
            "raw_grid_candidates.csv",
            "raw_grid_summary.json",
            "common_grid_candidates.csv",
            "common_grid_summary.json",
            "grid_comparison.json",
        ):
            assert (output_dir / name).exists()
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)
