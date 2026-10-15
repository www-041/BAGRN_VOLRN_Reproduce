import importlib
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

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


def test_batch_renames_nonempty_partial_output_before_retry(tmp_path):
    output_dir = tmp_path / "mst"
    output_dir.mkdir()
    (output_dir / "partial.tif").write_bytes(b"partial")

    renamed = batch.prepare_output_dir(output_dir)

    assert not output_dir.exists()
    assert renamed.name.startswith("mst_incomplete_")
    assert (renamed / "partial.tif").exists()


def test_protocol_verification_rejects_changed_source_config(tmp_path):
    source = tmp_path / "source.json"
    source.write_text("{}", encoding="utf-8")
    protocol = tmp_path / "protocol.json"
    protocol.write_text(
        json.dumps(
            {
                "dataset": "B9",
                "manifest_indices": [2, 3, 5, 8, 10],
                "pixel_size_m": 14.0,
                "radiometric_normalization": "NONE",
                "source_config": "source.json",
                "source_config_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "global_transform_hashes": [],
            }
        ),
        encoding="utf-8",
    )
    source.write_text('{"changed": true}', encoding="utf-8")

    with pytest.raises(ValueError, match="source config hash mismatch"):
        batch._verify_protocol(protocol, tmp_path)


def test_protocol_verification_rejects_different_source_config_argument(tmp_path):
    frozen = tmp_path / "frozen.json"
    actual = tmp_path / "actual.json"
    frozen.write_text("{}", encoding="utf-8")
    actual.write_text('{"different": true}', encoding="utf-8")
    protocol = tmp_path / "protocol.json"
    protocol.write_text(
        json.dumps(
            {
                "dataset": "B9",
                "manifest_indices": [2, 3, 5, 8, 10],
                "pixel_size_m": 14.0,
                "radiometric_normalization": "NONE",
                "source_config": "frozen.json",
                "source_config_sha256": hashlib.sha256(frozen.read_bytes()).hexdigest(),
                "global_transform_hashes": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="source config hash mismatch"):
        batch._verify_protocol(protocol, tmp_path, actual)
