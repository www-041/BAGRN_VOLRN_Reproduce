import hashlib
import json

import pytest

from scripts import run_b9_radiometric_batch as batch


def test_batch_preserves_nonempty_partial_output(tmp_path):
    output = tmp_path / "RAW"
    output.mkdir()
    (output / "partial.tif").write_bytes(b"partial")

    renamed = batch.prepare_output_dir(output)

    assert not output.exists()
    assert renamed.name.startswith("RAW_incomplete_")
    assert (renamed / "partial.tif").exists()


def test_batch_protocol_checks_frozen_source_config_hash(tmp_path):
    frozen_source = tmp_path / "frozen_source.json"
    frozen_source.write_text("{}", encoding="utf-8")
    source = tmp_path / "source.json"
    source.write_text("{}", encoding="utf-8")
    frozen_transform = tmp_path / "global_transforms.json"
    frozen_transform.write_text("{}", encoding="utf-8")
    protocol = tmp_path / "protocol.json"
    protocol.write_text(
        json.dumps({
            "dataset": "B9",
            "manifest_indices": [2, 3, 5, 8, 10],
            "pixel_size_m": 14.0,
            "source_config": str(frozen_source),
            "source_config_sha256": hashlib.sha256(frozen_source.read_bytes()).hexdigest(),
            "global_transform_hashes": [
                {"matcher": "efficient_loftr", "global_method": "translation_l2", "path": str(frozen_transform), "sha256": hashlib.sha256(frozen_transform.read_bytes()).hexdigest()},
                {"matcher": "sift", "global_method": "mst", "path": str(frozen_transform), "sha256": hashlib.sha256(frozen_transform.read_bytes()).hexdigest()},
            ],
        }),
        encoding="utf-8",
    )
    source.write_text('{"changed": true}', encoding="utf-8")

    with pytest.raises(ValueError, match="source config hash mismatch"):
        batch._verify_protocol(protocol, source, repo_root=tmp_path)
