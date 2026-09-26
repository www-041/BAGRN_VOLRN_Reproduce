"""Task10C strict protocol and status invariants."""

import hashlib
import json

import pytest

from scripts import run_b9_radiometric_batch as batch
from src.multiscene_sift.radiometric_protocol import (
    SCIENCE_STATUSES,
    build_task10_config,
    validate_task10_config,
)


def test_strict_protocol_rejects_each_frozen_volrn_mutation(tmp_path):
    base = build_task10_config(tmp_path)
    for key, value in {
        "lambda": 0.1,
        "block_size_pixels": 800,
        "rho": 2.0,
        "max_iter": 20,
        "tol": 1e-3,
    }.items():
        mutated = json.loads(json.dumps(base))
        mutated["volrn_params"][key] = value
        with pytest.raises(ValueError, match="volrn_params"):
            validate_task10_config(mutated)


def test_strict_protocol_rejects_control_mutation(tmp_path):
    config = build_task10_config(tmp_path)
    config["radiometric_control_idx"] = 1

    with pytest.raises(ValueError, match="radiometric_control_idx"):
        validate_task10_config(config)


def test_science_status_vocabulary_is_explicit():
    assert {
        "PASS_CONVERGED",
        "PASS_RAW",
        "PASS_BAGRN",
        "COMPLETED_NONCONVERGED",
        "FAILED_PROVENANCE",
        "FAILED_RUNTIME",
    } <= set(SCIENCE_STATUSES)


def test_success_update_removes_stale_error():
    row = {"status": "FAILED_RUNTIME", "error": "old failure"}

    batch._merge_success(row, {"status": "PASS_RAW", "valid_pixels": 4})

    assert row["status"] == "PASS_RAW"
    assert "error" not in row


def test_resume_provenance_checks_source_grid_transform_protocol_and_identity(tmp_path):
    source = tmp_path / "source.json"
    grid = tmp_path / "grid.json"
    transforms = tmp_path / "global_transforms.json"
    protocol = tmp_path / "protocol.json"
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    for path, payload in (
        (source, b"source"),
        (grid, b"grid"),
        (transforms, b"transforms"),
        (protocol, b"protocol"),
    ):
        path.write_bytes(payload)

    (run_dir / "geometry_source.json").write_text(
        json.dumps({
            "source_config_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "output_grid_sha256": hashlib.sha256(grid.read_bytes()).hexdigest(),
            "global_transforms_sha256": hashlib.sha256(transforms.read_bytes()).hexdigest(),
            "geometry_run": "sift_mst",
        }),
        encoding="utf-8",
    )
    (run_dir / "run_config.json").write_text(
        json.dumps({"protocol_sha256": hashlib.sha256(protocol.read_bytes()).hexdigest()}),
        encoding="utf-8",
    )
    (run_dir / "radiometric_method.json").write_text(
        json.dumps({"method": "RAW"}), encoding="utf-8"
    )

    assert batch.resume_provenance_matches(
        run_dir, source, grid, transforms, protocol, "sift_mst", "RAW"
    )
    protocol.write_bytes(b"changed")
    assert not batch.resume_provenance_matches(
        run_dir, source, grid, transforms, protocol, "sift_mst", "RAW"
    )
