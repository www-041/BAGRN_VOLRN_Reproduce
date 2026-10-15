"""RED tests for the Task10D runner summary separation."""

import hashlib
import json
from pathlib import Path

from src.multiscene_sift.radiometric_runner import run_fixed_geometry_radiometric
from tests.multiscene_sift.test_fixed_radiometric_runner import _inputs


PROTOCOL = Path("data/output/b9_five_scene_validation/task10d_metric_protocol.json")


def test_runner_emits_task10d_primary_schema_and_protocol_hash(tmp_path):
    source, global_dir, grid = _inputs(tmp_path)

    result = run_fixed_geometry_radiometric(
        source,
        global_dir,
        grid,
        tmp_path / "run",
        method="RAW",
        geometry_run="sift_mst",
        protocol_path=PROTOCOL,
    )

    expected_hash = hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
    assert result["paper_metrics"] == {
        "status": "UNVERIFIED",
        "cd": None,
        "gl": None,
    }
    assert result["task10d_metric_protocol_sha256"] == expected_hash
    assert set(result["task10d_primary_metrics"]) == {
        "mamd", "msdd", "rdd", "local_mamd", "local_rdd",
        "seam_mae", "seam_rmse", "seam_rdd", "cgl_rad",
    }
    assert result["task10d_primary_metrics"]["cgl_rad"]["value_rad"] == 0.0
    assert result["diagnostics"]["histogram_tv_distance"] is not None

    summary = json.loads((tmp_path / "run" / "radiometric_summary.json").read_text())
    assert summary["paper_metrics"]["status"] == "UNVERIFIED"
    assert summary["task10d_metric_protocol_sha256"] == expected_hash
