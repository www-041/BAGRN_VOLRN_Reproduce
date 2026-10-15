from pathlib import Path

from src.pipeline.artifacts import artifact_identity, stage_is_reusable, write_stage_manifest
from src.pipeline.provenance import canonical_sha256


def test_matching_config_and_upstream_identity_allows_resume(tmp_path: Path):
    stage_dir = tmp_path / "01_overlap"
    output = stage_dir / "overlap.json"
    output.parent.mkdir()
    output.write_text('{"ok": true}', encoding="utf-8")
    upstream = tmp_path / "00_preflight.json"
    upstream.write_text("preflight", encoding="utf-8")
    upstream_id = artifact_identity(upstream)
    write_stage_manifest(
        stage_dir,
        stage_name="01_overlap",
        config_sha256="cfg-a",
        upstream_artifacts={"00_preflight": upstream_id},
        outputs={"overlap": artifact_identity(output)},
        protocol_version="task15-v1",
    )

    assert stage_is_reusable(
        stage_dir,
        expected_config_sha256="cfg-a",
        expected_upstream_artifacts={"00_preflight": upstream_id},
        required_outputs=(output,),
        protocol_version="task15-v1",
    )


def test_config_change_or_partial_output_never_resumes(tmp_path: Path):
    stage_dir = tmp_path / "01_overlap"
    output = stage_dir / "overlap.json"
    stage_dir.mkdir()
    output.write_text("partial", encoding="utf-8")

    assert not stage_is_reusable(
        stage_dir,
        expected_config_sha256=canonical_sha256({"config": "new"}),
        expected_upstream_artifacts={},
        required_outputs=(output,),
        protocol_version="task15-v1",
    )

