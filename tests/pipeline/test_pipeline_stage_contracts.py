from pathlib import Path

from src.pipeline.stages import STAGE_NAMES, required_stage_output_names, stage_directory


def test_task15_has_all_frozen_stage_names_and_outputs():
    assert STAGE_NAMES == tuple(f"{index:02d}_{name}" for index, name in enumerate((
        "preflight", "spatial_index_graph", "matching_ransac", "global_adjustment",
        "canonical_warp", "valid_distance_cache", "bagrn", "pairwise_seam_local",
        "multiscene_labeling", "correction", "mosaics", "metrics", "report",
    )))
    assert required_stage_output_names("10_mosaics") == (
        "v0_bagrn_weighted.tif", "v1_multiscene_label_blend.tif", "v2_local_corrected_multiscene.tif",
    )


def test_stage_directory_is_rooted_and_does_not_alias_task14_output():
    root = Path("data/output/final_pipeline/b9_13scene")
    assert stage_directory(root, 0) == root / "stages/00_preflight"
    assert "task14" not in str(stage_directory(root, 12)).lower()
