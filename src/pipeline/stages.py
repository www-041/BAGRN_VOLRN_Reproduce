"""Task15 stage names and small output-contract helpers.

The computational implementations remain in the established registration,
seam/local, BAGRN, and mosaic modules.  This module only defines the stable
orchestration contract shared by the runner and tests.
"""

from __future__ import annotations

from pathlib import Path


STAGE_NAMES = (
    "00_preflight",
    "01_spatial_index_graph",
    "02_matching_ransac",
    "03_global_adjustment",
    "04_canonical_warp",
    "05_valid_distance_cache",
    "06_bagrn",
    "07_pairwise_seam_local",
    "08_multiscene_labeling",
    "09_correction",
    "10_mosaics",
    "11_metrics",
    "12_report",
)


_REQUIRED_OUTPUTS = {
    "08_multiscene_labeling": (
        "v1/source_label_map.tif",
        "v1/label_method_map.tif",
        "v1/score_margin.tif",
        "v1/coverage_count.tif",
        "v1/labeling_summary.json",
        "v2/source_label_map.tif",
        "v2/label_method_map.tif",
        "v2/score_margin.tif",
        "v2/coverage_count.tif",
        "v2/labeling_summary.json",
    ),
    "10_mosaics": (
        "v0_bagrn_weighted.tif",
        "v1_multiscene_label_blend.tif",
        "v2_local_corrected_multiscene.tif",
    ),
}


def stage_directory(root: str | Path, stage_index: int) -> Path:
    if not 0 <= int(stage_index) < len(STAGE_NAMES):
        raise ValueError(f"invalid Task15 stage index: {stage_index}")
    return Path(root) / "stages" / STAGE_NAMES[int(stage_index)]


def required_stage_output_names(stage_name: str) -> tuple[str, ...]:
    return tuple(_REQUIRED_OUTPUTS.get(stage_name, ()))


def stage_name_to_index(stage_name: str) -> int:
    try:
        return STAGE_NAMES.index(stage_name)
    except ValueError as exc:
        raise ValueError(f"unknown Task15 stage: {stage_name}") from exc
