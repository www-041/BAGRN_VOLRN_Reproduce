"""Artifact-level checks for the frozen Task13B label replay."""

from pathlib import Path

import numpy as np
import rasterio


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/output/b9_five_scene_validation/multiscene_task13b"


def test_task13b_label_artifacts_have_canonical_grid_and_valid_labels():
    labels = OUT / "labels/source_label_map.tif"
    if not labels.exists():
        return
    paths = [OUT / "labels" / name for name in (
        "source_label_map.tif", "label_method_map.tif",
        "label_score_margin.tif", "contributor_count.tif")]
    with rasterio.open(paths[0]) as src:
        profile = (src.width, src.height, str(src.crs), tuple(src.transform))
        values = src.read(1)
    assert profile[:2] == (5116, 6134)
    assert profile[2] == "EPSG:32650"
    assert np.all((values == -1) | ((values >= 0) & (values < 5)))
    for path in paths[1:]:
        with rasterio.open(path) as src:
            assert (src.width, src.height, str(src.crs), tuple(src.transform)) == profile


def test_task13b_labeling_summary_records_hard_stop_evidence():
    summary = OUT / "labeling_summary.json"
    if not summary.exists():
        return
    import json
    data = json.loads(summary.read_text(encoding="utf-8"))
    assert data["finite_union_pixels"] == 22167910
    assert data["invalid_label_pixels"] == 0
    assert data["two_scene_disagreement_pixels"] == 0
    assert data["unresolved_pixels"] == 27744
