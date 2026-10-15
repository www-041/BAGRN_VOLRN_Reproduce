from pathlib import Path

import numpy as np
import rasterio
from affine import Affine

from src.benchmark_v1_unified.provenance_audit import audit_mosaic_provenance
from tests.benchmark_v1_unified.test_submission_scoring import _write, make_thirteen_scene_fixture


def test_hard_source_provenance_passes_for_values_from_tagged_corrected_scene(tmp_path):
    freeze, submission, _ = make_thirteen_scene_fixture(tmp_path)

    result = audit_mosaic_provenance(submission, freeze, submission_dir=tmp_path / "method", block_shape=(31, 37))

    assert result["status"] == "PASS"
    assert result["checked_pixels"] > 0
    assert result["mismatch_pixels"] == 0


def test_hard_source_provenance_rejects_changed_mosaic_pixel(tmp_path):
    freeze, submission, masks = make_thirteen_scene_fixture(tmp_path)
    mosaic_path = Path(submission["output_mosaic_path"])
    with rasterio.open(mosaic_path) as ds:
        mosaic = ds.read(1)
    y, x = np.argwhere(np.logical_or.reduce(masks))[0]
    mosaic[y, x] += 7
    _write(mosaic_path, mosaic, np.logical_or.reduce(masks), Affine(14, 0, 633346, 0, -14, 3359874))

    result = audit_mosaic_provenance(submission, freeze, submission_dir=tmp_path / "method")

    assert result["status"] == "INVALID_SOURCE_VALUE_PROVENANCE"
    assert result["mismatch_pixels"] == 1


def test_hard_source_provenance_rejects_invalid_source_label(tmp_path):
    freeze, submission, masks = make_thirteen_scene_fixture(tmp_path)
    label_path = Path(submission["source_label_path"])
    with rasterio.open(label_path) as ds:
        labels = ds.read(1)
    y, x = np.argwhere(np.logical_or.reduce(masks))[0]
    labels[y, x] = 255
    _write(label_path, labels, np.logical_or.reduce(masks), Affine(14, 0, 633346, 0, -14, 3359874), dtype="uint8")

    result = audit_mosaic_provenance(submission, freeze, submission_dir=tmp_path / "method")

    assert result["status"] == "INVALID_SOURCE_VALUE_PROVENANCE"
    assert result["invalid_label_pixels"] == 1


def test_soft_seam_provenance_is_not_evaluable_as_hard_source_proof(tmp_path):
    freeze, submission, _ = make_thirteen_scene_fixture(tmp_path)
    submission["seam_type"] = "SOFT"

    result = audit_mosaic_provenance(submission, freeze, submission_dir=tmp_path / "method")

    assert result["status"] == "NOT_EVALUABLE_SOFT_PROVENANCE"
    assert result["checked_pixels"] == 0
