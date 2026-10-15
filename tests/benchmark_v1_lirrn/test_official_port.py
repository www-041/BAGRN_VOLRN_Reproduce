import numpy as np
import pytest
import hashlib
import json
from types import SimpleNamespace
from pathlib import Path

from src.benchmark_v1_lirrn.frozen_crop import RasterCrop
from src.benchmark_v1_lirrn.official_port import (
    RegressionResult,
    apply_lirrn_linear,
    classify_three_levels,
    draw_shared_ids,
    flatten_lirrn_values,
    fit_lirrn_linear,
    select_lirrn_samples,
)
from src.benchmark_v1_lirrn import frozen_crop


def _crop(values: np.ndarray, scene: int = 0) -> RasterCrop:
    values = np.asarray(values, dtype=np.float64)
    return RasterCrop(values, np.ones(values.shape, dtype=bool), None, "EPSG:32650", scene)


def test_three_level_otsu_separates_three_peaks_and_excludes_invalid():
    values = np.repeat(np.array([10.0, 11.0, 100.0, 101.0, 240.0, 241.0]), 20).reshape(10, 12)
    valid = np.ones(values.shape, dtype=bool)
    valid[0, 0] = False
    labels, audit = classify_three_levels(values, valid)
    assert len(audit["thresholds"]) == 2
    assert audit["thresholds"][0] < audit["thresholds"][1]
    assert set(np.unique(labels[valid])) == {1, 2, 3}
    assert labels[0, 0] == 0


@pytest.mark.parametrize("values,valid", [
    (np.ones((4, 4)), np.ones((4, 4), dtype=bool)),
    (np.array([[1.0, 2.0]]), np.ones((1, 2), dtype=bool)),
])
def test_three_level_otsu_rejects_constant_or_too_few_samples(values, valid):
    with pytest.raises(ValueError):
        classify_three_levels(values, valid)


def test_shared_ids_are_seeded_replacement_indices_with_official_sample_count():
    first = draw_shared_ids(1000, 20261014)
    second = draw_shared_ids(1000, 20261014)
    other = draw_shared_ids(1000, 20261015)
    assert first.shape == (100,)
    assert np.array_equal(first, second)
    assert not np.array_equal(first, other)
    assert np.all((first >= 0) & (first < 1000))


def test_sampling_repeats_the_same_max_distance_selection_three_times():
    # All three rows are separate strata with distinct values and stable rank order.
    ref_values = np.vstack([np.arange(1, 11), np.arange(101, 111), np.arange(201, 211)]).astype(float)
    sub_values = ref_values - 20.0
    ref_labels = np.repeat(np.arange(1, 4)[:, None], 10, axis=1)
    sub_labels = ref_labels.copy()
    valid = np.ones(ref_values.shape, dtype=bool)
    ref = RasterCrop(ref_values, valid, None, "EPSG:32650", 0)
    sub = RasterCrop(sub_values, valid, None, "EPSG:32650", 1)
    ids = np.array([0, 3], dtype=np.int64)
    x_sub, y_ref, audit = select_lirrn_samples(ref, sub, ref_labels, sub_labels, ids, p_n=10)
    assert x_sub.shape == y_ref.shape == (18,)  # 3 classes x 2 IDs x 3 duplicate calls
    assert audit["three_identical_max_sampling_calls"] is True
    for class_index in range(3):
        block = slice(class_index * 6, (class_index + 1) * 6)
        assert np.array_equal(x_sub[block][:2], x_sub[block][2:4])
        assert np.array_equal(x_sub[block][:2], x_sub[block][4:6])
        assert np.array_equal(y_ref[block][:2], y_ref[block][2:4])


def test_sampling_requires_pn_candidates_in_each_scene_and_stratum():
    values = np.arange(1, 13, dtype=float).reshape(3, 4)
    labels = np.repeat(np.arange(1, 4)[:, None], 4, axis=1)
    crop = _crop(values)
    with pytest.raises(ValueError, match="INSUFFICIENT_STRATUM_SAMPLES"):
        select_lirrn_samples(crop, crop, labels, labels, np.array([0]), p_n=5)


def test_ols_recovers_known_affine_model_and_rejects_rank_deficiency():
    x = np.arange(-5.0, 8.0)
    fit = fit_lirrn_linear(x, 1.25 * x + 20.0)
    assert fit.scale == pytest.approx(1.25)
    assert fit.offset == pytest.approx(20.0)
    with pytest.raises(ValueError, match="REGRESSION_DEGENERATE"):
        fit_lirrn_linear(np.ones(5), np.arange(5.0))


def test_numpy_order_f_flattening_is_recorded_by_sampling_audit():
    ref_values = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
    labels = np.repeat(np.array([1, 2, 3], dtype=np.uint8)[:, None], 3, axis=1)
    crop = _crop(ref_values)
    ids = np.array([0, 1, 2], dtype=np.int64)
    x_sub, y_ref, audit = select_lirrn_samples(crop, crop, labels, labels, ids, p_n=3)
    assert audit["flatten_order"] == "F"
    assert np.all(np.isfinite(x_sub)) and np.all(np.isfinite(y_ref))


def test_matlab_nonzeros_order_is_column_major():
    values = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
    valid = np.array([[True, False, True], [True, True, False]])
    assert np.array_equal(flatten_lirrn_values(values, valid), np.array([1, 4, 5, 3], dtype=float))


def test_apply_allows_nonfinite_values_only_outside_valid_mask():
    values = np.array([[2.0, np.nan], [4.0, np.inf]])
    valid = np.array([[True, False], [True, False]])
    crop = RasterCrop(values, valid, None, "EPSG:32650", 1)
    model = RegressionResult(2.0, 3.0, 0.0, 1.0, 2, 1.0, 3)
    result = apply_lirrn_linear(crop, model)
    assert np.array_equal(result[valid], np.array([7.0, 11.0], dtype=np.float32))
    assert np.isnan(result[0, 1]) and np.isinf(result[1, 1])


def test_freeze_audit_resolves_repo_from_code_not_relative_freeze_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    freeze = tmp_path / "freeze"
    freeze.mkdir()
    locked = tmp_path / "locked.txt"
    locked.write_text("immutable", encoding="utf-8")
    digest = hashlib.sha256(locked.read_bytes()).hexdigest()
    manifest = {"files": [{"path": str(locked), "sha256": digest} for _ in range(38)]}
    (freeze / "code_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(frozen_crop, "verify_lock", lambda root: SimpleNamespace(valid=True, errors=[]))
    result = frozen_crop.verify_frozen_b9(Path("freeze"))
    assert result["status"] == "PASS"
    assert result["code_manifest_count"] == 38
