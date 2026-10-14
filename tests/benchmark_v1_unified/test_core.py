from __future__ import annotations

import numpy as np
import pytest

from src.benchmark_v1_unified.aggregation import aggregate_pair_rows
from src.benchmark_v1_unified.radiometric import (
    audit_pair_output_support,
    compute_pair_direct_metrics,
    compute_pair_distribution_metrics,
    pair_support,
)
from src.benchmark_v1_unified.simulation import evaluate_s4_change, score_truth
from src.benchmark_v1_unified.structure import compute_sobel_structure, compute_range_retention
from src.benchmark_v1_unified.windows import WindowConfig, make_fixed_windows


def test_t01_permutation_separates_metrics():
    x = np.array([0, 1, 2], dtype=float)
    y = np.array([2, 0, 1], dtype=float)
    mask = np.ones(3, bool)
    direct = compute_pair_direct_metrics(x, y, mask)
    dist = compute_pair_distribution_metrics(x, y, mask)
    assert direct.mae == pytest.approx(4 / 3)
    assert direct.rmse == pytest.approx(np.sqrt(2))
    assert dist.mamd == 0
    assert dist.msdd == 0
    assert dist.rdd == 0


def test_t02_mean_std_w1_fixture():
    result = compute_pair_direct_metrics([0, 0, 4], [1, 1, 1], [1, 1, 1])
    dist = compute_pair_distribution_metrics([0, 0, 4], [1, 1, 1], [1, 1, 1])
    assert dist.mamd == pytest.approx(1 / 3)
    assert dist.msdd == pytest.approx(np.sqrt(32) / 3)
    assert dist.rdd == pytest.approx(5 / 3)
    assert result.mae == pytest.approx(5 / 3)
    assert result.rmse == pytest.approx(np.sqrt(11 / 3))


def test_t03_identity_all_discrepancies_zero():
    x = np.array([0, 3, 7, 1.])
    assert compute_pair_direct_metrics(x, x, np.ones(4, bool)).mae == 0
    d = compute_pair_distribution_metrics(x, x, np.ones(4, bool))
    assert (d.mamd, d.msdd, d.rdd) == (0, 0, 0)


def test_t04_output_zero_is_valid_when_explicit_mask_says_so():
    support = np.array([True, True])
    audit = audit_pair_output_support(support, np.array([0., 2.]), np.array([True, True]))
    assert audit.coverage == 1
    assert audit.status == "FULL_SUPPORT"


def test_t05_partial_output_has_no_primary_score():
    audit = audit_pair_output_support(np.ones(4, bool), np.array([1, 2, 3, 4]), np.array([1, 1, 1, 0], bool))
    assert audit.coverage == .75
    assert audit.status == "PARTIAL_OUTPUT"
    assert not audit.primary_eligible


def test_t06_disjoint_overlap_is_explicit():
    assert not pair_support(np.array([1, 0], bool), np.array([0, 1], bool)).any()
    result = compute_pair_direct_metrics([1, 2], [3, 4], [False, False])
    assert result.status == "NO_VALID_OVERLAP"
    assert result.mae is None


def test_t07_std_uses_population_ddof_zero():
    d = compute_pair_distribution_metrics([0, 2], [0, 0], [1, 1])
    assert d.msdd == 1


def test_t08_w1_ignores_spatial_permutation():
    d = compute_pair_distribution_metrics([0, 1, 2], [2, 0, 1], [1, 1, 1])
    assert d.rdd == 0
    assert compute_pair_direct_metrics([0, 1, 2], [2, 0, 1], [1, 1, 1]).mae > 0


def test_t09_sobel_direction_identity_inversion_and_flat():
    x = np.zeros((9, 9)); x[:, 4:] = 1
    valid = np.ones_like(x, bool)
    same = compute_sobel_structure(x, x, valid)
    inverted = compute_sobel_structure(x, -x, valid)
    flat = compute_sobel_structure(x, np.zeros_like(x), valid)
    assert same.direction_loss == pytest.approx(0)
    assert inverted.direction_loss == pytest.approx(2)
    assert flat.direction_loss == pytest.approx(1)


def test_t10_invalid_boundary_stencil_not_counted():
    x = np.arange(49, dtype=float).reshape(7, 7)
    mask = np.ones((7, 7), bool); mask[3, 3] = False
    result = compute_sobel_structure(x, x, mask)
    assert result.gradient_centers == 16  # 25 interior centers minus the nine 3x3 stencils touching the hole.


def test_t11_t12_seam_excess_fixture():
    from src.benchmark_v1_unified.seams import compute_seam_excess
    assert compute_seam_excess(np.array([[1, 1, 6, 6.]]), np.array([[0, 0, 1, 1]]))['mean_excess'] == 5
    assert compute_seam_excess(np.array([[1, 2, 3, 4.]]), np.array([[0, 0, 1, 1]]))['mean_excess'] == 0


def test_t13_t14_mosaic_coverage_and_no_seam_provenance():
    from src.benchmark_v1_unified.mosaic import compute_mosaic_coverage
    expected = np.array([True] * 10 + [False])
    out = np.array([True] * 9 + [False, True])
    result = compute_mosaic_coverage(expected, out)
    assert result.coverage == .9
    assert result.extras == 1
    assert result.holes == 1
    assert result.seam_status == 'NOT_APPLICABLE_NO_PROVENANCE'


def test_t15_window_selection_is_deterministic_and_method_independent():
    mask = np.ones((512, 512), bool)
    cfg = WindowConfig(max_per_pair=3)
    a = make_fixed_windows('00_01', mask, cfg)
    b = make_fixed_windows('00_01', mask.copy(), cfg)
    assert [w.window_id for w in a] == [w.window_id for w in b]
    assert len(a) == 3


def test_t16_time_strata_excludes_zero_pairs():
    from src.benchmark_v1_unified.support import time_strata_counts
    pairs = [{'status': 'OVERLAP', 'time_stratum': s} for s in ['SAME_DATE'] * 7 + ['CROSS_DATE_LT_7D'] * 13 + ['CROSS_DATE_7_TO_LT_30D'] * 6 + ['CROSS_DATE_GE_30D'] * 22]
    pairs += [{'status': 'NO_VALID_OVERLAP', 'time_stratum': 'SAME_DATE'}]
    assert time_strata_counts(pairs) == {'SAME_DATE': 7, 'CROSS_DATE_LT_7D': 13, 'CROSS_DATE_7_TO_LT_30D': 6, 'CROSS_DATE_GE_30D': 22}


def test_t17_pair_macro_differs_from_pixel_weighted_and_missing_pair_blocks():
    rows = [{'pair_id': 'a', 'value': 1, 'n': 1, 'status': 'EVALUABLE'}, {'pair_id': 'b', 'value': 3, 'n': 3, 'status': 'EVALUABLE'}]
    result = aggregate_pair_rows(rows, ['a', 'b'])
    assert result['pair_macro'] == 2
    assert result['pixel_weighted'] == 2.5
    assert result['all_pairs_eligible']
    assert not aggregate_pair_rows(rows[:1], ['a', 'b'])['all_pairs_eligible']


def test_t18_t19_s4_truth_and_suppression():
    before = np.array([0., 1, 2, 3]); after = before.copy(); after[2:] += 2
    assert score_truth(after, after)['mae'] == 0
    result = evaluate_s4_change(before, after, before, after, np.array([0, 0, 1, 1], bool))
    assert result['change_delta_rmse'] == 0
    suppressed = evaluate_s4_change(before, after, before, before, np.array([0, 0, 1, 1], bool))
    assert suppressed['change_delta_rmse'] > 0
    assert suppressed['change_magnitude_ratio'] == 0


def test_b3_evaluator_requires_common_anchor_and_reports_s4():
    from src.benchmark_v1_unified.evaluator import evaluate_synthetic_arrays
    before=np.array([0.,1,2,3]);after=before.copy();after[2:]+=1;mask=np.ones(4,bool);changed=np.array([0,0,1,1],bool)
    good=evaluate_synthetic_arrays(before,before,mask,truth_after=after,output_after=after,valid_after=mask,change_mask=changed)
    assert good['status']=='EVALUABLE' and good['before']['mae']==0 and good['change']['change_delta_rmse']==0
    bad=evaluate_synthetic_arrays(before,before,mask,anchor_matched=False)
    assert bad['status']=='UNMATCHED_OUTPUT_GAUGE'


def test_t20_degenerate_metrics_are_statused_not_nan():
    assert compute_range_retention(np.ones(4), np.ones(4), np.ones(4, bool)).status == 'NOT_EVALUABLE_FLAT_INPUT'
    result = evaluate_s4_change(np.ones(4), np.ones(4), np.ones(4), np.ones(4), np.zeros(4, bool))
    assert result['status'] == 'NOT_EVALUABLE_ZERO_TRUE_CHANGE'


def test_t21_grid_shift_is_invalid():
    from src.benchmark_v1_unified.submission import validate_grid
    assert validate_grid((0, 14, 0, 0, 0, -14), (14, 14, 0, 0, 0, -14)) == 'INVALID_GRID'


def test_t22_freeze_tamper_detected(tmp_path):
    from src.benchmark_v1_unified.freeze import write_lock, verify_lock
    item = tmp_path / 'source.json'; item.write_text('{"x":1}')
    write_lock(tmp_path, [item])
    assert verify_lock(tmp_path).valid
    item.write_text('{"x":2}')
    assert not verify_lock(tmp_path).valid


def test_t23_legacy_pair_formula_equivalence():
    from src.benchmark_v1_radiometric_r1.metrics import pair_metrics
    x = np.array([0., 0, 4]); y = np.array([1., 1, 1]); mask = np.ones(3, bool)
    legacy = pair_metrics(x, y, mask)
    new = compute_pair_distribution_metrics(x, y, mask)
    assert new.mamd == pytest.approx(legacy['MAMD'])
    assert new.msdd == pytest.approx(legacy['MSDD'])
    assert new.rdd == pytest.approx(legacy['RDD'])


def test_t24_method_mask_not_primary_input_support():
    from src.benchmark_v1_unified.support import validate_support_origin
    assert validate_support_origin('METHOD_SEAM') == 'INVALID_SCORING_SUPPORT'
    assert validate_support_origin('FROZEN_INPUT_MASK') == 'PASS'


def test_t25_missing_runtime_is_not_zero():
    from src.benchmark_v1_unified.resource import missing_runtime
    result = missing_runtime()
    assert result['status'] == 'NOT_MEASURED'
    assert result['wall_seconds'] is None


def test_t26_multiband_requires_normalization_for_cross_band_summary():
    from src.benchmark_v1_unified.radiometric import cross_band_summary
    result = cross_band_summary({'B9': 1., 'B14': 100.}, units_known=False, scales=None)
    assert result['status'] == 'NOT_EVALUABLE_UNKNOWN_UNITS'
    assert result['value'] is None


def test_distribution_properties_and_support_validation():
    rng = np.random.default_rng(91)
    x, y = rng.normal(size=31), rng.normal(size=31)
    mask = rng.random(31) > .2
    a = compute_pair_distribution_metrics(x, y, mask)
    b = compute_pair_distribution_metrics(y, x, mask)
    assert a.rdd == pytest.approx(b.rdd)
    assert compute_pair_distribution_metrics(x + 4, y + 4, mask).rdd == pytest.approx(a.rdd)
    with pytest.raises(ValueError):
        compute_pair_direct_metrics(x, y[:-1], mask)
    bad = x.copy(); bad[0] = np.nan
    with pytest.raises(ValueError):
        compute_pair_direct_metrics(bad, y, np.ones(31, bool))


def test_t15_cap_is_spatially_dispersed_across_supercells():
    mask = np.ones((2048, 2048), bool)
    result = make_fixed_windows('00_01', mask, WindowConfig(max_per_pair=4))
    cells={(r.row_off//512,r.col_off//512) for r in result}
    assert len(result)==4
    assert len(cells)==4


def test_aligned_input_identity_mapping_and_explicit_mask(tmp_path):
    import rasterio
    from rasterio.transform import Affine
    from src.benchmark_v1_unified.aligned import prepare_aligned_input
    from src.benchmark_v1_unified.geometry import GridSpec
    src=tmp_path/'source.tif';values=np.arange(1,17,dtype=np.uint16).reshape(4,4)
    with rasterio.open(src,'w',driver='GTiff',width=4,height=4,count=1,dtype='uint16',crs='EPSG:32650',transform=Affine(14,0,0,0,-14,56),nodata=0) as ds:ds.write(values,1)
    grid=GridSpec('EPSG:32650',(14,0,0,0,-14,56),4,4,196)
    item=prepare_aligned_input(src,0,grid,np.eye(3),tmp_path/'aligned')
    with rasterio.open(item['aligned_path']) as ds:assert np.allclose(ds.read(1),values)
    with rasterio.open(item['mask_path']) as ds:assert ds.read(1).all()
    with pytest.raises(FileExistsError):prepare_aligned_input(src,0,grid,np.eye(3),tmp_path/'aligned')

