# BAGRN-VOLRN Reliability Fixes — 2026-09-07

## Summary

This document records the reliability fixes applied to the BAGRN-VOLRN reproduction codebase on 2026-09-07.

- **Repair branch:** `fix/reliability-audit-20260907`
- **Base commit:** `cbde342137afbfd8f81fc506763e1c94aae36e79`
- **Final commit:** `446df83582907f1361c5bcdb88a15b7c4afeacff`

## Defects Fixed

### Phase 1 — Valid Pixels, Geometry, and Metrics

1. **Overlap geometry (`src/overlap.py`)**
   - `has_overlap` now requires positive intersection area (edge-touching returns False)
   - `get_overlap_window` uses floor/ceil for conservative pixel coverage

2. **Multi-resolution metrics (`src/metrics.py`)**
   - `_extract_overlap_pixels` returns independently filtered pixels per window
   - `compute_gl` divides by actual contributors, not all images
   - `compute_cd` recomputes valid counts per band for weights

### Phase 2 — BAGRN and VOLRN

3. **BAGRN valid-pixel handling (`src/bagrn.py`)**
   - Added `_valid_mask` helper that excludes both NaN/Inf AND nodata
   - Input validation: list lengths, 3D input, control_idx range
   - Single-image returns identity with zero compensation

4. **VOLRN NoData preservation (`src/volrn.py`)**
   - Normalized arrays use NaN for invalid pixels, not zero
   - Returned b coefficients converted to original DN units: `b_original = scale*b_n + (1-a)*vmin`

5. **VOLRN local IDW interpolation (`src/volrn.py`)**
   - Uses local 3x3 grid neighbourhood first
   - Falls back to global nearest only when no local candidates exist

### Phase 3 — Mosaic and Comparison

6. **Mosaic NoData and extent (`src/mosaic.py`)**
   - Empty inputs fail with ValueError before accessing arrays[0]
   - Canvas dimensions use ceil, not round
   - nodata=None no longer implies DN=0 is invalid
   - Invalid mask checks all bands

7. **Comparison method masking (`src/comparison.py`)**
   - Independent valid-pixel extraction for two overlap windows
   - Wallis skips control image
   - Valid masks exclude both NaN/Inf AND nodata

### Phase 4 — Pipeline and Coregistration

8. **Main pipeline validation (`src/main.py`)**
   - Transform/CRS consistency check before np.stack
   - VOLRN block count uses shape[1], not shape[0]
   - Metrics preserved when --compare used
   - control_idx validation added

9. **Multiband pipeline (`src/multiband_pipeline.py`)**
   - nodata=None no longer converted to 0
   - VOLRN called with return_diagnostics=True

10. **Coregistration (`src/coregistration.py`)**
    - Fixed rowcol() unpacking: `r0, c0 = rowcol(...)` instead of `c0, r0`
    - Zero inliers → confidence 0.0 (was 1.0)
    - Multi-resolution registration uses common grid via rasterio reproject

### Phase 5 — Diagnostics and Secondary Metrics

11. **Diagnostics coefficient statistics (`src/diagnostics.py`)**
    - a_rms now measures deviation from identity: `sqrt(mean((a-1)^2))`
    - Dynamic range uses reference value of 1.0

12. **Spectral metrics (`src/spectral_metrics.py`)**
    - `bands=` parameter now actually restricts computation to selected bands

13. **Seam metrics (`src/seam_metrics.py`)**
    - `_safe_array` always excludes NaN/Inf, not just nodata

### Phase 6 — Experiment Runner and Dependencies

14. **Experiment runner (`src/experiment_runner.py`)**
    - `--crop-size` defaults to None (was 512)
    - YAML config only overridden when CLI option explicitly provided

15. **Dependencies (`requirements.txt`)**
    - Added: PyYAML, scikit-image, scikit-learn, matplotlib, psutil

### Phase 7 — Legacy Scripts

16. **Metric call order**
    - Fixed `compute_all(before, after, ...)` argument order in 7 scripts
    - Never `compute_all(result, result, ...)` for normalized methods

17. **Script bugs**
    - Fixed `read_geotiff` unpacking order in test_norm_fix.py
    - Fixed reversed overlap slice order in generate_effect_figures.py
    - Replaced `> 0` validity checks with `np.isfinite` in 3 diagnostic scripts

## Tests Added

- `tests/test_overlap_regressions.py` — 8 tests
- `tests/test_metrics_regressions.py` — 5 tests
- `tests/test_bagrn_regressions.py` — 3 tests
- `tests/test_volrn_regressions.py` — 6 tests
- `tests/test_mosaic_regressions.py` — 4 tests
- `tests/test_comparison_regressions.py` — 3 tests
- `tests/test_main_regressions.py` — 4 tests
- `tests/test_multiband_pipeline_regressions.py` — 2 tests
- `tests/test_coregistration_regressions.py` — 4 tests
- `tests/test_diagnostics_regressions.py` — 3 tests
- `tests/test_spectral_metrics_regressions.py` — 2 tests
- `tests/test_seam_metrics_regressions.py` — 3 tests
- `tests/test_experiment_runner_regressions.py` — 2 tests
- `tests/test_script_contracts.py` — 1 test
- `tests/test_synthetic_end_to_end_smoke.py` — 1 test

**Total: 76 new regression tests, all passing.**

## Safe Verification Results

```
python -m compileall -q src tests     # Exit 0
pytest -q <all regression tests>      # 75 passed
pytest -q test_synthetic_end_to_end   # 1 passed
```

## Formal experiment status

No full real-data normalization, VOLRN, multi-scene mosaic, sensitivity sweep,
scale experiment, or formal cross-sensor experiment was executed during this
repair cycle. Formal experiment results must be regenerated manually after
code review.

## Manual Experiment Recommendations

After reviewing this branch, the following experiments should be regenerated:

1. Small real-data BAGRN sanity case (2 scenes, 1 band)
2. Small real-data BAGRN→VOLRN sanity case
3. Two-scene mosaic comparison (source_selection vs narrow_feather vs weighted)
4. Multi-band two-scene case
5. Cross-sensor DZ01S vs DZ01V (after common-grid coregistration review)
6. Four-image/multi-scene run
7. Parameter sensitivity sweeps
8. Scale experiment
9. Spectral/seam experiments
