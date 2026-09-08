# Task 5 Report — Final Independent Registration Validation

## Scope

Implemented configured independent final validation and final quality aggregation in the assigned project workspace. Tasks 1–4 changes and unrelated untracked report data were left untouched.

## TDD evidence

Added the three exact Task 5 tests:

- `test_final_quality_gate_uses_post_warp_residuals`
- `test_final_quality_gate_rejects_bad_rmse_or_p95`
- `test_independent_validation_uses_configured_thresholds`

The first focused red run failed for the intended missing behavior: the aggregate helper import was unavailable and the validator rejected the new explicit threshold keyword arguments. The RMSE/P95 classifier regression already passed because the preceding implementation already included those checks.

## Implementation

- Added `aggregate_final_validation_quality()` in `src/coregistration.py`.
  - Aggregates accepted final block residuals and confidence values when detailed blocks are available.
  - Supports the specified summary-statistics contract.
  - Classifies the aggregate through the configured pass/warn/fail thresholds.
- Extended `validate_registration_independent_grid()` with explicit `confidence_threshold`, `max_residual_shift`, and `min_accepted` parameters while preserving the previous defaults.
- Updated `MultibandPipeline.register_scenes()` to validate the final registered arrays on every spanning-tree edge, excluding that edge’s training block centers, and to pass configured final-validation thresholds.
- Exposed final quality and validation results in the registration result and diagnostics.
- No `experiment_config.py` change was needed because the required validation and quality defaults already existed.

## Verification

- Focused tests: `3 passed`.
- Adjacent registration tests excluding the known temporary-directory fixture: `22 passed, 1 deselected`.
- Full adjacent-file run: `22 passed, 1 error`; the error is pytest’s `tmp_path` setup failing with `PermissionError: [WinError 5]` while scanning the host temporary directory. Redirecting `--basetemp` produced the same environment-level cleanup permission error.
- `git diff --check`: passed.

## Concerns

- The host pytest environment cannot access/clean its temporary-directory roots, so the single `tmp_path`-based adjacent test could not complete in this session.
- A broad `compileall` check was also blocked by permission errors writing existing `__pycache__` files; the focused and adjacent tests imported the modified modules successfully.

## Review fix round

Addressed all four review findings:

1. Final validation exclusion points now combine matching controls from both the original global pair measurements and post-global residual pairs. Duplicate points are removed before passing them to the independent validator.
2. Added schema-backed defaults for `validation_step`, `validation_offset_row`, `validation_offset_col`, and `validation_min_distance_from_training` in `_DEFAULT_REGISTRATION_PARAMS`, so `_merge_registration_params()` preserves normal YAML values.
3. Added deterministic `register_scenes()` regression coverage proving that final registered arrays are validated on spanning-tree edges, original and post-global controls are excluded, configured thresholds are forwarded, and `nodata=None` is preserved.
4. Summary-only aggregation now uses the supplied metrics for a single edge and conservative maximum median/P95 bounds for multiple edges; it no longer averages median/P95 values across edge summaries. RMSE remains count-weighted because it is aggregable from per-edge sums of squares.

### Fix-round TDD and verification

- New review regressions were run before the fixes: `3 failed`, covering the summary-quantile fallback, discarded validation-grid config keys, and missing post-global exclusion.
- After implementation: review regressions `3 passed`.
- `tests/test_registration_quality.py`: `7 passed`.
- `tests/test_registration_refinement.py` excluding the pre-existing temporary-path case: `18 passed, 1 deselected`.
- Both adjacent files together: `25 passed, 1 error`; the remaining error is the host pytest `tmp_path` setup `PermissionError: [WinError 5]` while scanning `C:\Users\wang\AppData\Local\Temp\pytest-of-wang`.
- `git diff --check`: passed.
