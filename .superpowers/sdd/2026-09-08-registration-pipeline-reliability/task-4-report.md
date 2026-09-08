# Task 4 Report

## Changed files

- `src/coregistration.py`: extended local-control helpers with configurable minimum counts and explicit handling of post-global residual measurements while preserving target-pixel displacement signs.
- `src/multiband_pipeline.py`: added the local-control spatial gate, held-out RBF CV, post-global rematching, balanced target-coordinate controls, hull fade, component clipping, field statistics, fallback diagnostics, final one-shot local/global warp, and smoke-mode field cropping.
- `tests/test_registration_refinement.py`: added the two required Task 4 acceptance-gate tests.
- `.superpowers/sdd/2026-09-08-registration-pipeline-reliability/task-4-report.md`: this report.

Unrelated Task 1–3 changes, planning files, and the untracked `data/report_figures` tree were not staged or modified.

## Commits

- `feat: integrate gated local rbf registration refinement`

## TDD and verification

Required RED run:

```text
.venv\Scripts\python.exe -m pytest tests/test_registration_refinement.py::test_local_rbf_reduces_smooth_spatial_residual tests/test_registration_refinement.py::test_local_rbf_rejected_when_cv_does_not_improve -v
```

Observed result: both tests failed with the intended missing-helper `AttributeError` for `multiband_pipeline._accept_local_rbf_candidate`.

Required focused GREEN run: `2 passed` with one pre-existing pytest-cache permission warning.

Final focused run: `2 passed` with one pre-existing pytest-cache permission warning.

Adjacent registration run excluding the environment-blocked `tmp_path` test: `12 passed` with one pre-existing pytest-cache permission warning. The complete adjacent command reached `21 passed` and then errored during temporary-directory fixture setup/cleanup for `test_analyze_displacement_spikes_handles_distance_field` because the managed Windows environment denied access to pytest’s temp directory. The equivalent direct distance-field check passed.

Additional checks:

- Task 4 registration tests excluding the temp-path case: `9 passed, 1 deselected`.
- AST parsing of the three changed Python files: passed.
- Target-coordinate/sign smoke check for post-global parent controls: passed.
- Synthetic local CV/field smoke check: passed; field shape and clipping diagnostics were produced.
- `git diff --check`: passed with only Git’s existing LF/CRLF conversion warnings.

## Self-review

- Local fields are fitted only from rematches performed after global refinement on global-only arrays generated from the ORIGINAL inputs.
- `shift_dx` and `shift_dy` consistently mean the displacement applied to the target; flipped pair controls are negated and placed in the target/image pixel system.
- The acceptance gate requires local enablement, minimum control count, minimum spatial-group count, and both configured held-out RMSE and P95 improvements.
- Accepted fields use `fit_local_rbf()`, `compute_hull_fade_mask()`, component-wise `local_max_component` clipping, and per-scene field/clipping diagnostics.
- Rejected, unavailable, disabled, or failed-fit scenes fall back to global-only fields and are recorded in `fallback_scenes` with reasons.
- Final non-smoke warps combine global and local fields in one warp from ORIGINAL arrays; smoke mode crops the full-scene local fields before its final warp.
- Existing registration graph fields and Task 1–3 behavior remain in the returned schema.

## Concerns

- The repository’s managed Windows filesystem continues to deny pytest temp-directory scanning/cleanup, so the one `tmp_path`-based adjacent test cannot complete through pytest in this environment even though its underlying direct check passes.
- Local rematching adds work after global refinement; this is intentional to ensure local controls are post-global residuals rather than initial pair measurements.
- The local CV uses the configured first smoothing candidate; smoothing-model selection remains outside this Task 4 gate.

## Task 4 review-fix report

### Findings fixed

- CV now distinguishes attempted spatial groups from successful folds, reports both counts plus held-out-control coverage, and returns an unavailable result when configured minimum fold/control coverage is not met.
- Held-out RBF predictions are component-clipped with the configured `local_max_component` before RMSE/P95 scoring, matching the field applied during the final warp.
- Post-global rematch exceptions and unavailable results are converted into per-edge diagnostics and global-only scene fallbacks instead of propagating.
- The single-scene early return now includes `local_refinement` with `enabled`, `used_for_scenes`, `fallback_scenes`, and `cv_results`, plus zero local fields.

### Review-fix tests

- RED run: all five new regression tests failed for the expected pre-fix behaviors.
- Focused fix run: `5 passed` with one pre-existing pytest-cache permission warning.
- Full registration refinement file excluding the known temp-path case: `14 passed, 1 deselected`.
- Adjacent registration suites: `12 passed` with one pre-existing pytest-cache permission warning.
- Complete covering command: `26 passed, 1 error`; the error is the existing managed-Windows permission failure while pytest sets up `tmp_path` for `test_analyze_displacement_spikes_handles_distance_field`.
- AST parsing of changed Python/test files: passed.
- `git diff --check`: passed with only LF/CRLF conversion warnings.

### Review-fix commit

- `fix: harden gated local rbf review findings`

### Review-fix concerns

- The managed Windows environment still prevents pytest from scanning/cleaning its temp directory, so the one `tmp_path` test cannot complete through the normal fixture even though the underlying direct distance-field check passed previously.
- The fix intentionally rejects local refinement when held-out coverage is insufficient; affected scenes remain on the global-only fallback path.

## Task 4 scoped re-review fix report

### Finding fixed

- Post-global rematch failures and unavailable control edges are now associated with every affected scene before local-candidate acceptance. Each affected scene is explicitly rejected with a global-only fallback reason and cannot be added to `used_for_scenes`; unaffected scenes continue through local refinement.
- Added a focused regression test covering the case where one failed rematch edge coexists with otherwise sufficient local controls and passing CV metrics.
- Added an integration regression test verifying that the affected scene is absent from `used_for_scenes`, is recorded in `fallback_scenes`, and does not prevent an unaffected scene on a surviving edge from being refined.

### TDD and verification

- RED run: the new regression test failed with the intended missing `rematch_failures` helper argument.
- Focused regression run: `1 passed` with one pre-existing pytest-cache permission warning.
- Full registration refinement file excluding the known temp-path case: `16 passed, 1 deselected`.
- Adjacent registration suites: `12 passed` with one pre-existing pytest-cache permission warning.
- Complete covering command: `28 passed, 1 error`; the error is the existing managed-Windows permission failure while pytest sets up `tmp_path` for `test_analyze_displacement_spikes_handles_distance_field`.
- AST parsing passed and `git diff --check` passed with only existing LF/CRLF conversion warnings.

### Fix commit

- `fix: force global-only fallback for affected local rbf scenes`

### Concerns

- The managed Windows environment still prevents pytest from scanning the shared temp directory, so the one unrelated `tmp_path` test cannot complete through its normal fixture.
