# Task 1 report

## Changed files

- `src/coregistration.py`: added `build_robust_pair_measurement()` with a single confidence mask, joint 2-D Euclidean MAD rejection, confidence-weighted inlier estimate, residual statistics, and structured failure results.
- `src/multiband_pipeline.py`: block pair matching now uses the robust helper; network adjustment receives the robust shift, confidence, inlier count, and RMSE; phase correlation is used only after robust block estimation fails; the pre-warp pair-summary quality gate was removed.
- `tests/test_registration_refinement.py`: added the two required TDD tests.
- `.superpowers/sdd/2026-09-08-registration-pipeline-reliability/task-1-report.md`: this report.

## Commits

The implementation commit message is `fix: robustly estimate pair shifts from block matches`.

## TDD and verification

Focused red command:

```text
.venv\Scripts\python.exe -m pytest tests/test_registration_refinement.py::test_robust_pair_measurement_rejects_joint_xy_outliers tests/test_registration_refinement.py::test_n2_pair_uses_block_samples_not_one_pair_sample -v
```

Observed failure: test collection failed with `ImportError: cannot import name 'build_robust_pair_measurement'`, confirming the missing-helper red state.

Focused green command:

```text
.venv\Scripts\python.exe -m pytest tests/test_registration_refinement.py::test_robust_pair_measurement_rejects_joint_xy_outliers tests/test_registration_refinement.py::test_n2_pair_uses_block_samples_not_one_pair_sample -v
```

Observed output: `2 passed, 1 warning`.

Adjacent registration command:

```text
.venv\Scripts\python.exe -m pytest tests/test_registration_refinement.py tests/test_registration_quality.py tests/test_coregistration_regressions.py tests/test_coregistration_same_resolution.py tests/test_multiband_pipeline_regressions.py -q
```

Observed output: `14 passed, 1 warning`.

The full suite was also attempted: `205 passed, 23 failed, 8 errors`. The failures/errors are pre-existing managed-filesystem permission failures for temporary/output paths under Windows, including `tempfile`, pytest temp paths, and raster output; no changed registration test failed.

## Self-review

- The helper preserves the requested graph fields and adds the robust inlier count without changing network-adjustment interfaces.
- RMSE and P95 are calculated from residuals around the estimated 2-D shift.
- Confidence weighting is applied only to final inliers.
- Robust block failure continues to the explicit phase-correlation fallback.
- Unrelated user changes, including the untracked `data/report_figures` directory, were not staged.
- `git diff --check` completed without whitespace errors.

## Concerns

- The managed environment prevents cleanup of the temporary directory created for the full-suite workaround: `.pytest-temp-task1`. It is untracked and was not staged; manual cleanup may be required outside this restricted environment.
- The repository-wide suite cannot be reported fully green until the Windows temporary/output permission issue is resolved.
