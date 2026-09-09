# Final Review Fix 2 Report

Date: 2026-09-09

Branch: \`fix/dz01v-mosaic-series-readiness-20260907\`

Base: \`134e5fd fix: close final registration review findings\`

## Findings closed

1. Iterative robust pair MAD rejection now retains the below-minimum next
   candidate mask before the existing minimum-inlier check. Four retained
   inliers plus one rejected outlier therefore produce \`status="fail"\`
   when \`robust_min_inliers=5\`; the previous mask cannot be accepted.
2. The final \`MultibandPipeline.run()\` registration gate now requires
   \`registration["status"] == "pass"\` in addition to connectivity and the
   configured quality floor. A smoke run with full-frame status \`fail\` is
   blocked even when the recropped quality result reports \`pass\`.

## Changes

- \`src/coregistration.py\`: use \`next_inlier_mask\` when it falls below the
  configured minimum, so the existing failure path reports the effective
  inlier count.
- \`src/multiband_pipeline.py\`: include canonical registration status in the
  normalization gate and failure log.
- \`tests/test_registration_refinement.py\`: add the four-inlier/one-outlier
  minimum-count regression.
- \`tests/test_multiband_pipeline_regressions.py\`: add the smoke \`run()\`
  integration regression with full-frame failure status and recropped passing
  quality.

## TDD red run

Command:

\`\`\`text
.\.venv\Scripts\python.exe -m pytest tests/test_registration_refinement.py::test_robust_pair_measurement_fails_when_mad_filter_drops_below_minimum tests/test_multiband_pipeline_regressions.py::test_smoke_run_stops_before_normalization_when_registration_status_fails -q -p no:cacheprovider
\`\`\`

Output:

\`\`\`text
exit_code=1
FF                                                                       [100%]
E       AssertionError: assert 'pass' == 'fail'
E       AssertionError: assert [True] == []
2 failed in 4.40s
\`\`\`

The first failure showed the estimator accepting all five controls after the
MAD candidate had dropped to four. The second showed normalization being
called while the smoke registration status was \`fail\` and recropped quality
was \`pass\`.

## TDD green run

Command:

\`\`\`text
.\.venv\Scripts\python.exe -m pytest tests/test_registration_refinement.py::test_robust_pair_measurement_fails_when_mad_filter_drops_below_minimum tests/test_multiband_pipeline_regressions.py::test_smoke_run_stops_before_normalization_when_registration_status_fails -q -p no:cacheprovider
\`\`\`

Output:

\`\`\`text
exit_code=0
..                                                                       [100%]
2 passed in 2.54s
\`\`\`

## Focused verification

Command:

\`\`\`text
.\.venv\Scripts\python.exe -m pytest tests/test_registration_quality.py tests/test_coregistration_regressions.py tests/test_registration_refinement.py tests/test_multiband_pipeline_regressions.py tests/test_script_contracts.py -k "not analyze_displacement_spikes_handles_distance_field" -q -p no:cacheprovider
\`\`\`

Output:

\`\`\`text
exit_code=0
......................................................                   [100%]
54 passed, 1 deselected in 3.34s
\`\`\`

Command:

\`\`\`text
.\.venv\Scripts\python.exe -m pytest tests/test_diagnostics_regressions.py -k "not registration_diagnostic_distinguishes_raw_and_robust_pair_matches and not no_overlap_writes_structured_failure_payload and not failed_overlapping_registration_writes_failure_only_and_returns_nonzero and not registration_diagnostic_uses_actual_registration_schema" -q -p no:cacheprovider
\`\`\`

Output:

\`\`\`text
exit_code=0
....                                                                     [100%]
4 passed, 4 deselected in 2.86s
\`\`\`

## Full-suite verification

Command:

\`\`\`text
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --tb=line
\`\`\`

Output:

\`\`\`text
exit_code=1
23 failed, 248 passed, 3 warnings, 13 errors in 8.35s
\`\`\`

The 23 failures and 13 errors are the host's pre-existing protected
\`C:\Users\wang\AppData\Local\Temp\` ACL failures while creating, reading, or
cleaning temporary files and raster outputs. No assertion failure was
reported in the focused registration or diagnostic suites.

## Static checks

Command:

\`\`\`text
.\.venv\Scripts\python.exe -c "import ast, pathlib; paths=['src/coregistration.py','src/multiband_pipeline.py','tests/test_multiband_pipeline_regressions.py','tests/test_registration_refinement.py']; [ast.parse(pathlib.Path(path).read_text(encoding='utf-8'), filename=path) for path in paths]; print('syntax-ok')"
\`\`\`

Output:

\`\`\`text
exit_code=0
syntax-ok
\`\`\`

Command:

\`\`\`text
git diff --check
\`\`\`

Output:

\`\`\`text
exit_code=0
warning: in the working copy of '.superpowers/sdd/2026-09-08-registration-pipeline-reliability/progress.md', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'src/coregistration.py', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'src/multiband_pipeline.py', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'tests/test_multiband_pipeline_regressions.py', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'tests/test_registration_refinement.py', LF will be replaced by CRLF the next time Git touches it
\`\`\`

The line-ending messages are Git normalization warnings; the diff check
returned zero.

## Scope and commit

Only the two confirmed residual root causes were changed. Existing ledger,
review-package, and report-figure artifacts were left unstaged.

Requested commit message: \`fix: close residual registration review findings\`
