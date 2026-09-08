# Task 3 Report

## Changed files

- `src/coregistration.py`: added generic post-global residual refinement from original arrays; added 2-D compatibility for the multiband warp used by the refinement helper.
- `src/multiband_pipeline.py`: invokes residual refinement after the initial network adjustment and records refinement history and warnings in diagnostics.
- `tests/test_registration_refinement.py`: added the two required regression tests.

Unrelated untracked planning files and the existing `data/report_figures` tree were not staged or modified.

## Commits

- `fix: refine global registration from post-warp residuals`

## Test output

- Required RED run: 2 failed with the expected missing-helper `AttributeError`.
- Required focused GREEN run: 2 passed (one pre-existing pytest cache warning).
- Adjacent coregistration suites: 7 passed (one pre-existing pytest cache warning).
- Pipeline regression suites: 35 passed, 1 failed because the environment denied access to the existing Windows temporary directory during `test_save_and_reload`; the failure was unrelated to this change.
- `git diff --check` passed.

## Self-review

- Each iteration rebuilds temporary global-only arrays from `original_arrays` and never feeds a prior temporary warp into another warp.
- Residual edges are rematched, converted to the existing pair-measurement schema, solved through `multi_image_network_adjustment()`, and accumulated into the current global shifts.
- Corrections below the configured stop magnitude stop refinement without application; corrections above the configured maximum are rejected and recorded in warnings/history.
- The configured control/reference image remains fixed at zero shift.

## Concerns

- The test environment has persistent permission failures for pytest/tempfile directories under the user temp location, so suites using those fixtures cannot complete cleanly there without environment cleanup or permission changes.
- The refinement helper assumes the supplied edge indices and array/transform/nodata lists are aligned, matching the existing pipeline data contract.
