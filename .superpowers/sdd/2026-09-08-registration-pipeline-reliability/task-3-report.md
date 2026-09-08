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

## Review fixes

- Strengthened `test_global_refinement_rewarps_from_original_not_previous_warp` with a nonzero initial target shift, an explicit nonempty warp assertion, and an assertion that every warp input equals one of the original arrays.
- Strengthened `test_post_global_refinement_reduces_known_translation_residual` with a deterministic pre/post Euclidean translation residual comparison against the known final shift.
- Added `test_global_refinement_rejects_over_limit_before_stop` to cover an over-limit correction when the stop threshold is larger than the correction limit.
- Reordered `refine_global_residual_shifts_from_original()` so maximum-correction rejection and warning occur before stop-magnitude convergence handling.

## Review-fix verification

- TDD RED covering run: 2 strengthened tests passed; the new threshold-edge test failed because the old ordering omitted the required warning.
- TDD GREEN covering run: 3/3 passed.
- Adjacent coregistration suites after the fix: 7/7 passed.
- Commit: review-fix commit with the message `fix: enforce non-vacuous residual refinement checks`.

## Review-fix concerns

- The pre-existing pytest cache/temp-directory permission warnings remain environmental and unrelated to the review fix.
