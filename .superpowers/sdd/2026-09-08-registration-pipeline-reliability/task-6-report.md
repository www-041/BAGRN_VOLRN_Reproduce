# Task 6 report

## Changes

- Added `registration_quality_meets_requirement(actual, required)` with the existing `fail < warn < pass` ordering and invalid-value rejection.
- Rebuilt the public `register_scenes()` result around the stable schema: real `connected`, canonical final `quality`, `final_validation` with `edges` and `overall`, local-refinement diagnostics, and preserved graph fields.
- Kept final validation based on arrays produced by the final original-based global+local warp; temporary global-only arrays are created only when local refinement is enabled.
- Propagated registration quality and connectivity through smoke-mode registration results.
- Added a run-level gate that prevents BAGRN/VOLRN and downstream outputs when connectivity or final quality does not meet `required_quality`.
- Added schema, quality-order, and normalization-stop regression coverage using deterministic patched two-scene fixtures.

## Commit

- Required commit message: `fix: enforce final registration quality gate`

## Tests / output

- TDD red: the two new focused tests failed for the intended missing-schema and missing-helper reasons.
- TDD green: focused schema/helper tests passed (`2 passed`).
- Adjacent registration and pipeline checks passed (`37 passed, 1 deselected` when excluding the protected-temp test).
- Direct quality-gate selectors passed (`9 passed`).
- `git diff --check` passed.

## Self-review

- Final quality is derived from independent final validation, not pre-warp pair summaries.
- The configured `final_min_blocks` and PASS/WARN thresholds are used by the existing aggregator; the public quality dictionary exposes the required stable keys.
- Graph compatibility fields remain present, and `connected` is a boolean.
- Existing unrelated worktree artifacts were not removed or staged.

## Concerns

- The host denies access to pytest’s normal temporary-directory tree, so tests requiring `tmp_path` and a clean full-suite run cannot complete in this environment. This is an environment permission issue, not a source assertion failure.
- Smoke mode reuses the full-frame registration quality after recropping, matching the existing smoke execution design; a later task may choose to validate the recropped arrays independently.

## Review-fix report

### Changes

- Added required-edge completeness checking to `aggregate_final_validation_quality()`. Missing, failed, or unusable required validation edges now force `quality: fail` and are listed as `unavailable_edges`.
- Reused one final-validation helper for normal and smoke execution. Smoke mode now validates the recropped, actually gated arrays and replaces the copied full-frame quality/final-validation result.
- Replaced no-usable-pair and disconnected-graph exceptions with the stable registration failure schema (`connected=False`, `quality=fail`, zero shifts, preserved graph diagnostics), allowing `run()` to apply the normal gate.
- Corrected the smoke zero-shift branch so it does not append an undefined temporary warp result.

### Covering tests, commands, and outputs

- Review regressions before fixes: `pytest tests/test_registration_refinement.py::test_final_quality_fails_when_required_validation_edge_is_unavailable tests/test_registration_refinement.py::test_register_scenes_returns_stable_failure_schema_when_no_pair_is_usable tests/test_registration_refinement.py::test_register_scenes_returns_stable_failure_schema_when_graph_is_disconnected tests/test_registration_refinement.py::test_smoke_registration_quality_is_recomputed_for_recropped_arrays -q --tb=short` → `4 failed`, exposing the missing aggregator argument, both exception paths, and missing smoke validation.
- Aggregator fix: focused test → `1 passed`.
- Final review regressions: the same four-test command → `4 passed`.
- Relevant regression set: `pytest tests/test_registration_refinement.py tests/test_registration_quality.py tests/test_multiband_pipeline_regressions.py -q -k 'not analyze_displacement_spikes_handles_distance_field'` → `41 passed, 1 deselected, 1 warning`.
- The warning is the pre-existing pytest cache/temp-directory permission warning on this host.

### Self-review and concerns

- Required validation edges are taken from the registration spanning tree, so an unavailable critical edge cannot be hidden by good statistics from another edge.
- Smoke quality now describes the recropped registered arrays before the run-level quality gate.
- No-pair and disconnected results remain consumable by callers without exception handling changes.
- The standalone `tmp_path` regression remains unable to start because pytest’s protected temporary directory is inaccessible in this environment; it is unchanged from the original Task 6 verification limitation.

### Commit

- Fixes committed with the required Task 6 follow-up commit message.
