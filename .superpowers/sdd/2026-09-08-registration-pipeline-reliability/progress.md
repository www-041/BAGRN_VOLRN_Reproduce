# SDD ledger — plan: docs/superpowers/plans/2026-09-08-registration-pipeline-reliability.md

## Setup

- Ruling: execute in the current `fix/dz01v-mosaic-series-readiness-20260907` checkout — it is a dedicated non-main repair branch, and creating another worktree would add unnecessary repository state while the current checkout is already the user's scoped workspace. Cost if wrong: changes would be less isolated than a separate worktree, but every task is committed and the branch can be reviewed or reverted.
- Baseline: HEAD `33fe74e` after the approved design and implementation-plan commits.
- Pre-existing untracked artifact: `data/report_figures/` remains untouched.
- Baseline test evidence: `203 passed, 23 failed, 8 errors`; the failures/errors were dominated by permission denial under the system temporary directory.

## Plan conflict scan

| Tasks | Shared file or interface | What producer/consumer requires | Finding and ruling |
|---|---|---|---|
| 1 ↔ 3 | `src/coregistration.py`; pair measurements and rematch results | Task 3 consumes pair/rematch dictionaries created or normalized by Task 1 | Compatible. Ruling: preserve `shift_dx/shift_dy/confidence/matches` keys and add counts without removing old keys. |
| 1 ↔ 6 | `src/multiband_pipeline.py`; `register_scenes()` | Task 1 supplies robust pair edges; Task 6 replaces orchestration | Compatible if Task 6 retains Task 1's pair schema. Ruling: Task 6 must call the Task 1 helper rather than duplicate estimation. |
| 2 ↔ 4 | `src/coregistration.py`; displacement fields and confidence | Task 4 consumes fields and uses NoData-safe final warp | Compatible. Ruling: local fields are target-image sized, component-clipped, and passed to the one-shot multiband warp. |
| 2 ↔ 5 | `src/coregistration.py`; warp and validation masks | Task 5 validates arrays produced by Task 2 | Compatible. Ruling: validation treats `nodata=None` with finite masks only. |
| 3 ↔ 4 | `src/multiband_pipeline.py`; global refinement then local RBF | Task 4 must use post-global residual matches from Task 3 | Compatible and sequential. Ruling: no local controls are built from pre-global matches. |
| 3 ↔ 6 | `src/multiband_pipeline.py`; refinement history and final warp | Task 6 consumes Task 3 global shifts/history | Compatible. Ruling: all temporary arrays are disposable diagnostics; final arrays are rebuilt from ORIGINAL. |
| 4 ↔ 5 | `src/multiband_pipeline.py`; local diagnostics then final validation | Task 5 validates the combined global+local result | Compatible. Ruling: local fallback remains a valid path to final validation. |
| 5 ↔ 6 | quality dictionary and requirement comparison | Task 6 must gate normalization using final quality | Compatible. Ruling: `registration["quality"]["quality"]` is authoritative; connectivity is a separate required condition. |
| 6 ↔ 7 | stable registration result and diagnostic JSON | Task 7 serializes Task 6 fields | Compatible. Ruling: diagnostic code reads actual schema and uses `.get()` only for optional compatibility fields. |
| 1 | robust pair tests vs estimator implementation | Tests require joint XY rejection, block-level counts, weighted inlier mean | Internally consistent. |
| 2 | bug tests vs fixes | Tests cover distance field, `>=` confidence, and `nodata=None` zero preservation | Internally consistent; exact existing key is `n_spike_pixels`, used in the plan. |
| 3 | refinement tests vs helper | Tests require ORIGINAL inputs and iteration history | Internally consistent; monkeypatched rematch/warp isolate the behavior. |
| 4 | RBF tests vs gate | Tests require accepted/rejected CV outcomes | Internally consistent; helper must expose the stated acceptance result. |
| 5 | validation tests vs thresholds | Tests require configured threshold parameters and final aggregation | Internally consistent; validation signature must add explicit threshold arguments. |
| 6 | orchestration tests vs schema | Tests require `connected`, `local_refinement`, and `final_validation` | Internally consistent after adding the named quality helper. |
| 7 | diagnostic test vs script | Test requires `build_diagnostic_payload()` and actual schema fields | Internally consistent; payload builder is the testable boundary. |
| 8 | verification commands vs project | Existing preflight CLI requires `--scene-count`; the plan uses it | Ruling: use the corrected command in the saved plan and classify system-temp permission failures separately. |

## Task checklist

- Task 1: complete (commits 33fe74e..14fdf83, review clean)
- Task 2: complete (commits 14fdf83..de78393, review clean)
- Task 3: complete (commits de78393..121c869, review clean after one fix round)
- Task 4: complete (commits 121c869..32f3e2b, review clean after two fix rounds)
- Task 5: complete (commits 32f3e2b..8af1ced, review clean after one fix round)
- Task 6: complete (commits 8af1ced..1b3f79a, review clean after one fix round)
- Task 7: complete (commits 1b3f79a..d2d20b5, review clean after one fix round)
- Task 8: complete (verification report recorded; no source correction required)

Task 1: complete (commits 33fe74e..14fdf83, review clean)
Task 2: complete (commits 14fdf83..de78393, review clean)
Task 3: fix round 1/5 (2 Important findings open, 1 Minor noted; implementer 66c5fd2)
Task 4: fix round 1/5 (4 Important/P2 findings open; implementer 3ca33d6)
Task 4: fix round 2/5 (1 Important finding open; implementer 2c8659e)
Task 5: fix round 1/5 (1 Critical/P1 and 3 Important/P2 findings open; implementer 5d8794b)
2026-09-08 — Task 6 complete — commit baa9c86; report recorded schema/gate changes, focused tests passed, full-suite temp-permission limitation noted.
Task 6: fix round 1/5 (2 Critical and 1 Important findings open; implementer resumed 01a0817a)
Task 6: fix round 1/5 (3 findings addressed, 0 open; commits baa9c86..1b3f79a)
Task 6: complete (commits 8af1ced..1b3f79a, review clean after one fix round)
Task 7: fix round 1/5 (2 Important findings open; implementer resumed 01a08192)
Task 7: fix round 1/5 (2 findings addressed, 0 open; commits 40b4607..d2d20b5)
Task 7: complete (commits 1b3f79a..d2d20b5, review clean after one fix round)
Task 8: complete (focused non-temp tests 46 passed; full suite classified as environment-limited; config checks recorded)
2026-09-09 — Final review fix wave complete — the Critical and four Important findings are fixed with focused regressions; the active robust pair estimator also now follows the plan's iterative MAD wording, and the Task 1/2 checklist contradiction is corrected. Final focused verification: 52 passed, 1 deselected. Full-suite non-pass results remain environment-limited to protected temporary/raster output and console encoding paths; see `final-review-fix-report.md`. Remaining minor observations (local smoothing-candidate selection and broader mocked orchestration coverage) are recorded as non-blocking concerns.
