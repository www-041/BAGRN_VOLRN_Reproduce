# DZ01V Benchmark V1 Phase4 preparation status

Scope: four frozen trial pairs only. Full 15-pair Benchmark and remaining five B9 version checks remain open. No formal matcher, radiometric, Task15 or Task16 run is part of this phase.

## Frozen and human-review state

- Phase3: 43 DEVELOPMENT, 84 TEST candidates, 33 UNASSIGNED; 10 km EPSG:32650 grid, fixed `(0,0)` origin, 500 m edge margins and role formula unchanged.
- Eight trial-scene B9 SHA256 values matched Phase3 records in the Phase4 read-only check. The other five scenes still require version verification before the full 13-scene freeze.
- The accepted four-pair Phase3 audit itself is pinned at SHA256 `44765f95717b742f8e0c08bc6bc14d51bb3574b5c921d8cdd444c17877c20c43`; Phase4 refuses a changed audit even if its internal hashes were redefined.
- Current second-review preflight: 84 pending, 0 human-approved-spatial-valid, no formal TEST truth written. A distinct *real* second reviewer must inspect each chosen TEST point; software validates declared identities but cannot authenticate people.
- Existing first-review quality caution: all 160 candidate rows say HIGH; none is marked human-adjusted. The localization note has 155 approximately ±0.5-pixel entries in one formatting, one additional ±0.5-pixel formatting variant, and four approximately ±1-pixel entries. These are human labels, not independently measured precision. Second reviewers should pay particular attention to ambiguous objects, low texture, shadows and bright spots and supply their own uncertainty estimates.
- Frozen role and human decision are separate. An approved event remains excluded if coordinates leave the original safe core, fail support/isolation, have a prior invalid adjustment, or remain disputed. A signed non-use declaration is additionally required before formal TEST export.
- The formal TEST gate reprojects every logged native row/column through the verified B9 affine and requires its stored projected coordinate to agree (absolute tolerance `1e-6` projected units). It recomputes each event's `adjusted`, spatial reasons and sticky ineligibility from the whole history; a stored `VALID` string cannot conceal a failed adjustment. The second-review GUI verifies that its displayed B9 scene paths and SHA256 match the Phase3 audit before opening controls.

## DEVELOPMENT readiness

Four-pair DEVELOPMENT counts are 11/9/9/14. The common 2D translation result schema and synthetic metrics/FAIL handling are prepared, but no six-method run was started. Local dependency discovery found `cv2`/SIFT, Kornia, PyTorch and LightGlue packages; KARIOS and AROSICS packages were not found in `.venv-registration`. **Every method remains `NOT_VERIFIED_END_TO_END` for this Phase4 trial.** AROSICS global `COREG` belongs to the main global-translation comparison; `COREG_LOCAL` is separate.

## Verification and limits

Phase4 engineering-verification rerun (2026-10-11): the seven first-review GUI regressions were individually traced to tests that loaded the real, already-approved `00_01` candidate CSV while asserting an UNVERIFIED trial state. Their tests now use disposable synthetic GeoTIFFs, scene metadata and 40-point candidate CSVs; the visibility, counts, review, batch-edit and save assertions remain active. Both GUI modes bind Tcl/Tk resources to the active Python installation, retry only transient resource-read initialization errors, and destroy the Tk root when the main loop returns. Synthetic second-review GUI tests cover selection, approval, coordinate adjustment, rejection, append-only event save and reload. Focused Phase4/GUI checks: 40 passed. Full repository suite: **1058 passed, 4 skipped, 0 failed**. `compileall` and `git diff --check` passed. Four-pair TEST remains at 84 pending and zero formal truth; no actual second review or matcher experiment was run. Tcl/Tk still depends on the local Python installation, so a manual launch on the researcher's machine remains advisable.

## Commands and next human actions

Read-only TEST count: `python scripts/export_benchmark_v1_test_truth.py --preflight`.

Start independent review, with the actual new reviewer's own identity: `python scripts/annotate_benchmark_v1.py --second-review --pair 00_01 --reviewer "SECOND_PERSON"`. Replace `00_01` for the other three pairs. Detailed workflow: [second_review_usage.md](second_review_usage.md).

Read-only DEVELOPMENT dependency/input check: `python scripts/prepare_benchmark_v1_development.py --check-env`. Optional small manifest: add `--write-manifest docs/benchmark/v1/phase4/development_input_manifest.json`. There is no Phase4 command that runs all six algorithms; a later experimental instruction must supply one after adapter validation and installation.

Before formal TEST scoring, actual independent reviews must be completed, disputes resolved by humans, exclusion counts checked, all required B9 identities verified, and a truthful non-use declaration supplied. No synthetic event or guessed reviewer identity may be promoted into the real event directory.
