# TASK10C_BLOCKED

Date: 2026-09-26

## Hard-stop reason

Task10C is stopped before Task12 real-data execution. The repository does not
contain the paper PDF, equation source, supplementary material, or a trusted
reference implementation that uniquely verifies the paper definitions of
Eq.(38) CD and Eq.(39) GL. The implementation in
[`src/metrics.py`](src/metrics.py) and internal reports are not sufficient
source material for that verification.

The plan explicitly forbids guessing a paper formula. Therefore no LoFTR
replay, VOLRN science gate on real B9 data, or final six-row B9 experiment was
run.

## Completed before the stop

- Tasks 0–5: frozen protocol/preflight, matcher fairness/session fixes,
  geometry/global hard-fails, geometry provenance, and BAGRN valid-overlap
  semantics.
- Task 6: explicit ADMM primal/dual residual convergence, CG failure
  propagation, finite-iterate retention, solver diagnostics, and removal of
  unverified global percentile scaling.
- Task 7: explicit valid-mask propagation, normalized NoData semantics, and
  cloud-mask non-mutation.
- Task 8: histogram-after fix, Sobel stencil validity guard, and formula audit
  saved at `docs/audits/2026-09-26-task10c-metric-formula-audit.md`.
- Task 9: strict protocol parameters, source/grid/transform/protocol resume
  checks, partial-output preservation, explicit science statuses, and fixed
  radiometric-control metadata.
- Task 10: explicit scientific float32 mosaic output while preserving the
  geometry-only dtype contract.
- Task 11: synthetic local-radiometry, nonconvergence, and six-metric-schema
  invariants.

## Verification evidence

- Focused Task10C suite: `449 passed, 2 skipped, 8581 warnings`.
- Full repository suite: `807 passed, 4 skipped, 10182 warnings`; exit code 0.
- Static checks: `git diff --check` and `python -m compileall -q src scripts`.

## Not run

Task12 LoFTR replay, Task13 real-data VOLRN gate, Task14 formal six-row B9
rerun, and Task15 final real-artifact audit were not run because they depend on
the unverified paper metric definitions.

`push_performed=false`. No merge or push was performed.
