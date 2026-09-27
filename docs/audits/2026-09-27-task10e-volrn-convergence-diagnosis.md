# Task10E VOLRN ADMM Convergence Diagnosis

Date: 2026-09-27  
Branch: `exp/2026-09-24-b9-five-scene-validation`  
Start HEAD: `d821db2`  
Final HEAD: recorded after this report commit

## Scope and frozen protocol

Task10E diagnosed the Task10D B9 `EfficientLoFTR + Translation-L2` VOLRN
nonconvergence without changing matcher, RANSAC, global geometry, BAGRN, or
VOLRN scientific parameters. The frozen parameters were:

```text
block_size_pixels = 400
lambda = 0.5
rho = 1.0
max_iter = 200
tol = 1e-4
```

The Task10D metric protocol SHA-256 was
`dadd7ff9fbe3702b93be97070b549754d2eeccc921a1c4f6575eee97d09e73a0`; the
frozen source-config SHA-256 was
`b9181afcc972b29716d309f7aa3e82d3fbdba57fd5d466e2c0aa04d25060c9ba`.

## Evidence by task

### Task 0 — baseline

`docs/audits/2026-09-27-task10e-volrn-convergence-baseline.md` records the
Task10D failure as `NONCONVERGED_BASELINE`.

### Task 1 — iteration history

`src/volrn.py` now records all requested per-iteration fields, including the
update norms and CG diagnostics. The fixed runner writes
`volrn_solver_history.json` and `volrn_solver_history.csv`.

The frozen B9 diagnostic run has 200 records. The run remains
`COMPLETED_NONCONVERGED`; CG did not fail and all states were finite.

### Task 2 — convergence curves

The diagnostic directory contains the four requested PNG curves and
`convergence_summary.json`. Objective and dual residual decrease throughout;
primal residual decreases overall with oscillations. The classifier is
`slow convergence`. At iteration 200:

```text
objective = 42304.73224541148
primal residual = 0.011755229749852653
dual residual = 23178.59068649828
dual tolerance = 2.899373405241831
relative x change = 4.3642382943778173e-07
```

### Task 3 — equation audit

`docs/audits/2026-09-27-task10e-admm-equation-audit.md` audits B/A construction,
x update, L1 shrinkage, scaled-dual update, residuals, tolerances, objective,
and science gating. No equation mismatch was found.

### Task 4 — matrix conditioning

`volrn_matrix_diagnostics.json` reports:

```text
num_blocks = 461
num_pairs = 618
num_variables = 922
matrix_density = 0.00798509323784473
condition_estimate = 193076169765.823
row/column norm range = 2282.8271484479305 .. 57909685.59782624
normalization_applied = false
```

This is evidence of severe scale/conditioning pressure, not an automatic
normalization or a protocol change.

### Task 5 — application audit

`volrn_application_audit.json` confirms the solver coefficients were applied
to all five scenes. Every scene had `changed_fraction = 1.0`; mean absolute
differences were `25.77758932690131`, `920.4402143148616`,
`289.12316959171716`, `297.6353707829898`, and `121.63572549076328`.
Coefficient ranges were `a=[0.17097971223261124, 1.3890055691227192]` and
`b=[-1462.5169222963455, 1480.4836628564187]`, with 922 finite coefficients.

### Task 6 — synthetic checks

`synthetic_volrn_convergence_report.md` covers all three requested cases:

| case | result |
|---|---|
| global gain `image2 = 1.2 * image1` | converged; pair MAE `23.72 → 0.510036` |
| local gain variation | converged; pair MAE `23.72 → 5.99626`; non-constant local gain |
| identical images | converged in 1 iteration; exact identity coefficients |

## Decision gate

Outcome A applies: synthetic cases pass, while the frozen B9 case exhibits slow
convergence. The evidence does not support an implementation-issue or equation-
mismatch conclusion, and does not justify changing parameters.

## Verification

```text
focused Task10E suite: 32 passed
full pytest: 831 passed, 4 skipped, 0 failed
python -m compileall -q src scripts: passed
git diff --check: passed
```

The four skips are the existing environment/data-dependent skips; no pytest
failure occurred in this run. Existing untracked temporary directories were
preserved. No push, merge, Task14 expansion, seamline optimization, or large-
scale extension was performed.
