# Task10E ADMM Equation Implementation Audit

Date: 2026-09-27

## Model audited

The implementation uses the split form

```text
min_x 1/2 ||B x||_2^2 + lambda ||A x - b||_1
z = A x - b
```

with scaled dual variable `u`. The audit covers `src/volrn.py` and does not
change the frozen Task10D protocol.

## Equation-by-equation result

| Item | Implementation | Result |
|---|---|---|
| `B` mean constraint | Each overlap pair row is `mu_i*a_i+b_i-mu_j*a_j-b_j`; sigma row is `sigma_i*a_i-sigma_j*a_j` | PASS |
| `A`, `b` identity prior | Each block has `mu*a+b = mu` and `sigma*a = sigma`, so `b=[mu,sigma]` | PASS |
| x update | `M=B.T@B+rho*A.T@A`; `rhs=rho*A.T@(z-u+b)`; solve `M x = rhs` | PASS |
| z update | `soft_threshold(A@x-b+u, lambda/rho)` | PASS |
| scaled dual update | `u = u + A@x-b-z` | PASS |
| primal residual | `r = A@x-b-z` | PASS |
| dual residual | `s = rho*A.T@(z-z_previous)` | PASS |
| primal stopping tolerance | `sqrt(m)*tol + tol*max(||A@x-b||, ||z||)` | PASS |
| dual stopping tolerance | `sqrt(n)*tol + tol*||rho*A.T@u||` | PASS |
| objective diagnostic | `0.5*||B@x||^2 + lambda*||A@x-b||_1` | PASS |
| science stop | Requires both residual tests, no CG failure, and finite state | PASS |

The x update follows from the augmented Lagrangian stationarity equation for
the stated split. The soft-threshold threshold and sign handling are correct
for the L1 term. The residual definitions are the standard scaled-ADMM
residuals and use the newly updated `z`/`u` values in the stopping test.

## Findings

No equation mismatch or implementation error was found in the audited ADMM
updates. The Task10E history shows a monotonically decreasing dual residual,
but it remains `23178.59068649828` at iteration 200 versus a dual tolerance of
`2.899373405241831`. The final relative x change is only
`4.3642382943778173e-07`; therefore a small x update must not be mistaken for
ADMM convergence. This supports the Task2 classification of slow convergence,
not an equation-alignment repair.

The CG diagnostic records the residual of the returned linear solve and the
callback iteration count; it is diagnostic instrumentation only and does not
alter the x-update tolerances or the scientific parameters.
