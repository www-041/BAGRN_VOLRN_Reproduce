# Task10D Real B9 VOLRN Science Gate

Date: 2026-09-27

## Gate decision

`SCIENCE_GATE_NONCONVERGED`

The fixed EfficientLoFTR + Translation-L2 B9 gate was executed with the
frozen Task10C parameters:

```text
block_size_pixels = 400
lambda = 0.5
rho = 1.0
max_iter = 200
tol = 1e-4
```

BAGRN completed with `PASS_BAGRN`. BAGRN+VOLRN reached the iteration cap with
`science_status = COMPLETED_NONCONVERGED`, so the plan hard-stop applies.
Task14 final six-run expansion and Task15/16 finalization were not executed.
No VOLRN parameter tuning was attempted.

## Provenance checks

Both gate runs use the same frozen provenance:

- source config SHA-256: `b9181afcc972b29716d309f7aa3e82d3fbdba57fd5d466e2c0aa04d25060c9ba`
- canonical output-grid SHA-256: `fb27e5cafbb05cfc48103abc5500e7640eee189c40eebcf7509fb171cbf7c0cd`
- EfficientLoFTR Translation-L2 global-transform SHA-256: `7bc3cd5c43912f43f8daa5ce233ded12f1ce5444210361bf9293f71af620de8a`
- Task10D metric protocol SHA-256: `dadd7ff9fbe3702b93be97070b549754d2eeccc921a1c4f6575eee97d09e73a0`
- manifest indices: `[2, 3, 5, 8, 10]`; band: `B9`; pixel size: `14 m`
- control scene index: `0`; weighted feather mosaic; cloud mask disabled; geometry immutable

The gate outputs are under
`data/output/b9_five_scene_validation/radiometric_gate_1024_task10d/efficient_loftr_translation_l2/`.

## Solver evidence

The BAGRN+VOLRN run produced 461 blocks and 618 block pairs. Its single B9
solver record reports:

| field | value |
|---|---:|
| iterations | 200 |
| objective | 42304.73224541148 |
| primal residual | 0.011755229749852653 |
| primal tolerance | 0.4322821717610789 |
| dual residual | 23178.59068649828 |
| dual tolerance | 2.899373405241831 |
| CG failed | `false` |
| finite state | `true` |
| converged | `false` |
| science pass | `false` |

The dual residual remains far above its tolerance at the frozen iteration cap;
this is a genuine convergence failure, not a reporting omission.

## System-state and non-identity diagnostics

The output remains finite on all valid pixels: all five scenes report zero
non-finite valid pixels. The VOLRN coefficient array is finite with 461
`(a,b)` blocks. For the final coefficients:

- `a` range: `[0.17097971223261124, 1.3890055691227192]`
- `max |a-1|`: `0.8290202877673888`; mean `|a-1|`: `0.07022100254678043`
- `b` range: `[-1462.5169222963455, 1480.4836628564187]`
- `max |b|`: `1480.4836628564187`; mean `|b|`: `173.63315302399033`
- non-identity blocks: 404 for `a`, 404 for `b`
- changed valid-pixel fraction: `1.0` for each scene

Thus finite-state and non-identity checks pass, but they do not override the
failed convergence gate.

## Task10D primary metrics (diagnostic only for this gate)

Metrics were computed under the frozen protocol for both BAGRN and
BAGRN+VOLRN. They are not used to waive the convergence gate and are not
interpreted as paper Eq.(38) CD or Eq.(39) GL.

| metric | BAGRN | BAGRN+VOLRN |
|---|---:|---:|
| weighted MAMD | 21.4285575265 | 7.18612550685 |
| weighted MSDD | 9.98587332712 | 4.52234026043 |
| weighted RDD | 31.1826282677 | 14.1438766039 |
| Local MAMD median | 57.0313244384 | 31.6382335989 |
| Local RDD median | 63.9667731128 | 38.8690037978 |
| seam MAE weighted | 123.659349085 | 104.766230386 |
| seam RMSE weighted | 177.337445238 | 156.652826887 |
| seam RDD weighted | 28.6460913159 | 14.5002684945 |
| CGL radians | 1.16182553305e-15 | 0.0057649364856 |

These values are retained for diagnosis only. The hard-stop is caused by
nonconvergence at the frozen cap, independently of whether the metrics move
in a favorable direction.

## Stop point

Stopped before Task14. No final six-run root was created, no scale-up or
seamline work was performed, and no push or merge was performed.

