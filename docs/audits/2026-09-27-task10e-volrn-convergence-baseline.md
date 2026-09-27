# Task10E VOLRN Nonconvergence Baseline

Date: 2026-09-27

## Frozen experiment

This baseline records the Task10D B9 science-gate failure before any Task10E
diagnostic instrumentation. The geometry is the immutable EfficientLoFTR with
Translation-L2 replay, using the B9 manifest indices `[2, 3, 5, 8, 10]`, band
`B9`, 14 m pixels, scene `0` as control, weighted-feather mosaic, and cloud
mask disabled.

The VOLRN parameters are frozen at:

```text
block_size_pixels = 400
lambda = 0.5
rho = 1.0
max_iter = 200
tol = 1e-4
```

The Task10D metric protocol SHA-256 is:

```text
dadd7ff9fbe3702b93be97070b549754d2eeccc921a1c4f6575eee97d09e73a0
```

The corresponding fixed gate output is
`data/output/b9_five_scene_validation/radiometric_gate_1024_task10d/efficient_loftr_translation_l2/`.
BAGRN passed; BAGRN+VOLRN completed at the 200-iteration cap without
convergence. The recorded solver state was objective
`42304.73224541148`, primal residual `0.011755229749852653`, dual residual
`23178.59068649828`, `cg_failed=false`, and finite state `true`.

## Machine-readable status

```json
{
  "status": "NONCONVERGED_BASELINE",
  "max_iter": 200,
  "lambda": 0.5,
  "rho": 1.0
}
```

No matcher, RANSAC, global geometry, BAGRN, or VOLRN scientific parameter was
changed to create this baseline. It is the reference for Task10E diagnosis.
