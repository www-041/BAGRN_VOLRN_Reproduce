# DZ01V MODEL-C1: B12 robust-affine residual causal diagnostic

## Frozen contract

MODEL-C1 compares the existing B12 global-only result with a diagnostic-only
counterfactual that adds one TRAIN-selected robust affine residual field.
Both use the same B14 validation band and the same seven reserved 384×384
HOLDOUT windows from the BAND-C1 manifest.

The control is `ORIGINAL + frozen B12 global shift`. The treatment is
`ORIGINAL + the same frozen B12 global shift + affine residual field` in one
call to `warp_multiband_with_displacement_field`. The treatment contains no
RBF field and does not use `warp_affine_once`.

Affine fitting, buffered spatial CV, safety checks, and model selection use
TRAIN controls only. Final HOLDOUT is evaluated only after the TRAIN gate
passes and is never used to tune thresholds or choose a model. The canonical
production arrays, quality, status, and failure fields remain unchanged.

The base registration-parameter fingerprint filters only the explicit
`AFFINE_CAUSAL_PARAM_KEYS`, so adding diagnostic parameters does not invalidate
the BAND-C1 base manifest. Safety limits are engineering gates, not paper
quality thresholds. A supported result only motivates a later AFFINE-P1 plan;
it does not authorize N=4/N=6, BAGRN/VOLRN, DEM, or terrain-aware changes.
