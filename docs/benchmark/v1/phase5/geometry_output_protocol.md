# Phase5 unified geometry output protocol

## Coordinate convention

- Image coordinates are zero-based native B9 pixel-center `(row, col)` values; displacement is scene-i to scene-j, in native pixels.
- The common matching view is a fixed, shared-overlap raster. Its affine, crop, resize, stretch, resampling and mask provenance are recorded in the protocol lock/input provenance.
- Correspondence-producing methods use the same frozen translation estimator in `configs/benchmark_v1_phase5.yaml`. Direct AROSICS global COREG output is converted from its native corrective-shift convention and retains separate solver provenance.
- `GEOREF_ONLY` maps between the two native grids via their affines. It is a diagnostic baseline, not a Matcher and is excluded from the six-method macro ranking.

## Per-run records

Each explicit one-pair/one-method run writes to a new child of `data/output/benchmark_v1_phase5/`:

| File | Purpose |
|---|---|
| `protocol_lock.json` | Frozen config values/hash, method, pair, execution mode |
| `input_provenance.json` | B9 identities, sizes/hashes, shared-view and mask provenance |
| `method_result.json` | Status, native i-to-j translation, matches/inliers/coverage where applicable, timings and resource observations |
| `development_metrics.json` | Approved DEVELOPMENT-only independent checkpoint metrics, when requested and available |
| `georef_only_metrics.json` | Separate diagnostic GEOREF_ONLY score |
| `status.json` | Run/evaluation state and classified failure/unavailability reason |
| worker/resource logs | Captured stdout/stderr and supervisor resource samples |

Execution status, evaluation status and data availability are separate. `ENV_UNAVAILABLE`, `WEIGHTS_UNAVAILABLE`, `NO_VALID_OVERLAP`, `PREPROCESSING_FAIL`, `ALGORITHM_FAIL`, `SUCCESS`, and `NO_INDEPENDENT_TRUTH` must not be collapsed into one failure count. Metrics remain null when there is no eligible independent truth. TEST coordinates and second-review records are never passed to matcher workers.

AROSICS `COREG_LOCAL` is a different model and is not part of the global translation table. AROSICS global COREG also uses a different native solver from correspondence-based methods; results do not isolate feature extraction alone.
