# Stage 1: Four-Scene Multiband Baseline Reproduction

## Objective

Reproduce the BAGRN-VOLRN method using a verified connected four-scene subset of DZ01V imagery.
This stage establishes a stable, reproducible baseline before any innovation work.

## Deferred Experiments

The following experiments are postponed until after the baseline is established:

- **sensitivity**: Parameter sensitivity analysis (block_size, lambda, rho sweeps)
- **scale**: 8-scene and 12-scene expansion experiments
- **diagnostics**: Full VOLRN coefficient and RBF displacement field diagnostics
- **seam-width sweep**: Narrow feather width scanning
- **full spectral experiment**: Standalone spectral fidelity validation
- **5/8/12-scene expansion**: Multi-scene connectivity and scaling studies

**Reason for deferral**: Current stage establishes a credible four-scene multiband baseline.
Later experiments will be redesigned based on confirmed innovation directions from the paper.

## Stage 1 Scope

### Methods (4 only)
1. `original` — registered, no radiometric normalization (baseline reference)
2. `bagrn` — global radiometric normalization only
3. `volrn_only` — local radiometric normalization only (no BAGRN pre-processing)
4. `bagrn_volrn` — BAGRN + VOLRN (paper main method)

### Excluded from Stage 1
- `histogram_matching`, `moment_matching`, `wallis` — traditional baselines (run after innovation direction confirmed)
- `narrow_feather` and feather width scanning
- Parameter tuning (block_size, lambda, rho)
- Multiple mosaic modes (only `weighted`)

### Four-Scene Subset
Selected based on verified connectivity through block matching:

| Index | Scene ID | Date | Role |
|-------|----------|------|------|
| 0 | scene_20251114 | 2025-11-14 | Control scene |
| 1 | scene_20251120 | 2025-11-20 | — |
| 2 | scene_20251208 | 2025-12-08 | — |
| 3 | scene_20251215 | 2025-12-15 | — |

**Excluded**: `scene_20251214` — not part of the verified connected four-scene subset
(the edge 2→3 from the 5-scene registration involves scene_20251208→scene_20251214, but
scene_20251214 is only reachable through scene_20251208, which itself becomes unreachable
on small crops in smoke mode).

### Overlap Graph (4 scenes)
- Geometric overlaps: (0,1), (0,3), (1,2), (1,3), (2,3)
- Valid matching edges: (0,1), (0,3), (1,2)
- Spanning tree: (0→1), (1→2), (0→3)
- Control scene: scene_20251114 (index 0)

### Bands
B01–B14 (14 bands), registration on B14, strict mode.

### Mosaic
Only `weighted` mode, identical output grid/resolution/projection for all methods.

### VOLRN Parameters (fixed)
- block_size: 200
- lambda: 0.1
- rho: 1.0
- max_iter: 200
- tol: 1e-4

## Success Criteria
1. Four scenes loaded, all B01–B14 present
2. Registration connected, all scenes reachable from control
3. All four methods complete successfully
4. All methods share the same registration result
5. All methods share the same mosaic mode (weighted)
6. All methods share the same visualization stretch
7. Full radiation metrics (ADM, ADSD, CD, GL, RDOA, Ave) per band, per pair, per pair-band
8. Full spectral metrics (SAM, spectral RMSE, correlation)
9. Data quality checks pass (no NaN/Inf in valid regions)
10. Exit code 0
