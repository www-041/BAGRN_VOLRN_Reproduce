# DZ01V Unified Benchmark V1 — protocol draft / Phase 1 freeze

Protocol ID: `DZ01V_UNIFIED_A_B_V1_20261013`

Scope: 13 processed DZ01V V-sensor B9 scenes; one band; pixel unit `UNKNOWN`; no claim of surface reflectance.

## Evaluation regimes

- **A — final mosaic diagnostics:** common input-union coverage, holes/extras, and structure on the unique-source-only area. Seam excess is reported only with source provenance and a frozen seam definition. Method-specific seams remain secondary diagnostics, never the sole common comparison support.
- **B1 — full pair support:** all 48 predeclared positive canonical-grid pairs, using each frozen input-mask intersection. The other 30 of 78 pairs remain `NO_VALID_OVERLAP`, not matcher failures.
- **B2 — fixed local windows:** 128×128 windows, 128-pixel stride and `(row=0,col=0)` grid anchor; at least 95% input pair support; at most 32 per pair; spatially dispersed across 512-pixel supercells using deterministic hash tie-breaking and seed `20261013`. Labels are `UNLABELED_LOCAL_CONSISTENCY`, not stable/changed truth.
- **B3 — known-truth synthetic S1–S4:** existing R1 generator with fixed seed/config and arbitrary synthetic units. It is not evidence of real seasonal generalization or real-scene change preservation.

## Inputs and support

Input identity is the source B9 path + SHA256 + scene identity in the frozen source inventory. Geometry is the Task15 V2 world-transform set and canonical north-up grid: EPSG:32650, 14 m, affine `(14,0,633346,0,-14,3359874)`, 7992×12404. Intensity uses the Task15-verified bilinear warp; valid masks are derived from declared source nodata and finite reprojected values and compared to the persisted Task15 input masks. Scores use identical frozen input support for every method; an output may not choose a smaller scoring mask.

Primary pair scores require full frozen support. `PARTIAL_OUTPUT` remains diagnostic-only and is not ranked. Missing and mathematically undefined values are `null` with explicit status, never numeric zero. Every table retains all 78 expected pair IDs and separately reports positive support, no-overlap and method-output failure states.

## B1 metric definitions

For paired values `x,y` on the same frozen valid pair support: MAE=`mean(|x-y|)`, RMSE=`sqrt(mean((x-y)^2))`, BIAS=`mean(x-y)`, P95AE=`linear_quantile(|x-y|,.95)`, MAMD=`|mean(x)-mean(y)|`, MSDD=`|std_ddof0(x)-std_ddof0(y)|`, and RDD=`exact empirical 1-D Wasserstein-1`. Primary aggregate is pair macro; pixel-weighted, pair median/P95 and full per-pair distributions are secondary. Pair consistency is not accuracy against truth.

The exact paper `CD` and `GL` formulas are not verified in accessible Section 3 full text. Do not call W1 `CD`, gradient NCC or V1 Sobel direction loss `GL`, or compute paper `RDOA`/`Ave` from substitutes.

## Structural and range diagnostics

Use fixed 3×3 Sobel kernels divided by 8; require all 9 pixels valid; use a one-pixel halo. Input-only edge threshold is `max(1e-12, P50(input gradient magnitude))`. Report `SOBEL_DIRECTION_LOSS_V1=mean(1-cos)` on input edges and `REL_GRAD_MAE_V1` over valid centers. Report robust range ratio `(Q95-Q05)_output/(Q95-Q05)_input` using linear quantiles. These are preservation diagnostics, not direct quality truth. SSIM is disabled until a source-only range contract is verified; gradient NCC is separate optional context.

For regime A, the primary structure comparison uses `U_single`, the pixels with exactly one valid source, comparing the mosaic only to that unique source. It does not penalize a legitimate source choice in multicoverage areas. Seam metrics require valid source labels/weights and record eligible edge counts/length.

## B3 synthetic known truth

Reuse `src/benchmark_v1_radiometric_r1/synthetic.py::generate_case`; record seed, generator/config, NPZ hash, truth anchor and support. Report truth MAE/RMSE, changed-area delta MAE/RMSE, stable-area spurious delta MAE, and change magnitude ratio only with nonzero truth change. No correction algorithm is run in Phase 1.

## Time, quality and ranking

Report per-pair results and pair-macro values stratified by the frozen temporal bins and scene quality flags. Runtime begins before method-specific preprocessing and ends after closing output artifacts. RAM/GPU fields are observed measurements or `NOT_MEASURED`, never zero by default. Cross-method ranking requires the same protocol/input/geometry hashes, complete frozen pair/window population, full support, valid values and complete runtime provenance. No universal weighted score.

## Split and generalization caveat

This protocol's pair consistency uses all declared overlap pixels; it does not assert spatially held-out or cross-region generalization. Existing development/test checkpoint roles do not enter radiometric metric support or method fitting. Internal 13-scene data do not constitute independent cross-region validation.

## Freeze state

See `data/output/benchmark_v1_unified/freeze_20261013_v3/freeze_readiness.json` and `freeze_lock.json`. A freeze locks data identity, transform/grid, input masks, pair/window inventory, metric specification, code/test manifest and synthetic fixture provenance only. It evaluates no radiometric method.
