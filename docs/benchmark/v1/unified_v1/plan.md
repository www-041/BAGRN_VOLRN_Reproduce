# DZ01V Unified Benchmark V1.0 -- Phase 1 Freeze and Evaluator Implementation Plan

**Document purpose:** implementation contract for Codex; not evidence of completed real-data scoring.  
**Date:** 2026-10-13  
**Proposed protocol ID:** `DZ01V_UNIFIED_A_B_V1_20261013`  
**Status:** `PROPOSED_FOR_REPO_AUDIT`, subsequently `FROZEN` only after the gates in section 14 pass.  
**Scope:** 13 processed DZ01V optical VNIR scenes; B9 first; 16-band extension only after independent per-band identity, calibration and registration review.

## 0. Non-negotiable research and implementation boundaries

1. **Two distinct settings**: **A** evaluates the finished 13-scene mosaic on a fixed geometry; **B** evaluates radiometric normalization when individual corrected scenes are available. They share frozen source inputs and geometry but **do not share one universal score**.
2. **Three evaluation layers for B**: B1 = complete pre-frozen valid overlap of all 48 positive-overlap pairs; B2 = predefined deterministic spatial windows within those overlaps; B3 = S1--S4 controlled synthetic scenarios with known numerical ground truth. No manual land-cover annotations are prerequisite for these layers.
3. Do **not** develop, optimize, modify or re-run the abandoned in-house Ours method. Keep Task15, Task16, radiometric R1/R1.5, Phase3/4/5 and their histories strictly read-only. BAGRN+VOLRN (2024) is an optional archival/reference baseline, not the development center.
4. Candidate external methods: Spectralmatch (2026; A+B where scene products exist), Harmoni-Planet (2025; B and A with an explicitly named shared mosaicking backend), RS-NormGAN (2025; pair-specific B only), LIRRN (2024; pair-specific B only). No method is declared runnable until its real interface has been audited. A later geometry/mosaicking study may examine RT-UDRSIS (2025), separately.
5. Do not mix a native full pipeline, a fixed-geometry pipeline, and a pair-only pipeline into one ranking. A combined method must include the shared backend's runtime and be named `algorithm + backend`; it is **not** the paper's original end-to-end method.
6. Existing units are `UNKNOWN`, not confirmed surface reflectance. All real-scene errors are values in a source digital-value domain. No real-scene ground-truth RMSE, absolute reflectance accuracy, genuine-change F1, or four-season generalization claim.
7. Existing R1 manual stable/change gating applies to **truth-labeled** experiments only. It must not block the explicitly unlabeled consistency, coverage, structure or synthetic experiments defined here. Never relabel unknown regions as stable.
8. Stage 1 develops only the neutral benchmark code, input freeze/verification, output contracts, numerical metric tests and tiny **synthetic evaluation smoke**. Do not install/run external radiometric algorithms or launch long 13-scene experiments in this stage.
9. Algorithm outputs cannot change any benchmark mask, window, geometric transform, bin edge, metric parameter, temporal stratum, or source identity.

### Evidence already present in project -- verify in the real repository

- `metric_definition_audit.md` (read-only Task15/16 code audit): MAMD = `abs(mean(A)-mean(B))`; MSDD = `abs(std(A, ddof=0)-std(B, ddof=0))`; RDD = exact `scipy.stats.wasserstein_distance(A,B)`. Historic Task15/16 weighted summaries are **not** pair-macro summaries.
- `multitemporal_radiometric_protocol.md`, `R1_SUMMARY.md`, `R1_READINESS_REPORT.md`, `temporal_pair_inventory.csv`, `input_identity_manifest.csv`, Task15 geometry/grid manifests, R1 metric and S1--S4 code.
- Existing B9 inventory: 13 scenes, 78 unordered pairs, 48 actual shared-valid overlaps, 30 `NO_VALID_OVERLAP`. Temporal strata within the positive 48: 7 `SAME_DATE`, 13 `CROSS_DATE_LT_7D`, 6 `CROSS_DATE_7_TO_LT_30D`, 22 `CROSS_DATE_GE_30D`.
- Expected inventory checkpoints: `00_01` overlap = **1,291,672** pixels; `00_10` = **1,374,076** pixels, **only under the existing native-B9 overlap rule**. The new aligned target grid may produce different counts: record both and do not force-match one without auditing definitions.
- Task15 Stage04 does not persist all unnormalized aligned scene rasters; the read-only B9 input source appears in a neighboring repository. Hash and replay the exact frozen transform; do not silently use already BAGRN-normalized Stage06 as input.
- Task16 `COMPLETED_FINITE_NONCONVERGED` and historical `solver=0s` timing are **not** proof of convergence or a valid comparable cold-start full runtime.

**Required source audit before implementation:** inspect actual source code and its exported signatures; this plan proposes new names and interfaces only. Resolve differences by recording an explicit compatibility table. Do not silently change historical formulas or overwrite any old artifact.

## 1. Comparison regimes and eligible method outputs

| Evaluation | Input | Required output | Main comparisons | Ineligible claim |
|---|---|---|---|---|
| A / `FIXED_GEOMETRY_MOSAIC` | Same 13 radiometrically unadjusted, geometry-aligned scenes | Final full-grid GeoTIFF + explicit valid mask; provenance logs | completeness, seam/transition where measurable, structure diagnostics, end-to-end processing time, memory | Registration accuracy; absolute reflectance accuracy |
| B1 / `ALL_OVERLAPS` | Same fixed aligned source images and same 48 pairs | Corrected scene raster **for each participating scene** (or explicit pair-specific outputs) + output valid masks | MAE, RMSE, MAMD, MSDD, RDD, reference-to-input structural distortion | Full 13-scene capability inferred from pair-only runs |
| B2 / `FIXED_WINDOWS` | Same aligned scenes + pre-frozen spatial windows | Same as B1 | distributions, median/P90/worst windows, representative comparisons | Stable surface truth without labels |
| B3 / `SYNTHETIC_TRUTH` | Frozen synthetic reference + known perturbations | Corrected arrays in synthetic reference coordinate/value domain | truth MAE/RMSE, gradient/texture error, known change retention | Real-earth absolute accuracy |

In fixed-geometry A, turn off any algorithm-specific joint-coregistration stage (or account for its explicit changes in a separate native-registration regime). "Native workflow" here means native **radiometric/seam/mosaic stages on common aligned inputs**, not an assertion of native raw-image-to-map registration performance.

A method with only a final mosaic reports B1/B2 `NOT_APPLICABLE_MISSING_SCENE_OUTPUT`, **not zero**. A method with only corrected scenes can enter B and can enter A only when paired with an explicit, identically configured mosaicking backend. A pair-only method has results for the predeclared eligible pairs; never extrapolate missing 48-pair results.

A's **fixed-geometry** timing starts from the already aligned common-input files and includes **all method-specific** preprocessing, radiometric estimation/application, seam generation, blend/merge and final output file writes. Common geometry construction cost is tracked separately, not charged to one method. Any experiment that allows method-specific geometric registration must be a new, separately named regime.

## 2. Input, identity, geometry and quality freeze

### 2.1 Canonical input bundle

- `scene_count=13`, `band=B9`, initial `source_dtype=uint16`, physical unit `UNKNOWN`, source nodata convention `0` **as verified per source**, `crs=EPSG:32650`, nominal pixel size `14m`.
- Canonical target grid `G=(CRS, affine[6], width, height, axis order, pixel-center convention)` must be replayed from the **verified Task15 formal V2 geometry**. Freeze exact floating-point affine values; nominal resolution alone is not sufficient.
- Raw source images and their recorded Task15 transforms are read-only. A new alignment adapter may reconstruct and cache aligned **radiometrically unmodified** B9 input on the canonical grid using verified source transforms and one explicit image resampler/mask policy. No invented shifts.
- Source files need current actual SHA256 checked against frozen recorded hashes (where present); path, file size, scene timestamp, source scene ID, band name, CRS, affine, NoData, dtype, and transform version are stored. If hashes, grid or transform provenance fail, freeze is `BLOCKED_INPUT_IDENTITY` / `BLOCKED_GEOMETRY_PROVENANCE`.
- All 13 scene masks, 48 positive-overlap pair IDs, 30 zero-overlap pair IDs, time strata and the three RISK scene flags are included, not cherry-picked.
- B1--B16 extension is a separate versioned band manifest, with 208 file identities, per-band output values and per-band valid masks; never average unscaled errors across bands. SAM/SAC remain `NOT_EVALUABLE` until bandwise physical scales and content-level co-registration are independently confirmed.

### 2.2 Validity is defined from input, never from algorithm scores

Let `V_i(p)` mean that frozen aligned scene `i` has a valid **source** pixel at target-grid pixel `p`. Determine this using dataset masks/alpha, source nodata, explicit source-to-target validity resampling and finite numeric checks. Use nearest-neighbor resampling for masks. Do not infer cloud-free or temporally stable pixels from `V_i`.

- Pair scoring support: `Omega_ij = {p | V_i(p) and V_j(p)}`, with `n_ij = |Omega_ij|`.
- Mosaic expected support: `U = union_i V_i`. Pixel `p` in `U` is expected to have a final output.
- A method's output validity `Q_i^m` / `Q_M^m` must derive from **declared output mask/alpha and finiteness**, not an inferred `output_value != 0`: zero can be a legitimate corrected value.
- Pair output coverage: `c_ij^m = |Omega_ij intersect Q_i^m intersect Q_j^m| / n_ij`.
- Mosaic coverage: `c_M^m = |U intersect Q_M^m| / |U|`; holes = `|U minus Q_M^m|`; extraneous support = `|Q_M^m minus U|`.
- Define separate `input_support_count`, `output_support_count`, `missing_count`, `coverage_ratio`, `nonfinite_count`, `masked_invalid_count` for every score.

**Formal score fairness rule:** B1/B2 primary numerical entries require the method to supply valid values on the **entire frozen input support**. If `0 < coverage < 1`, report `PARTIAL_OUTPUT`, coverage and explicitly labeled `diagnostic_available_pixels` scores, but **do not rank** those partial-score values against full-support results. If `coverage=0`, report `OUTPUT_MISSING`. An optional later coverage-threshold ranking must be a distinct pre-registered protocol, not a silent relaxation. This strict rule deliberately prevents algorithms from improving scores by dropping difficult pixels.

No post-hoc intersection of successful methods' masks may redefine `Omega_ij`; no method-selected PIF or seam may redefine the primary scoring support. Report method-specific supports only in additional diagnostic tables.

### 2.3 Spatial processing contract

- Read images in bounded windows with halos when required. Each aligned output must have **exact** canonical CRS, affine, dimensions, band, coordinate orientation and pixel-center convention.
- If an external algorithm produces a different grid, its explicitly declared output adapter must reproject to the common grid using the frozen intensity resampling method, and **adapter runtime must be included in the method time**. An adapter must never fix registration by an unreported extra warp.
- Intensity resampling: freeze the verified Task15 interpolation kernel; if historical kernel cannot be recovered, stop for researcher approval rather than silently choosing bilinear. Mask resampling is nearest-neighbor and avoids interpolation through NoData. Record any radiometric effect from resampling.
- Use `float64` internally for scalar metrics. Never silently clip, stretch, rescale, histogram-equalize or quantize outputs in the evaluator. The final A mosaic is evaluated as the method actually delivered it, including output quantization effects.

## 3. B1 radiometric metrics: exact mathematical definitions

For any positive-overlap pair `(i,j)` and band `b`, let `x_p = C_i^m(p,b)` and `y_p = C_j^m(p,b)` on the **frozen full** `Omega_ij,b`, `n=|Omega_ij,b|`. These are corrected numeric values, not reflectance truth. `mu_x = n^-1 sum_p x_p`, `mu_y` likewise; `sigma_x = sqrt(n^-1 sum_p (x_p-mu_x)^2)`, using **population** standard deviation (`ddof=0`).

**Direct spatial correspondence metrics:**

```
MAE_ij   = (1/n) * sum_p abs(x_p-y_p)
RMSE_ij  = sqrt((1/n) * sum_p (x_p-y_p)^2)
BIAS_ij  = (1/n) * sum_p (x_p-y_p)       # signed, diagnostic
P95AE_ij = quantile_0.95({abs(x_p-y_p)}) # numpy method='linear'
```

**Existing-project distribution/statistics definitions, proposed frozen metric IDs:**

```
MAMD_ij = abs(mu_x-mu_y)
MSDD_ij = abs(sigma_x-sigma_y)
RDD_ij  = W1(P_x,P_y)
        = integral_0^1 abs(F_x^-1(u)-F_y^-1(u)) du
        = scipy.stats.wasserstein_distance(x, y)  # equal empirical weights
```

- MAMD is **not** MAE. MSDD is **not** `std(x-y)`.
- RDD is a *distribution-only* 1D earth-mover distance; permuting pixel positions does not change it. Neither RDD nor MAMD alone detects geometric misalignment.
- All metric values are in the input digital-value domain except dimensionless structural ratios and coverages. Lower is better for MAE/RMSE/P95AE/MAMD/MSDD/RDD when interpreted **only as consistency**. Do not call this known-truth radiometric accuracy.
- Exact RDD requires all samples (or exact external sorting/merging). Use bounded on-disk scratch arrays or a strict memory preflight; never silently replace it with binned/histogram approximate W1 or a random subset. If exact computation cannot fit, status `RESOURCE_LIMIT_EXACT_RDD` while other valid metrics may continue. Any approximate alternative must use a different metric ID and explicit error bounds.
- An optional fixed-bin **new** distribution metric may be reported as `HIST_TV_V1 = 0.5 sum_k abs(h_x,k-h_y,k)` for probability-normalized histogram bins predetermined from uncorrected inputs with underflow/overflow bins. **This is not paper CD** and is supplementary only. Do not implement unless bin policy is separately frozen.

### 3.1 Relationship to the 2024 Lin et al. paper -- no false synonyms

Original publication: Lin et al., *ISPRS Journal of Photogrammetry and Remote Sensing*, 218 (2024), 187--203, DOI `10.1016/j.isprsjprs.2024.08.016`, section 3.1, Eqs. (36)--(41). The accessible author-hosted text describes:

- `ADM`: mean over overlapping **pairs** of absolute overlap-region mean differences.
- `ADSD`: mean over overlapping **pairs** of absolute overlap-region standard-deviation differences.
- `CD`: a weighted histogram-bin difference, with weights proportional to overlap area and a bin-count normalization.
- `GL`: input-versus-corrected image gradient **orientation** difference averaged over image pixels, then images.
- Paper `RDOA=(ADM+ADSD+CD)/3`, `Ave=(ADM+ADSD+CD+GL)/4`.

`MAMD_ij` and `MSDD_ij` have the same **per-pair mathematical quantities** as the described ADM/ADSD ingredients, but historical Task15/16 use **pixel-count weighted** aggregation; Lin et al.'s ADM/ADSD are pair-average. Their paper CD is **not** our Wasserstein RDD, and GL is **not** gradient NCC. The author-hosted math rendering does **not** fully disambiguate the histogram discrepancy's bin-level functional or the precise angle-difference operator. Therefore:

- Implement only project-defined identifiers and explicitly pair-macro `ADM_STYLE_MACRO = mean_pairs(MAMD_ij)` and `ADSD_STYLE_MACRO = mean_pairs(MSDD_ij)` as accurately named descriptive summaries.
- Keep metric IDs `PAPER_CD_EXACT`, `PAPER_GL_EXACT`, `PAPER_RDOA_EXACT`, `PAPER_AVE_EXACT` **disabled/FORMULA_UNVERIFIED** until the original PDF and a reference numerical fixture define every operator, histogram normalization and its value range. Never relabel project RDD/NCC or a proxy as exact paper CD/GL.
- Do not combine errors with unrelated units by an uncalibrated weighted total score; present the vector of metrics.

### 3.2 Global aggregation (48 pairs; or explicitly registered subset)

For metric `M`, successful, full-support pair set `K_m`, `k_m=|K_m|`:

```
PAIR_MACRO_m(M) = (1/k_m) * sum_(ij in K_m) M_ij
PIXEL_WEIGHTED_m(M) = sum_(ij in K_m) n_ij*M_ij / sum_(ij in K_m) n_ij
PAIR_MEDIAN_m(M) = median({M_ij : ij in K_m})
PAIR_P95_m(M) = quantile_0.95({M_ij : ij in K_m})
```

- **Primary descriptive summary** = `PAIR_MACRO` plus pair distribution and `successful_pairs / eligible_pairs`. `PIXEL_WEIGHTED` is supplementary and must be named distinctly. Never pool input samples across all pairs then run MAMD/MSDD/RDD: that is a different metric.
- Every pair is in `all_pairs.csv`, even `NO_VALID_OVERLAP`, `NOT_RUN`, `FAILED`, and `PARTIAL_OUTPUT`; failing/partial pairs never count as a zero error, and must not disappear from denominators.
- Cross-method rank eligibility requires **the same frozen pair set**, same `band`, same evaluation mode, all full-support outputs and identical normalization. If methods cover different pair subsets, label `PARTIAL_PAIR_SET`; show per-method results descriptively but **do not compare their means as a main ranking**. A common subset comparison may be published only as separately named post-freeze sensitivity analysis with all exclusions disclosed.
- Predeclare time strata `SAME_DATE` (7), `CROSS_DATE_LT_7D` (13), `CROSS_DATE_7_TO_LT_30D` (6), `CROSS_DATE_GE_30D` (22); also RISK-containing vs all-PASS pairs. Report per stratum, with sample counts and failures. Same-date and long-gap numbers answer different questions.
- Pair errors are statistically dependent (shared scene observations, spatial autocorrelation). Do not report naive pixelwise independent confidence intervals; an optional bootstrap must use prespecified **scene-cluster** or spatial-block resampling, marked exploratory, with seed and uncertainty caveats.

## 4. B2 fixed geographic windows (no semantic labels)

Define the windows **before** seeing any external algorithm's output, anchored to the canonical global grid:

```
window_size_px = 128
window_stride_px = 128
window_anchor = (target_grid_row=0, target_grid_col=0)
min_input_pair_valid_fraction = 0.95
max_windows_per_pair = 32
selection = deterministic spatially dispersed, balanced by canonical-grid cells;
            score/order use only input masks, location and a fixed hash of pair_id/window_id/seed
selection_seed = 20261013
```

A window `W` is eligible for pair `(i,j)` when `|W intersect Omega_ij| / 16384 >= 0.95`, and the 128x128 window lies fully within the canonical grid. `W` always stores its immutable `(row_off,col_off,height,width)`, polygon bounds, selected mask digest, number of valid pixels, pair/time/quality flags. When eligible windows are numerous, use deterministic spatial coverage, not method scores, to select <=32; when none qualify, record `NO_ELIGIBLE_WINDOW`. A pair can have <32 windows without padding.

Compute B1 **identical metric functions** on `W intersect Omega_ij` (full support required after masking). Aggregate per pair over windows as `median`, `mean`, `P90`, `max` with the number of windows; then aggregate the *per-pair* statistics across the predeclared pair set. Do not conflate this window distribution with historical Task15's 256x256 tiles (min 4096 valid pixels). Do not reuse R1.5's 34 semantic candidate ROIs as the main fixed-window scoring set.

Window score labels must say `UNLABELED_LOCAL_CONSISTENCY`, not `STABLE_LANDCOVER`, `CHANGE_ACCURACY`, or `SHADOW_RESTORATION`.

## 5. B3 S1--S4 synthetic known-truth evaluation

Reuse and audit the existing R1 deterministic synthetic generators; do not fabricate physical radiance, terrain or seasonal ground truth. Synthetic arrays carry explicit arbitrary units and an identity/anchor protocol.

Let `T_i(p,b)` be the clean synthetic truth of scene `i`; let `D_i = f_i(T_i)` denote the **known nuisance radiometric corruption** given to the method; let `Y_i` be the corrected output. For S4, let `T_before` and `T_after` include the **designed actual ground-truth change** in mask `H`, and corruption is added on top of that designed truth.

```
TRUTH_MAE_i = mean_(p in H_eval) abs(Y_i(p)-T_i(p))
TRUTH_RMSE_i = sqrt(mean_(p in H_eval) (Y_i(p)-T_i(p))**2)
TRUE_DELTA(p) = T_after(p)-T_before(p)
OUT_DELTA(p)  = Y_after(p)-Y_before(p)
CHANGE_DELTA_RMSE = sqrt(mean_(p in H) (OUT_DELTA-TRUE_DELTA)**2)
CHANGE_DELTA_MAE = mean_(p in H) abs(OUT_DELTA-TRUE_DELTA)
SPURIOUS_DELTA_MAE = mean_(p in H_eval minus H) abs(OUT_DELTA-TRUE_DELTA)
CHANGE_MAG_RATIO = sum_(p in H) abs(OUT_DELTA(p)) /
                   sum_(p in H) abs(TRUE_DELTA(p))
```

- `CHANGE_MAG_RATIO` near 1 indicates similar total change magnitude **only when nonzero true change exists**. This ratio does not establish correct spatial change; use together with change-delta RMSE. Zero denominator -> `NOT_EVALUABLE_ZERO_TRUE_CHANGE`.
- Optionally calculate `PSNR_truth = 20 log10(L/TRUTH_RMSE)` and `SSIM_truth` with a frozen truth-value range `L`. `TRUTH_RMSE=0` yields `+inf` PSNR by mathematical convention but not a JSON nonfinite number: serialize as `{value:null, status:"PERFECT_RECONSTRUCTION", is_positive_infinity:true}` or equivalent. Do not claim PSNR/SSIM truth on real unlabelled mosaics.
- Fix synthetic gain, offset, spatial trend, band coefficients, noise, seeds, window, train/calibration/test masks, and clean-reference anchor **before** seeing algorithms. Record the exact versioned generator function plus NPZ SHA256. S1 global gain/offset; S2 smooth spatial trend; S3 independent known band perturbations; S4 nuisance plus actual changed area.
- Relative-normalization solutions can have an unconstrained global gain/offset gauge. For comparable *absolute synthetic truth errors*, require every method to receive the same declared **unaltered anchor scene/reference convention**. Do not secretly solve an extra affine calibration against held-out truth. Methods that cannot honor the anchor must report `UNMATCHED_OUTPUT_GAUGE`, or join a separately frozen gauge-invariant/calibration-track experiment.
- Pairwise change evaluation in S4 uses the same geometry/mask in before/after. No synthetic image should be promoted to evidence of real DZ01V change preservation.

## 6. Structural preservation (B and restricted A diagnostics)

Core problem: even a constant-gray output can achieve zero MAMD/MSDD/RDD and low pairwise MAE. Every radiometric consistency table must therefore carry input-vs-output structure and dynamic-range diagnostics.

### 6.1 Fully specified V1 Sobel gradients

For image `I` on the canonical grid, use fixed zero-mean 3x3 Sobel kernels:

```
Kx = [[-1,0,1], [-2,0,2], [-1,0,1]] / 8
Ky = Kx.T
G(I,p) = (Kx*I at p, Ky*I at p)
mag(I,p) = sqrt(Gx**2 + Gy**2)
```

A gradient center is eligible only if **all nine** pixels of its 3x3 support are finite, input-valid and output-valid; never zero-pad over NoData, a tile boundary, or outside image. Use at least a one-pixel halo when streaming. Compute on each individual input scene `X_i` and corresponding corrected scene `Y_i` at identical positions, not across two differently acquired dates.

Select edge pixels `E_i` from the input only: `mag(X_i,p) > tau_i`, where `tau_i = max(1e-12, P50(mag(X_i,q) for all eligible ORIGINAL input gradient centers q))`. `tau_i` is precomputed and frozen from original input, not output. Record `n_gradient_centers`, `n_edge_centers`, and `tau_i`; if no edge pixels, edge orientation metric is `NOT_EVALUABLE_NO_EDGES`.

```
cos_i(p) = dot(G(X_i,p), G(Y_i,p)) /
           (mag(X_i,p)*mag(Y_i,p))    # only if both magnitudes > 1e-12
          = 0                        # if corrected gradient is zero
DIR_LOSS_i = mean_(p in E_i) (1-cos_i(p))  # range [0,2], lower better
REL_GRAD_MAE_i = sum_(p in eligible centers) abs(mag(Y_i,p)-mag(X_i,p)) /
                 sum_(p in eligible centers) mag(X_i,p)
```

If the denominator of `REL_GRAD_MAE` is exactly zero: `NOT_EVALUABLE_FLAT_INPUT`. `DIR_LOSS=0` for exactly matched orientations; a flat corrected image gives direction loss 1 on nonzero input edges. This is our **`SOBEL_DIRECTION_LOSS_V1` proxy**, **not** the paper's exactly reproduced GL metric. `REL_GRAD_MAE` is sensitive to legitimate overall contrast changes, so it is reported as a distortion diagnostic, not an unqualified quality score.

Add dynamic-range retention on the SAME input-valid region:

```
ROBUST_RANGE(I) = Q95(I)-Q05(I)   # numpy quantile method='linear'
RANGE_RATIO_i = ROBUST_RANGE(Y_i) / ROBUST_RANGE(X_i)
```

Report `RANGE_RATIO`, not an assumption that exactly 1 is always optimal; values near 0 expose severe collapse and large values expose aggressive contrast expansion. `ROBUST_RANGE(X_i)=0` -> `NOT_EVALUABLE_FLAT_INPUT`. Quantiles are exact or clearly named approximate with different metric ID.

For two processed scenes, the gradient structure scores must be calculated per scene and aggregated scene-macro, with each scene counted exactly once; do not overweight a scene by how many overlap pairs it participates in.

### 6.2 SSIM and gradient NCC -- optional, separately named

If included, `SSIM_INPUT_OUTPUT_V1` uses identical 7x7 windows, `K1=0.01`, `K2=0.03`, sample covariance setting frozen (`use_sample_covariance=False`), Gaussian window setting explicitly frozen (`gaussian_weights=False` for standard uniform 7x7), `data_range=L_b` frozen **from the original input** (`L_b > 0`) and a 3-pixel erosion of valid input/output masks. Do not pass NaN/NoData through SSIM neighborhoods. SSIM is input-to-output distortion, **not** an external ground-truth quality score.

For each eligible 7x7 SSIM patch, use the exact population-form local statistics:

```
SSIM_patch(x,y) = ((2*mu_x*mu_y + C1)*(2*sigma_xy+C2)) /
                  ((mu_x**2+mu_y**2+C1)*(sigma_x**2+sigma_y**2+C2))
C1=(0.01*L_b)**2; C2=(0.03*L_b)**2
sigma_xy = mean_patch((x-mu_x)*(y-mu_y))  # population covariance
SSIM_IMAGE = mean_(all valid 7x7 centers) SSIM_patch
```

The frozen `L_b` must be a positive source-only fixed value with its exact derivation recorded in the source/metric manifest, **before** external method scoring. Without such a verified `L_b`, mark SSIM `FORMULA_CONFIG_UNVERIFIED` and omit it from V1 main results; never choose `data_range` independently from each corrected result.

`GRADIENT_NCC_V1` (if retained from R1) is Pearson's centered correlation between matching input/output Sobel gradient-magnitude vectors over frozen valid gradient centers:

```
NCC(a,b) = sum_p (a_p-mean(a))*(b_p-mean(b)) /
           sqrt(sum_p(a_p-mean(a))**2 * sum_p(b_p-mean(b))**2)
```

If either variance is zero, `NOT_EVALUABLE_ZERO_VARIANCE`, not NCC=1. Numerical directions: high NCC suggests texture association; NCC alone can remain high under undesirable gain changes. Explicitly audit whether the existing R1 code computes gradients of magnitudes, separate x/y, or orientation; if different, use versioned separate IDs and do not change historical files.

### 6.3 A: meaningful structural comparison on final mosaic

For the *final single mosaic* a direct comparison to **one arbitrarily chosen input over multi-coverage pixels is not valid**, because the method may legitimately choose a different source there. Define instead a universal diagnostic on the frozen **single-source-only** region:

```
U_single = {p | sum_i 1[V_i(p)] == 1}
ref(p) = source image X_i(p) of the unique valid scene at p
```

Compute Sobel direction loss, relative gradient MAE and optional SSIM between final mosaic and `ref` only where a 3x3 or 7x7 neighborhood is also strictly inside the **same** unique-scene region and valid in both rasters. Report `U_single` coverage and sample count. If insufficient, status `INSUFFICIENT_SINGLE_SOURCE_AREA`, not a fabricated score. Method-specific provenance/label maps may support a **separate** overlapping-scene source-structure diagnostic, but that cannot replace the common A metric.

## 7. Experiment A: final-mosaic, seam and completeness measurements

### 7.1 Always-comparable final output criteria

- `MOSAIC_COVERAGE = |U intersect Q_M|/|U|` (higher better).
- `HOLE_PIXELS = |U minus Q_M|` (lower better).
- `EXTRA_PIXELS = |Q_M minus U|` (flag unexpected additional coverage, not automatically error until diagnosed).
- `NONFINITE_PIXELS`, `UNDECLARED_NODATA_PIXELS`, output band/type/CRS/dimensions/affine validation; preserve method output's actual quantization and mask semantics.
- `VALID_AREA_M2 = covered_input_pixel_count * abs(det(pixel_affine_2x2))`; pixel area nominally `196m2` for orthogonal 14m square pixels, but use the actual affine determinant.
- Single-source-region structure from section 6.3, plus fixed input-based visual comparison windows. Full mosaic may be visually rated from a blinded, preselected set; do **not** pretend such subjective ratings are radiometric ground truth.

### 7.2 Seam-interface metrics when provenance exists

A seam cannot in general be reconstructed exactly from a final raster with no source/provenance. Therefore declare output capability:

- `HARD_SOURCE_LABELS`: a single source ID for each output pixel (hard cutline).
- `SOFT_BLEND_WEIGHTS`: source weights/alpha per pixel. A dominant-source interface is only an **approximate transition proxy**, not a true hard seam.
- `NO_SEAM_PROVENANCE`: no source/weight/cutline maps; seam-specific source-pair metrics must be `NOT_APPLICABLE_NO_SEAM_PROVENANCE`.

For a hard-label map `L`, derive unordered 4-connected **adjacent-pixel edges** `e=(p,q)` inside the frozen expected mosaic support, with `L(p) != L(q)`. Count each horizontal/vertical edge only once; attribute the edge to the actual pair of source IDs and report edge count and geometric length (14m for orthogonal pixels; otherwise use affine-derived horizontal/vertical center distance). Never infer the seam from the reference method's labels.

**Observable final-image seam-excess diagnostic** (new V1 proxy, not a paper metric): take the pixels `p_in` and `q_in` each one grid step farther into its own side along the interface normal, require `L(p_in)=L(p)` and `L(q_in)=L(q)`, and all four output values valid.

```
cross_e = abs(M(p)-M(q))
within_e = 0.5*(abs(M(p)-M(p_in)) + abs(M(q)-M(q_in)))
excess_e = max(0, cross_e-within_e)
SEAM_EXCESS_MEAN = mean_e(excess_e)
SEAM_EXCESS_P95 = quantile_0.95({excess_e})
SEAM_CROSS_MEAN = mean_e(cross_e)
SEAM_EXCESS_SUM = sum_e(excess_e)
```

A high value suggests an unusually abrupt transition relative to its immediate surroundings, but natural boundaries, genuine changes, orientation and differing seam lengths are confounders. Report both mean and total with seam length; **never use this proxy alone** to decide best mosaic. If the map represents dominant blend weights, suffix metric IDs with `_DOMINANCE_PROXY`, not `_HARD_SEAM`.

When separate corrected scenes *and* a provenance seam exist, supplement with `SEAM_PAIR_OVERLAP_MAE` on a method-specific frozen-width **8-pixel** buffer on both sides of its seam intersected with the relevant source-pair `Omega_ij`. Report mask size and whether 3+ sources overlap. Method-specific seam selection produces different pixels, so this is an **additional native-seam diagnostic**, not the primary common-region ranking. Require exact source-pair IDs; do not invent source assignments.

### 7.3 Method-independent fixed regions and comparisons

- Preselect method-independent fixed 128x128 windows via section 4, and predefined map transects/crops using only frozen input coverage/geometry. Export **identical crop coordinates and display stretch** for every method.
- On those fixed areas, calculate final-mosaic gradients and tile summaries as **spatial artifact diagnostics**; a raw high gradient can be genuine topography and is not automatically an error.
- For a comparable transition/visual assessment, show algorithm-blinded image crops, quality/coverage masks and available seam overlays. A small visual inspection is allowed; no dense manual stable/land-cover ROI annotation is required.
- If a method lacks seam provenance, still score common A coverage, runtime, single-source structure and fixed-area mosaics. Record seam metrics as `N/A`, **never as zeros or as evidence that the method has no seams**. No arbitrary single 'mosaic quality score' is sanctioned.

## 8. Runtime, memory, throughput and scaling contract

For a runnable method, measure `TOTAL_WALL_S = t_after_fully_flushed_final_outputs - t_before_method_specific_preprocess`, using a monotonic clock. Include load/inference/model loading (pretrained only), statistics, normalization, core solver, seam, blend, output reprojection/adapter and all file writes. Verify that file handles are closed and the product exists before stopping the timer. Record separately:

- `CORE_COMPUTE_S` if instrumentable and **actually measured**, `IO_S` if instrumentable; never infer `CORE_COMPUTE_S=0` from a missing log.
- `GEOMETRY_COMMON_PREP_S` measured and excluded from fixed-geometry A method timing; `TRAINING_S` separately for methods trained for this experiment; pretrained model identity and download/installation time separately, not disguised as free training.
- Peak process-tree RSS (including children) sampled at a stated frequency (e.g., 0.5s); note potential shared-memory double counting. Where feasible record peak private/PSS, but never equate an unavailable memory field to zero. Peak GPU VRAM includes children and records the GPU measurement method/device; if unavailable -> `NOT_MEASURED`.
- `INPUT_BYTES`, `OUTPUT_BYTES`, `VALID_MPIX`, `TIME_PER_VALID_MPIX_S`, hardware, CPU thread caps, RAM, GPU model, package commits, environment lock, container/OS, process return code, solver convergence, timeout and whether cache was cold/warm or resume.
- Formal A: same physical machine and explicit cache/repetition policy; a method preflighted with insufficient RAM becomes `BLOCKED_RESOURCE_PREFLIGHT`, not a successful fast run. Successful finite-nonconverged solver has its own status and warning.
- Optional independent scale track uses **preselected nested** subsets of 3, 5, 13 scenes (fixed inclusion rule, connectivity checked, no method-driven cherry-picking). Never compare full 13-scene time to another method's 2-scene time as if equivalent.

For GPU polling finer than the sampling resolution, describe peak VRAM as **observed sampled peak** rather than absolute microsecond maximum. Missing runtime measurements in historical reports cannot be filled with the cited values from unrelated phases.

## 9. Proposed software module tree and typed I/O API

All names below are **new interfaces** for Codex to implement after auditing the real repository; do not assert they already exist. Implementation should use Python type annotations, dataclasses or Pydantic-style validated models, NumPy arrays, Rasterio window IO, SciPy for exact W1, psutil (if available) for telemetry, and pure computation functions with no hidden method-specific branches.

Proposed isolation:

```
src/benchmark_v1_unified/
    __init__.py
    schemas.py                  # versioned typed contracts
    config.py                   # config parsing + frozen parameter validators
    identity.py                 # SHA256 + source input provenance
    geometry.py                 # common grid and source-to-grid transform audit
    support.py                  # input masks, overlap inventory, method coverage
    windows.py                  # deterministic B2 window selection
    radiometric.py              # B1/B2 pure pair statistics
    structure.py                # input/output gradients, SSIM and range
    mosaic.py                   # A completeness + single-source region
    seams.py                    # source label/weight seam optional diagnostics
    simulation.py               # B3 synthetic known-truth scorers
    resource.py                 # subprocess wall/memory/GPU instrumentation
    aggregation.py              # pair/scene/window/temporal reports
    submission.py               # external algorithm output adapter/validation
    reporting.py                # schema-stable CSV/JSON and scientific caveats
    freeze.py                   # immutable manifest checks and SHA lock
scripts/benchmark/
    audit_unified_v1.py
    freeze_unified_v1.py
    check_unified_submission.py
    evaluate_unified_v1.py
    smoke_unified_v1.py
configs/benchmark_v1_unified.yaml
tests/benchmark_v1_unified/
docs/benchmark/v1/unified_v1/
data/output/benchmark_v1_unified/
```

The evaluator must not import model-specific frameworks (PyTorch, MATLAB, Spectralmatch, VOLRN) into its core modules. External wrappers feed immutable artifacts to the same metric functions.

### 9.1 Required typed data objects

```
GridSpec:
  crs: str                              # EPSG:32650
  affine: tuple[float,float,float,float,float,float]
  width: int; height: int
  pixel_area_m2: float
  pixel_center: Literal['col_plus_half_row_plus_half']

SceneRef:
  scene_id: str; scene_index: int; band: str
  source_path: Path; source_sha256: str; source_dtype: str
  acquisition_utc_or_source_time: str; quality_flag: str
  source_nodata: float | None; source_mask_semantics: str
  aligned_input_path: Path | None; aligned_mask_path: Path
  alignment_transform_provenance: dict

PairRef:
  pair_id: str; scene_i: str; scene_j: str
  time_stratum: str; days_apart: float; risk_stratum: str
  input_overlap_pixels: int; support_mask_path: Path | None
  expected_status: Literal['OVERLAP','NO_VALID_OVERLAP']

WindowRef:
  window_id: str; pair_id: str
  row_off: int; col_off: int; height: int; width: int
  input_valid_count: int; input_support_mask_sha256: str
  stratum: str; immutable: bool

MaskedRasterChunk:
  data: ndarray[float64]                # shape [H,W], target-grid coordinates
  valid: ndarray[bool]                  # same shape; input or output, explicit
  window: RasterioWindow; scene_id: str; band: str

MetricValue:
  value: float | None                   # None for not evaluated
  unit: str; status: str
  n_input: int; n_output: int; diagnostics: dict

MethodSubmission:
  method_id: str; run_id: str; method_kind: str
  algorithm_commit: str | None; environment_digest: str
  input_freeze_sha256: str; geometry_freeze_sha256: str
  output_mosaic_path: Path | None
  output_mosaic_mask_path: Path | None
  corrected_scene_paths: dict[str,Path] # absent for mosaic-only
  corrected_scene_mask_paths: dict[str,Path]
  pair_specific_outputs: dict[str,dict] # for pair-only methods
  source_label_path: Path | None
  blend_weight_paths: dict[str,Path]
  seam_type: str                        # HARD, SOFT, NONE
  output_grid: GridSpec; pixel_value_unit: str
  process_log: Path | None; status: str; config_sha256: str

FrozenBenchmark:
  protocol_id: str; protocol_sha256: str
  grid: GridSpec; band: str; unit: str
  scenes: list[SceneRef]                # exactly 13 unique B9 IDs
  pairs: list[PairRef]                  # exactly 78 (48/30 if audit matches)
  windows: list[WindowRef]
  mask_manifest_sha256: str
  source_manifest_sha256: str
  method_independent_metric_spec_sha256: str
```

The actual implementation may split these objects into smaller models, but the serialized fields and meanings must remain stable for V1. Every computational result uses `MetricValue` semantics rather than conflating zero with missingness.

### 9.2 Exact function signatures and contracts -- freeze/inventory

| Callable signature | Inputs and validation | Return type and side effects |
|---|---|---|
| `load_benchmark_config(config_path: Path) -> BenchmarkConfig` | Typed YAML, all required fields; reject unknown metric IDs and ambiguous defaults | Immutable validated config; no files written |
| `sha256_file(path: Path, chunk_bytes: int=8_388_608) -> str` | Existing readable file | 64-char lower-case SHA256, streaming |
| `audit_input_sources(cfg: BenchmarkConfig) -> InputAudit` | Manifest + actual 13 B9 sources and recorded SHA/metadata | Per-scene identity status, errors, no changes to sources |
| `load_and_check_geometry(cfg: BenchmarkConfig, audit: InputAudit) -> GeometryAudit` | Task15 transform records/grid and source metadata | Exact `GridSpec`, per-scene mapping audit or BLOCKED |
| `prepare_aligned_input(scene: SceneRef, grid: GridSpec, resampling: str, out_dir: Path) -> AlignedInputArtifact` | Verified mapping and untouched source radiometry | Chunked aligned GeoTIFF + native validity mask + hashes; writes only isolated output; refuses unverifiable alignment |
| `create_input_masks(scenes: list[SceneRef], grid: GridSpec, out_dir: Path) -> MaskManifest` | Explicit masks and source finite checks | 13 Boolean GeoTIFF masks + hashes, input-union/single-source masks |
| `enumerate_overlap_pairs(scenes: list[SceneRef], masks: MaskManifest, times: dict) -> list[PairRef]` | 13 scenes, 78 combinations, exact streaming bitwise mask intersections | Full 78-pair immutable inventory (including zeros and strata) |
| `make_fixed_windows(pairs: list[PairRef], grid: GridSpec, masks: MaskManifest, cfg: WindowConfig) -> list[WindowRef]` | Input-only masks/grid; 128x128/stride128/min0.95/cap32/seed | Frozen B2 windows + per-window mask hashes; no method outputs accepted |
| `freeze_benchmark(cfg: BenchmarkConfig, audit: InputAudit, geo: GeometryAudit, pairs: list[PairRef], windows: list[WindowRef], out_dir: Path) -> FreezeResult` | Every gate passed; no existing lock overwrite | Writes versioned manifest, metric spec, SHA checksums and lock; otherwise returns NOT_FROZEN reasons |
| `verify_freeze(frozen_root: Path) -> FreezeVerification` | Locked V1 artifacts | Recompute hashes; reject drift before all scoring |

**Important:** A current alignment may change the exact overlap inventory relative to source-native masks. If so `enumerate_overlap_pairs` must emit both inventories and halt until verified; never overwrite an old number to make a test pass. `freeze_benchmark` must refuse a source hash mismatch and a missing Task15 transform provenance.

### 9.3 Exact function signatures and contracts -- common evaluator

| Callable signature | Inputs | Return, sample/exception behavior |
|---|---|---|
| `validate_submission(method: MethodSubmission, frozen: FrozenBenchmark) -> SubmissionAudit` | Method manifest, exact grid, output masks, band/value domain, declared version | Eligible regimes, per-scene/pair product checks, `INVALID_GRID` / `INPUT_HASH_MISMATCH` / `NOT_APPLICABLE` instead of implicit reprojection |
| `read_masked_aligned_window(scene: SceneRef, window: Window, role: str) -> MaskedRasterChunk` | Verified aligned raster, mask and canonical target window | Float64 array and independent Boolean mask; input zero can be invalid, output zero may be valid |
| `pair_support(mask_i: ndarray[bool], mask_j: ndarray[bool]) -> ndarray[bool]` | Equal shape, frozen input masks only | Bitwise intersection; deterministic |
| `audit_pair_output_support(input_pair_mask: ndarray[bool], output_i: MaskedRasterChunk, output_j: MaskedRasterChunk) -> CoverageAudit` | Same target window/grid and explicit output validity | Input count, eligible output count, missing count, coverage, status |
| `compute_pair_direct_metrics(x: ndarray, y: ndarray, support: ndarray[bool]) -> PairDirectMetrics` | Identical shapes; valid, finite, **full** support | MAE/RMSE/BIAS/P95AE, n; null + explicit status if n=0 |
| `compute_pair_distribution_metrics(x: ndarray, y: ndarray, support: ndarray[bool], exact_w1: bool=True) -> PairDistributionMetrics` | Same support and float64 samples, ddof=0 | MAMD/MSDD/exact RDD, means/std/counts; W1 resource failure is per-metric |
| `evaluate_pair_band(method: MethodSubmission, pair: PairRef, band: str, frozen: FrozenBenchmark) -> PairEvaluation` | Registered scene artifacts + masks, streamed windows | Single pair record with all metrics, coverage, timing provenance, method eligibility, no score on partial support |
| `evaluate_fixed_window(method: MethodSubmission, w: WindowRef, frozen: FrozenBenchmark) -> WindowEvaluation` | Pre-frozen window and exact same metric functions | All B1 metrics for one B2 tile; `NO_ELIGIBLE_WINDOW` when appropriate |
| `compute_sobel_structure(original: MaskedRasterChunk, corrected: MaskedRasterChunk, params: StructureConfig) -> StructureStats` | 3x3 halo, input-frozen threshold and masks | Direction loss, relative magnitude MAE, counts, status; no padded nodata |
| `compute_range_retention(original: ndarray, corrected: ndarray, valid: ndarray[bool]) -> MetricValue` | Same original-valid centers, exact Q05/Q95 | Dimensionless output/input robust range ratio or flat-input status |
| `compute_ssim_input_output(original: MaskedRasterChunk, corrected: MaskedRasterChunk, params: SSIMConfig) -> MetricValue` | Frozen 7x7, data range, eroded valid mask | SSIM with n_valid_centers, or not evaluable; optional implementation |
| `evaluate_scene_structure(method: MethodSubmission, scene: SceneRef, frozen: FrozenBenchmark) -> SceneStructureEvaluation` | Input + scene-corrected artifact, fixed original valid region | Scene-level structure metrics, not duplicated for each overlap pair |
| `compute_mosaic_coverage(mosaic: MaskedRasterRef, frozen_union_mask: RasterRef) -> MosaicCoverage` | Same grid, output mask and union input mask | coverage, holes, extras, missing/nonfinite, area |
| `evaluate_single_source_structure(mosaic: MaskedRasterRef, frozen: FrozenBenchmark) -> SingleSourceStructure` | Original input scene + unique-source mask + halos | Structure on method-independent single-source region; n and status |
| `extract_label_boundaries(label_map: RasterRef, union_mask: RasterRef, grid: GridSpec) -> BoundaryEdgeIterator` | Explicit hard-source or source-dominance provenance map | Unique four-neighbor source-change edges with pair IDs and distances |
| `compute_seam_excess(mosaic: RasterRef, edges: BoundaryEdgeIterator, labels: RasterRef) -> SeamExcessEvaluation` | Four valid pixels/edge, same-source interior neighbors | raw jumps, excess jumps, P95, edge count/length; N/A without seam provenance |
| `evaluate_native_seam_pair_residual(method: MethodSubmission, frozen: FrozenBenchmark, buffer_px: int=8) -> SeamPairResiduals` | Labels/weights plus available corrected scenes | Native seam-buffer pixelwise MAE and valid support; method-specific diagnostic |
| `evaluate_synthetic(method_result: SyntheticMethodOutput, synthetic_truth: SyntheticTruthBundle) -> SyntheticEvaluation` | Fixed S1--S4 truth/perturbation/mask/reference anchor | Truth MAE/RMSE, S4 change errors/ratio, optional PSNR/SSIM with statuses |
| `aggregate_pair_rows(rows: list[PairEvaluation], pair_manifest: list[PairRef], cfg: AggregationConfig) -> AggregateReport` | All 78 expected pair IDs and eligibility; no hidden filtering | Pair macro, pixel weighted, pair median/P95, temporal/quality strata, failure rates |
| `aggregate_window_rows(rows: list[WindowEvaluation], frozen_windows: list[WindowRef]) -> WindowAggregate` | All frozen B2 windows | Per-pair window distribution then pair macro; eligibility counts |
| `aggregate_scene_structure(rows: list[SceneStructureEvaluation], frozen_scenes: list[SceneRef]) -> SceneAggregate` | One row per scene | Scene macro plus counts, no overlap-pair replication |
| `write_report_bundle(report: UnifiedReport, output_dir: Path) -> ReportPaths` | Valid schema, lock/provenance, null-status discipline | Atomic JSON/CSV and summary markdown, checksums; no edits to R1/Task15/16 |

All floating-point functions must reject incompatible shapes, off-grid arrays, unknown source band, nonfinite samples in an allegedly valid input, and accidental mismatched mask semantics. No score function is permitted to infer input masks from radiometric output values.

### 9.4 Exact function signatures -- subprocess timing

```
run_instrumented_method(
    command: list[str],
    cwd: Path,
    env: dict[str,str],
    time_limit_s: int,
    rss_limit_bytes: int,
    gpu_device_ids: list[int],
    output_dir: Path,
    sample_interval_s: float = 0.5,
) -> MethodRunRecord
```

`MethodRunRecord` must include `start/end monotonic timestamps`, `total_wall_s`, resource sampling source/rate/coverage, peak process-tree RSS, observed peak GPU memory or `NOT_MEASURED`, process return status, timeouts, stdout/stderr file paths, input/output artifact SHA256, and complete provenance. Use actual process execution measurements; never derive zero seconds from an absent log. This wrapper is built now, but **do not invoke external algorithms** during Phase 1 freeze.

## 10. Frozen YAML config -- required values, not silent defaults

Proposed initial content of `configs/benchmark_v1_unified.yaml` (Codex fills verified source paths and exact affine from repo metadata; the placeholders MUST block freeze):

```yaml
schema_version: 1
protocol_id: DZ01V_UNIFIED_A_B_V1_20261013
status: PROPOSED_FOR_REPO_AUDIT
scope:
  scene_count: 13
  band: B9
  sensor: DZ01V_L2_VNIR
  physical_unit: UNKNOWN
  expected_total_pairs: 78
  expected_positive_overlaps_native: 48
  expected_no_overlap_native: 30
geometry:
  crs: EPSG:32650
  pixel_resolution_m: 14
  canonical_grid_manifest: REQUIRED_VERIFIED_TASK15_PATH
  resampling_source: REQUIRED_VERIFIED_TASK15_KERNEL
  input_mask_resampling: nearest
  mask_definition: source_dataset_mask_and_declared_nodata_plus_finite
  output_mask_definition: explicit_mask_plus_finite
  allow_undocumented_extra_warp: false
identity:
  source_inventory: REQUIRED_VERIFIED_PATH_TO_INPUT_IDENTITY_MANIFEST
  transform_inventory: REQUIRED_VERIFIED_PATH_TO_TASK15_TRANSFORMS
  require_current_sha256_match: true
  freeze_all_13_b9_hashes: true
pair_scoring:
  source_support: frozen_input_mask_intersection
  full_support_required_for_primary: true
  metric_ids: [MAE_V1, RMSE_V1, BIAS_V1, P95AE_V1, MAMD_V1, MSDD_V1, RDD_W1_EXACT_V1]
  std_ddof: 0
  quantile_method: linear
  exact_wasserstein: true
  ranking_pair_set_policy: same_frozen_full_set_only
  aggregation_primary: pair_macro
  aggregation_secondary: [pixel_weighted, pair_median, pair_p95]
windows:
  width_px: 128
  height_px: 128
  stride_px: 128
  origin_row_col: [0, 0]
  min_input_pair_valid_fraction: 0.95
  max_per_pair: 32
  selection_seed: 20261013
  selection_algorithm: input_only_spatial_hash_distributed_v1
structure:
  kernel: sobel_3x3_divisor_8
  required_halo_pixels: 1
  direction_loss: 1_minus_cos_on_input_edges
  input_edge_threshold: max_1e_minus_12_and_input_p50_gradient_mag
  relative_grad_mae: true
  robust_range_quantiles: [0.05, 0.95]
  optional_ssim: false
  optional_gradient_ncc: false
seams:
  provenance_required_for_seam_metric: true
  hard_seam_connectivity: 4
  native_buffer_pixels: 8
  seam_excess: positive_excess_over_adjacent_within_side_gradients
  report_length_and_edge_count: true
simulation:
  cases: [S1, S2, S3, S4]
  verify_existing_r1_truth_generator: true
  require_fixed_clean_anchor: true
  test_split_seed: 20261013
runtime:
  measurement_start: before_method_specific_preprocessing
  measurement_end: after_final_output_close
  ram: observed_process_tree_rss
  sample_interval_s: 0.5
  missing_memory_value: NOT_MEASURED
status_policy:
  require_reason_for_missing: true
  no_renaming_historical_results: true
  no_manual_roi_requirement: true
  no_paper_cd_or_gl_claim_without_verified_reference: true
```

The config parser must demand explicitly filled values instead of making a plausible but wrong guess. Once frozen, `configs/...yaml` and the full `metric_spec.json` receive hashes and may be read by experiments but not edited. A change requires a new protocol ID/version and a changelog entry.

## 11. Submission and report file contracts

### 11.1 External method manifest (`method_submission.json`)

```json
{
  "schema_version": 1,
  "method_id": "SPECTRALMATCH_2026",
  "run_id": "UNIQUE_IMMUTABLE_RUN_ID",
  "method_kind": "FULL_MOSAIC",
  "experiment_regime": "FIXED_GEOMETRY_MOSAIC",
  "method_version": "REQUIRED_REAL_VERSION_OR_COMMIT",
  "algorithm_modified": false,
  "algorithm_modification_log": null,
  "license_audit": "REQUIRED",
  "input_freeze_sha256": "REQUIRED_SHA256",
  "geometry_freeze_sha256": "REQUIRED_SHA256",
  "band": "B9",
  "value_unit": "UNKNOWN",
  "outputs": {
    "mosaic": "REQUIRED_FINAL_TIF_PATH",
    "mosaic_mask": "REQUIRED_MASK_PATH",
    "corrected_scenes": {},
    "corrected_scene_masks": {},
    "source_labels": null,
    "blend_weights": {}
  },
  "seam_provenance": "NONE",
  "runtime_record": "REQUIRED_REAL_PROCESS_TELEMETRY_JSON",
  "config_sha256": "REQUIRED_SHA256",
  "status": "NOT_RUN"
}
```

For multi-scene B, `corrected_scenes` contains exact 13 scene-ID keys mapped to immutable output rasters and masks. Pair methods provide keyed pair outputs **with reference direction/anchor recorded**, e.g. `00_10 -> {before_scene, after_scene, reference_scene_id, corrected_pair_paths, masks}`. Never treat pair results that normalize the same scene differently across pairs as one globally coherent 13-scene solution.

Accepted method kinds: `FULL_MOSAIC`, `MULTISCENE_RRN`, `PAIR_RRN`, `GEOMETRY_ONLY`. External adapters may emit a union of supported capabilities. `GEOMETRY_ONLY` has no radiometric B score by itself.

### 11.2 Artifact layout (all new; read-only old work)

```
docs/benchmark/v1/unified_v1/
    plan.md                             # approved copy of this document
    source_formula_crosswalk.md         # paper/project/new-metric definitions
    source_and_transform_audit.md
    freeze_readiness.md
    changelog.md
    metric_spec.json
    freeze.lock.json

data/output/benchmark_v1_unified/
    freeze/
        source_manifest.csv
        grid.json
        input_mask_manifest.csv
        scene_union_mask.tif
        scene_single_source_mask.tif
        input_overlap_inventory.csv
        fixed_windows.csv
        frozen_manifest.json
        sha256sums.txt
    synthetic_smoke/
        known_arrays_metrics.json
        smoke_report.md
    runs/<method_id>/<run_id>/
        method_submission.json
        submission_audit.json
        per_pair_metrics.csv
        per_window_metrics.csv
        per_scene_structure.csv
        aggregate_pair_metrics.csv
        aggregate_temporal_metrics.csv
        aggregate_window_metrics.csv
        mosaic_metrics.json
        seam_metrics.json
        synthetic_truth_metrics.json
        runtime_metrics.json
        failure_inventory.csv
        report.md
        run_provenance.json
```

**Required `per_pair_metrics.csv` columns**, one record per eligible `(method,run,pair,band)` including failures:

```
protocol_id,freeze_sha256,method_id,run_id,method_kind,
regime,band,scene_i,scene_j,pair_id,days_apart,time_stratum,risk_stratum,
input_overlap_pixels,output_overlap_pixels,coverage_ratio,missing_output_pixels,
mae,rmse,bias,p95ae,mamd,msdd,rdd_w1_exact,
metric_status,method_status,formula_spec_sha256,mask_sha256,notes
```

For CSV: non-evaluated metric cells are empty/null with mandatory explicit `metric_status` and reason; **never** `0`, string `nan`, negative sentinel, or an invented result. JSON numeric values must be finite, except explicitly structured infinite special cases. Keep full floating precision; rounding happens only in report rendering.

`aggregate_pair_metrics.csv`: `(method_id,band,regime,metric_id,aggregation_id,time_stratum,risk_stratum,n_eligible_pairs,n_complete_pairs,n_partial_pairs,n_failed_pairs,expected_48_pairs,metric_value,unit,status,protocol_hash)`.

`per_window_metrics.csv`: `(window_id,pair_id,row_off,col_off,height,width,input_valid_count,output_coverage,all B1 metric fields,status)`.

`mosaic_metrics.json`: `coverage, hole count, extra count, nonfinite count, output CRS/grid, single-source structural metrics, provenance/comparability statuses`. `seam_metrics.json`: hard/soft/none source type; seam length/edge count, excess statistics, native seam-buffer support; `N/A` reason if no seam.

`synthetic_truth_metrics.json`: each `(scenario S1..S4, band, seed, anchor, truth mask hash, output mask hash, corruption parameters, truth RMSE/MAE, S4 delta errors, ratio, status)`.

`runtime_metrics.json`: measured timestamps, wall seconds, RSS/VRAM collection method, memory sample interval and observed peaks, method status and full subprocess/IO provenance.

Provide `report.md` with TWO distinct tables: **A (full mosaic)** and **B (radiometric)**, then B1/B2/B3 results, all unavailable statuses, full per-pair details, status counts, temporal strata, limitations and no raw cross-method numeric values from unrelated experiments.

## 12. Deterministic algorithm for window selection, pair aggregation and status handling

### 12.1 Spatially dispersed B2 windows -- exact tie-breaking contract

1. Generate every 128x128 **non-overlapping** grid-aligned window at row/column offsets that are multiples of 128 from frozen grid origin, fully inside target grid.
2. Retain for each pair only windows with `n_input_pair_valid >= ceil(0.95*16384)`.
3. Place eligible windows in supercells `(floor(row_off/512), floor(col_off/512))`.
4. Within each supercell choose the candidate with the smallest **unsigned lexical 64-char SHA256** of ASCII key `"20261013|PAIR_ID|ROW_OFF|COL_OFF"`; break identical hashes by row then col.
5. Sort the per-supercell representatives by that SHA256 then row/col; select up to 32.
6. If representatives number below 32, fill remaining slots from the unused eligible window list sorted by the same hash key; never duplicate a window.
7. Sort final `WindowRef` rows by `(pair_id,row_off,col_off)`, store full support-mask digests and selection-algorithm version.

This algorithm makes sampling **method-independent and fully reproducible**; it is not a random pixel-sample scheme. No image appearance, noise, segmentation, semantic land-cover label, or algorithm output is used to select windows.

### 12.2 Numerics and reproducibility

- One statistic definition per metric ID, with a recorded `formula_spec_sha256`.
- Mean/variance/MAE/RMSE computed with float64 and reproducible chunk accumulation (stable parallel Welford/Chan variance merges or equivalent). Summation order must be documented; test tolerance for different chunk sizes.
- `np.std(ddof=0)` behavior is the compatibility reference on fixture arrays. Never divide by `n-1`.
- Quantile convention = NumPy `method="linear"` for P95, Q05, Q95. If exact quantiles require storage beyond memory limits, fail that metric explicitly rather than rebrand an approximate quantile as exact.
- W1 reference = SciPy exact `wasserstein_distance` for one-dimensional empirical samples with uniform weights.
- `n=0`: `NO_VALID_OVERLAP`; `n>0` and all output missing: `OUTPUT_MISSING`; partially missing: `PARTIAL_OUTPUT`; valid zero variance: MAMD/MSDD/RDD/MAE can still be well-defined, but NCC/ratio have their own gating.
- Distinguish wrong-grid output (`INVALID_GRID`), wrong input (`INPUT_HASH_MISMATCH`), unsupported output type (`NOT_APPLICABLE`), incomplete pair set (`PARTIAL_PAIR_SET`), computational resource failure (`RESOURCE_LIMIT`), nonfinite contamination (`NONFINITE_VALID_OUTPUT`), mathematical degeneracy (`NOT_EVALUABLE_*`), unverified paper formula (`FORMULA_UNVERIFIED`), not attempted (`NOT_RUN`), run failure (`METHOD_FAILED`), and valid but nonconverged output (`COMPLETED_FINITE_NONCONVERGED`).
- For `METHOD_FAILED` and `NOT_RUN`, never substitute previous-task numbers. For a valid incomplete solver, metrics may be reported with a conspicuous convergence warning, not silently as converged.

### 12.3 Qualification and fairness checks

For each metric and any reported A/B ranking table, print an **eligibility matrix** with: method output capability, band, same input freeze hash, same geometry freeze hash, same pair/window population, full output support, valid numeric range, runtime completeness, and method status. Missing any required property -> no ranking cell. The user may compare quality/cost tradeoffs only after all relevant statuses are visible. No universal weighted score or post-hoc hand-picked favorable windows.

## 13. Automated tests and acceptance fixtures

Write small independent tests that do not access private DZ01V imagery. At minimum:

| Test | Frozen input / operation | Expected result |
|---|---|---|
| T01 permutation separates metrics | `x=[0,1,2]`, `y=[2,0,1]`, both masks all true | `MAE=4/3`, `RMSE=sqrt(2)`, `MAMD=0`, `MSDD=0`, `RDD=0` |
| T02 mean/std/W1 fixture | `x=[0,0,4]`, `y=[1,1,1]` | `MAMD=1/3`, `MSDD=sqrt(32)/3`, `RDD=5/3`, `MAE=5/3`, `RMSE=sqrt(11/3)` |
| T03 identity | `x=y=[0,3,7,1]` | All pair discrepancy metrics equal 0, including RDD |
| T04 nodata output zero | source masks exclude source nodata `0`, but method declares `0` as a valid corrected output | Method zero is **not** automatically dropped; support remains frozen from source masks |
| T05 incomplete result | Frozen pair support 4 pixels, corrected output only 3 valid | `coverage=0.75`, `PARTIAL_OUTPUT`, no primary score and no hidden smaller common support |
| T06 disjoint overlap | Two masks no shared pixels | `NO_VALID_OVERLAP`, no NaN/zero metric masquerading as valid |
| T07 ddof contract | `x=[0,2]`, `y=[0,0]` | `MSDD=1`, not `sqrt(2)` |
| T08 W1 spatial permutation | Any two permutations of same multiset | W1=0, even if pixelwise MAE>0 |
| T09 edge orientation | Input has nonzero Sobel edges; corrected identical, then sign-inverted, then constant | Direction loss 0, 2, and 1 on eligible input edge centers, respectively |
| T10 no artificial boundary | Tile processed with one-pixel halo and masked nodata neighbors | Gradient exactly matches whole-array valid centers; no tile-edge false gradients |
| T11 seam excess jump | A 1D hard seam `[1,1 | 6,6]` with two constant sides | One interface with `cross=5`, `within=0`, `excess=5` |
| T12 natural linear ramp | Hard interface `[1,2 | 3,4]` | `cross=1`, `within=1`, `excess=0` |
| T13 A coverage | Expected union 10 pixels, output 9 inside and 1 outside | `coverage=0.9`, holes 1, extras 1 |
| T14 A no provenance | Mosaic available, source labels/weights absent | A coverage/runtime still valid; seam source-specific fields are N/A, not zero |
| T15 windows deterministic | Same input masks/config, any insertion order | Identical ordered windows and hashes; changing **method output** does not change windows |
| T16 time strata | 7+13+6+22 predeclared positive pairs | Counts sum to 48; zero-overlap pairs remain in separate set |
| T17 aggregation distinction | Pair metric scores `[1,3]`, pixel counts `[1,3]` | pair-macro 2, pixel-weighted 2.5; method with missing pair not eligible for all-pair ranking |
| T18 S4 known truth | `Y_before=T_before`, `Y_after=T_after` with nonzero change | truth MAE/RMSE/delta MAE/RMSE all 0; magnitude ratio 1 |
| T19 changed suppressed | `Y_after=Y_before` in a region with true nonzero change | S4 change-delta RMSE>0; magnitude ratio 0 where denominator nonzero |
| T20 degenerate quantities | Constant source, no edges; zero true changes in S4 | corresponding `NOT_EVALUABLE_*`, no division by zero or JSON NaN |
| T21 spatial-coordinate tampering | Shift output affine by 14m but keep same shape | `INVALID_GRID`; evaluator must not silently repair |
| T22 input/lock tampering | Modify one byte in source manifest or mask manifest | `verify_freeze` fails before scoring |
| T23 historical audit compatibility | Controlled array fixture read by both new and existing Task15/16/R1 functions | Per-pair MAMD/MSDD/RDD match within documented float64 tolerance; old weighted aggregate correctly distinguished from new macro |
| T24 method leakage guard | Attempt to pass method-specific seam/PIF mask as primary support | Reject as `INVALID_SCORING_SUPPORT` |
| T25 run timing unavailable | No existing wall/RAM logs | `NOT_MEASURED`, never `0.000s` |
| T26 16-band units | B9 and another band have unrelated scales | No raw cross-band mean without explicit frozen normalization and source units |

For T02: `RMSE = sqrt((1+1+9)/3) = sqrt(11/3)` and `MSDD = sqrt(32)/3`. Tests must use those exact values; never write expected numbers derived from the function under test itself. Add property-based cases for W1 symmetry, shift-equivariance, permutation invariance; mask support invariance and exact chunk-order consistency with documented tolerances.

**No synthetic test is equivalent to a successful real 13-scene run.** If actual source manifests are unavailable in the current worktree, tests can still pass, while freeze state remains explicitly blocked.

## 14. Execution phases and strict stopping conditions for Codex

### P0. Read-only audit (mandatory, no algorithm runs)

- Inspect `metric_definition_audit.md`, actual `src/multiscene_sift/radiometric_metrics.py`, existing R1 metric functions/tests and S1--S4 generator, Task15 formal V2 grid/transform metadata, `temporal_pair_inventory.csv`, source file paths/hashes, and any existing normalized vs original input confusion.
- Write `source_formula_crosswalk.md`: name, precise legacy function, legacy formula, sampling/masking/aggregation, this document's new metric ID, equivalence **per pair** or **only if different aggregation**, validation fixture result. Explicitly flag original paper CD/GL not yet numerically reproduced.
- Verify source availability and current SHA hashes. Check canonical geometry reconstruction, including whether 48/30 overlaps survive the aligned geometry. Any mismatch must be explained, not altered to meet expected values.
- Deliver `source_and_transform_audit.md` with specific verified paths, methods and blockers.

### P1. Implement isolated core and freeze candidates

- Build typed models/config loader, source/hash/geometric audit, source masks, pair inventory, fixed B2 window selector, versioned metric formulas and submission audit.
- Compute and store input-only lock candidates. Do not inspect any algorithm outcome to choose samples.
- If verified assets are missing, emit `FREEZE_BLOCKED_*` and stop *only the real-data freeze step*; continue pure functions and synthetic unit tests.

### P2. Implement B1/B2/B3 + A diagnostic evaluation

- Implement exact B1 pair metrics, macro/weighted/temporal reports, B2 deterministic windows, B3 known-truth scoring, gradients and dynamic-range metrics, A union coverage and single-source structure, conditional seam edge diagnostics, and run telemetry wrapper.
- Implement CLI `evaluate_unified_v1.py --freeze <path> --method-submission <method_submission.json> --regime A|B1|B2|B3 --output-root <path>`.
- This phase only runs **fixture method submissions** created by tests. External methods must remain not run.

### P3. Validation and immutable freeze

- Pass all tests T01--T26 plus representative shape/mask/IO/large-window streaming tests.
- Confirm a real 13-scene input identity and transform (when accessible); write precise `frozen_manifest.json`, 78-row overlap inventory, 48/30 audit or justified divergence, deterministic window manifest and all SHA locks.
- Run read-only tiny synthetic smoke; verify result CSV/JSON schema, N/A states and method capability branching.
- Inspect time/memory of evaluator itself in one bounded fixture and verify no unintended 13-scene algorithm launches.
- Generate `freeze_readiness.md` with pass/fail for each gate; **do not unilaterally change metric definitions to pass**.

### Required Phase-1 delivery checklist

- [ ] This plan copied to project `docs/benchmark/v1/unified_v1/plan.md` **after researcher approval**.
- [ ] All proposed evaluation functions, concrete type signatures and runtime wrapper implemented in isolated package.
- [ ] Frozen YAML, JSON schemas/CSV schema, formula spec hash, input/mask/pair/window artifacts and re-verification command.
- [ ] Read-only legacy metric crosswalk with ADM/ADSD vs project MAMD/MSDD weighted-vs-macro distinctions, and CD/GL never mislabeled.
- [ ] All relevant unit tests pass, including fixtures T01--T26.
- [ ] Explicit separation of Experiment A vs Experiment B (B1/B2/B3), pair-only vs multi-scene methods, and methods lacking corrected scene outputs.
- [ ] Frozen mask outputs are computed independently from methods, masks/scenes immutable, 13 scenes + 78-pair inventory audited.
- [ ] No automated semantic ROI labels or human review gate added to unlabeled scoring.
- [ ] No Task15/16/R1/R1.5 files changed; no Ours edits; no Spectralmatch/Harmoni-Planet/RS-NormGAN actual experiment in this Phase 1.
- [ ] Valid protocol lock SHA256, code revision, source provenance and precise `FROZEN` state, **or** a factual `NOT_FROZEN` report identifying blocking input/geometry conditions.

## 15. CLI contract and final freeze decision

Expected entry points:

```powershell
# Read-only scan and explicit blockers; no algorithm runs
python scripts/benchmark/audit_unified_v1.py --config configs/benchmark_v1_unified.yaml

# Freeze input masks, canonical geometry, all 78 pairs and B2 windows;
# fail safely if required original source/transform identities are missing
python scripts/benchmark/freeze_unified_v1.py --config configs/benchmark_v1_unified.yaml --output-root data/output/benchmark_v1_unified/freeze

# Run ONLY fixture-based evaluator validation in Phase 1
python scripts/benchmark/smoke_unified_v1.py --fixture-only

# Future external method adapter validation -- not launched by Phase 1
python scripts/benchmark/check_unified_submission.py --freeze data/output/benchmark_v1_unified/freeze --submission METHOD_SUBMISSION_JSON

# Future scoring of a successfully run algorithm -- not launched by Phase 1
python scripts/benchmark/evaluate_unified_v1.py --freeze data/output/benchmark_v1_unified/freeze --method-submission METHOD_SUBMISSION_JSON --regime B1 --output-root METHOD_RESULTS_ROOT
```

The initial CLI is shell-agnostic and should work from the repository root. Scripts must resolve paths against the repository/config root, not the caller's incidental working directory; error messages print the actual resolved path and expected hashes.

**`FROZEN` means:** formula IDs and parameter values + exact geometry/aligned input identity + 13 source files + masks + 78 pair IDs/overlap audit + B2 windows + output submission schema + aggregation/coverage rules are all verified and hashed, and T01--T26 pass. It does **not** mean any 2025--2026 third-party algorithm has succeeded, nor that calibrated reflectance or real stable/change truth exists.

**If any critical gate fails:** emit `NOT_FROZEN` plus an actionable list of missing assets, inconsistent formulas, unsupported geometry or damaged hashes. Preserve all successful pure tests and code, but do not call the benchmark frozen. Request researcher review rather than changing source data, resampling parameters or the predeclared window sampling policy.

## 16. Readings and immutable provenance sources

Repository-local sources to audit, *not* overwrite:

- `metric_definition_audit.md` -- Task15/16 per-pair formulas and weighting, RDD exact W1, local 256 tiles and distinct seam metrics.
- `multitemporal_radiometric_protocol.md` -- unit/gauge/quality/true-change restrictions.
- `R1_SUMMARY.md`, `R1_READINESS_REPORT.md` -- existing S1--S4 truth generators, independently implemented metrics and actual code readiness.
- `temporal_pair_inventory.csv`, `input_identity_manifest.csv` -- the current 13 B9 source identities, dates and 78 pairs.
- `TASK16_13SCENE_VOLRN_COMPARISON_REPORT.md` -- only historical results; finite-nonconverged status and non-comparable timing.

External scientific sources:

- Lin et al., 2024, *ISPRS Journal of Photogrammetry and Remote Sensing* 218:187--203, DOI https://doi.org/10.1016/j.isprsjprs.2024.08.016 ; section 3.1 Eqs. 36--41. An author-hosted full-text rendering can be found at https://www.researchgate.net/publication/386992292_Joint_block_adjustment_and_variational_optimization_for_global_and_local_radiometric_normalization_toward_multiple_remote_sensing_image_mosaicking . Interpret any incompletely rendered mathematical operators as *unverified*.
- Lindiwe et al., 2026, *JOSS*, Spectralmatch, https://doi.org/10.21105/joss.08974 ; official code https://github.com/spectralmatch/spectralmatch . Feature listing establishes program capabilities, not numerical metric correctness on DZ01V.

## 17. Separate future registration and multi-band protocol (NOT Phase-1 numeric gate)

The above A+B freeze is fundamentally a **radiometric/mosaicking** benchmark with fixed geometry. Preserve independent Phase5 registration scoring, do not reinterpret a full-mosaic quality score as tie-point accuracy. When properly independent approved ground-truth correspondences eventually exist, use the frozen Phase5 pixel-center convention:

```
e_k = sqrt((pred_col_k-gt_col_k)**2 + (pred_row_k-gt_row_k)**2)
REG_RMSE_PX = sqrt(mean_k(e_k**2))
REG_P95_PX = quantile_0.95({e_k})
REG_MAX_PX = max_k(e_k)
PCK_at_t_px = count_k(e_k<=t) / n_checked, for t in {1,2,3}
```

Scores are computed over approved **held-out** check points only, with per-pair status, pair-macro summaries, pair success denominators and no matcher-fit-point substitution. Until the independent second human TEST review is valid, these formal scores are `NOT_EVALUABLE_NO_INDEPENDENT_TRUTH`. They are not required to perform fixed-geometry A/B. Method-specific geometric transforms require a new explicit regime and geometry-scoring adapter, not adjustment of the radiometric lock.

When calibrated, co-registered B1--B16 products become available, a future registered spectral metric can use the standard **spectral angle mapper** definition, for per-pixel spectral vectors `s_p` and `t_p`:

```
SAM_RAD = mean_p arccos(clip(dot(s_p,t_p)/(
          norm(s_p)*norm(t_p)), -1, 1))
```

Gate on reliable **band-to-band physical-unit comparability**, joint finite masks, all 16 band identities and nonzero vector norms. `SAC` is not an unambiguous universal acronym without the target paper's explicit definition: mark `SAC_FORMULA_UNVERIFIED` rather than inventing a metric. No raw multi-band MAE pooling in V1.

## 18. Researcher-facing result interpretation rules

1. B1's overlap MAE/other consistency metrics measure **numerical inter-image agreement** only, NOT known-reflectance accuracy. B2 evaluates local heterogeneity, not independently labeled stable land cover. B3 is controlled-model fidelity, not real seasonal truth.
2. A's coverage and comparable timing apply to essentially every full-mosaic submission; A seam/source-specific scores require provenance; missing those metadata produces a declared limitation, not an algorithmic zero. Never claim a complete algorithm ranking based on seam excess alone.
3. A method may have lower pairwise discrepancy yet destroy edge detail. Report structure and range diagnostics alongside each consistency score. Keep the Pareto tradeoff of quality, speed and resource use; do not optimize an arbitrary weighted scalar.
4. 13 B9 scenes have been used repeatedly for researcher development. Treat them as a **development benchmark**; independent geography/date data would be needed for strong out-of-distribution generalization claims.
5. Candidate methods are registered based on real supported capabilities; no numerical method superiority is asserted at freeze time.

**End-of-phase instruction:** deliver implementation diff, formula crosswalk, synthetic fixture results, source/geometry audit and freeze readiness. Stop before installing or running new algorithms; obtain researcher authorization for Spectralmatch real-data testing in Phase 2.
