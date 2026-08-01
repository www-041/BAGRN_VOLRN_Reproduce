# Stage 2: BAGRN Gain/Offset Ablation Report

**Date:** 2026-08-01
**Branch:** stage2-gain-offset-ablation
**Dataset:** DZ01 VNIR four-scene subset (scene_20251114, scene_20251120, scene_20251208, scene_20251215)
**Sensor:** DZ01 VNIR (B01–B14, 410–860 nm)
**Crop size:** 1024 × 1024 px

---

## 1. BAGRN Mathematical Formula

### Actual Implementation (`src/bagr n.py`)

The BAGRN normalization uses **moment matching** in raw pixel space (not standardized space).

**Step 1:** For each overlap pair (i, j), compute per-band overlap mean and std:
```
mu_i[band], mu_j[band] = overlap means (excluding nodata)
sigma_i[band], sigma_j[band] = overlap stds (excluding nodata)
```

**Step 2:** For each band independently, solve weighted least squares:
```
theta_i - theta_j = mu_j - mu_i       (Eq. 1)
theta_r = 0                            (control scene constraint, Eq. 5)
```
Returns: `theta_mu[band, scene]` and `theta_sigma[band, scene]`

**Step 3:** For each image, compute global mean/std, then apply moment matching:
```
mu_target  = mu_orig + theta_mu[band, scene]
sigma_target = sigma_orig + theta_sigma[band, scene]
omega  = sigma_target / sigma_orig       # gain (Eq. 11)
upsilon = mu_target - omega * mu_orig    # offset (Eq. 11)
output = omega * input + upsilon         # moment matching (Eq. 10)
```

### Key Properties

| Property | Value |
|----------|-------|
| Gain/offset shape | `(n_bands, n_images)` per component |
| Control scene | index 0 (scene_20251114): omega=1, upsilon=0 |
| NoData handling | masked pixels skipped, left unchanged |
| Clipping | none (output is float64) |
| Standardization | NOT used — raw pixel space |

---

## 2. Full Reconstruction Validation

| Scene | Status | RMSE | Max Abs Error | Allclose |
|-------|--------|------|---------------|----------|
| 20251114 | PASS | 0.000000 | 0.000000 | True |
| 20251120 | PASS | 0.000000 | 0.000000 | True |
| 20251208 | PASS | 0.000000 | 0.000000 | True |
| 20251215 | PASS | 0.000000 | 0.000000 | True |

**Full reconstruction matches Stage 1 BAGRN output exactly.** Threshold: rtol=1e-5, atol=1e-4.

---

## 3. BAGRN Global Coefficients

### Per-Scene Per-Band Gain (omega)

| Band | 20251114 (ctrl) | 20251120 | 20251208 | 20251215 |
|------|-----------------|----------|----------|----------|
| B01 | 1.000 | 1.200 | 1.179 | 1.349 |
| B02 | 1.000 | 1.107 | 1.052 | 1.121 |
| B03 | 1.000 | 1.077 | 1.065 | 1.130 |
| B04 | 1.000 | 0.969 | 1.023 | 0.983 |
| B05 | 1.000 | 0.983 | 1.053 | 0.963 |
| B06 | 1.000 | 0.963 | 1.050 | 0.954 |
| **B07** | **1.000** | **0.867** | **0.614** | **0.302** |
| **B08** | **1.000** | **0.909** | **0.862** | **0.509** |
| B09 | 1.000 | 0.991 | 1.107 | 0.988 |
| B10 | 1.000 | 0.837 | 1.097 | 0.845 |
| B11 | 1.000 | 0.887 | 1.106 | 1.024 |
| B12 | 1.000 | 0.888 | 1.159 | 0.991 |
| B13 | 1.000 | 0.863 | 1.124 | 0.937 |
| B14 | 1.000 | 0.928 | 1.187 | 1.002 |

### Key Findings

1. **scene_20251215 B07:** gain = 0.302 (70% reduction) — the most extreme gain deviation
2. **scene_20251215 B08:** gain = 0.509 (49% reduction) — coincides with 2× integration time
3. **scene_20251208 B07:** gain = 0.614 (39% reduction)
4. **scene_20251120 B07:** gain = 0.867 (13% reduction) — moderate

### Per-Scene Average Gain Deviation

| Scene | Mean |gain-1| | Max |gain-1| | Worst Band |
|-------|----------------|---------------|------------|
| 20251114 | 0.000 | 0.000 | (control) |
| 20251120 | 0.079 | 0.163 | B10 (0.163) |
| 20251208 | 0.077 | 0.386 | B07 (0.386) |
| 20251215 | 0.105 | 0.698 | B07 (0.698) |

---

## 4. Radiometric Metrics (Ave Decomposition)

### Global

| Method | ADM | ADSD | Ave |
|--------|------|------|------|
| original | 326.85 | 60.86 | 193.85 |
| bagrn_gain_only | 314.12 | **14.59** | 164.35 |
| bagrn_offset_only | **259.56** | 60.86 | 160.21 |
| bagrn_full_reconstructed | 28.47 | 14.59 | 21.53 |
| bagrn_volrn_reference | 21.15 | 12.75 | 16.95 |

### Interpretation

- **Gain-only** primarily reduces **ADSD** (60.86 → 14.59, −76%) but barely affects ADM
- **Offset-only** primarily reduces **ADM** (326.85 → 259.56, −20%) but does not affect ADSD
- **Full BAGRN** synergistically reduces **both** ADM (→28.47) and ADSD (→14.59)
- The synergy is super-additive: full Ave (21.53) << gain_only Ave (164.35) + offset_only Ave (160.21) − original Ave (193.85)

---

## 5. Spectral Metrics (SAM)

| Method | SAM Mean | SAM Median | SAM P90 | SAM P95 |
|--------|----------|------------|---------|---------|
| original | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| bagrn_gain_only | **1.9570** | 0.0000 | 5.871 | 5.885 |
| bagrn_offset_only | **1.4868** | 0.0000 | 4.460 | 4.570 |
| bagrn_full_reconstructed | **0.9852** | 0.0000 | 2.956 | 2.999 |
| bagrn_volrn_reference | 1.0516 | 0.154 | 2.960 | 3.003 |

### SAM Fractions

| Component | SAM | Fraction of Full |
|-----------|-----|-----------------|
| gain_only | 1.957 | 1.986 |
| offset_only | 1.487 | 1.509 |
| full | 0.985 | 1.000 |

Both gain-only and offset-only produce **higher SAM** than full BAGRN. This means gain and offset **partially cancel** in spectral angle space. The full BAGRN achieves lower SAM because the multiplicative and additive corrections offset each other's spectral distortion.

---

## 6. Dominant Effect Determination

**Classification: joint_effect**

| Criterion | Threshold | Actual | Met? |
|-----------|-----------|--------|------|
| gain_frac >= 0.70 | 0.70 | 1.986 | Yes |
| offset_frac <= 0.40 | 0.40 | 1.509 | No |
| offset_frac >= 0.70 | 0.70 | 1.509 | Yes |
| gain_frac <= 0.40 | 0.40 | 1.986 | No |

Both gain and offset produce significant spectral changes, and neither alone can reproduce the full BAGRN effect. However, they work on **different components**:

- **Gain (omega)** adjusts **standard deviation** → addresses contrast/scale differences
- **Offset (upsilon)** adjusts **mean** → addresses brightness/level differences
- **Full BAGRN** combines both → addresses both simultaneously

---

## 7. scene_20251120 Analysis

### Coefficient Comparison

| Metric | 20251120 | Median of others | Difference |
|--------|----------|------------------|------------|
| Mean gain deviation | 0.079 | 0.060 | +0.019 |
| Max gain deviation | 0.163 (B10) | 0.386 (B07) | −0.223 |
| Mean |offset| | 647 | 354 | +293 |
| Ave (BAGRN) | 25.35 | 23.50 | +1.85 |

### Key Findings

1. scene_20251120's coefficients are **not** the most extreme — scene_20251215 has larger deviations
2. scene_20251120's poor Ave (25.35 vs 14.44 control) is **not** explained by extreme gain/offset
3. The gain deviations for scene_20251120 are moderate (0.837–1.200)
4. No single acquisition parameter explains its poor performance
5. **scene_20251120's problem is NOT acquisition-parameter-driven** — evidence supports other explanations (overlap composition, surface change, registration, etc.)

---

## 8. scene_20251215 B08/B09 Analysis

### Integration Parameter Anomaly

| Scene | Band | Integration Time | Level | Gain |
|-------|------|-----------------|-------|------|
| 20251114 | B08 | 8.2784 | 4 | 1.000 |
| 20251120 | B08 | 8.064 | 4 | 0.909 |
| 20251208 | B08 | 8.1656 | 4 | 0.862 |
| **20251215** | **B08** | **16.8704** | **8** | **0.509** |
| 20251114 | B09 | 8.2784 | 4 | 1.000 |
| 20251120 | B09 | 8.064 | 4 | 0.991 |
| 20251208 | B09 | 8.1656 | 4 | 1.107 |
| **20251215** | **B09** | **33.7408** | **16** | **0.988** |

### Findings

1. **scene_20251215 B08** has gain=0.509 (49% reduction) and 2× integration time — **coincides with** integration anomaly
2. **scene_20251215 B09** has gain=0.988 (near 1.0) despite 4× integration time — the gain is **not** affected because sigma_orig is large (543.3)
3. **scene_20251215 B07** has gain=0.302 (70% reduction) but **no integration anomaly** — this cannot be explained by integration settings
4. The integration anomaly for B08 is a **candidate explanatory factor** for its gain deviation, but not the sole cause
5. B07's extreme gain deviation has **no comparable integration-setting anomaly**

---

## 9. BAGRN+VOLRN vs BAGRN

| Metric | BAGRN | BAGRN+VOLRN | Change |
|--------|-------|-------------|--------|
| Ave | 21.53 | 16.95 | −21% |
| SAM mean | 0.985 | 1.052 | +7% |

VOLRN further reduces Ave by 21% but slightly increases SAM (7%). This suggests VOLRN's local corrections improve radiometric consistency but introduce mild additional spectral distortion.

---

## 10. Stage 3 Direction Recommendation

### Primary Direction: Spatial quality-aware local overlap weighting for BAGRN

**Evidence:** evidence supported

The current BAGRN uses only global moment constraints (mean, std) solved via weighted least squares. The multiwindow experiment shows max/min Ave ratio ~17–18×, meaning certain overlap regions have much worse statistics. The gain/offset ablation shows the global coefficients are well-behaved, but the spatial heterogeneity is extreme.

**Research focus:**
- Local window reliability scoring
- Effective pixel ratio weighting
- Texture-based quality assessment
- Initial residual-based weighting
- NoData proportion handling
- Registration quality awareness

### Secondary Direction: Reliability-constrained BAGRN gain estimation

**Evidence:** partially supported

B07 (633.5 nm) shows extreme gain deviations (0.302 in scene_20251215, 0.614 in scene_20251208). Red/NIR bands have larger residuals. Cross-band gain smoothing or reliability constraints could prevent extreme gain values.

**Research focus:**
- Limit extreme gain values per band
- Cross-band gain smoothness constraints
- Overlap quality-weighted gain estimation
- Avoid over-scaling in red/NIR bands

### Not Recommended as Primary: Offset-focused approach

**Evidence:** not supported as primary

Offset-only reduces ADM but not ADSD. The dominant radiometric issue is spatial heterogeneity (high Ave variance), not mean bias. Offset constraints alone would not address the core problem.

---

## 11. What Current Evidence Does NOT Support

| Claim | Evidence Status |
|-------|----------------|
| "scene_20251120 has abnormal sensor settings" | **not supported** — metadata shows no extreme |
| "B08/B09 gain anomaly is caused by integration time" | **partially supported** — coincides with, but B09 gain is near 1.0 |
| "B07 high residual is caused by integration settings" | **not supported** — no integration anomaly for B07 |
| "Gain and offset effects are linearly additive" | **not supported** — they partially cancel in SAM space |
| "scene_20251120 is worst because of acquisition conditions" | **insufficient evidence** — 4 scenes too few for attribution |

---

## 12. Files Generated

```
data/output/stage2_problem_discovery/dz01_stage2_problem_discovery/gain_offset_ablation/
├── coefficients/
│   ├── bagrn_global_coefficients.csv
│   └── bagrn_global_coefficients.json
├── validation/
│   ├── full_reconstruction_validation.csv
│   └── full_reconstruction_summary.json
├── metrics/
│   ├── radiometric_global.csv
│   ├── radiometric_per_band_*.csv
│   ├── radiometric_per_scene_*.csv
│   ├── radiometric_scene_band_*.csv
│   ├── spectral_global.csv
│   ├── spectral_per_scene_*.csv
│   └── spectral_per_band_*.csv
└── summary.json
```

---

## 13. Experiment Completion Status

| Experiment | Status | Note |
|------------|--------|------|
| band_attribution | completed | |
| scene_attribution | completed | |
| spatial_attribution | partially_completed | boundary/interior completed |
| multiwindow | completed | |
| nan_trace | completed_for_primary_question | no new NaN |
| gain_offset_ablation | **completed** | coefficient_type=BAGRN_global, full_reconstruction_verified=true |
| acquisition_attribution | diagnostic_added | exploratory only (n_scenes=4) |

**overall_stage2_status:** ready_for_stage3_model_design

---

## 14. Git Information

- **Branch:** stage2-gain-offset-ablation
- **Base tag:** stage2-metadata-correction-done
- **Commit:** (to be filled after commit)
- **Tag:** stage2-gain-offset-ablation-done
