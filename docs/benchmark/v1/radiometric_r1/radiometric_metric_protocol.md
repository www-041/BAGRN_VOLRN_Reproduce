# R1 independent radiometric metric protocol (draft, pre-method)

R1 metrics operate per human-reviewed ROI, per scene pair, and per band on identical shared finite valid support. Each row reports status, support count, pair/ROI/band/date-stratum identity, and all applicable values. Empty support returns `NO_VALID_SUPPORT` with null metrics, never zero. No R1 metrics have been run on real method outputs.

For reference samples `x` and corrected samples `y` on the same mask:

- `MAMD = |mean(x) − mean(y)|`.
- `MSDD = |population_std(x) − population_std(y)|` (`ddof=0`).
- `RDD = exact 1D Wasserstein-1 distance` over all valid samples, using `scipy.stats.wasserstein_distance`; not a histogram approximation.
- `MAE = mean(|x−y|)`, `RMSE = sqrt(mean((x−y)^2))`, `P95 = percentile_95(|x−y|)`, and `MAX_ABS_ERROR = max(|x−y|)`.

Stable-region consistency is reportable only for independently `APPROVED` stable ROI labels. A cross-date ROI without such a label remains a proposal, not a stable sample. Real-change magnitude MAE/RMSE and binary F1/IoU are conditional on an independently approved change truth. Without approved truth, output is `CONDITIONAL_NOT_EVALUABLE`; there is no proxy “change accuracy”.

## Structure and spectral metrics

Gradient-magnitude NCC uses Sobel gradients and finite shared support eroded by one pixel. SSIM uses 7×7 local windows, population local moments, and `C1=(0.01L)^2`, `C2=(0.03L)^2`; only fully valid windows are included. The data dynamic range `L` and common-grid/co-registration status must be explicitly confirmed by the caller. Otherwise both structure metrics return `CONDITIONAL_NOT_EVALUABLE`. These are independent R1 definitions; they do not claim to duplicate Task15/16 structure metric implementations.

Multiband outputs always preserve every band row. Raw MAE values are never pooled across bands. A caller must supply a pre-frozen positive scale for every band; the only cross-band scalar currently exposed is the macro mean of per-band `MAE/scale`. SAM is only computed if common spectral co-registration and scale are explicitly confirmed, with jointly valid nonzero spectra. SAC is `CONDITIONAL_NOT_EVALUABLE` because its exact R1 definition is not frozen; no formula is invented.

## Aggregation

The aggregation order is equal-weight ROI mean within each pair/band/time stratum, followed by equal-weight pair macro mean. Report the per-pair distribution, number evaluable, and non-evaluable denominators. Do not weight one pair more because it has more pixels or more ROIs. Pixel-level distributions may be supplemental but are not a substitute for pair-level macro results. No composite score is defined.

All physical-unit values remain in source digital units, currently `UNKNOWN`; never call them reflectance or absolute reflectance error. Cross-date masks, geometry, sample role, and approval history must be frozen before any method comparison.
