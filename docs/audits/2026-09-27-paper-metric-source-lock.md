# Task10C paper-metric source audit

Date: 2026-09-27

Status: `AMBIGUOUS`; Task11.5A hard-stopped before implementation.

## Sources inspected

- 2024 paper, DOI `10.1016/j.isprsjprs.2024.08.016`, public author-uploaded full text, Section 3.1, Eq. (38)--(41).
- Xia et al. 2017, *Color Consistency Correction Based on Remapping Optimization for Image Stitching*, public ICCV Workshop paper, Eq. (9).
- `MenghanXia/ColorConsistency` at commit `05d03d87ea67dd92de7ccea09dd450579f6d991e`.

The 2024 paper states that CD is an overlap-area-weighted histogram difference divided by `N_b`, and that GL is the mean over images of the per-image gradient-orientation-map difference divided by `N_p`. It also states that overlap weights are normalized to sum to one. These facts are sufficient to reject the current histogram-TV implementation as an unverified paper formula, but not sufficient to implement a uniquely specified paper metric.

## Verified implementation-relevant evidence

The cited public implementation contains `Utils::getGradientOrientationMaps` and `Utils::calStructuralDiffs` in `Source/Utils/util.cpp`. At the frozen commit it:

1. uses the current pixel minus its right and down neighbors as `gx` and `gy`;
2. skips the last row/column and skips zero right/down neighbors;
3. computes orientation with `atan2l(gy, gx)`;
4. computes the diagnostic as the mean of `fabs(orientation_1 - orientation_2)`.

This records the reference behavior, including the fact that the code does not apply an angular wrap correction.

## Remaining hard-stop ambiguities

The available sources do not uniquely determine:

- the Eq. (38) histogram channel, value range, invalid-pixel treatment, and normalization;
- the numeric `N_b` used by the paper experiments;
- the exact `DeltaH` bin-to-bin norm/aggregation (the sources only say “difference between histograms by bins” or “bin-to-bin distance”);
- whether Eq. (39) intends the cited implementation's raw absolute angle difference or a shortest-arc angular distance at the `-pi/pi` boundary.

Consequently, this audit intentionally does not promote either the existing histogram-TV metric or a newly invented histogram/angular convention to verified paper status.

See the machine-readable field-by-field record in [`2026-09-27-paper-metric-source-lock.json`](2026-09-27-paper-metric-source-lock.json).
