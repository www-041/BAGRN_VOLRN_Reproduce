# Candidate research hypotheses for later validation

These are research questions, not established findings or claims of novelty. R1 has not run any radiometric method on real images and has no approved stable/real-change ROI labels yet.

## Established by current metadata/records

- The V-sensor input comprises 13 scenes and B1–B16; metadata gives center wavelength/bandwidth for all 208 scene-band rows, while all physical units remain UNKNOWN.
- The scene window spans roughly 2025-11-14 to 2026-01-14 (about 61 days), not a full annual or multi-season cycle.
- The frozen inventory contains 48 pairs with real B9 valid-mask overlap: 7 same-date and 41 cross-date. Three scenes are flagged QUALITY_STATUS=RISK.
- The 00_10 (~61 d), 02_10 (~55 d), and 04_11 (~37 d) candidates have existing B9 mask-supported candidate windows. This verifies input support only, not stable landcover or seasonal change truth.
- Existing literature/software already explores PIF selection, masking and global-to-local radiometric matching. Spectralmatch publicly documents global regression, local block adjustment, PIF/mask functions and time-series workflows. These concepts themselves are not claimed as new here.

## Falsifiable hypotheses to test after R1 annotation

| ID | Hypothesis | Required independent evidence / comparison | What would weaken it |
|---|---|---|---|
| H1 | Longer acquisition gaps and vegetation change increase contamination of automatically selected pseudo-invariant samples, causing real changes to be partially normalized away. | Human-reviewed STABLE and REAL_CHANGE ROIs, including long-gap pairs; compare PIF membership/fit influence and change-amplitude retention on held-out approved ROIs. | Stable/change labels show no meaningful interaction with temporal gap, or change amplitude remains preserved across methods. |
| H2 | Cloud/shadow boundaries and terrain illumination produce local radiometric degradation that is hidden by scene-average metrics. | Independent CLOUD/SHADOW/TERRAIN labels; report fixed ROI and local-block/seam distributions stratified by class and QUALITY_STATUS. | No repeatable class-specific/local tail degradation on held-out ROIs. |
| H3 | Independently fitting each V band can lower within-band overlap differences while distorting cross-band spectral shape. | Verified cross-band co-registration and calibration/scale semantics; approved same-location multi-band stable and change spectra; per-band metrics plus conditionally valid SAM. | No material spectral-shape loss under scale-confirmed co-registered validation. |
| H4 | A global correction can improve stable-region agreement while overcorrecting a spatially localized real-change region. | Separate calibration ROIs from held-out approved REAL_CHANGE ROIs; compare global-only, local-stage, and full routes. | Change retention is comparable while stable consistency improves, across independent regions/pairs. |
| H5 | Quality-risk scenes have heavier spatial/temporal error tails than PASS scenes, and overall averages can obscure that behavior. | Keep all 13 scenes; predeclare PASS/RISK strata and pair-macro distributions before method results. | Quality-stratified distributions are similar or differences do not repeat across pairs. |

## Novelty discipline

The hypotheses concern evaluation/behavior, not proposed algorithms. Do not claim that a method solves seasonal change, terrain, shadow, or spectral distortion until independent labels and held-out evidence support the claim. Do not call the November-to-January sample a four-season/general seasonal benchmark. Internal spatial partitions from this one region are not cross-region generalization. Unknown physical units preclude absolute reflectance-error claims. Every result must state which hypothesis was tested, which remained untestable, and how many independently reviewed ROIs supported it.
