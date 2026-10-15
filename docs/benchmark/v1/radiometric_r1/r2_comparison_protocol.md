# R2 comparison protocol handoff (not an execution authorization)

R2 begins only after the R1 readiness checklist is satisfied. No algorithm is run by this document.

## Comparison A — fixed public BAGRN input, local-module ablation

Freeze and hash the exact BAGRN-normalized images, geometry, masks, band mapping, and common evaluation windows once. Compare LMM, Ours, and VOLRN as local-stage alternatives on that same public input. Do not allow one alternative to alter another method's input, PIF/ROI labels, masks, or geometry. Preserve each method's status and output provenance, including failures. This is a local-module ablation, not an end-to-end comparison.

## Comparison B — common aligned, non-BAGRN input, complete radiometric/mosaic pipelines

Start every route from the same geometrically aligned but not BAGRN-normalized source version and common valid masks. Candidate routes include BAGRN+Ours, BAGRN+VOLRN, BAGRN-only control, and Spectralmatch as its own published complete workflow. Any route that performs its own joint coregistration/resampling must report that as part of its full pipeline; the fixed ROI evaluation is reprojected/read only for scoring, never used to tune the route. The two comparisons answer different questions and must not be pooled into one ranking.

## Evaluation and reproducibility rules

- Freeze the pre-method shortlist and manually reviewed ROI polygons/roles before running methods. All stable/real-change truth stays human-approved; estimation ROIs and evaluation ROIs are disjoint. TEST regions need independent second review.
- Use the same source bands, masks, valid support policy, fixed geographic evaluation regions, and data versions. Report each band separately and each pair/time stratum; no raw MAE pooling across bands.
- Report stable-region MAMD/MSDD/RDD/MAE/RMSE/P95, approved-change magnitude retention and (only with approved labels) F1/IoU, structure metrics under stated scale/grid conditions, and both fixed-region and method-seam scores.
- Record complete cold-run wall time, preprocessing, CPU RAM, GPU VRAM if used, hardware/software versions, parameters, random seeds, input/output hashes, and status. Do not label a `--resume`/cached run as first-run timing.
- Keep UNKNOWN/NO_VALID_SUPPORT/failed/insufficient-support denominators visible; do not discard them after seeing outcomes. Report the three RISK scenes overall and in a predeclared quality stratum; do not remove them.
- The 13-scene corpus is one regional sample, not independent cross-region generalization. Synthetic inverse recovery is a separate controlled condition and never evidence of real seasonal robustness.

## Conditions that must be met first

1. Researcher confirms the pre-method radiometric pair shortlist and every candidate geometry.
2. Human labels distinguish stable, real change, cloud, shadow and unknown; classes remain `UNKNOWN/UNVERIFIED` until reviewed with evidence. Establish independent DEVELOPMENT/TEST ROI sets and complete second review of TEST.
3. Complete pairwise spatial leakage/overlap audit; freeze ROI geometry, role, review history, input hashes and evaluation-mask rule.
4. Confirm physical units and calibration history before absolute reflectance claims. Until then, report source-unit relative differences only.
5. Decide the immutable Spectralmatch tag/commit and isolated compatible environment; confirm raster band stacking, output affine/grid, NoData and resampling in synthetic and tiny-window checks.
6. Freeze parameters, public BAGRN input, all route inputs, mask, aggregation, timing protocol, and report schema before R2 method outputs are inspected.

Until these checks pass, status remains `R1_INFRASTRUCTURE_READY_MANUAL_REVIEW_PENDING`; do not start R2 runs.
