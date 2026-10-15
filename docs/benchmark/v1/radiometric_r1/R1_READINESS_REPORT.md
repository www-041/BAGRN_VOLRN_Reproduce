# R1 readiness report

**Disposition: `R1_INFRASTRUCTURE_READY_MANUAL_REVIEW_PENDING`.** R1 infrastructure is present, but R2 experiment execution is not yet authorized by readiness: the stable/change truth and evaluation masks do not exist as approved human records.

## Ready now

- Read-only inventory and 10-pair pre-method draft for the 48 real B9-overlap pairs among 78 combinations.
- Usable ROI annotation GUI, candidate GeoJSON, schema/template, append-only review log, and source-hash provenance. Inputs are script-rooted, not dependent on PowerShell's current working directory.
- Independent metric implementations with explicit support/status, synthetic tests, controlled S1–S4 truth generation, metadata-only B1–B16 readiness inventory, and formula compatibility audit.
- Spectralmatch compatibility review without clone/install/run.
- Verification: `40 passed` in `tests/radiometric_r1`; R1 `compileall`, `git diff --check`, and scoped whitespace scan clean. The suite emits 11 upstream Rasterio/Affine pending-deprecation warnings.

## Not ready for R2

1. `APPROVED` stable and real-change ROIs: **0**. Candidate regions are not truth. Manual review on both dates/bands plus evidence is required.
2. Separate estimation/development and evaluation ROI sets: not yet manually drawn/frozen. The broad ten proposals are navigation windows and currently often `UNASSIGNED` under the strict frozen grid; draw smaller polygons/rectangles within them and let the unchanged formula classify each.
3. Pairwise ROI spatial-overlap/leakage audit: pending; live records keep `spatial_overlap_audit=PENDING_PAIRWISE_AUDIT`, which prevents formal TEST eligibility.
4. Independent second reviewer for TEST evaluation ROIs: none. Identity must be a different real reviewer; second-class disagreement must stay doubtful/disputed.
5. Confirmed physical unit/calibration: absent (`UNKNOWN`). No absolute reflectance accuracy claim.
6. Inter-band co-registration: metadata grid matches within each scene, but no subpixel cross-band check; SAM/SAC remain conditional.
7. Source identity freeze: B9 has 13 existing hashes; hashes for other 195 scene-band files remain ungenerated/unfrozen. R1 did not hash large rasters.
8. Spectralmatch isolated pinned environment and tiny integration compatibility: not prepared; current environment is unsuitable and must not be modified for it.

## Researcher actions to move to R2

1. Use the documented PowerShell command to launch `00_01`; inspect the two displayed dates, draw appropriately small regions, and record `STABLE`, `REAL_CHANGE`, `CLOUD`, `SHADOW`, or `UNKNOWN` only when supported. Do not rely on proposal geometry or score as truth.
2. Review enough selected pairs to cover same-date and short/medium/long gaps; confirm the shortlist. Manually allocate distinct calibration and evaluation regions before any method output is inspected.
3. Independently double-review all TEST evaluation ROIs and resolve class disagreements without automatic preference.
4. Run the cross-pair overlap/leakage audit; freeze approved geometry, labels, roles, review histories, ROI source hashes and evaluation-mask/support policy.
5. Decide whether/when to establish pixel calibration and validate cross-band geometry. Until then, keep metrics in unknown source units and omit absolute reflectance/SAM/SAC claims.
6. Only then freeze R2 inputs/parameters, test the pinned Spectralmatch version in a new isolated environment, and explicitly authorize the R2 experiment run.

## Start human ROI annotation

From the repository root in PowerShell:

```powershell
.\.venv-registration\Scripts\python.exe scripts\benchmark\prepare_radiometric_r1_rois.py
.\.venv-registration\Scripts\python.exe scripts\benchmark\annotate_radiometric_rois.py --pair 00_01 --reviewer "你的真实姓名或审核身份"
```

Output annotations go only to `data/output/benchmark_v1_radiometric_r1/roi_reviews/`; candidate GeoJSON and frozen input manifests are not edited.
