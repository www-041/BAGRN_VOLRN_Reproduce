# DZ01V Radiometric Benchmark R1.5 — Automatic ROI Proposals and Human Review

## Status

Design specification for review. This is an annotation-support stage only; it does not establish scientific ground truth or run radiometric normalization.

## 1. Purpose and success criteria

R1.5 reduces the manual effort required to locate useful radiometric regions in four already-known B9-overlap pairs: `00_01`, `02_03`, `05_07`, and `08_10`. It generates spatially distributed ROI proposals and cautious machine-generated suggestions that a researcher can inspect, edit, approve, mark doubtful, or reject.

Success means that the four pair-specific candidate collections and quality summaries can be generated reproducibly; every suggestion is traceable to input bands, versions, parameters, and evidence; the existing Tkinter workflow can rapidly review proposals; and no automatic output is mistaken for a human label or formal Benchmark truth.

The four pairs are tool-development samples only. They do not replace or reorder R1's ten-pair radiometric shortlist.

## 2. Scope and non-goals

### In scope

- Read only the inventory-identified, identity-checkable DZ01V V-sensor B1–B16 GeoTIFFs for the four specified pairs.
- Use B9 as the mandatory initial geometry, common-valid-support, and browsing band. Auxiliary bands may be used only after verifying wavelength metadata, CRS, transform/grid relation, masks, and data type/unit metadata.
- Generate small, spatially distributed ROI proposals from common valid coverage, local texture/homogeneity, edge proximity, and available multispectral evidence.
- Assign explicitly non-authoritative class and land-cover suggestions with evidence and uncertainty explanations.
- Apply the already-frozen EPSG:32650 10 km grid, 500 m safety buffer, and Phase3 role formula without changes.
- Extend the existing annotation entry point in an opt-in mode while keeping all existing R1 annotation inputs and outputs untouched.
- Produce candidate inventories, quality/exclusion summaries, spatial visualizations, instructions, and focused tests.

### Explicitly out of scope

- Running BAGRN, Ours, VOLRN, Spectralmatch, or any radiometric correction, mosaicking, or formal evaluation.
- Declaring any generated candidate or label to be approved truth.
- Rewriting existing R1 candidate GeoJSON, R1 review CSV/JSONL, Phase3–Phase5 artifacts, Task15/Task16 code/results, or source images.
- Applying MTL calibration coefficients or asserting surface-reflectance units.
- Inferring terrain from brightness or shadow appearance; DEM-based terrain classification is unavailable unless a separately verified source is later supplied.
- Expanding generation to all 48 overlap pairs or replacing the existing ten-pair R1 shortlist.

## 3. Existing project interfaces and constraints

The current R1 implementation provides `RoiPairReader` for map-aligned paired windows, display-only stretching, valid-mask fractions, the ROI role classifier, editable polygon/rectangle geometry, and a CSV snapshot plus append-only JSONL event history. Its current CLI passes only B9 and uses `data/output/benchmark_v1_radiometric_r1/roi_reviews/` for human decisions. R1's B1–B16 readiness inventory identifies all 16 bands for each of the four requested pairs and reports their files present; data units remain `UNKNOWN`.

The frozen Phase3 role rule is EPSG:32650, origin `(0,0)`, 10,000 m cells, a strict 500 m margin on all cell edges, and `DEVELOPMENT if (ix + 2*iy) % 3 == 0 else TEST`. R1.5 must call or exactly reproduce this existing rule; it must not alter the rule, cell size, buffer, origin, or role formula. Any ROI that is invalid, crosses a cell, enters a buffer, or fails the common-support checks remains `UNASSIGNED` or is excluded with a reason. R1.5 will not auto-shrink/crop ROI geometries.

The workspace contains unrelated pre-existing untracked files. Implementation must preserve them and limit edits to the R1.5 modules, the opt-in annotation integration, R1.5-specific tests, and new R1.5 output/docs paths.

## 4. Architecture and data flow

Use a dedicated implementation module, `src/benchmark_v1_radiometric_r15/`, so candidate generation, suggestion logic, spatial auditing, and GUI state are isolated from the R1 measurement and metric implementations. Add a generator CLI at `scripts/benchmark/generate_radiometric_r1_5_auto_rois.py`. Add an opt-in `--auto-candidates` mode to the existing `scripts/benchmark/annotate_radiometric_rois.py`; without that flag, its current behavior and paths remain unchanged. The auto mode launches the R1.5 review flow and writes only to the separate R1.5 review directory.

Pipeline:

1. Read the existing scene inventory, R1 pair inventory, B1–B16 readiness inventory, and the existing B9 identity manifest. Resolve the requested pair by its frozen `pair_id` and `product_id` records; reject unknown pairs, missing files, missing B9 overlap, duplicate band records, and mismatched input identities before reading pixels.
2. Inspect GeoTIFF headers for all available bands. Verify CRS, affine transform, native resolution, wavelength metadata, dtype, nodata/mask, and declared unit. Do not infer missing values. B9 is required in both scenes. An auxiliary band is usable only if both scenes have it and its metadata and geospatial mapping can be verified. Record each band's inclusion/exclusion reason.
3. Define the working grid from scene-i B9 in EPSG:32650. Map scene-j pixels by georeferencing; do not assume equal row/column indices. If a band must be reprojected to the reference grid for feature calculation, record the exact resampling method and keep the corresponding mask resampling nearest-neighbor. No source file is rewritten.
4. Read only bounded windows. Build the common B9-valid mask and per-band common-valid supports. Search fixed-size 32×32-native-pixel proposal windows (448 m at 14 m pixels); the size is configurable before a generation run and must be recorded. Do not relax validity, homogeneity, edge, or coverage criteria to force a target count.
5. Score candidate windows for support, within-image homogeneity, texture sufficiency, distance from strong boundaries, and spatial coverage. Use deterministic tie-breaking and a recorded seed if any randomized tie-breaking is required. Limit proposals to 15–30 per pair when evidence permits; fewer is a valid result. Candidate ranking must not use any corrected image, seam, matcher, registration residual, or algorithm outcome.
6. Create immutable candidate records with `UNVERIFIED` review state. Compute the frozen spatial role for the entire proposal geometry and run the cross-pair leakage audit against all R1.5 proposals generated so far. Do not adjust geometry or role to improve counts.
7. For `00_01`, save a small initial candidate set and quality report; verify candidate loading, display, role calculation, edit/review persistence, and reload with automated checks plus a bounded GUI smoke check. Only after this check passes, generate `02_03`, `05_07`, and `08_10` with the same parameter configuration and code version. The researcher performs the substantive visual suitability review after delivery.

## 5. Suggestion semantics and conservative decision rules

Candidate suggestions are prioritization cues, not calibrated probabilities or truths. `suggestion_confidence` is an evidence tier (`LOW`, `MEDIUM`, `HIGH`), never a claim of statistical accuracy. Under current inputs, `HIGH` is prohibited. Every proposal starts with `human_review_status=UNVERIFIED`; the generator cannot write an approval event, reviewer identity, review time, or human precision estimate.

Supported suggested labels are `STABLE`, `REAL_CHANGE`, `CLOUD`, `SHADOW`, and `UNKNOWN`.

- `STABLE` may be suggested only when multiple independent cues agree: adequate common support, low internal heterogeneity in each date, strong spatial-structure agreement, and compatible multispectral shape over the available verified bands. Brightness similarity alone is insufficient. Unknown units, atmospheric differences, and imperfect subpixel correspondence must lower the tier and appear in `uncertainty_reason`.
- `REAL_CHANGE` means a suspected change proposal only. It requires coherent spatial or multi-band structural change evidence rather than a scalar brightness difference alone. Unknown units and possible illumination/atmosphere effects must be called out. Only independent human evidence can approve `REAL_CHANGE`.
- `CLOUD` and `SHADOW` may be suggested only from conservative multi-band and local-context cues. Bright bare ground/rock, water, vegetation, and terrain shadow are known confounders. A dark pixel alone cannot suggest `SHADOW`; when cloud/shadow cannot be distinguished from those cases, use `UNKNOWN` and state the ambiguity.
- `suggested_landcover_type` remains `UNKNOWN` unless a reliable, verified land-cover source is available. `terrain` remains `UNKNOWN` without a reliable DEM or verified terrain source. Spectral appearance alone must not be represented as a named land-cover or terrain class.
- Missing unit, wavelength, band, mask, DEM, land-cover, or cross-band co-registration evidence must be explicit in the record and report. Do not apply calibration coefficients or claim reflectance.

Suggestion evidence is stored as machine-readable cue names plus a concise explanation; uncertainty reasons are separately enumerated. Human changes never overwrite the generated suggestion fields.

## 6. Candidate record and review persistence

Generate one GeoJSON FeatureCollection and a tabular summary per pair under:

`data/output/benchmark_v1_radiometric_r1/auto_roi_candidates/`

Candidate records include at minimum:

- stable unique `candidate_id`, `pair_id`, `scene_i`, and `scene_j`;
- polygon geometry in EPSG:32650 and an explicit CRS field;
- fixed-grid cell ID, computed role, and spatial status;
- source band paths, per-band wavelength/dtype/unit/mask/grid checks, input identity/hash status, and resampling record;
- common support counts and fractions, homogeneity/texture/edge evidence, and proposal rank;
- `suggested_label`, `suggested_landcover_type`, `suggestion_confidence`, `suggestion_evidence`, and `uncertainty_reason`;
- generator name/version, configuration, seed, creation timestamp, and `human_review_status=UNVERIFIED`.

The candidate collection is immutable after generation. R1.5 human review output and append-only event history use a separate directory:

`data/output/benchmark_v1_radiometric_r1/auto_roi_reviews/`

Each review event refers to `candidate_id` and preserves the original geometry, any adjusted geometry, prior suggestion, human label, status, evidence notes, real reviewer identity, and timestamp. Manual geometry changes create a review-side record; they never rewrite the source candidate. Formal TEST use still requires a distinct real second reviewer and the existing R1 eligibility checks, including a passing pairwise spatial-overlap audit.

## 7. GUI behavior

The opt-in auto mode retains the current paired B9 display, common geospatial window, zoom/pan, band selector, geometry tools, and audit history, and adds:

- a candidate queue with previous/next, pending/reviewed/all filters, and progress counts;
- display of candidate ID, suggestion, land-cover suggestion, evidence tier, evidence, uncertainty, support fractions, usable bands, and frozen spatial role/status;
- synchronized display of the proposal geometry in both scenes; band switching changes only the visualization/feature layer, not source values;
- manual polygon/rectangle modification saved as a new review-side geometry with history;
- human choices `APPROVED`, `DOUBTFUL`, or `REJECTED`, plus label override and required evidence notes for approval;
- explicit confirmation before human approval; no batch approval;
- save/reopen recovery without changing the immutable candidate collection or existing R1 review files.

Only the R1.5 opt-in path uses the queue and separate review files. Existing R1 CLI behavior remains backward compatible.

## 8. Quality report and visualization

For each pair and overall, report proposed, eligible, and excluded counts; exclusion reasons; suggestion and confidence-tier counts; common valid support; usable/missing bands; wavelength/unit/grid checks; resampling; cell/role counts; duplicate/overlapping proposals; candidate spatial coverage; and any cross-pair development/test conflict. The spatial map uses distinct symbols/colors for DEVELOPMENT, TEST, and UNASSIGNED, labels the pair and candidate ID, and includes the frozen grid safety margins. The map is for review only, not evidence of label correctness.

## 9. Testing and acceptance

Use synthetic rasters and small temporary stores for automated tests. Required coverage:

1. no valid overlap, nodata edges, and partial windows;
2. geographic mapping where scene row/column grids differ;
3. missing bands, unknown units, absent wavelength metadata, transform mismatch, and mask handling;
4. deterministic candidate generation, spatial coverage, minimum support, edge exclusion, duplicate IDs, and fewer-than-target behavior;
5. candidate geometry/metadata save and reload; immutable suggestion fields and never-approved generated status;
6. frozen role calculation, buffer/cross-cell `UNASSIGNED`, and cross-pair overlap/leakage report;
7. manual geometry edits preserve candidate geometry and append before/after review events;
8. GUI candidate loading, navigation, display-band changes, label override, approve/doubt/reject, and resume after reload;
9. ordinary R1 annotation mode remains unchanged and all existing R1 tests remain green;
10. verify Phase3 files, R1 candidate/review data, Task15/16, Phase4/5 data and source imagery are not modified by candidate generation or automated tests.

Real-image validation is limited to bounded windows from `00_01`, then the other three pairs after the researcher validates the first pair's proposal quality and GUI flow. Do not scan or load full images into memory. Do not run any radiometric algorithm or formal score.

## 10. Deliverables and stop condition

Deliver:

- isolated R1.5 candidate-generation, spatial-audit, and review modules;
- `scripts/benchmark/generate_radiometric_r1_5_auto_rois.py`;
- opt-in auto-candidate mode in `scripts/benchmark/annotate_radiometric_rois.py`;
- per-pair candidate GeoJSON/CSV, quality reports, and spatial maps for the four pairs;
- `radiometric_r1_5_auto_roi_usage.md` with PowerShell generation/review commands;
- focused tests and regression results.

The phase ends after the researcher can begin genuine review and all four proposal sets are generated and validated. It must report actual proposal counts and limitations, explicitly state that all proposals remain unverified, and stop. It must not start R2 or declare radiometric Benchmark truth complete.
