# Task 10 Radiometric Pipeline Audit

Date: 2026-09-26

## Scope

Task 10 compares `RAW`, `BAGRN`, and `BAGRN+VOLRN` under frozen B9 geometry. The
geometry stage is therefore an input contract for this audit, not a stage that
may be rerun by the radiometric runner.

## Existing entry points

| Component | Current entry point | Input | Output | Task 10 disposition |
|---|---|---|---|---|
| BAGRN | `src.bagrn.bagrn_normalize` | registered arrays `(bands, rows, cols)`, nodata values, overlap records, control index | normalized arrays, `theta_mu`, `theta_sigma` | use directly; record the returned parameters |
| VOLRN | `src.volrn.volrn_normalize` | BAGRN arrays, aligned transforms/bounds, nodata values, block and ADMM parameters | normalized arrays, block coefficients; optional diagnostics | use directly after BAGRN; request diagnostics |
| Existing multi-scene adapter | `src.multiscene_sift.radiometric.normalize_registered_band` | `Scene` objects, persisted world transforms, optional fixed registration grid | `BandRadiometricResult` containing BAGRN/VOLRN arrays and metrics | useful reference and compatibility surface; Task 10 runner must not invoke matching or global adjustment |
| Weighted feather mosaic | `src.mosaic.create_mosaic` with `mode="weighted"` | registered arrays, aligned transforms, CRS, nodata, fixed output grid | GeoTIFF and optional coverage diagnostics | reuse unchanged semantics and fixed Task 9 grid |
| Task 9 persisted-geometry runner | `scripts/run_b9_weighted_mosaic.py::run_weighted_mosaic` | frozen source config, global transform directory, canonical output grid | `mosaic.tif`, preview, masks, summary | extend the same source/grid/warp contract for radiometric variants |

## Data and geometry contract

The frozen B9 source configuration selects manifest indices `[2, 3, 5, 8, 10]`
and band `B9` at 14 m in `EPSG:32650`. The primary geometry is the persisted
`efficient_loftr/translation_l2` global solution and the control geometry is
the persisted `sift/mst` solution. The repository's current directory name
for Equal-L2 Translation is `translation_l2`; Task 10 reports the protocol
method name rather than inventing a new transform.

Every radiometric variant must use the same:

- five source rasters and source order;
- persisted global transforms;
- corrected transforms and north-up warp;
- canonical output transform, width, height, CRS, and 14 m resolution;
- nodata handling and weighted distance-feather mosaic;
- seam-zone definition used by Task 9 diagnostics.

No matcher, RANSAC, graph/MST, global adjustment, seamline optimization, or
large-scale output-range change belongs in this runner.

## Format and adapter findings

1. Source B9 inputs are single-band GeoTIFFs. Task 9 reads band 1 and keeps
   source nodata as the invalid value; the radiometric runner should project
   each source to the fixed grid as floating-point data with `NaN` as the
   in-memory invalid marker, then pass the same aligned transform/bounds to
   BAGRN and VOLRN.
2. BAGRN and VOLRN operate on in-memory registered arrays, not on file paths.
   They already return parameter arrays and block-level coefficients, so the
   runner needs an artifact adapter rather than new radiometric formulas.
3. `normalize_registered_band` is coupled to the general `Scene` pipeline and
   performs its own source reads and fixed-grid reprojection. For Task 10 the
   lower-level registered-array path is preferable because it makes the
   no-rematching/no-reoptimization constraint auditable and allows RAW to use
   exactly the same projected arrays.
4. Existing metrics in `src.metrics` accept overlap records and optional cloud
   masks. Task 10's B9 source is a single band; the report adapter must expose
   pairwise overlap MAE/RMSE/Bias and histogram mean/std/distance, plus the
   Task 9 seam-zone gradient metric.
5. Existing multi-scene cloud-mask generation requires B2/B5/B7/B14 and MTL
   metadata. It is not applicable to the B9-only fixed-geometry experiment
   unless a separately frozen cloud-mask artifact is supplied. The Task 10
   runner therefore records cloud masking as disabled for this B9 protocol and
   must not silently synthesize a B9 cloud mask.

## Baseline test observations

The focused baseline before Task 10 changes was `13 failed, 1 passed` for the
existing radiometric/runner/import/config selection. The actionable production
compatibility findings are:

- adapter tests still call the established `reference_idx` keyword while the
  implementation only accepts `radiometric_control_idx`;
- synthetic full-run fixtures exercise the general cloud-aware runner without
  the required Landsat-style B2/B5/B7/MTL inputs;
- `src/gen_report_figures.py` performs file writes during import;
- the optional-fields config test also performs real-data path validation even
  though its assertion is about config fields, while the referenced external
  sample dataset is not part of this repository.

These are kept separate from the B9 fixed-geometry experiment artifacts. They
will be repaired or explicitly isolated with regression tests before final
verification; no existing full-suite failure will be relabeled as a Task 10
scientific result.

## Task 10 adapter decision

Implement one small fixed-geometry runner that has three radiometric modes:

- `RAW`: projected registered arrays directly to weighted feather;
- `BAGRN`: BAGRN output to weighted feather;
- `BAGRN_VOLRN`: BAGRN output, then VOLRN output, to weighted feather.

The runner will persist the frozen source/geometry/grid identities in every
experiment directory and will fail on an existing non-empty output directory,
leaving resume/partial-output policy to the batch wrapper. It will write
`mosaic.tif`, `preview.png`, `radiometric_summary.json`, plus the diagnostic
arrays required by the metrics/report stages.
