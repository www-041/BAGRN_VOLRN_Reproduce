# DZ01V multi-temporal radiometric Benchmark R1 — summary

R1 delivered independent infrastructure and readiness evidence. **No BAGRN/VOLRN/Spectralmatch/other radiometric algorithm was run on real imagery; no full-scene or 13-scene experiment was run.** The package namespace and default run outputs are distinct: `src/benchmark_v1_radiometric_r1/` and `data/output/benchmark_v1_radiometric_r1/`.

## Data and pair scope

- 13 processed DZ01V scenes; 16 V bands per scene; 208 metadata records; all scenes report all B1–B16.
- Existing B9 inventory: 78 scene combinations, 48 real shared-valid-mask overlaps and 30 no-valid-overlap pairs. The latter are `NO_VALID_OVERLAP`, not matcher failures.
- Valid overlaps: 7 same-date, 41 cross-date; temporal strata T1 (<7 d)=13, T2 (7–<30 d)=6, T3 (≥30 d)=22.
- Ten pre-method pair proposals: `00_01`, `10_11`, `04_06`, `06_08`, `02_03`, `02_08`, `00_10`, `02_10`, `04_11`, `06_11`. Required 00_01 and long-gap 00_10/02_10/04_11 are included. Selection used only time separation, overlap support, quality flag and deterministic coverage—not algorithm scores.
- Three pairs (`06_08`, `02_08`, `06_11`) have navigation polygons derived from bbox intersection only; they remain explicitly unverified until GUI mask fractions and a human check. Existing region proposals also remain unverified.

## Tools and limited validation

- ROI candidate builder emits ten proposal features and a separate review template. Every label is `UNKNOWN`, every review status is `UNVERIFIED`; zero human approvals were created.
- Tkinter/Pillow/Rasterio GUI is available in the current venv. `--check-env` succeeds; `--validate-pair --pair 00_01` verifies source files and headers without launching a window. Synthetic tests cover geospatial alignment, bounded window reads, validity fractions, save/reload, independent review labels and spatial-role safety. The interactive screen was not visually operated by this run; the researcher must open it and review images.
- A candidate-centered native 1024×1024 display-window smoke read for 00_01 found 727,508 valid pixels in scene i, 725,505 in scene j, and 412,629 shared-valid pixels. This was a bounded window read, not a registration or radiometric experiment.
- Final R1 automated suite: 40 passed; 11 Rasterio/Affine pending-deprecation warnings, no failures. `compileall`, `git diff --check`, and the R1 scoped trailing-whitespace scan passed.
- S1–S4 deterministic synthetic NPZ examples were generated at 64×64 (S3 three bands) under the R1 output root. They are arbitrary synthetic units, not reflectance and not algorithm results.
- Independent metrics now expose MAMD/MSDD/exact Wasserstein RDD, fixed-ROI MAE/RMSE/P95/max, conditional approved-change retention, gated structure measures, per-band scaled summaries and pair-macro aggregation. Raw MAE is never pooled across bands.
- Existing B9 manifest hashes were reused for 13 scenes. No full-image hash was newly computed. Other 195 scene-band file hashes are not frozen.

## Scientific status

- All data units remain `UNKNOWN`; MTL coefficients were not applied. Wavelength centers/bandwidths are present in existing metadata, but this does not establish radiometric calibration or inter-band subpixel co-registration.
- Three scenes remain `QUALITY_STATUS=RISK` and must stay in overall results with a prespecified quality stratum.
- No independently approved stable or real-change ROI exists yet. Synthetic known truth is separate from real-image truth.
- Spectralmatch is a promising secondary full-pipeline R2 candidate, but not ready in `.venv-registration`: current repo main advertises v1.6.0, the latest observed tagged release is v1.4.1, and this environment lacks several required Python dependencies and has GDAL 3.10.3 vs the documented system requirement ≥3.11. No install or run was performed.

## Current phase state

`R1_INFRASTRUCTURE_READY_MANUAL_REVIEW_PENDING`. Begin by opening `radiometric_roi_annotation_guide.md`, running the candidate preparation command if needed, then manually inspecting 00_01. R2 is not started; its prerequisites and two comparison layers are in `r2_comparison_protocol.md`.
