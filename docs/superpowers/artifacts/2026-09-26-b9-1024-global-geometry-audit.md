# B9 1024 Global Geometry Audit

Status: Task 8C analysis-only audit. Existing matcher, RANSAC, Global, radiometric, and mosaic code is not rerun or changed by this report.

## Frozen experiment

- Repository branch: `exp/2026-09-24-b9-five-scene-validation`
- Starting HEAD: `2b68c043ab2907c7bdab51246f714a91f34ee5a8`
- Dataset: frozen B9 five-scene manifest indices `[2, 3, 5, 8, 10]`
- Protocol: `match_max_side=1024`, `pixel_size_m=14.0`, coordinate frame `pair_common_grid`, transform direction `target_to_reference`
- Methods: `sift`, `loftr`, `efficient_loftr`, `lightglue_disk`; Global methods `MST`, `Translation-L2`
- Inputs are the existing `matcher_runs_1024_globalready/*` and `global_runs_1024/*` artifacts.

## Task 0 provenance

`08_global_method_comparison.json` and `.csv` contain exactly eight rows (four matchers × two Global methods); all eight have `status=PASS`, `accepted_edges=10`, and `connected=true`. Each run has `global_transforms.json`, `global_edge_consistency.{json,csv}`, and `global_registration.json`; MST runs additionally have `spanning_tree.json` and `01_mst_point_residuals.csv`, while Translation-L2 runs have `05_translation_point_residuals.csv` and `06_translation_solution.json`. The ten pair bundles are stored once per matcher under `matcher_runs_1024_globalready/<matcher>/geometry/pair_*.npz` with matching sidecars.

The downstream sections are filled by Tasks 1–10 from these frozen artifacts only.
