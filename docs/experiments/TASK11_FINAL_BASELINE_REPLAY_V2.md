# Task11 Final End-to-End Baseline Replay v2

Date: 2026-09-27

## Frozen inputs

- Selection: B9 manifest indices `[2, 3, 5, 8, 10]`.
- Canonical grid: EPSG:32650, 14 m, 5116 x 6134 pixels.
- Main geometry: persisted `efficient_loftr/translation_l2` transforms.
- Traditional geometry: persisted `sift/mst` transforms.
- No matcher, RANSAC, MST, Translation-L2, or VOLRN process was run.

## Replay result

Both historical radiometric directories lacked per-scene BAGRN rasters, so the replay performed exactly two BAGRN-only runs and wrote separate replay caches. Each cache contains five canonical-grid normalised scenes, BAGRN parameters, provenance, RAW/BAGRN Task10D snapshots, and a weighted-feather mosaic rebuilt from the persisted scenes. The hardened rerun was written to `data/output/b9_five_scene_validation/final_replay_v2_hardened/` and the non-overwriting paper package to `data/output/b9_five_scene_validation/final_results_v2/`.

Exact execution (registration/matcher and Global stages were not invoked):

```text
python scripts/run_final_baseline_replay.py --frozen-config data/output/b9_five_scene_validation/04_frozen_five_scene_config_1024.json --output-grid data/output/b9_five_scene_validation/mosaic_runs_1024/protocol/canonical_output_grid.json --main-global-run-dir data/output/b9_five_scene_validation/global_runs_1024/efficient_loftr/translation_l2 --traditional-global-run-dir data/output/b9_five_scene_validation/global_runs_1024/sift/mst --replay-root data/output/b9_five_scene_validation/final_replay_v2_hardened --output-root data/output/b9_five_scene_validation/final_results_v2 --main-pairwise-summary data/output/b9_five_scene_validation/matcher_runs_1024_globalready/efficient_loftr/pairwise_summary.csv --traditional-pairwise-summary data/output/b9_five_scene_validation/matcher_runs_1024_globalready/sift/pairwise_summary.csv
```

Source config SHA-256: `b9181afcc972b29716d309f7aa3e82d3fbdba57fd5d466e2c0aa04d25060c9ba`; canonical grid SHA-256: `fb27e5cafbb05cfc48103abc5500e7640eee189c40eebcf7509fb171cbf7c0cd`.

| Baseline | Cache status | Final mosaic SHA-256 |
|---|---|---|
| EfficientLoFTR + Translation-L2 + BAGRN | REGENERATED | `3a5ee630ceaeb668b4b6e3885ae05bc86a36f14b4d024788d1bfd30b5592b0ee` |
| SIFT + MST + BAGRN | REGENERATED | `0725866b3b22842207f8c4ff28d2d585461c3a194893299c4ce32c477a509542` |

CLI delivery verification (including source/global/grid hashes, scene-cache provenance, exact support, all package artifact hashes, replay/final mosaic equality, and final mosaics) returned `status=PASS`. The final package's paper tables report frozen-geometry pairwise aggregates and RAW-versus-BAGRN MAMD/MSDD/RDD; the complete snapshots are in each `02_radiometric/radiometric_metrics.json`. Task10D values remain frozen-protocol descriptive metrics; this replay does not resolve the separately recorded paper CD/GL formula ambiguity.
