# Task11 Final End-to-End Baseline Replay v2

Date: 2026-09-27

## Frozen inputs

- Selection: B9 manifest indices `[2, 3, 5, 8, 10]`.
- Canonical grid: EPSG:32650, 14 m, 5116 x 6134 pixels.
- Main geometry: persisted `efficient_loftr/translation_l2` transforms.
- Traditional geometry: persisted `sift/mst` transforms.
- No matcher, RANSAC, MST, Translation-L2, or VOLRN process was run.

## Replay result

Both historical radiometric directories lacked per-scene BAGRN rasters, so the replay performed exactly two BAGRN-only runs and wrote separate replay caches. Each cache contains five canonical-grid normalised scenes, BAGRN parameters, provenance, Task10D metrics, and a weighted-feather mosaic.

| Baseline | Cache status | Final mosaic SHA-256 |
|---|---|---|
| EfficientLoFTR + Translation-L2 + BAGRN | REGENERATED | `6d2eabba6ea2563a58f438c8c32f6bbb737e85e195706c31c1d8aa8fab9cbb07` |
| SIFT + MST + BAGRN | REGENERATED | `bece287cc3031256771fecc6114d6f49ed80576b84042c9b4a0a6d0f4965e218` |

The final package is `data/output/b9_five_scene_validation/final_results/`. CLI checksum verification returned `status=PASS`. Task10D values in the package remain frozen-protocol descriptive metrics; this replay does not resolve the separately recorded paper CD/GL formula ambiguity.
