# B9 unified 1024 matcher protocol audit

## Repository state

- Repository: `D:\科研\地质一号\文献\BAGRN_VOLRN_Reproduce_two_image_flat`
- Branch: `exp/2026-09-24-b9-five-scene-validation`
- Start HEAD: `7c99721bd95a385cebe0eab864c27fead4750a50`
- Existing untracked scratch directories were preserved.
- No real B9 matcher or GPU experiment was run for this audit.

## Scale propagation owners

| Owner | Location | Responsibility |
|---|---|---|
| `CONFIG_OWNER` | `data/output/b9_five_scene_validation/04_frozen_five_scene_config.json` | Frozen `registration.match_max_side` value; currently `1600`. |
| `CLI_OWNER` | `scripts/run_b9_registration.py`, `src/multiscene_sift/b9_runner.py` | Loads the frozen config and forwards the registered scale to the shared runner. |
| `PAIR_VIEW_OWNER` | `src/multiscene_sift/pairwise.py::register_pair` | Calls `build_match_view(pair, max_side=match_max_side)`. |
| `MATCHER_DISPATCH_OWNER` | `src/multiscene_sift/pairwise.py::_run_matcher` | Dispatches SIFT, LoFTR, EfficientLoFTR, and LightGlue+DISK with the same `MatchView`. |
| `RESIZE_OWNER` | `src/registration_benchmark/common_grid.py::build_match_view` | Crops the pair-common-grid overlap and applies one shared aspect-preserving resize. |
| `INVERSE_COORDINATE_OWNER` | `src/registration_benchmark/models.py::MatchView.to_common_grid` and matcher adapters | Converts matcher-frame coordinates back to `pair_common_grid` before shared geometry. |

The B9 path therefore has one shared scale owner. The four matchers enter the same `PairView` and then the same `fit_affine_ransac` path; no matcher-specific scale override was found.

## Frozen protocol values

- Frozen scene manifest indices: `[2, 3, 5, 8, 10]`.
- Frozen graph candidates: `(2,3)`, `(2,5)`, `(2,8)`, `(2,10)`, `(3,5)`, `(3,8)`, `(3,10)`, `(5,8)`, `(5,10)`, `(8,10)`.
- Existing configuration: `04_frozen_five_scene_config.json`.
- Existing `registration.match_max_side`: `1600`.
- Coordinate frame: `pair_common_grid`.
- RANSAC configuration is unchanged: AffineTransform, residual threshold 2.0 px, 5000 trials, seed 0, minimum 20 inliers, minimum inlier ratio 0.3.

## Task 7C memory evidence frozen for Task 7D

- At 1600, worst single FP32 `sim_matrix`: approximately `5861.8 MiB`.
- At 1024, worst single FP32 `sim_matrix`: approximately `961.0 MiB`.
- Rough four-sim FP32 estimate at 1024: approximately `3844.0 MiB`.
- Available GPU class: RTX 4060 Laptop GPU, approximately 8 GB total VRAM.

The 1600 worst-pair evidence from Task 7C was tied by manifest pairs `(2,3)` and `(2,5)` at the model-size level. For deterministic continuation, the gate uses `(2,3)`, the pair used as the Task 7C worst-pair reference. The 1024 shape-only audit confirms the same tie:

| Manifest pair | Experiment/local pair | 1024 model input | Coarse grid | Tokens | One FP32 sim |
|---|---:|---:|---:|---:|---:|
| `(2,3)` | `(0,1)` | `1024x992` | `128x124` | `15872` | `961.0 MiB` |
| `(2,5)` | `(0,2)` | `992x1024` | `124x128` | `15872` | `961.0 MiB` |
| `(2,8)` | `(0,3)` | `704x1024` | `88x128` | `11264` | `484.0 MiB` |
| `(2,10)` | `(0,4)` | `960x1024` | `120x128` | `15360` | `900.0 MiB` |
| `(3,5)` | `(1,2)` | `1024x864` | `128x108` | `13824` | `729.0 MiB` |
| `(3,8)` | `(1,3)` | `672x1024` | `84x128` | `10752` | `441.0 MiB` |
| `(3,10)` | `(1,4)` | `928x1024` | `116x128` | `14848` | `841.0 MiB` |
| `(5,8)` | `(2,3)` | `640x1024` | `80x128` | `10240` | `400.0 MiB` |
| `(5,10)` | `(2,4)` | `960x1024` | `120x128` | `15360` | `900.0 MiB` |
| `(8,10)` | `(3,4)` | `896x1024` | `112x128` | `14336` | `784.0 MiB` |

Model dimensions include EfficientLoFTR's existing floor-to-32 handling. These are geometry-only estimates; they are not a substitute for the user-run one-pair GPU gate.

## Scope guard

This task made no production-code, RANSAC, threshold, matcher, mosaic, BAGRN, or VOLRN changes. The old 1600 configuration and historical outputs remain untouched.
