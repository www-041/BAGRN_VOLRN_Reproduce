# Memory-Safe Shared Affine RANSAC Verification

## Implementation

The only production-code change in this task is in the shared modern geometry
owner, `src/registration_benchmark/geometry.py`:

```text
all matchers
  -> shared fit_affine_ransac()
  -> same skimage.measure.ransac() sampling policy
  -> _MemorySafeAffineRansacModel
  -> numpy.linalg.lstsq() affine fit
  -> same Euclidean residual and inlier mask semantics
```

The estimator solves a three-column design matrix with
`numpy.linalg.lstsq(..., rcond=None)` and returns the existing homogeneous
`AffineTransform` interface after RANSAC. It does not construct the large
full-SVD matrix used by the installed `AffineTransform._estimate`.

```text
RANSAC threshold changes: NONE
min_samples changes: NONE
max_trials changes: NONE
stop probability changes: NONE
random seed changes: NONE
MIN_INLIERS changes: NONE
MIN_INLIER_RATIO changes: NONE
coverage/acceptance changes: NONE
matcher-specific geometry branch: NONE
subsampling/candidate cap: NONE
fallback gate: SKIP_FALLBACK
```

## Compatibility and direction

```text
exact affine: PASS
noisy affine: PASS
normal-scale legacy estimator comparison (N=200): PASS
fixed-seed RANSAC comparison (N=1000, 80% inliers/20% outliers): PASS
src -> dst direction with translation/scale/rotation/shear: PASS
```

The fixed-seed comparison accepted the same quality status, kept the inlier
count within 1%, kept RMSE and P95 differences within 0.05 px, and kept
control-point predictions within 0.05 px.

## Dense stress

```text
20k correspondences: PASS, 1 test passed
50k correspondences: PASS, 1 test passed
MemoryError: not observed
full-size numpy.linalg.svd in shared estimator: not used
```

Both stress cases used approximately 0.97 inliers, fixed seeds, the frozen
2.0 px residual threshold, and max_trials=5000. All correspondences were
retained; no cap or subsampling was introduced.

## Regression tests

Fresh focused results during this task include:

```text
memory-safe affine + legacy geometry + pairwise: 33 passed, 31 warnings
multiscene affine/global selector: 38 passed, 13 warnings
LoFTR/LightGlue/matcher contract/memory group: 26 passed, 1 warning
EfficientLoFTR adapter: 9 passed, 1 skipped, 1 warning
local affine diagnostics: 20 passed
```

The exact broad selector from the plan,
`pytest tests -k "global_consistency or registration_benchmark or affine or ransac" -q`,
was attempted. In this Windows environment it terminated after earlier tests
without a pytest failure traceback or summary. The same selected areas were
then split by test family; the listed suites completed with no new test
failure. The skipped EfficientLoFTR test is the existing official-import skip
when `EFFICIENT_LOFTR_REPO` is not configured for that test process.

## CUDA observation

```text
torch version: 2.14.0+cpu
torch CUDA build: None
torch.cuda.is_available(): False
torch.cuda.device_count(): 0
nvidia-smi: RTX 4060 visible, driver 560.94, CUDA 12.6 reported by driver
CUDA repaired in this task: NO
```

The CPU-only PyTorch environment is independent of the shared CPU geometry
MemoryError and remains out of scope.

## Scope confirmation

```text
No matcher inference changed
No matcher confidence changed
No pair_common_grid conversion changed
No RANSAC threshold changed
No acceptance threshold changed
No MST changed
No Equal-L2 Translation changed
No global metric changed
No mosaic changed
No BAGRN/VOLRN changed
No formal B9 matcher rerun
No model download
No dependency reinstall
No push
```
