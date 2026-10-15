# Memory-Safe Shared Affine RANSAC Root-Cause Audit

## Scope and repository state

- Branch: `exp/2026-09-24-b9-five-scene-validation`
- HEAD at audit start: `d065eec504b01c7a541d7418aa61a36c49be9433`
- The worktree has only pre-existing untracked temporary directories; no
  tracked files were modified during this audit.
- The formal B9 EfficientLoFTR experiment was not rerun.

## Shared geometry owner and callers

The shared modern-matcher geometry owner is:

```text
src/registration_benchmark/geometry.py::fit_affine_ransac
```

Its tests are primarily:

```text
tests/registration_benchmark/test_geometry.py
tests/multiscene_sift/test_pairwise.py
```

Current callers found by repository search are:

```text
src/multiscene_sift/pairwise.py::register_pair
src/registration_benchmark/runner.py
```

`src/coregistration.py::fit_affine_ransac` is a separate legacy backend. It
was identified by the search but is not the shared modern-matcher path and
was not modified.

## Frozen geometry contract

The current shared entry point maps `tgt_xy` to `ref_xy`:

```text
src = matches.tgt_xy
dst = matches.ref_xy
```

The current frozen policy is:

```text
model class: skimage.transform.AffineTransform
matrix: [[a,b,tx],[c,d,ty],[0,0,1]]
min_samples: 3
residual_threshold: caller value, normally 2.0 px
max_trials: caller value, normally 5000
stop_probability: scikit-image default 1
random seed: 0, passed as rng on the installed API
inlier rule: Euclidean model residual < residual_threshold
returned mask: bool array aligned with all raw correspondences
post-RANSAC refit: scikit-image estimates a final model from all inliers
acceptance: MIN_INLIERS=20 and MIN_INLIER_RATIO=0.30, plus existing geometry checks
```

No threshold, trial count, seed, coordinate convention, or acceptance rule
was changed in this audit.

## Installed scikit-image call chain

Installed versions:

```text
numpy: 2.4.6
scikit-image: 0.26.0
```

The installed `skimage.measure.ransac` API calls:

```text
model_class.from_estimate(*samples)
model.residuals(*data)
...
model_class.from_estimate(*data_inliers)  # final all-inlier refit
```

The installed `AffineTransform._estimate` constructs a matrix with a row
count proportional to the number of correspondences and calls:

```text
np.linalg.svd(A)
```

The observed failure therefore occurs in the final all-inlier
`AffineTransform.from_estimate` path, after RANSAC has classified the dense
correspondences. The failure stack is:

```text
fit_affine_ransac
  -> skimage.measure.ransac
    -> AffineTransform.from_estimate
      -> AffineTransform._estimate
        -> numpy.linalg.svd
          -> MemoryError / init_gesdd failed init
```

## Existing EfficientLoFTR evidence

The existing output at
`data/output/b9_five_scene_validation/matcher_runs/efficient_loftr` shows:

```text
raw_matches on successful dense edge: 20989
inliers: 20434
inlier_ratio: 0.974
failed edges: 9/10
successful edge: (2,3)
accepted_edges: 1
connected_components: [[0], [1], [2,3], [4]]
network: NETWORK_DISCONNECTED
```

The successful edge spent approximately 1230.55 seconds in geometry and
produced finite residual metrics. This shows the EfficientLoFTR adapter can
load, infer, and emit usable common-grid correspondences; the shared CPU
geometry backend is the blocking component.

## Independent CUDA observation

The current registration environment reports:

```text
torch version: 2.14.0+cpu
torch CUDA build: None
torch.cuda.is_available(): False
torch.cuda.device_count(): 0
```

`nvidia-smi` does see an RTX 4060 and a compatible driver, so the machine has
a GPU, but this virtual environment contains a CPU-only PyTorch build. This
is an independent environment issue and is explicitly out of scope for this
task.

## Root-cause conclusion

```text
Problem A (this task): dense EfficientLoFTR correspondences enter the shared
CPU AffineTransform SVD refit and can exhaust memory. The planned fix is a
shared memory-safe affine estimator while preserving the existing skimage
RANSAC sampling, residual threshold, inlier mask, and acceptance semantics.

Problem B (not this task): the environment has CPU-only PyTorch despite a
visible NVIDIA GPU. CUDA availability is recorded only; it is not repaired
or changed here.
```
