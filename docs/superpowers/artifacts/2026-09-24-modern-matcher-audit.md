# Modern Matcher Literature and Engineering Audit

Date: 2026-09-24

Scope: Task 1 only. This document records the candidate audit and freezes one
fourth benchmark method. No matcher package, checkpoint, or production code was
added in this task.

## Existing benchmark core

| Method | Paper/year | Matching regime | Official implementation and weights | License | Engineering notes |
|---|---|---|---|---|---|
| SIFT | Lowe, 2004; OpenCV implementation | Sparse, detector/descriptor based | OpenCV `SIFT_create`; no learned checkpoint | OpenCV 4.5+ Apache-2.0 | CPU baseline; native pixel keypoints and descriptor distance/response, no CUDA requirement; image size is not model-constrained; runtime and memory are hardware/data dependent. |
| LoFTR | CVPR 2021 / TPAMI 2022 | Semi-dense, detector-free | `zju3dv/LoFTR`; indoor/outdoor DS/OT checkpoints, approximately 45 MB each | Apache-2.0 repository | PyTorch/Kornia/einops/yacs; official inference uses CUDA; outputs matched keypoint coordinates and matching confidence; grayscale pair input is supported by the reference example; no fixed high-resolution runtime or peak-memory figure is published in the README. |
| EfficientLoFTR | CVPR 2024 | Semi-dense, detector-free, sparse-like speed | `zju3dv/EfficientLoFTR`; `eloftr_outdoor.ckpt`, full/opt configurations | Apache-2.0 repository | Official environment pins PyTorch 2.0.0 + CUDA 11.8; full/opt and fp32/mixed/fp16 modes; input is resized to dimensions divisible by 32 in the reference path; outputs `mkpts0_f`, `mkpts1_f`, `mconf`; outdoor MegaDepth pretraining creates a remote-sensing domain-gap risk; no fixed high-resolution runtime or peak-memory figure is published in the README. |

SIFT remains the classical baseline. LoFTR and EfficientLoFTR remain benchmark
baselines, not preselected winners.

## Audited modern candidates

| Method | Paper/year | Sparse/dense and detector status | Official code / weights / license | Python/PyTorch/CUDA and input behavior | Runtime / memory class | Confidence and coordinates | B14 suitability and maintenance assessment |
|---|---|---|---|---|---|---|---|
| LightGlue + DISK | ICCV 2023 | Sparse local-feature matcher; detector/descriptor is supplied separately | `cvg/LightGlue`; official LightGlue and DISK weights; Apache-2.0 for code/weights and DISK | PyTorch package; official examples use CUDA tensors; accepts keypoints/descriptors and returns match indices; grayscale B14 can be replicated to 3 channels for DISK | Lightweight/adaptive matcher; official README describes adaptive width/depth pruning, but gives no fixed high-resolution runtime or peak-memory number; expected lower memory than dense matchers | Returns matched indices and matcher scores/probabilities from the local-feature pipeline; coordinates remain the extractor's native pixel coordinates and can be mapped into the common grid | Strong engineering fit: clean pairwise interface, reproducible weights, different from detector-free EfficientLoFTR, and avoids the restrictive SuperPoint dependency by freezing DISK. Selected as fourth comparator. |
| XFeat | CVPR 2024 | Sparse and semi-dense learned local features; detector/descriptor based | `verlab/accelerated_features`; official `weights/xfeat.pt`; Apache-2.0 repository | Minimal PyTorch dependency; official README reports CPU sparse inference and optional OpenCV; supports grayscale array conversion in the implementation path, but the benchmark adapter must explicitly define the channel conversion | Lightweight; official README reports real-time CPU VGA inference, over 150 FPS single-batch VGA GPU inference and about 1,400 FPS extraction on RTX 4090; high-resolution five-scene cost must still be measured | Produces keypoints/descriptors/reliability and match coordinates; confidence semantics are reliability/cosine/refinement scores rather than a calibrated probability | Attractive low-resource comparator, but its detector/descriptor and semi-dense modes require more adapter decisions than LightGlue for a fair shared matcher contract. Not selected for the frozen fourth slot. |
| RoMa | CVPR 2024; RoMa v2 preprint 2025 | Dense, detector-free correspondence/warp estimation | `Parskatt/RoMa` and `Parskatt/RoMaV2`; pretrained outdoor model; RoMa code MIT except DINOv2 component Apache-2.0 | PyTorch; official package tested on Linux Python 3.12; CUDA/fused local-correlation option is available; default path uses 560 initial resolution and 864 upsampled resolution | Dense model with substantially higher GPU/memory risk than sparse matchers; exact five-scene high-resolution runtime/memory is not fixed by the project | Produces normalized warp in `[-1,1]` plus certainty; must convert through `to_pixel_coordinates`; certainty is matchability, not directly comparable to LightGlue confidence | Scientifically relevant, but dense warp semantics and resolution/memory behavior make a fair first benchmark adapter more invasive. Not selected for Task 1. |
| MASt3R | ECCV 2024; MASt3R-SfM additions through 2025 | Dense/3D pointmap correspondence, detector-free | `naver/mast3r`; official checkpoints and Hugging Face loading | Python 3.11, PyTorch/CUDA 12.1 example, recursive submodules and optional compiled CUDA kernels; fixed model resolutions around 512-wide; outputs 3D pointmaps/descriptors and reciprocal matches rather than a simple 2D match set | Heavy ViT-L/ViT-B class model; GPU-oriented and materially higher memory/installation complexity | Produces 3D points, descriptors and reciprocal matches; confidence/coordinate conversion is not equivalent to the common 2D pair contract | Not selected: CC BY-NC-SA 4.0 and training-data restrictions, plus 3D semantics would confound the first two-view geometry benchmark. |

## 2025-2026 maturity screen

| Candidate | Evidence and practical issue | Decision |
|---|---|---|
| UFM, NeurIPS 2025 | Official code and checkpoints exist; unified dense flow/wide-baseline matching, 0.3–1B parameter variants and published RTX 5090 timings. Code is BSD-3-Clause, but checkpoints inherit non-commercial training-data restrictions; output is dense flow/covisibility rather than a sparse pair match set. | Defer. Revisit only after the four-method benchmark and license review. |
| SwinMatcher, TGRS 2025 | Official remote-sensing cross-modal repository and weights are available through Google Drive/Baidu. The repository exposes an Apache-2.0 code label, but the small maintenance footprint, external weight hosting, and incomplete published runtime/memory/confidence contract make reproducibility uncertain. | Defer. Do not silently replace the frozen fourth candidate. |
| RoMa v2, 2025 preprint / 2025 release line | Official repository/releases and new weights exist, but the dense CUDA/DINOv3-oriented pipeline is heavier and less directly aligned with the common sparse 2D MatchSet contract. | Defer. |

## Frozen fourth comparator

The fourth benchmark method is:

```text
LightGlue + DISK
```

Reasoning, in the required priority order:

1. Official inference code and pretrained weights are available.
2. Code and selected DISK dependency have a permissive Apache-2.0 path.
3. It fits the existing pairwise correspondence problem and has a clear
   keypoints/descriptors-to-match-indices interface.
4. Its sparse local-feature architecture is materially different from
   detector-free LoFTR/EfficientLoFTR while remaining compatible with the same
   downstream RANSAC and network evaluator.
5. It avoids selecting a heavier dense/3D model before the five-scene gate.

The benchmark must not use SuperPoint for this slot without a separate license
review, because the official LightGlue README calls out a restrictive
SuperPoint license. The planned adapter will explicitly convert the B14 image
to the extractor's accepted channel format and record that conversion.

## Frozen fair-comparison requirements for later tasks

Every method must eventually use the same:

- B14 image pair and pair-common-grid coordinate frame;
- coordinate back-conversion and crop-origin audit;
- RANSAC affine implementation, threshold, max trials, and acceptance rules;
- graph connectivity, MST, per-edge residual, cycle, and Equal-L2 Translation
  global evaluation;
- runtime and peak-GPU-memory measurement protocol.

Matcher-specific confidence values must be recorded for diagnostics only; they
must not be compared as if they were calibrated probabilities across methods.
Disconnected graphs are failures, not reasons to relax thresholds.

## Sources

- [OpenCV license](https://opencv.org/license/)
- [LoFTR official repository](https://github.com/zju3dv/LoFTR)
- [EfficientLoFTR official repository](https://github.com/zju3dv/EfficientLoFTR)
- [LightGlue official repository](https://github.com/cvg/LightGlue)
- [XFeat official repository](https://github.com/verlab/accelerated_features)
- [RoMa official repository](https://github.com/Parskatt/RoMa)
- [RoMa v2 official repository](https://github.com/Parskatt/RoMaV2)
- [MASt3R official repository](https://github.com/naver/mast3r)
- [UFM official repository](https://github.com/UniFlowMatch/UFM)
- [SwinMatcher official repository](https://github.com/LotrL/SwinMatcher)

