# TPS-SUPPORT-C1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不改变师兄 GFTT/KLT/TPS 拟合参数和现有 production `klt_tps` backend 默认行为的前提下，实现一个严格的 N=2 `TPS-SUPPORT-C1` 诊断实验：同一次 raw TPS fit 同时生成 KLT median translation baseline 与 convex-hull inside-only supported TPS treatment，并在固定 B14 HOLDOUT 上比较其几何安全性与配准精度。

**Architecture:** 算法数学函数继续放在 `src/klt_tps_registration.py`；新增 `src/klt_tps_support_c1.py` 专门承载 C1 protocol、array fingerprint、field integrity 与固定 HOLDOUT 比较，不做 I/O；`MultibandPipeline` 新增独立的 diagnostic-only `run_klt_tps_support_c1_n2()`，不修改 production `register_scenes()` 的默认 KLT/TPS 路径；`scripts/diagnose_registration_pair.py` 新增显式 CLI opt-in、artifact writer 与独立 exit semantics。

**Tech Stack:** Python 3.x, NumPy, SciPy (`RBFInterpolator`, `distance_transform_edt`), OpenCV, Rasterio, Matplotlib, pytest, existing DZ01V registration pipeline.

**Spec:** `docs/superpowers/specs/2026-09-13-tps-support-c1-design.md`

## Global Constraints

- 工作分支固定为 `feat/dz01v-klt-tps-n2-integration-20260912`。
- 当前设计/spec commit 为 `492946f1ac6adc6d80a53f023627a8140009d2e6`；执行前必须 `git pull`，并验证该 commit 是当前 HEAD 的祖先。不要 reset/rebase/force-push。
- Phase-I 仍只支持 **N=2**；禁止实现 N=4/N=6、多父节点 TPS、field composition。
- registration band 固定 **B12**；independent validation band 固定 **B14**。
- C1 固定复用 `configs/dz01_model_c1_holdout_manifest.json` 中现有 7 个 `384 x 384` HOLDOUT windows；不得在 C1 中重新 reserve final HOLDOUT。
- support taper 固定 **64 px**；本阶段禁止参数搜索，也禁止从 final HOLDOUT 选择 taper width。
- hull 外 continuation 固定为 TRAIN-only KLT displacement 的 component-wise median translation `g=(median(dx), median(dy))`，不是 0、不是 affine、不是重新拟合。
- raw TPS 只拟合 **一次**；translation 和 supported treatment 必须从同一 raw field 派生。
- `klt_tps_window=9`、`klt_tps_pyramid_level=3`、`klt_tps_max_corners=4000`、`klt_tps_min_corner_distance=5.0`、`klt_tps_fb_threshold=0.5`、`klt_tps_min_points=30`、`klt_tps_neighbors=80`、`klt_tps_smoothing=3.0`、`klt_tps_field_step=4`、`klt_tps_max_shift=50.0`、`klt_tps_threads=4` 不得修改。
- Jacobian safety gate 不得放宽；supported field 必须 `fold_pixels == 0` 且 `max_displacement_pixels <= 50.0` 才允许 Stage 2 warp。
- Stage 1 translation-only 与 Stage 2 supported TPS 都必须从 **ORIGINAL moving multiband array** 各自一次 warp；禁止串行 warp。
- final HOLDOUT 只做最终评价；不得影响 KLT controls、TPS fit、translation、support mask、taper 或任何超参数。
- 禁止修改 BAGRN、VOLRN、mosaic、legacy backend、registration quality thresholds、`required_quality`。
- Codex/agent 禁止运行真实 DZ01V N=2/N=4/N=6；真实 N=2 仅由用户在所有 synthetic/unit/full regression 通过后手工运行。
- 不允许把 full raw/support flow、Jacobian、weight 或 fold mask 序列化进 JSON；大数组只允许在当前 process 中作为 private artifact arrays 使用。

---

## File Structure

**Create**
- `src/klt_tps_support_c1.py` — C1 protocol validation、array fingerprints、field-level causal integrity、fixed-HOLDOUT comparison。
- `tests/test_klt_tps_support_c1.py` — C1 pure protocol/field/comparison tests。

**Modify**
- `src/klt_tps_registration.py` — 新增 median continuation、constant field、inside-hull taper、supported-flow composition 纯函数。
- `src/multiband_pipeline.py` — 新增 diagnostic-only `run_klt_tps_support_c1_n2()`；production `register_scenes()` 默认路径保持不变。
- `scripts/diagnose_registration_pair.py` — 新 CLI flag、严格 manifest preflight、C1 artifacts、payload、logging、exit semantics。
- `tests/test_klt_tps_registration.py` — support 数学函数 TDD。
- `tests/test_klt_tps_backend_n2.py` — pipeline C1 orchestration、one-fit、one-pass、same-HOLDOUT、unsafe-no-warp tests。
- `tests/test_diagnostics_regressions.py` — CLI fail-fast、payload/CSV/serialization 回归。
- `docs/KLT_TPS_INTEGRATION.md` — C1 使用说明、解释边界、用户命令。

**Read-only / reuse; do not modify unless a failing test proves an actual bug**
- `configs/dz01_klt_tps_n2_b12.yaml`
- `configs/dz01_model_c1_holdout_manifest.json`
- `src/registration_band_ab.py`
- `src/registration_model_c1.py`

---

### Task 0: Preflight and Branch Safety

**Files:** none.

**Interfaces:**
- Consumes: current branch history and committed spec.
- Produces: recorded starting HEAD in Codex final report; no code changes.

- [ ] **Step 1: Verify branch and clean working tree**

```powershell
git switch feat/dz01v-klt-tps-n2-integration-20260912
git pull
git status --short
git rev-parse HEAD
```

Expected: branch is correct and `git status --short` is empty.

- [ ] **Step 2: Verify the approved spec is an ancestor of current HEAD**

```powershell
git merge-base --is-ancestor 492946f1ac6adc6d80a53f023627a8140009d2e6 HEAD
if ($LASTEXITCODE -ne 0) { throw "Approved TPS-SUPPORT-C1 spec is not an ancestor of HEAD" }
```

- [ ] **Step 3: Verify the frozen HOLDOUT manifest really contains the expected pair**

```powershell
$j = Get-Content .\configs\dz01_model_c1_holdout_manifest.json -Raw | ConvertFrom-Json
$j.validation_band
$j.scene_ids
$j.pairs.'0-1'.selected_block_size
$j.pairs.'0-1'.reserved_windows.Count
```

Expected exactly:
- validation band `B14`
- scenes `scene_20251114`, `scene_20251120`
- block size `384`
- reserved window count `7`

Do not copy or regenerate this manifest.

---

### Task 1: Add the Frozen TPS-SUPPORT-C1 Protocol Contract

**Files:**
- Create: `src/klt_tps_support_c1.py`
- Create: `tests/test_klt_tps_support_c1.py`

**Interfaces:**
- Consumes: config object/dict, loaded HOLDOUT manifest.
- Produces:
  - `TPS_SUPPORT_C1_TAPER_PIXELS: int = 64`
  - `validate_tps_support_c1_protocol(config, holdout_manifest, *, validation_band="B14") -> dict[str, Any]`
  - `array_sha256(array: np.ndarray) -> str`
  - `validation_block_keys(validation: dict) -> list[tuple[int, int, int, int, int]]`

- [ ] **Step 1: Write protocol tests first**

Add tests with these exact names:

```python
def test_tps_support_c1_protocol_accepts_frozen_b14_manifest(): ...
def test_tps_support_c1_protocol_requires_klt_tps_backend(): ...
def test_tps_support_c1_protocol_requires_b12_registration_and_b14_validation(): ...
def test_tps_support_c1_protocol_rejects_empty_pair_windows(): ...
def test_tps_support_c1_protocol_requires_exactly_seven_384_windows(): ...
def test_tps_support_c1_protocol_rejects_scene_id_mismatch(): ...
def test_tps_support_c1_array_sha256_is_shape_dtype_and_content_sensitive(): ...
```

For the positive test, load the real committed JSON fixture `configs/dz01_model_c1_holdout_manifest.json`; do not open imagery.

- [ ] **Step 2: Run the new tests and verify red state**

```powershell
pytest tests/test_klt_tps_support_c1.py -q
```

Expected: FAIL because `src.klt_tps_support_c1` does not exist.

- [ ] **Step 3: Implement the frozen protocol**

Use this exact contract:

```python
TPS_SUPPORT_C1_TAPER_PIXELS = 64


def validate_tps_support_c1_protocol(
    config,
    holdout_manifest: dict,
    *,
    validation_band: str = "B14",
) -> dict[str, Any]:
    """Validate TPS-SUPPORT-C1 without opening imagery."""
```

Validation rules:

```text
registration_band == "B12"
registration_params.registration_backend == "klt_tps"
selected_bands == ["B12", "B14"]
validation_band == "B14"
manifest.validation_band == "B14"
manifest.source_registration_band == "B14"
manifest.scene_ids == first two config scene IDs
manifest.pairs["0-1"] exists
selected_block_size == 384
len(reserved_windows) == 7
each window has height == width == 384
```

Do **not** compare `base_registration_params_sha256` from this old manifest to current KLT/TPS params; that hash belongs to the earlier legacy/MODEL-C1 protocol and is not a valid KLT/TPS parity hash.

Return:

```python
{
    "valid": not errors,
    "errors": errors,
    "registration_band": registration_band,
    "validation_band": validation_band,
    "scene_ids": manifest_ids,
    "reserved_count": len(windows),
    "selected_block_size": pair.get("selected_block_size"),
    "taper_pixels": TPS_SUPPORT_C1_TAPER_PIXELS,
}
```

- [ ] **Step 4: Implement deterministic array fingerprint**

Use dtype + shape + raw contiguous bytes, not `repr()`:

```python
def array_sha256(array: np.ndarray) -> str:
    arr = np.ascontiguousarray(np.asarray(array))
    digest = hashlib.sha256()
    digest.update(str(arr.dtype).encode("utf-8"))
    digest.update(np.asarray(arr.shape, dtype=np.int64).tobytes())
    digest.update(arr.tobytes())
    return digest.hexdigest()
```

- [ ] **Step 5: Implement fixed validation block key extraction**

Return sorted keys `(idx_i, idx_j, validation_row, validation_col, block_size)` using the existing final-validation schema. Do not use accepted/rejected status to decide whether a key exists.

- [ ] **Step 6: Run tests green**

```powershell
pytest tests/test_klt_tps_support_c1.py -q
```

- [ ] **Step 7: Commit**

```powershell
git add src/klt_tps_support_c1.py tests/test_klt_tps_support_c1.py
git commit -m "feat: add frozen tps support c1 protocol"
```

**Acceptance:** protocol accepts the committed 7-window manifest and rejects empty/wrong manifests before imagery is loaded.

**Non-goals:** no field generation, no warp, no CLI, no real data.

---

### Task 2: Implement Pure Translation Continuation and Inside-Hull Taper Math

**Files:**
- Modify: `src/klt_tps_registration.py`
- Modify: `tests/test_klt_tps_registration.py`

**Interfaces:**
- Consumes: TRAIN-only `displacement_xy`, moving-native control points, raw TPS flow.
- Produces:
  - `compute_klt_translation_continuation(displacement_xy) -> np.ndarray`
  - `build_translation_flow(translation_xy, shape, *, dtype=np.float32) -> np.ndarray`
  - `build_inside_hull_taper_weight(control_points_xy, shape, taper_pixels) -> dict[str, Any]`
  - `compose_supported_tps_flow(raw_flow, translation_xy, support_weight) -> np.ndarray`

- [ ] **Step 1: Add failing math tests**

Add these exact tests:

```python
def test_klt_translation_continuation_uses_componentwise_median(): ...
def test_klt_translation_continuation_keeps_extreme_outlier_out_of_median(): ...
def test_inside_hull_taper_is_zero_outside_and_on_boundary(): ...
def test_inside_hull_taper_reaches_one_deep_inside(): ...
def test_inside_hull_taper_is_bounded_and_monotone_inward(): ...
def test_supported_tps_equals_translation_outside_hull(): ...
def test_supported_tps_equals_raw_flow_where_weight_is_one(): ...
def test_supported_tps_constant_translation_is_identity_counterfactual(): ...
def test_supported_tps_rejects_shape_mismatch(): ...
```

- [ ] **Step 2: Verify red state**

```powershell
pytest tests/test_klt_tps_registration.py -k "translation_continuation or inside_hull_taper or supported_tps" -q
```

- [ ] **Step 3: Implement component-wise median continuation**

Contract:

```python
def compute_klt_translation_continuation(displacement_xy: np.ndarray) -> np.ndarray:
    displacement = np.asarray(displacement_xy, dtype=float)
    if displacement.ndim != 2 or displacement.shape[1] != 2 or len(displacement) == 0:
        raise ValueError("KLT displacement must have shape (N, 2) and be non-empty")
    if not np.isfinite(displacement).all():
        raise ValueError("KLT displacement must be finite")
    return np.median(displacement, axis=0)
```

Do not clip or filter controls.

- [ ] **Step 4: Implement constant translation field**

`build_translation_flow([dx,dy], (h,w))` must produce `(h,w,2)` with identical vectors at every pixel. Reject non-finite translation or non-positive shape.

- [ ] **Step 5: Implement inside-only hull taper**

Use the existing `build_control_hull_mask(...)` to rasterize the same convex hull used by TPS-FOLD-D1. Then use `scipy.ndimage.distance_transform_edt` on the hull mask.

Boundary semantics are fixed as:

```python
distance_inside = np.maximum(distance_transform_edt(hull_mask) - 1.0, 0.0)
u = np.clip(distance_inside / float(taper_pixels), 0.0, 1.0)
weight = 3.0 * u ** 2 - 2.0 * u ** 3
weight[~hull_mask] = 0.0
```

Return:

```python
{
    "weight": weight.astype(np.float32),
    "hull_mask": hull_mask,
    "distance_inside": distance_inside.astype(np.float32),
    "taper_pixels": int(taper_pixels),
    "deep_inside_mask": hull_mask & (distance_inside >= taper_pixels),
}
```

`taper_pixels <= 0` must raise `ValueError`.

- [ ] **Step 6: Implement supported composition**

Formula is exactly:

```python
supported = translation + weight[..., None] * (raw - translation)
```

No refit, no clamp, no extra smoothing. Preserve raw flow shape and return `float32`.

- [ ] **Step 7: Run full KLT/TPS unit file**

```powershell
pytest tests/test_klt_tps_registration.py -q
```

- [ ] **Step 8: Commit**

```powershell
git add src/klt_tps_registration.py tests/test_klt_tps_registration.py
git commit -m "feat: add supported tps continuation math"
```

**Acceptance:** hull outside is exact translation to tolerance, deep interior is raw TPS to tolerance, weight range is `[0,1]`, and raw flow input remains unchanged.

**Non-goals:** do not change `build_tps_dense_flow`, KLT, RBFInterpolator, safety gates, or production estimator logic.

---

### Task 3: Build the C1 Field Bundle and Integrity Gate

**Files:**
- Modify: `src/klt_tps_support_c1.py`
- Modify: `tests/test_klt_tps_support_c1.py`

**Interfaces:**
- Consumes: one raw TPS field, moving-native controls, control displacements, current max-shift gate.
- Produces:
  - `build_tps_support_c1_fields(...) -> dict[str, Any]`
  - `summarize_tps_support_field_integrity(...) -> dict[str, Any]`

- [ ] **Step 1: Add failing field-bundle tests**

```python
def test_tps_support_field_bundle_uses_one_raw_flow_without_refit(): ...
def test_tps_support_field_bundle_records_raw_translation_supported_geometry(): ...
def test_tps_support_field_integrity_requires_outside_translation_match(): ...
def test_tps_support_field_integrity_requires_deep_inside_raw_match(): ...
def test_tps_support_field_bundle_reports_supported_unsafe_without_warping(): ...
```

Use synthetic arrays only.

- [ ] **Step 2: Implement field bundle**

Exact signature:

```python
def build_tps_support_c1_fields(
    raw_flow: np.ndarray,
    control_points_xy: np.ndarray,
    displacement_xy: np.ndarray,
    *,
    max_shift: float,
    taper_pixels: int = TPS_SUPPORT_C1_TAPER_PIXELS,
) -> dict[str, Any]:
```

Required sequence:

```text
translation_xy = median(TRAIN KLT displacement)
translation_flow = constant g
support = inside-hull 64 px taper
supported_flow = g + w(raw-g)
raw_analysis = analyze_tps_dense_flow(raw_flow)          # diagnostic only
translation_analysis = analyze_tps_dense_flow(...)
supported_analysis = analyze_tps_dense_flow(...)
translation safety = inspect_tps_dense_flow(...)
supported safety = inspect_tps_dense_flow(...)
```

Catch safety `ValueError` only to record `*_geometry_safe=False` and rejection reason. Do not suppress invalid input errors from field construction.

- [ ] **Step 3: Compute causal integrity numerically**

Use tolerance `1e-6` for flow equality checks.

Record at least:

```python
{
    "raw_flow_sha256": array_sha256(raw_flow),
    "controls_sha256": array_sha256(control_points_xy),
    "displacements_sha256": array_sha256(displacement_xy),
    "weight_min": float(weight.min()),
    "weight_max": float(weight.max()),
    "outside_hull_nonzero_weight_pixels": int(...),
    "outside_max_abs_supported_minus_translation": float(...),
    "deep_inside_pixel_count": int(...),
    "deep_inside_max_abs_supported_minus_raw": float(...) if any else None,
    "support_formula_pass": bool(...),
}
```

`support_formula_pass` requires:
- `weight_min >= 0`
- `weight_max <= 1`
- zero nonzero support outside hull
- outside difference <= `1e-6`
- if deep-inside pixels exist, deep-inside difference <= `1e-6`

Do not require raw TPS itself to be safe.

- [ ] **Step 4: Keep full arrays private**

Return compact summaries plus:

```python
"_arrays": {
    "raw_flow": raw_flow,
    "translation_flow": translation_flow,
    "support_weight": support["weight"],
    "supported_flow": supported_flow,
    "supported_jacobian": supported_analysis["jacobian_determinant"],
    "supported_fold_mask": supported_analysis["fold_mask"],
}
```

These arrays are for pipeline/artifact use and must later be stripped before JSON serialization.

- [ ] **Step 5: Run tests and commit**

```powershell
pytest tests/test_klt_tps_support_c1.py -q
git add src/klt_tps_support_c1.py tests/test_klt_tps_support_c1.py
git commit -m "feat: build tps support c1 field counterfactuals"
```

**Acceptance:** one input raw field deterministically yields translation and supported fields; supported safety can fail without mutating gates.

---

### Task 4: Add Fixed-HOLDOUT Comparison Helpers

**Files:**
- Modify: `src/klt_tps_support_c1.py`
- Modify: `tests/test_klt_tps_support_c1.py`

**Interfaces:**
- Consumes: Stage 1 and Stage 2 validation dicts from `_validate_final_registration_arrays`.
- Produces:
  - `compare_tps_support_validations(translation_validation, supported_validation) -> dict[str, Any]`

- [ ] **Step 1: Add failing paired comparison tests**

```python
def test_compare_tps_support_validations_pairs_identical_fixed_windows(): ...
def test_compare_tps_support_validations_preserves_rejected_blocks(): ...
def test_compare_tps_support_validations_reports_translation_minus_supported_improvement(): ...
def test_compare_tps_support_validations_marks_holdout_mismatch_unavailable(): ...
```

- [ ] **Step 2: Implement comparison without selection bias**

Mirror the established MODEL-C1 idea but use translation/supported naming. Pair blocks by exact key:

```text
(idx_i, idx_j, validation_row, validation_col, block_size)
```

For every measurable common block record:

```python
{
    "key": [...],
    "translation_residual": ...,
    "supported_residual": ...,
    "translation_accepted": ...,
    "supported_accepted": ...,
    "translation_reject_reason": ...,
    "supported_reject_reason": ...,
    "improvement": translation_residual - supported_residual,
}
```

Do not drop a block merely because one stage is rejected if the residual is still measurable.

Return summaries for all measurable paired blocks and `holdout_keys_match`.

- [ ] **Step 3: Run and commit**

```powershell
pytest tests/test_klt_tps_support_c1.py -q
git add src/klt_tps_support_c1.py tests/test_klt_tps_support_c1.py
git commit -m "feat: compare fixed holdout tps support stages"
```

**Acceptance:** improvement sign is always `translation - supported`; positive means supported TPS improved residual.

---

### Task 5: Add Diagnostic-Only N=2 Pipeline Orchestration

**Files:**
- Modify: `src/multiband_pipeline.py`
- Modify: `tests/test_klt_tps_backend_n2.py`

**Interfaces:**
- Consumes: scene data, overlaps, explicit fixed HOLDOUT overrides, B12 registration index, B14 validation.
- Produces:
  - `MultibandPipeline.run_klt_tps_support_c1_n2(...) -> dict[str, Any]`
- Must not alter `register_scenes()` production dispatch behavior.

- [ ] **Step 1: Add pipeline tests before implementation**

Add these exact tests:

```python
def test_tps_support_c1_calls_klt_tps_estimator_exactly_once(monkeypatch): ...
def test_tps_support_c1_can_use_raw_flow_from_geometry_gate_rejection(monkeypatch): ...
def test_tps_support_c1_rejects_tps_fit_failure_without_counterfactual(monkeypatch): ...
def test_tps_support_c1_warps_translation_and_supported_from_original(monkeypatch): ...
def test_tps_support_c1_reuses_same_holdout_context_for_both_validations(monkeypatch): ...
def test_tps_support_c1_does_not_warp_supported_when_geometry_unsafe(monkeypatch): ...
def test_tps_support_c1_records_raw_fit_count_one(monkeypatch): ...
def test_production_klt_tps_register_scenes_does_not_enable_support_c1(monkeypatch): ...
```

- [ ] **Step 2: Add a separate method; do not add a hidden production flag**

Exact signature:

```python
def run_klt_tps_support_c1_n2(
    self,
    scene_data: Dict[str, Any],
    overlaps: List[dict],
    registration_band_idx: int,
    *,
    diagnostic_validation_band: str | int = "B14",
    holdout_reservation_overrides,
) -> Dict[str, Any]:
    """Run diagnostic-only TPS-SUPPORT-C1 for exactly two scenes."""
```

Do not route this through `register_scenes()` and do not change the normal `registration_backend: klt_tps` behavior.

- [ ] **Step 3: Require fixed HOLDOUT overrides at pipeline level**

If `holdout_reservation_overrides` is missing/empty, raise:

```python
ValueError("TPS-SUPPORT-C1 requires fixed HOLDOUT overrides")
```

Prepare `holdout_contexts` once with `_prepare_pair_holdout_contexts(...)`; require edge `(0,1)` available and all 7 reserved windows accepted by reservation geometry. Do not call random reservation again.

- [ ] **Step 4: Estimate KLT + raw TPS exactly once**

Call current `estimate_klt_tps_pair(...)` once with `training_mask=context["train_sampling_mask"]`.

Raw flow extraction rule:

```text
A. estimation.available == True:
   raw_flow = estimation["flow"]

B. estimation.available == False AND
   failure_diagnostics.stage == "tps_geometry_gate" AND
   _failure_diagnostic_arrays["flow"] exists:
   raw_flow = preserved pre-gate flow
   continue C1 because raw unsafe geometry is the Stage-0 control

C. KLT failure / mapping failure / tps_fit failure / no raw flow:
   C1 unavailable; do not fabricate a field
```

Record `raw_fit_count = 1`.

- [ ] **Step 5: Build the three stage fields from the same raw result**

Use Task 3 `build_tps_support_c1_fields(...)` with fixed taper `64` and existing max shift `50.0` from config.

Stage 0: raw geometry only; never raw warp if unsafe.

Stage 1: translation flow.

Stage 2: supported flow.

- [ ] **Step 6: Build one KLT pair measurement and reuse it**

Use the same controls/FB errors for both validation calls. The measurement's summary shift is exactly `translation_xy`.

- [ ] **Step 7: Warp Stage 1 from ORIGINAL and validate on B14**

Call:

```python
translation_registered, translation_valid = warp_multiband_with_tps_flow(
    arrays[1], translation_flow, nodata_values[1]
)
```

Then validate:

```python
translation_quality, translation_validation = _validate_final_registration_arrays(
    [np.asarray(arrays[0]).copy(), translation_registered],
    registration_band_idx,
    transforms,
    nodata_values,
    [edge],
    [pair_measurement],
    reg_params,
    holdout_contexts=holdout_contexts,
    validation_band_idx=validation_band_idx,
)
```

- [ ] **Step 8: Gate Stage 2 before warp**

If supported geometry is unsafe, do not call warp for Stage 2. Return C1 result with:

```python
"supported_geometry_safe": False,
"supported_quality": None,
"supported_validation": None,
```

If safe, warp **the same `arrays[1]` ORIGINAL** using `supported_flow`, then validate using the exact same `holdout_contexts` object.

- [ ] **Step 9: Build comparison and overall integrity**

Overall `integrity_pass` requires:

```text
raw_fit_count == 1
field support_formula_pass == True
same fixed HOLDOUT keys for translation and supported (when supported exists)
registration band == B12
validation band == B14
taper_pixels == 64
```

Do not require supported quality class to be PASS for integrity.

- [ ] **Step 10: Return dedicated diagnostic schema**

Top-level result must include:

```python
{
    "registration_backend": "klt_tps",
    "diagnostic_mode": "tps_support_c1",
    "scene_ids": scene_data["scene_ids"],
    "registration_band_name": "B12",
    "validation_band_name": "B14",
    "tps_support_causal": {
        "available": ...,
        "integrity": ...,
        "translation_xy": ...,
        "taper_pixels": 64,
        "raw_geometry": ...,
        "translation_geometry": ...,
        "supported_geometry": ...,
        "translation_quality": ...,
        "translation_validation": ...,
        "supported_quality": ...,
        "supported_validation": ...,
        "comparison": ...,
        "paired_holdout_blocks": ...,
    },
    "_tps_support_causal_arrays": {
        "translation_registered": ...,
        "supported_registered": ... or None,
        "translation_flow": ...,
        "support_weight": ...,
        "supported_flow": ...,
        "supported_jacobian": ...,
        "supported_fold_mask": ...,
    },
}
```

Do not pretend this is production `registered_arrays` output.

- [ ] **Step 11: Run focused backend tests and commit**

```powershell
pytest tests/test_klt_tps_backend_n2.py -q
pytest tests/test_klt_tps_support_c1.py -q
git add src/multiband_pipeline.py tests/test_klt_tps_backend_n2.py
git commit -m "feat: orchestrate n2 tps support causal diagnostic"
```

**Acceptance:** estimator is called once; both stages use original arrays; supported unsafe means no Stage-2 warp; production registration path remains unchanged.

---

### Task 6: Add Strict CLI Opt-In and Manifest Fail-Fast

**Files:**
- Modify: `scripts/diagnose_registration_pair.py`
- Modify: `tests/test_diagnostics_regressions.py`

**Interfaces:**
- Consumes: `--tps-support-causal-test`, fixed manifest, `--validation-band B14`.
- Produces: diagnostic C1 result and explicit completion exit code.

- [ ] **Step 1: Add parser/protocol tests**

```python
def test_tps_support_causal_cli_requires_holdout_manifest(): ...
def test_tps_support_causal_cli_requires_explicit_b14_validation(): ...
def test_tps_support_causal_cli_rejects_empty_manifest_pair(tmp_path): ...
def test_tps_support_causal_cli_rejects_export_holdout_manifest(): ...
def test_tps_support_causal_cli_is_mutually_exclusive_with_other_causal_modes(): ...
```

- [ ] **Step 2: Add CLI flag**

```python
parser.add_argument(
    "--tps-support-causal-test",
    action="store_true",
    help="Run diagnostic-only KLT translation vs supported TPS C1 on a fixed B14 HOLDOUT.",
)
```

- [ ] **Step 3: Add strict preflight before imagery loads**

When flag is set:

```text
require --holdout-manifest
require --validation-band B14
reject --export-holdout-manifest
reject simultaneous --hull-causal-test
reject simultaneous --affine-causal-test
load manifest
validate_tps_support_c1_protocol(...)
if invalid: raise ValueError("invalid TPS-SUPPORT-C1 protocol: ...")
```

- [ ] **Step 4: Route to the dedicated pipeline method**

After `scene_data` and `overlaps` are loaded:

```python
if args.tps_support_causal_test:
    registration = pipeline.run_klt_tps_support_c1_n2(
        scene_data,
        overlaps,
        pipeline.registration_band_idx,
        diagnostic_validation_band="B14",
        holdout_reservation_overrides=holdout_overrides,
    )
else:
    registration = pipeline.register_scenes(...existing behavior...)
```

Do not add the C1 flag to production `register_scenes()`.

- [ ] **Step 5: Define diagnostic-specific exit semantics**

For `--tps-support-causal-test` only:

```text
exit 1 if protocol/integrity unavailable
exit 1 if supported geometry unsafe (Gate C cannot run)
exit 1 if either Stage 1 or Stage 2 fixed-HOLDOUT validation missing
exit 0 if C1 fully completed and comparison exists
```

Quality class `FAIL` by itself must **not** cause exit 1 in this diagnostic mode; C1 may complete scientifically even if both stages fail the production quality threshold.

- [ ] **Step 6: Run parser tests and commit**

```powershell
pytest tests/test_diagnostics_regressions.py -k "tps_support_causal" -q
git add scripts/diagnose_registration_pair.py tests/test_diagnostics_regressions.py
git commit -m "feat: add strict tps support causal cli"
```

**Acceptance:** C1 cannot run without the frozen manifest and explicit B14; no fresh HOLDOUT is generated.

---

### Task 7: Write C1 Stage Metrics, Paired HOLDOUT CSV, Integrity JSON, and Visual Artifacts

**Files:**
- Modify: `scripts/diagnose_registration_pair.py`
- Modify: `tests/test_diagnostics_regressions.py`
- Optionally extend existing synthetic artifact fixture in `tests/test_klt_tps_failure_diagnostics.py`; do not duplicate fixture logic unnecessarily.

**Interfaces:**
- Consumes: `registration["tps_support_causal"]` plus private arrays.
- Produces required C1 artifacts.

- [ ] **Step 1: Add artifact tests first**

```python
def test_tps_support_c1_writes_stage_metrics_csv(tmp_path): ...
def test_tps_support_c1_writes_paired_holdout_csv(tmp_path): ...
def test_tps_support_c1_writes_integrity_json_without_full_arrays(tmp_path): ...
def test_tps_support_c1_writes_support_weight_and_supported_geometry_artifacts(tmp_path): ...
def test_tps_support_c1_writes_translation_and_supported_b14_overlays(tmp_path): ...
def test_tps_support_c1_skips_supported_overlay_when_geometry_unsafe(tmp_path): ...
```

- [ ] **Step 2: Add stage metrics CSV**

Create `_write_tps_support_c1_stage_metrics_csv(registration, output_dir)` with rows:

```text
raw_tps
klt_translation
supported_tps
```

Columns:

```text
stage
geometry_safe
fold_pixels
max_displacement_pixels
quality
median
rmse
p95
confidence
n_blocks
```

Raw TPS quality fields are blank because raw unsafe flow is never warped.

- [ ] **Step 3: Add paired HOLDOUT CSV**

Create `_write_tps_support_c1_holdout_pairs_csv(...)` using `comparison["paired_blocks"]`.

Columns:

```text
idx_i
idx_j
row
col
block_size
translation_residual
supported_residual
improvement
translation_accepted
supported_accepted
translation_reject_reason
supported_reject_reason
```

- [ ] **Step 4: Add compact integrity JSON**

Write `tps_support_c1_integrity.json` from only compact integrity fields. Assert serialized text does not contain keys:

```text
_tps_support_causal_arrays
raw_flow
translation_flow
supported_flow
support_weight
supported_jacobian
supported_fold_mask
```

- [ ] **Step 5: Add geometry artifacts**

Required files:

```text
klt_translation_field_magnitude.png
klt_tps_support_weight.png
klt_tps_supported_displacement_magnitude.png
klt_tps_supported_jacobian.png
klt_tps_supported_fold_mask.tif
```

Use moving scene native transform/CRS for the TIF. Jacobian PNG may clip p01-p99 for visualization only; numeric statistics remain unclipped.

- [ ] **Step 6: Add B14 stage overlays**

Write:

```text
klt_translation_b14_red_green_overlay.tif
klt_tps_supported_b14_red_green_overlay.tif
translation_holdout_validation_blocks.png
supported_holdout_validation_blocks.png
```

Use the fixed B14 validation band and the stage-specific registered moving array. If supported geometry is unsafe, omit supported registered overlay/holdout image and clearly report path as absent.

- [ ] **Step 7: Add one artifact entry point**

```python
def write_tps_support_c1_artifacts(
    registration,
    scene_data,
    scene_ids,
    output_dir,
    *,
    registration_band_idx,
    validation_band_idx,
) -> dict[str, str | None]:
```

Do not call the normal production `write_diagnostic_artifacts()` for this C1 mode.

- [ ] **Step 8: Run and commit**

```powershell
pytest tests/test_diagnostics_regressions.py -k "tps_support_c1" -q
pytest tests/test_klt_tps_failure_diagnostics.py -q
git add scripts/diagnose_registration_pair.py tests/test_diagnostics_regressions.py tests/test_klt_tps_failure_diagnostics.py
git commit -m "feat: write tps support c1 diagnostic artifacts"
```

**Acceptance:** all required files are written from synthetic data; unsafe supported flow never produces a registered supported overlay.

---

### Task 8: Add JSON Payload and Logging Without Serializing Private Arrays

**Files:**
- Modify: `scripts/diagnose_registration_pair.py`
- Modify: `tests/test_diagnostics_regressions.py`

**Interfaces:**
- Consumes: completed C1 result.
- Produces: `registration_diagnostics.json` containing compact `tps_support_causal` and artifact paths.

- [ ] **Step 1: Add serialization regression tests**

```python
def test_tps_support_c1_payload_contains_compact_causal_result(): ...
def test_tps_support_c1_payload_excludes_private_arrays(): ...
def test_tps_support_c1_logging_reports_raw_translation_supported_stages(caplog): ...
```

- [ ] **Step 2: Extend `build_diagnostic_payload`**

Add:

```python
"tps_support_causal": registration.get("tps_support_causal"),
```

Before calling the payload builder in C1 mode, construct a shallow sanitized dict that omits `_tps_support_causal_arrays`.

- [ ] **Step 3: Add C1 summary logger**

Create `_log_tps_support_c1_summary(registration)` and report:

```text
integrity_pass
translation_xy
taper_pixels
raw fold_pixels / raw max displacement
supported fold_pixels / supported max displacement
translation B14 median / RMSE / P95
supported B14 median / RMSE / P95
translation-minus-supported RMSE/P95 improvement
n paired fixed-HOLDOUT blocks
```

Do not print a conclusion such as “TPS solved” automatically.

- [ ] **Step 4: Run and commit**

```powershell
pytest tests/test_diagnostics_regressions.py -q
git add scripts/diagnose_registration_pair.py tests/test_diagnostics_regressions.py
git commit -m "feat: serialize tps support c1 comparison safely"
```

**Acceptance:** JSON is compact and auditable; no full field arrays appear.

---

### Task 9: Production Isolation and Regression Guard

**Files:**
- Modify: `tests/test_klt_tps_backend_n2.py`
- Modify: `tests/test_klt_tps_registration.py`
- Modify: `tests/test_diagnostics_regressions.py`

**Interfaces:** no new production interfaces; this task proves C1 remains opt-in.

- [ ] **Step 1: Add/strengthen isolation tests**

Ensure these behaviors are explicitly tested:

```python
def test_normal_klt_tps_backend_still_rejects_raw_folding(): ...
def test_normal_klt_tps_backend_does_not_apply_support_taper(): ...
def test_tps_support_c1_does_not_change_klt_or_tps_parameters(): ...
def test_tps_support_c1_does_not_change_quality_thresholds(): ...
def test_legacy_backend_dispatch_remains_unchanged(): ...
```

The raw production path must still report the same folding rejection for a synthetic folding raw field.

- [ ] **Step 2: Run all registration-focused tests**

```powershell
pytest tests/test_klt_tps_registration.py -q
pytest tests/test_klt_tps_backend_n2.py -q
pytest tests/test_klt_tps_support_c1.py -q
pytest tests/test_klt_tps_failure_diagnostics.py -q
pytest tests/test_diagnostics_regressions.py -q
pytest tests/test_registration_quality.py -q
```

- [ ] **Step 3: Commit only if new tests/code were needed**

```powershell
git add tests/test_klt_tps_registration.py tests/test_klt_tps_backend_n2.py tests/test_diagnostics_regressions.py
git commit -m "test: lock tps support c1 production isolation"
```

If all required isolation assertions already exist from earlier tasks, do not create an empty commit.

**Acceptance:** support continuation exists only under explicit C1 CLI path; production raw-TPS semantics remain unchanged.

---

### Task 10: Document the C1 Protocol and Manual Command

**Files:**
- Modify: `docs/KLT_TPS_INTEGRATION.md`

**Interfaces:** documentation only.

- [ ] **Step 1: Add a `TPS-SUPPORT-C1` section**

Document exactly:

```text
Stage 0 = raw TPS geometry-only, unsafe raw field never warped
Stage 1 = KLT median translation-only
Stage 2 = g + w(Fraw-g), fixed 64 px inside-only taper
fixed HOLDOUT = configs/dz01_model_c1_holdout_manifest.json
registration = B12
validation = B14
production backend unchanged
```

- [ ] **Step 2: Include the user-only real command**

```powershell
.\.venv\Scripts\python.exe scripts\diagnose_registration_pair.py `
  --config configs\dz01_klt_tps_n2_b12.yaml `
  --scene-i 0 `
  --scene-j 1 `
  --validation-band B14 `
  --holdout-manifest configs\dz01_model_c1_holdout_manifest.json `
  --tps-support-causal-test `
  --output-dir outputs\klt_tps_n2_b12_tps_support_c1_v1
```

Explicitly say Codex must not execute this command.

- [ ] **Step 3: Document interpretation**

Only three valid post-run branches:

```text
supported safe + better than translation -> design production adoption next
supported safe + equal/worse -> local TPS residual has limited/negative value; investigate before adoption
supported unsafe -> return to geometry/root-cause; no taper search and no safety relaxation
```

- [ ] **Step 4: Commit**

```powershell
git add docs/KLT_TPS_INTEGRATION.md
git commit -m "docs: document tps support c1 protocol"
```

---

### Task 11: Full Verification, Diff Audit, and Push

**Files:** all files touched above; no new implementation.

**Interfaces:** produces final pushed implementation commit and verification report.

- [ ] **Step 1: Run the combined focused suite**

```powershell
pytest `
  tests/test_klt_tps_registration.py `
  tests/test_klt_tps_backend_n2.py `
  tests/test_klt_tps_support_c1.py `
  tests/test_klt_tps_failure_diagnostics.py `
  tests/test_diagnostics_regressions.py `
  tests/test_registration_quality.py `
  -q
```

Expected: zero failures.

- [ ] **Step 2: Run full regression**

```powershell
pytest -q
```

Expected: zero failures. Existing warnings may remain; report their exact count. Do not suppress warnings to make output look cleaner.

- [ ] **Step 3: Audit changed paths**

```powershell
git diff 492946f1ac6adc6d80a53f023627a8140009d2e6..HEAD -- `
  src/klt_tps_registration.py `
  src/klt_tps_support_c1.py `
  src/multiband_pipeline.py `
  scripts/diagnose_registration_pair.py `
  tests `
  docs/KLT_TPS_INTEGRATION.md
```

Verify manually:

```text
no BAGRN/VOLRN/mosaic code changed
no legacy backend behavior changed
no KLT parameter defaults changed
no TPS smoothing/neighbors/field_step/max_shift changed
no quality thresholds changed
no holdout config changed
no N>2 implementation
no affine/DEM/optical-flow additions
no supported field applied in production path
no full arrays serialized to JSON
```

- [ ] **Step 4: Verify branch state and commit history**

```powershell
git status --short
git log --oneline -12
```

Working tree must be clean.

- [ ] **Step 5: Push**

```powershell
git push origin feat/dz01v-klt-tps-n2-integration-20260912
```

No merge, no rebase, no force push.

- [ ] **Step 6: STOP before real data**

Do not run the real N=2 command. Return a report containing:

```text
branch
start HEAD
final HEAD
commits created
focused pytest counts
full pytest counts
warning counts
confirmation real DZ01V not run
confirmation N=4/N=6 not run
confirmation BAGRN/VOLRN/mosaic unchanged
confirmation KLT/TPS parameters unchanged
confirmation Jacobian/max-shift gate unchanged
confirmation fixed manifest unchanged
confirmation production klt_tps behavior unchanged
```

---

## Manual User Checkpoint After Codex Stops

Only after Codex has pushed and the implementation has been reviewed should the user run:

```powershell
git switch feat/dz01v-klt-tps-n2-integration-20260912
git pull

.\.venv\Scripts\python.exe scripts\diagnose_registration_pair.py `
  --config configs\dz01_klt_tps_n2_b12.yaml `
  --scene-i 0 `
  --scene-j 1 `
  --validation-band B14 `
  --holdout-manifest configs\dz01_model_c1_holdout_manifest.json `
  --tps-support-causal-test `
  --output-dir outputs\klt_tps_n2_b12_tps_support_c1_v1
```

The real C1 run is valid only if `integrity_pass=true`, the fixed 7-window HOLDOUT is reused, and Stage 0/1/2 all originate from the same one-time raw KLT/TPS estimate. A supported geometry failure is a scientifically valid negative result; do not respond by tuning 64 px from final HOLDOUT or relaxing the safety gate.

## Expected Real-Run Artifacts

```text
registration_diagnostics.json
tps_support_c1_stage_metrics.csv
tps_support_c1_holdout_pairs.csv
tps_support_c1_integrity.json
klt_translation_field_magnitude.png
klt_tps_support_weight.png
klt_tps_supported_displacement_magnitude.png
klt_tps_supported_jacobian.png
klt_tps_supported_fold_mask.tif
klt_translation_b14_red_green_overlay.tif
klt_tps_supported_b14_red_green_overlay.tif
translation_holdout_validation_blocks.png
supported_holdout_validation_blocks.png
```

If Stage 2 remains geometry-unsafe, its registered overlay and holdout image must be absent; geometry diagnostics must still be present.

## Decision Rule After the Real Run

- **Geometry first:** supported field must have `fold_pixels == 0`, finite values, and `max_displacement_pixels <= 50.0` before accuracy is interpreted.
- **Then accuracy:** compare translation-only versus supported TPS on exactly the same 7 fixed B14 windows. Primary quantities are RMSE, P95, median, paired per-window residuals, and accepted/rejected status transitions.
- **No production adoption in this plan:** even a successful C1 result only authorizes a subsequent production-adoption design; it does not silently change the backend here.
