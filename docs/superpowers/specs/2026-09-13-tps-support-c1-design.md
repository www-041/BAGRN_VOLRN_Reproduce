# TPS-SUPPORT-C1 设计规格

日期：2026-09-13  
基线分支：`feat/dz01v-klt-tps-n2-integration-20260912`  
设计基线 commit：`1d0f7de56ae0bb716c1e2aad38eb1ce04074b5ad`

## 1. 背景与已知证据

TPS-FOLD-D1 已经把当前 KLT/TPS Phase-I N=2 失败定位到 TPS dense-flow 的无支撑外推，而不是 KLT 无法在 DZ01V B12 上获得稳定控制点。

真实 N=2 结果：

- KLT accepted controls：3962 / 4000；
- FB median：约 0.016 px；
- FB P95：约 0.087 px；
- control hull 只覆盖 moving scene 约 10.31%；
- raw TPS full-scene field 出现 599,908 个 fold pixels；
- 其中仅 9 个 fold pixels 位于 control hull 内，599,899 个位于 hull 外；
- 99.9985% folding 位于 control hull 外；
- control displacement magnitude 最大约 19.69 px，而 full-scene TPS field 最大约 146.48 px。

因此本阶段不再调整 KLT 参数、TPS smoothing、neighbors、max_shift 或 Jacobian gate。设计目标是只改变“TPS 在控制点无支撑区域如何延拓到整景”的几何语义。

## 2. 目标

TPS-SUPPORT-C1 是一个严格的 N=2 因果实验，回答两个问题：

1. **安全性问题**：把 raw TPS 的局部残差限制在 control hull 支撑域内，并在无支撑区域平滑退回稳定的 KLT median translation，是否足以消除当前 full-scene TPS folding？
2. **精度价值问题**：在相同 TRAIN-only KLT controls、相同 raw TPS fit、相同 B14 fixed HOLDOUT 下，supported TPS 是否比 KLT translation-only 更有配准精度价值？

本阶段仍然不是 production rollout。只有真实 C1 结果通过后，才讨论是否把 support continuation 写成 KLT/TPS backend 的生产行为。

## 3. 不在本阶段做的事情

本阶段明确禁止：

- 不重新设计或替换师兄的 GFTT + bidirectional KLT；
- 不修改 B12 registration / B14 validation；
- 不修改 TPS kernel；
- 不修改 `klt_tps_smoothing`；
- 不修改 `klt_tps_neighbors`；
- 不修改 `klt_tps_field_step`；
- 不修改 `klt_tps_max_shift`；
- 不修改 Jacobian folding gate；
- 不增加 KLT displacement outlier rejection；
- 不增加 RANSAC / affine / DEM / optical flow；
- 不扩展到 N=4/N=6；
- 不修改 BAGRN、VOLRN、mosaic；
- 不用 final HOLDOUT 选择 taper width 或任何超参数。

## 4. 推荐方案：KLT median translation continuation + supported TPS residual

### 4.1 同一批 controls 产生两个量

对 TRAIN-only KLT controls 的 moving-native displacement：

`d_i = (dx_i, dy_i)`

定义稳健常量 continuation：

`g = (median(dx_i), median(dy_i))`

这里的 `g` 不是额外模型搜索，也不是从 final HOLDOUT 得到；它只来自与 raw TPS 完全相同的 TRAIN controls。

### 4.2 raw TPS 保持不变

继续用当前 `build_tps_dense_flow(...)` 得到：

`F_raw(x, y)`

TPS fit 必须只做一次；C1 不允许为不同 stage 重新拟合 TPS。

### 4.3 代数分解为 translation + local residual

定义：

`R_tps(x, y) = F_raw(x, y) - g`

这一步不改变 raw TPS，只是代数分解。

### 4.4 control-hull inside-only support weight

从 moving-native KLT control convex hull 构造权重 `w(x,y)`：

- hull 外：`w = 0`；
- hull 边界：`w = 0`；
- 从 hull 边界向内部的 `T` 像素内平滑增大；
- 距离边界 >= `T`：`w = 1`。

第一轮固定：

`T = 64 px`

不做参数搜索。

推荐 smoothstep：

`u = clip(d_inside / T, 0, 1)`

`w = 3u^2 - 2u^3`

其中 `d_inside` 为 hull 内像素到 hull 边界的欧氏距离；hull 外直接为 0。

### 4.5 supported TPS treatment

最终 treatment field：

`F_supported = g + w * (F_raw - g)`

其语义：

- hull 外：严格退回 `g`；
- hull 边界：仍为 `g`；
- hull 内部过渡带：逐渐开启 TPS residual；
- hull 深处：完全恢复 `F_raw`。

这比 `flow[~hull]=0` 更合理，因为无支撑区域不会重新引入“完全不做几何修正”的 discontinuity。

## 5. C1 三阶段实验

### Stage 0 — RAW TPS geometry-only

- 使用与当前完全相同的 KLT controls；
- 使用完全相同的 raw TPS fit；
- 不允许 unsafe raw TPS warp；
- 只记录 raw geometry statistics；
- 预期复现当前 folding，作为 treatment 的同-run control。

### Stage 1 — KLT translation-only baseline

全景 field：

`F_translation(x,y) = g`

- 从 ORIGINAL moving multiband 一次 warp；
- 使用固定 B14 HOLDOUT；
- 作为 supported TPS 的最小、稳定 baseline。

### Stage 2 — SUPPORTED TPS treatment

全景 field：

`F_supported = g + w(F_raw-g)`

- 先通过当前完全不变的 geometry safety gate；
- 若 unsafe：STOP，不允许 warp；
- 若 safe：从 ORIGINAL moving multiband 一次 warp；
- 用与 Stage 1 完全相同的 B14 HOLDOUT 验证。

## 6. 因果完整性要求

C1 必须记录并验证：

- same KLT initial/accepted control count；
- same KLT control coordinates；
- same KLT displacement values；
- same raw TPS fit；
- same smoothing / neighbors / field step；
- same raw TPS field checksum；
- same support taper fixed at 64 px；
- same fixed HOLDOUT manifest；
- same B14 validation band；
- Stage 1 和 Stage 2 都从 ORIGINAL 一次 warp；
- final HOLDOUT 不参与任何 model/taper selection。

数值完整性：

- deep-inside `w=1` 区域：`F_supported == F_raw` 到数值容差；
- outside-hull `w=0` 区域：`F_supported == g` 到数值容差；
- `0 <= w <= 1`；
- support mask 不得改变 raw TPS 本身。

## 7. HOLDOUT 协议

TPS-FOLD-D1 导出的 manifest 当前 `pairs` 为空，因此不能用于 C1。

C1 必须要求用户提供一份已有、真正冻结的 BAND-C1 B14 HOLDOUT manifest，其中必须包含当前 N=2 pair 的固定验证窗口。

CLI 必须 fail-fast：

- `--tps-support-causal-test` 必须同时提供 `--holdout-manifest`；
- validation band 必须显式为 `B14`；
- manifest 必须包含当前 pair；
- manifest pair windows 必须非空；
- 不允许在 C1 内重新 reserve 一套新 final HOLDOUT。

如果用户无法找到旧 BAND-C1 manifest，应先恢复/重建该固定 manifest 的来源，不应直接开始 C1 真值比较。

## 8. 安全 gate

### Gate A — Causal integrity

若 controls、raw TPS、HOLDOUT 或参数不一致，本次实验无效，直接标记 `integrity_pass=false`。

### Gate B — Geometry safety

SUPPORTED TPS 必须继续通过现有安全条件：

- finite field；
- `fold_pixels == 0`；
- `max_displacement_pixels <= klt_tps_max_shift`。

禁止为 C1 放宽这些门。

如果 supported field 仍然 unsafe：

- 不进行 Stage 2 warp；
- 不调 taper width；
- 输出 geometry diagnostics；
- 实验结论只能是“固定 64 px support continuation 不足以使 field safe”。

### Gate C — Accuracy value

只有 Stage 2 geometry safe 后才比较 fixed B14 HOLDOUT：

- translation-only median / RMSE / P95 / confidence；
- supported TPS median / RMSE / P95 / confidence；
- paired fixed-HOLDOUT block residuals；
- accepted/rejected block 状态。

判读：

- supported 显著优于 translation：局部 TPS residual 有实际价值；
- supported ≈ translation：support 解决安全问题，但 local TPS 增益有限；
- supported 更差：local TPS residual 本身仍需调查，不得通过调 final HOLDOUT 参数救模型。

## 9. 组件边界

### `src/klt_tps_registration.py`

新增纯函数，建议职责：

- `compute_klt_translation_continuation(displacement_xy)`；
- `build_inside_hull_taper_weight(control_points_xy, shape, taper_pixels)`；
- `compose_supported_tps_flow(raw_flow, translation_xy, support_weight)`；
- support/continuation integrity statistics。

这些函数不得做 I/O，也不得读取 HOLDOUT。

### `src/multiband_pipeline.py`

新增 C1 diagnostic orchestration；production `registration_backend: klt_tps` 默认行为保持当前 raw TPS 语义不变。

C1 应显式 opt-in，不应静默改变正常 backend。

### `scripts/diagnose_registration_pair.py`

新增 CLI：

`--tps-support-causal-test`

并要求固定 manifest + B14 validation；输出三阶段 C1 diagnostics/artifacts。

### tests

新增 pure-array synthetic tests、pipeline causal-integrity tests、CLI/manifest fail-fast tests、diagnostic serialization tests。

## 10. 诊断产物

建议至少输出：

- `tps_support_c1_stage_metrics.csv`
- `tps_support_c1_holdout_pairs.csv`
- `tps_support_c1_integrity.json`（或嵌入 registration_diagnostics.json）
- `klt_translation_field_magnitude.png`
- `klt_tps_support_weight.png`
- `klt_tps_supported_displacement_magnitude.png`
- `klt_tps_supported_jacobian.png`
- `klt_tps_supported_fold_mask.tif`（若 fold=0 仍可写零图，便于审计）
- Stage 1 / Stage 2 的 B14 red-green overlay
- fixed-HOLDOUT block visualization

禁止把 full raw/support arrays 直接序列化进 JSON。

## 11. 实验结果 schema

建议在 registration result 中增加 diagnostic-only：

`registration["tps_support_causal"]`

至少包含：

- `available`
- `integrity`
- `translation_xy`
- `taper_pixels`
- `raw_geometry`
- `supported_geometry`
- `translation_quality`
- `supported_quality`
- `comparison`
- `paired_holdout_blocks`
- `artifacts`

production `quality` 字段不应被 C1 偷偷替换，除非 CLI diagnostic 明确构造独立 comparison payload。

## 12. 测试策略

必须先 synthetic/TDD，再允许用户真实 N=2。

核心 synthetic cases：

1. constant translation raw flow：supported 应等于 translation/raw；
2. hull 外 raw flow 极端发散：supported hull 外严格等于 translation；
3. hull 深处：supported 与 raw TPS 相等；
4. taper 区 smoothstep 单调、连续、范围 [0,1]；
5. supported field geometry safe case；
6. supported field 仍 unsafe case：禁止 warp；
7. fixed manifest 缺失/空 pair：fail-fast；
8. Stage 1/2 确实复用同一 HOLDOUT；
9. one-pass warp from ORIGINAL；
10. legacy backend / BAGRN / VOLRN / mosaic regression 不变。

## 13. C1 结束后的决策

真实 N=2 运行后只允许三种下一步：

1. **supported safe 且优于 translation**：再设计 production adoption；
2. **supported safe 但≈translation/更差**：保留 KLT translation，继续分析 local TPS residual；
3. **supported 仍 unsafe**：回到 geometry/root-cause，不做 taper 参数搜索，不放宽 safety gate。

本设计明确不预设 TPS-SUPPORT-C1 会成功。它的目的，是用单一变量、固定 HOLDOUT、同一 raw TPS fit，验证“support continuation”是否既解决 geometry folding，又保留有价值的 local correction。
