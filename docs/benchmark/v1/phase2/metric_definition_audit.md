# Task15/Task16 辐射指标公式与支持域只读审计

审计对象是现有源码，不重算正式结果，也不修改其实现。当前 TIFF 像元物理单位为 `UNKNOWN`，下述量均在现有数值域，不是地表反射率误差。

| 指标 | 实际公式/实现 | 样本支持和聚合 | 源码位置 |
|---|---|---|---|
| MAMD | `abs(mean(A)-mean(B))`，**不是**逐像元 MAE | 同一对景的 shared-valid 且两者有限；每对一个值，以 `valid_pixels` 加权平均，也保留 pair median/明细 | `src/multiscene_sift/radiometric_metrics.py::compute_pair_mamd`, `aggregate_weighted_pair_metric` |
| MSDD | `abs(std(A)-std(B))`，`numpy.std` 默认总体标准差 ddof=0 | 同上；不是逐像元差的标准差 | 同文件 `compute_pair_msdd` |
| RDD | `scipy.stats.wasserstein_distance(A,B)`，两组共享支持像元值的精确一维 Wasserstein-1/分布距离；不依赖像元一一对应的差值 | 同上；不是百分比，也不是 Task15 边界 MAE | 同文件 `compute_pair_rdd` |
| Local MAMD/RDD | 上述公式在 canonical-grid 固定 256×256 tile 上计算，tile 至少 4096 个双有效有限像元 | 汇总为 tile median/mean/P95；不能与全局 pair 加权均值混称 | `compute_local_pair_metrics`、`task10d_metric_protocol.py` |
| Seam MAE/RMSE/RDD | MAE=`mean|A-B|`，RMSE=`sqrt(mean(A-B)^2)`，RDD 同 Wasserstein | pair feather 权重归一后 `min(wA/(wA+wB),wB/(wA+wB))≥0.25` 且双有效；每对后按支持像元加权 | `compute_seam_zone_metrics`、`src/registration_benchmark/mosaic_diagnostics.py::build_seam_zone_mask` |
| Task15 Stage11 transition | 在路线各自权重的 pair-normalized transition zone 计算 per-pair MAE/RMSE/RDD，RDD 仍为 Wasserstein | V0/V1/V2 **各自**权重与支持；不能直接当跨方法固定区域独立评估。`_transition_metrics` 汇总 MAE/RDD 时按 pair 像元数加权 | `scripts/run_task14a_resume_13.py::_transition_metrics`, `_stage11` |
| Task15/16 label boundary | 在 source-label map 相邻标签变化处取相应两景源与校正值，计算 MAE/RMSE/RDD；RDD 为 Wasserstein | label 边界为方法相关支持，pair 像元加权及 pair median；Task16 A/B 复用 Task15 label helper。不同于固定地理评价区 | `scripts/run_task14a_resume_13.py::_boundary_metrics`, `src/task16_volrn_comparison.py` |

Task16 `_stream_task10d_metrics` 对 BAGRN/Ours/VOLRN 使用共同 13 景 mask，global MAMD/MSDD/RDD 经同一共享有效支持和函数，local 在 canonical 256 tile 上至少 4096 像元；A/B 固定 BAGRN 输入消融沿用同一 mask/weights。Task15 Stage11 主体是 route transition/boundary 与结构评估，不能把其中 RDD 误称 Task10D 全局 RDD。Task16 E2E 另有固定 **V1** transition 支持用于 Ours/VOLRN 对比，必须与各自 route seam 支持分开引用。旧代码不变；V1 正式公共区域与标注尚未完成，因此这份审计只确认公式/支持约定，**不确认跨日期稳定区或变化区评价已经成立**。

## 多波段规则

B1–B16 每个波段单独记录单位、有效支持、MAE/RMSE/MAMD/MSDD/RDD、样本量和失败状态。**禁止把不同波段的原始 MAE 直接算术平均**：波段动态范围、定标与响应不同。若以后要跨波段汇总，必须事先冻结同一尺度规则（例如只用独立 DEVELOPMENT 稳定参考样本的固定尺度 `s_b`，汇总 `MAE_b/s_b`；`s_b>0`，测试集不得参与求尺度），同时始终保留原始逐波段结果。尺度来源、波段权重及异常波段处理要预注册；单位 UNKNOWN 或内容级共址未验证时 SAM/SAC 只能列为 `NOT_EVALUABLE`。真实地表反射率绝对精度不可报告。
