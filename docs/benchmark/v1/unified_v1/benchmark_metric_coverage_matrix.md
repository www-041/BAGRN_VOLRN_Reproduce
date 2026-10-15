# Unified Benchmark V1 指标—产物覆盖矩阵

Protocol：`DZ01V_UNIFIED_A_B_V1_20261013`；本文件只映射冻结V1定义，不增加或替换指标。检查依据为 `benchmark_protocol_draft.md/.yaml`、`configs/benchmark_v1_unified.yaml`、冻结目录 `metric_spec.json`/`frozen_manifest.json`/`pair_overlap_inventory.csv`/`fixed_windows.csv`、`method_submission_schema.json` 与 `src/benchmark_v1_unified/`。

| 指标 / 协议ID | 精确定义与支持域 | 所需产物 | 已实现 / 单测 | GeoTIFF端到端验证 | 正式排名条件 / 无法计算状态 |
|---|---|---|---|---|---|
| B1 MAE_V1 | `mean(abs(x-y))`，冻结输入掩膜交集；像元单位UNKNOWN | 13景逐景校正B9 GeoTIFF及明确有效掩膜 | `radiometric.compute_pair_direct_metrics`；核心单测与提交适配器测试 | 合成13景GeoTIFF，66个有重叠夹具对 | 48个冻结有效对均全支持；缺失为`OUTPUT_MISSING`，部分为`PARTIAL_OUTPUT`，保留78对行；不可用为null |
| B1 RMSE_V1 | `sqrt(mean((x-y)^2))`，同上 | 同上 | 同上 | 合成GeoTIFF | 同上 |
| B1 BIAS_V1 | `mean(x-y)`，有符号诊断 | 同上 | 同上 | 合成GeoTIFF | 同上 |
| B1 P95AE_V1 | `linear_quantile(abs(x-y), .95)` | 同上 | 同上 | 合成GeoTIFF | 同上 |
| B1 MAMD_V1 | `abs(mean(x)-mean(y))` | 同上 | `compute_pair_distribution_metrics`；核心单测 | 合成GeoTIFF | 同上；ddof规则不适用 |
| B1 MSDD_V1 | `abs(std_ddof0(x)-std_ddof0(y))` | 同上 | 同上 | 合成GeoTIFF | 同上；总体标准差ddof=0 |
| B1 RDD_W1_EXACT_V1 | 完整样本上的精确经验1D Wasserstein-1 | 同上 | SciPy exact W1；核心单测 | 合成GeoTIFF小样本 | 同上；资源不足须报告`RESOURCE_LIMIT_EXACT_RDD`，不得用近似直方图替代 |
| B1 pair-macro / weighted / median / P95 | 对每对先算指标；主摘要是有效重叠对宏平均，另报像元加权、对级中位数/P95和逐对全表 | 全部78对清单及每对产物状态 | `aggregation.aggregate_pair_rows`；测试适配器 | 合成78行（66重叠、12无重叠） | 30个无重叠对状态为`NO_VALID_OVERLAP`，不是失败；正式冻结数据预期48正、30零 |
| B2 固定窗口七指标 | 在每个冻结128×128窗口内同样计算七项；同时报告每对窗口中位数/均值/P90/最大值及窗口数 | 逐景校正输出、固定窗口CSV和原始冻结掩膜 | 窗口支持校验SHA、`aggregate_windows`；适配器测试 | 合成夹具66个窗口全部验证；真实冻结清单为1533个，未用真实方法输出评分 | 1533个窗口必须逐项存在；不完整窗口标`OUTPUT_MISSING`/`PARTIAL_OUTPUT`，不缩小支持域 |
| B逐景 / B2结构保持 | 冻结Sobel `/8`、3×3有效支撑、输入梯度P50阈值的`SOBEL_DIRECTION_LOSS_V1`与`REL_GRAD_MAE_V1`；P95-P05动态范围保持率 | 每景输入、校正后GeoTIFF和掩膜 | `structure.py`；核心与提交级测试 | 13景合成GeoTIFF及132条场景-窗口结构记录 | 每景保留状态/支持像元；不足时明确不可评价。不是论文GL；SSIM关闭，Gradient NCC不作V1主指标 |
| A覆盖 / 空洞 / 额外覆盖 | `coverage=|U∩Q|/|U|`，holes=`|U\Q|`，extras=`|Q\U|`；U为13景输入掩膜并集 | 最终Mosaic、显式Mosaic有效掩膜、冻结输入掩膜 | `mosaic.compute_mosaic_coverage`；已有及适配器测试 | 合成GeoTIFF：coverage=1、holes=0、extras=0 | 正式Mosaic必须全覆盖且无额外范围；输出缺失/网格错误显式拒绝 |
| A单源区结构 | 仅在恰有一个输入源有效的区域，将该源与最终Mosaic做冻结结构比较 | 13景校正图、最终Mosaic及掩膜 | `compute_single_source_structure`；适配器测试 | 合成GeoTIFF | 记录单源支持像元和结构状态；不扩展到多源区像元真值 |
| A接缝超额跳变诊断 | 冻结代码的硬来源标签接口；跨标签像元跳变与同源邻域基线之差，并报边数 | Mosaic、有效掩膜、与稳定scene ID映射一致的来源标签图 | `seams.compute_seam_excess`；适配器测试 | 有来源标签合成GeoTIFF可评价；无来源标签测试为`NOT_EVALUABLE_NO_PROVENANCE` | 有来源证明方可评价；不能把缺标签当0，也不单独代表整体质量 |
| 运行时间 / RSS / GPU / 预算 | 保留观测值、采样数量与测量来源；未测量必须null+`NOT_MEASURED`；资源预估不是OS硬限制 | 方法日志与`method_submission.json.runtime/resource_budget` | 适配器验证来源/采样一致性 | 合成流程明确标记未测量；未声称实测资源 | 正式排名需要有效wall计时来源及RSS/GPU“实测或明确未测”记录；硬内存限制单独标记。预估不足则在打开大栅格前`RESOURCE_PREFLIGHT_FAILED` |
| B3 S1–S4真值指标 | 既有R1合成器已知真值：truth MAE/RMSE；S4变化区域误差、稳定区伪变化及变化幅度比等 | 固定seed/config、真值/扰动/支持掩膜及方法校正后的同域合成产物 | `evaluator.evaluate_synthetic_arrays`、`simulation.py`；R1专项测试 | 本任务未运行B3方法实验 | 当前状态`NOT_RUN_SYNTHETIC_METHOD_OUTPUT_REQUIRED`；不得用真实重叠区B1/B2替代 |

冻结协议中论文`CD`、`GL`公式标为`FORMULA_UNVERIFIED`，`RDOA/Ave`为`NOT_COMPUTED`；W1不得改称CD，Sobel方向损失不得改称GL。历史Task15/16和R1指标仅做公式交叉核对，不导入本次评分。

## 当前结论

原`evaluate_unified_v1.py`仍是冻结历史入口的占位实现（返回`NOT_RUN_MISSING_REGIME_ARTIFACTS`），本次没有修改它。新增只读提交适配器`submission_scoring.py`和独立CLI接入现有冻结指标原语。合成端到端通过不等于真实方法已通过评分：当前没有任何新的完整13景RAW/BAGRN/BAGRN+VOLRN GeoTIFF提交。
