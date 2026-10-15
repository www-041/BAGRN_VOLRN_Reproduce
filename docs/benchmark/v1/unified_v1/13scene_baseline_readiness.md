# 13景正式Baseline运行前就绪报告

日期：2026-10-14  
Protocol：`DZ01V_UNIFIED_A_B_V1_20261013`  
结论：**NO-GO：完整13景正式方法实验的指标验收门尚未满足。** 本次只证明小型合成端到端路径可用，没有启动方法实验。

## 冻结内容复核

只读复核`freeze_20261013_v3`：freeze lock验证PASS（54项锁定文件）；冻结manifest为13景、B9、单位UNKNOWN，网格EPSG:32650、14m、7992×12404；影像对78，其中有效重叠48、无重叠30；固定B2窗口1533。`configs/benchmark_v1_unified.yaml`、metric_spec、protocol和Schema相符。本次未写入冻结目录，也未运行BAGRN/VOLRN/外部算法。

## 端到端合成验收

新建合成13景小型GeoTIFF、显式mask、mosaic和来源标签；运行A/B1/B2，并单独检查B3状态。`evaluator_smoke_report.json`记录：A覆盖率1.0、空洞0、额外覆盖0、来源接缝可评价；B1保留78行（夹具中66正重叠、12无重叠），七项指标齐全；B2夹具的66个窗口均有记录并生成132条场景-窗口结构诊断；另有13条逐景结构/动态范围记录；无来源标签时状态明确；人为不完整支持保留为`PARTIAL_OUTPUT`且不排名；资源测量样例明确为`NOT_MEASURED`；B3为`NOT_RUN_SYNTHETIC_METHOD_OUTPUT_REQUIRED`。这些是合成接口证据，不是真实方法成绩，也未验证真实方法输出GeoTIFF。

**验收缺口：**小型fixture不是冻结完整地理网格，B1夹具覆盖66个合成正重叠对而非真实48对，B2端到端计算66个fixture窗口而非实际1533个冻结窗口。真实freeze清单及1533条记录已只读核验，生产适配器也包含逐窗固定支持哈希校验，但尚未用完整规范网格的13份提交产物跑通全部记录。因此不能将本次称为“完整13景A/B1/B2端到端验收通过”，正式运行gate保持NO-GO。

专项测试：`tests/benchmark_v1_unified/test_submission_scoring.py`。原冻结脚本`evaluate_unified_v1.py`未改，本次使用新增`score_unified_v1_submission.py`。

验证记录（2026-10-14）：Benchmark V1核心目录 **38 passed**；最终全仓pytest **1353 passed, 4 skipped**（2项合成配准图断连条件跳过，2项外部matcher集成依赖未配置/转集成运行）；该次未触发Tk环境跳过。`compileall`通过；`git diff --check`通过（Git对工作区既有Spectralmatch文件提示LF/CRLF转换警告，不是空白错误）。

## 启动正式实验前剩余硬门

1. 为真实冻结网格补充轻量/流式全1533窗端到端验收（或提供足以安全运行现有整幅评价器的资源），并覆盖真实48正对/30零对清单。
2. 每个方法需单独提供符合`benchmark_artifact_contract.md`的13景逐景输出及A Mosaic/mask；RAW也需固定、可复现的统一Mosaic后端。
3. 每方法在开始正式计算前提供资源预算、运行状态schema和测量来源。当前评分器保守数组工作集规划值约为全网格像元数×466 bytes，即约46 GB；这不是OS硬限制或精确峰值承诺。低预算或可用RAM不足会在读取方法栅格前以`RESOURCE_PREFLIGHT_FAILED`停止。逐景归一化算法自身的内存预算仍须独立核实。
4. 最终方法输出完成后，用CLI在原冻结目录运行全量A/B1/B2，并确认13景、48正对、30零对、1533窗口、支持哈希及所有输出完整；合成通过不能替代真实文件验收。
5. A接缝指标需要来源标签/权重出处；无来源信息时只报告`NOT_EVALUABLE_NO_PROVENANCE`。B3仍是独立S1–S4真值模拟，本次未运行，不由真实重叠区评分替代。

**执行决策：**本任务完成了评分入口和产物契约初版，但严格gate仍为NO-GO。需先补足完整窗口验收与资源预检证据；不得据此宣称任何算法完成正式13景运行或排名。本次未启动算法。
