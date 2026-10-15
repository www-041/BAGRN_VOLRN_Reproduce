# Unified Benchmark V1：13景方法产物契约

适用Protocol：`DZ01V_UNIFIED_A_B_V1_20261013`。本契约面向RAW、BAGRN、BAGRN＋VOLRN及未来外部13景方法；不改冻结输入/掩膜/78对清单/1533个窗口/指标公式。

## 建议的独立输出布局

```text
data/output/benchmark_v1_unified/submissions/<method_id>/<run_id>/
  method_submission.json
  scenes/scene_000_B9_corrected.tif ... scene_012_B9_corrected.tif
  mosaic_B9.tif
  mosaic_valid_mask.tif
  source_labels.tif                 # 有来源图时必须提供
  solver_status.json
  runtime.json
  logs/...
  evaluation_A_B1_B2.json           # 由独立评分CLI生成
```

方法及run必须隔离目录，不能覆盖历史产物。RAW的逐景输入可以直接引用冻结对齐GeoTIFF（只读、以冻结哈希识别），不必复制；RAW仍需声明使用的公共接缝/拼接方式和Mosaic产物，不能把“无校正”写成“无评价”。

## 逐景GeoTIFF

- 恰好13个，`scene_id`与冻结`pair_overlap_inventory.csv`及manifest逐一对应；不得漏景或加入额外scene。
- 单波段B9，规范网格EPSG:32650，仿射`(14,0,633346,0,-14,3359874)`，宽7992、高12404；像元方向/中心约定与冻结输入一致。禁止评分时重投影、裁剪或对齐修正。
- 提供GeoTIFF显式有效mask（不能仅以像元值非零判有效）；浮点NoData/NaN和mask语义一致。不得通过输出mask缩小主指标支持域。
- 输出数值使用实数型；建议float32并保留实际dtype/NoData。不得由评分器拉伸、截断或重新定标。物理单位继续是`UNKNOWN`。
- 全冻结支持必须有有限有效输出；缺失状态如实记录，不能临时改用方法自选共同支持。

## Mosaic、掩膜、来源

- A要求规范网格单波段Mosaic和单独显式有效掩膜。预期支持是13个冻结输入mask的并集；报告coverage、holes、extras，正式完整覆盖检查不允许自动缩域。
- 硬/软接缝方法如欲计算接缝诊断，应交付同网格来源标签图和明确的ID映射。标签值`0..12`按照冻结scene顺序映射；Mosaic有效处必须有唯一合法标签且像元来自该场景的有效输出。缺失/非法来源图时 seam 状态为不可评价/无效，不填零分。
- 保存接缝/权重来源文件（如算法提供），声明`seam_type=HARD|SOFT|NONE`；额外方法特定seam只能作为辅助说明，不能改变固定B1/B2支持。

## 求解状态、时间和资源

`solver_status.json`至少记录：方法/版本/配置hash、每景和全局阶段状态、系数/参数（若适用）、异常类别、输入/输出数量及退出状态。失败项保留`FAIL`或精确状态，不删除。

`runtime.json`和提交JSON需记录：完整计时起止定义、wall秒数/计时器来源、RSS峰值/采样数/采样间隔/监控器、GPU峰值/采样数/来源，或明确`NOT_MEASURED`；另记录声明资源预算及`hard_os_memory_limit=true|false`。轮询预算不可伪称OS硬限制。算法运行耗时必须覆盖方法特定预处理、估计/应用、拼接和输出关闭；公共几何已冻结，不计入某个方法。

当前评价器采用整幅数组指标原语，运行前用冻结网格像元数、13景数、声明RSS预算和可取得的可用RAM执行保守预检。该预估不是内存保证/OS限制；当前全网格估值约为几十GB量级，低预算会在读方法大栅格前以`RESOURCE_PREFLIGHT_FAILED`停止。精确RDD不得改成抽样/分箱近似；资源不足应保留`RESOURCE_LIMIT_EXACT_RDD`或停止状态。

## `method_submission.json`

遵循既有`method_submission_schema.json`，必须提供：`schema_version=1`、`method_id`、`run_id`、`method_kind`、冻结的`input_freeze_sha256`与`geometry_freeze_sha256`、`band=B9`、`pixel_value_unit=UNKNOWN`和状态。路径可相对该JSON目录。B1/B2提交必须有完整13个scene路径；A必须有mosaic与mask；A seam评分需要source label路径。记录参数、环境、时间、RSS/GPU和资源预算。不得向Schema或冻结配置写入猜测值。

## 评分状态

- 78个pair行均保留；30个`NO_VALID_OVERLAP`不算失败，48个正重叠对必须完整。
- B1/B2支持不完整分别标`OUTPUT_MISSING`或`PARTIAL_OUTPUT`，缺失指标null，不进入正式主排名。
- B2读取冻结CSV全部1533窗口，先核对每窗`valid_count`与`support_sha256`；任何不符为冻结支持错误。
- A缺source label时仅接缝字段标`NOT_EVALUABLE_NO_PROVENANCE`；不构造代理标签。
- B3是分开的已知真值合成运行；无真值输入/输出时返回`NOT_RUN_SYNTHETIC_METHOD_OUTPUT_REQUIRED`，与真实B1/B2分开。
- 正式排名另要求完成的run状态、A/B1/B2完整支持、全部冻结pair/window人口、有效方法状态和可追溯时间/资源记录。未测资源可如实保留，但排名门槛应按冻结status policy逐项审查。

## 路径与命令

```powershell
python scripts/benchmark/score_unified_v1_submission.py `
  --freeze data/output/benchmark_v1_unified/freeze_20261013_v3 `
  --method-submission data/output/benchmark_v1_unified/submissions/<method_id>/<run_id>/method_submission.json `
  --regimes A B1 B2 `
  --output data/output/benchmark_v1_unified/submissions/<method_id>/<run_id>/evaluation_A_B1_B2.json
```

评分器会重验冻结lock、方法freeze hashes、输入/输出栅格、scene身份、有效掩膜、固定窗口SHA和运行测量来源；失败会输出明确状态，不自动缩小评价支持。
