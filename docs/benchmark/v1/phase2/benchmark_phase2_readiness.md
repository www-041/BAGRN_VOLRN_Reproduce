# DZ01V Benchmark V1 第二阶段就绪报告（2026-10-10）

结论：**人工标注工具与独立评分单元测试可用；正式六算法测试仍 NOT READY**。第一阶段文件保持原样，第二阶段全部新增在 `phase2/` 或 `scripts/`/`tests/`。未运行 Matcher、BAGRN/VOLRN、辐射归一化训练或完整镶嵌；未修改原始影像及 Task15/Task16 正式结果。

## 第一阶段复核及更正

78 对是 13 景完整组合；48 对有真实 B9 双有效掩膜重叠。其余 30 对为 `NO_VALID_OVERLAP`，其中 18 对 bbox 不交、12 对 bbox 交但掩膜不交，**不是配准方法 FAIL**。15 对首轮配准候选保持原顺序/成员，不按 Task15 成绩重排；其最大间隔约 30 天，不能代表整个 13 景跨度。本阶段单独增加三对 37–61 天辐射区域候选，不改变首轮配准集合。第一阶段 15 个区域全部仍 `UNVERIFIED`，新三处也未自动审批；人工标签均 `UNKNOWN`。

输入身份：Task15 `scene_manifest.csv` 已记录 13 景 B9 源路径、文件大小、当时的 SHA256 及 MTL SHA256；`input_identity_manifest.csv` 逐景转录并标 `TASK15_FORMAL_V2_RECORDED_SOURCE`，**本阶段没有重新哈希源文件，不能声称当前路径内容仍与记录 SHA 一致**。B1–B16 除 B9 外尚未冻结当前 SHA256，也无本阶段逐波段输入版本号；这些是正式实验前的待冻结项。Phase1 YAML 的相对路径从 `docs/benchmark/v1/` 解析至仓库 Task15 目录；本期脚本同用该 manifest 的完整产品 ID 与 B9 路径。

## 三个长间隔候选的实际 B9 支持

在原生 EPSG:32650/14 m 网格上遍历两景共同范围的 512×512 候选 tile，按双 mask 同时有效比例选一个窗口；这不是稳定地物/真实变化判定。全 pair 有效像元数来自第一阶段精确条带计数，窗口像元数是本阶段重新读取掩膜的计数。

| Pair | 时间差 | 整对 B9 有效交集 | 512×512 窗口双有效 | 比例 | 人工标签 |
|---|---:|---:|---:|---:|---|
| `00_10` | 61.001974 天 | 1,374,076 | 236,974 / 262,144 | 90.3984% | UNKNOWN / UNVERIFIED |
| `02_10` | 54.999443 天 | 9,173,415 | 262,144 / 262,144 | 100% | UNKNOWN / UNVERIFIED |
| `04_11` | 36.993539 天 | 5,088,227 | 262,144 / 262,144 | 100% | UNKNOWN / UNVERIFIED |

逐项日期、场景号、PASS/RISK、窗口投影边界、源 SHA 记录见 `extended_temporal_pair_candidates.csv`；几何见 `extended_temporal_regions.geojson`；独立拉伸的双窗小图在 `previews/`。全像素有效比例不等于云自由或稳定区比例，长日期差不保证季节变化。

## 人工审核与空间隔离

`radiometric_region_review_template.csv` 有原 15 加新 3，共 18 行，全部 `label=UNKNOWN`、`review_status=UNVERIFIED`、`role=UNASSIGNED`。人工 `label` 只可取 `STABLE/REAL_CHANGE/CLOUD/SHADOW/UNKNOWN`，须登记证据、审核人与时间；`CLOUD/SHADOW` 默认从稳定性评分排除但仍可作质量分层。人工审核后再给 `DEVELOPMENT`（校正/PIF/阈值）或 `TEST`（冻结后评价）。不能让同一地物或相邻窗口跨两者。`scripts/check_benchmark_phase2_leakage.py` 把全部候选 GeoJSON 转为 EPSG:32650，以默认 140 m 缓冲检查 DEV/TEST 多边形相交或相邻，并要求两类角色均存在且都已批准；当前空模板应返回 `ready_for_spatial_split=false`。该检查不替代人工地物/检查点分割审核。

研究者编辑审核 CSV 的副本、填写 `label/review_status/role/reviewer/review_time_utc/evidence_notes` 后，可运行 `python scripts/check_benchmark_phase2_leakage.py --review <审核CSV路径> --buffer-m 140`。发现冲突时退出码为 2；没有冲突但仍未审核齐全时 `ready_for_spatial_split=false`。这个命令只读，不自动改写审核决定。

## 指标可计算条件与阻塞

合成坐标/变换下，检查点 RMSE、P95、最大误差、PCK@1/2/3、对级成功/失败分母及宏平均已有可测试实现。**正式值仍无法计算**，因为 APPROVED TEST 独立点为零，四对仅供人工试验。MAMD/MSDD/RDD 的历史公式和支持域已在 `metric_definition_audit.md` 核查；但固定独立辐射区还没有 STABLE/REAL_CHANGE 真值，不能宣称真实变化保持或稳定区误差。多波段逐波段结果可设计，跨波段原始 MAE 不可直接平均；SAM/SAC 等待单位/内容级共址确认。现有像元单位仍 UNKNOWN，绝对地表反射率精度不可报告。

进入正式六算法测试前：至少完成四对工具实点试标、第二人复核并冻结 APPROVED TEST 与空间拆分；确认其余 11 对的标注方案；审核 18 个辐射区域及云/阴影/稳定/变化类别；冻结 B9 当前 SHA 和其他波段版本；确认 KARIOS/AROSICS 等环境、统一输入/几何模型、各法失败规则、辐射指标尺度和空间隔离。当前所有自动区保持候选，不得自动升级真值。完成本阶段后等待研究者人工检查。

## 自动候选辅助工具补充（2026-10-10）

四对现已各生成 40 条自动建议、共 160 条，全部仍为 `UNVERIFIED`，并未增加正式 APPROVED TEST 点。各对网格分布与输入 SHA 核验见 `auto_candidate_generation_report.md`。GUI 能把候选与原手工点并列显示、分别定位、保留原建议坐标进行人工修正、保存候选审核和独立的已批准真值；`00_01` 的自动点击/审核/重载只在临时副本上验证，**不替代实际人工视觉审核**。正式测试的阻塞条件维持不变。
