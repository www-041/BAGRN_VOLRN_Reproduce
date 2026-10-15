# Benchmark V1 第一阶段就绪审查（2026-10-09）

结论：**DATA/PROTOCOL DRAFT READY；FORMAL BENCHMARK NOT READY**。本次仅建立只读清单、候选、协议与少量预览；没有运行 SIFT、KARIOS、AROSICS、LoFTR、EfficientLoFTR、LightGlue＋DISK、BAGRN、VOLRN、Spectralmatch 或完整镶嵌，也没有训练模型。

## 数据与跨日期重叠

- 13 景、每景 V B1–B16 齐全；B9 是 EPSG:32650/14 m 的现有实验基础。13 景采集在 2025-11-14 至 2026-01-14，3 景 `RISK` 仍保留。
- 全部 78 对中 60 对外接矩形交、48 对 B9 有效掩膜真正重叠；同日 7、跨日期 41。跨日期按实际时间差：<7 天 13、7–<30 天 6、≥30 天 22。`temporal_pair_inventory.csv` 有逐对精确 B9 双 mask 像元计数。这个时间分布不能证明多完整季节或四季泛化。
- 15 对第一轮人工配准检查候选：`00_01 00_05 01_06 02_03 02_04 02_07 03_08 04_06 04_09 05_06 05_07 08_10 09_12 10_11 11_12`。覆盖 13 景、同日/跨日期及 PASS/RISK；选择没有使用 Matcher 指标。它们的地形均 `TERRAIN_UNKNOWN`，实际适宜性仍需人工确认。详见 `pair_candidates.csv`。

## 六配准方法接口判断

| 方法 | 第一阶段结论 |
|---|---|
| SIFT | 仓库已有路径；统一 2D 平移和独立点评价可设计，尚未执行。 |
| KARIOS | 官方 KLT 可输出点/位移；需对每对构造同 footprint/geotransform/14m 网格、单波段整型输入，并隔离 Python 3.12+/conda 环境。不得把其 SIFT/ECC `align` 当 KLT。 |
| AROSICS | `COREG` 全局平移可进共同模型主榜；`COREG_LOCAL` 是局部模型，只能分榜，另需掩膜/正射/网格和重采样适配。 |
| LoFTR | 仓库已有适配；统一输出几何模型和失败分母，未执行。 |
| EfficientLoFTR | Task15 旧配准产物可辅助设计，不可作独立检查点真值；V1 未执行。 |
| LightGlue＋DISK | 仓库已有适配；同输入/模型/独立检查点，未执行。 |

KARIOS/AROSICS 约束分别据其[官方仓库](https://github.com/telespazio-tim/karios)、[官方仓库](https://github.com/GFZ/arosics)；适配结论不是在本机的运行结果。

## 当前可用与缺失

可以立即建立：完整场景/波段/质量/路径清单、精确 B9 有效 overlap、跨日期分层、失败对分母规则、统一输入/变换/资源日志接口、已知扰动模拟设计及现有 B9 Task15/16 历史指标的复用索引。可复用 Task15 原始清单、60 条 bbox 重叠、Stage02 配准记录、网格/变换、Stage06 BAGRN、V0/V1/V2 与 Stage11；Task16 VOLRN 历史产物须带非严格收敛标签。

目前**没有可以直接用于正式配准精度的独立人工审核检查点**；模板零条点、推荐 pair 的预览仅供标注。还缺独立稳定地物与真实变化标签、云/阴影/质量掩膜、DEM/山区分类、跨波段内容级共址验证、当前像元物理单位/定标链、MAMD/MSDD/RDD 公式冻结、SAM/SAC 启用条件及外部方法许可证/环境确认。因此真实变化 F1/IoU、物理反射率误差、可靠多波段 SAM/SAC、真实季节泛化均不能正式评分。13 景内部空间拆分也不能称独立地理泛化。

## 研究者需确认（进入正式 Benchmark 前）

1. 15 对是否代表论文目标，哪些是山区/平原、稳定地物/真实变化、云或阴影；人工复核候选评价区域及掩膜。
2. 双人审核浮点检查点、定位不确定度和开发/测试空间隔离；冻结方法不参与真值制作的规则。
3. B1–B16 当前像元单位、跨波段真实共址、是否存在未归档的 QA/云掩膜及 DEM；不追溯处理前工艺。
4. LMM 具体实现/论文身份，外部方法代码/许可证与运行环境；尤其 RS-NormGAN 许可证、KARIOS 网格、AROSICS 两模式自由度。
5. 各辐射指标的原始公式、公共支持域/融合规则、真实变化标注及可接受的训练/运行资源预算。

产物：`benchmark_protocol_draft.md/.yaml`、`pair_candidates.csv`、`registration_checkpoints_template.csv`、`evaluation_region_candidates.geojson`、`baseline_manifest.csv`、`temporal_pair_inventory.csv`、`multitemporal_radiometric_protocol.md`、`radiometric_methods_review.md`、`innovation_hypotheses.md`、本报告和 `previews/`。`scripts/build_benchmark_v1_inputs.py` 可重复生成清单；单元测试在 `tests/test_benchmark_v1_inputs.py`。
