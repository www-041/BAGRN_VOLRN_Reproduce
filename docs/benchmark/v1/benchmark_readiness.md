# 13 景 V 传感器统一 Benchmark 数据条件

审计日期：2026-10-09。此页只判断现有数据及已有 Task15/Task16 产物的可复用性，不执行新实验。

## 当前条件

1. **景与波段：**处理后目录有 13 景 V 传感器影像，每景 B1–B16 完整，共 208 个单波段 GeoTIFF。B9 已有正式 13 景实验基础。
2. **B9 数据单位：**现存 B9.TIF 的明确像元单位为 `UNKNOWN`。MTL 提供 DN 到辐亮度可能使用的乘数/加数，但没有证明现存 TIFF 尚为 DN，或已经转换成辐亮度/反射率。研究者需提供处理链或产品说明确认。
3. **范围、时间、坐标：**全部 TIFF 为 EPSG:32650、14 m；各景宽高、Affine 原点和覆盖范围不同。采集时间跨 2025-11-14 至 2026-01-14，不是同期同范围影像。详见两个 CSV 中逐景的投影与经纬度外接矩形。
4. **景内多波段几何：**每景的 16 个波段文件头有相同的 CRS、宽高和 Affine Transform，可作为共网格实验设计的前提。像元内容是否达到跨波段配准精度仍需独立抽样或控制点检验，不能只凭文件头确认。
5. **光谱和质量：**16 个波段在 13 景的 MTL/IMD 中均有最小、中心、最大波长；208 个波段均有辐亮度定标乘数/加数。未发现明确的现存像元单位、反射率定标系数或独立云/质量掩膜。TIFF 有 NoData 掩膜和场景级 `QUALITY_STATUS` 标签，3 景标为 `RISK`；标签不提供像元级云/质量标注。

## 不新增数据时可做的工作

- 复用已完成的 B9 Task15/Task16 结果，做固定支持域上的内部拼缝、辐射相对一致性、结构保持和运行资源比较；结论应注明指标支持域及 Task16 收敛状态。
- 对 16 个波段做文件头/元数据一致性比较，并在有限窗口内检查跨波段几何对齐与相对亮度。后续多波段归一化实验可设计，但需先确认波段单位与输入协议；本审计没有运行该实验。
- 利用现有几何网络的匹配点与回环残差做内部一致性分析，但这些并非独立地面真值。

目前不能从这些输入独立证明绝对地理精度、真实地表反射率误差、云区表现或跨日期的绝对辐射稳定性。相应 Benchmark 需要可审计的独立控制/检查点坐标、物理单位与定标链、云/质量标注或参考观测。处理后 TIFF 中的检查点**数量标签**不是可重新计算误差的点坐标。

## 现有产物可复用位置

Task15 根目录为 `data/output/final_pipeline/b9_13scene_formal_v2/`；`pipeline_status.json` 标记 `SUCCESS`，`pipeline_manifest.json` 标记 `VALID_EXPERIMENT_COMPLETED`。以下均为该目录下的现有文件或目录：

| 用途 | 已有产物 |
| --- | --- |
| 输入景清单、scene index | `00_preflight/input_inventory.json` |
| 配准匹配和全局几何 | `registration/pairwise_summary.json`、`stages/03_global_adjustment/global/translation_l2/global_transforms.json` |
| 共用网格 | `stages/04_canonical_warp/canonical_output_grid.json` |
| 有效距离缓存 | `stages/05_valid_distance_cache/distance_scene_000.tif` 至 `distance_scene_012.tif` |
| BAGRN 参数及归一化景 | `stages/06_bagrn/bagrn/bagrn_parameters.npz`、`normalized_scenes_manifest.json` |
| seam 与局部系数 | `stages/07_pairwise_seam_local/pairwise_results.json`、`pairs/*/local_coefficients.csv` |
| 多景标签、局部校正 | `stages/08_multiscene_labeling/source_label_map.tif`、`stages/09_correction/corrected_scene_*.tif` |
| B9 三路线拼接与指标 | `stages/10_mosaics/v0_bagrn_weighted.tif`、`v1_multiscene_label_blend.tif`、`v2_local_corrected_multiscene.tif`、`stages/11_metrics/` |

Task16 根目录为 `data/output/b9_13scene_volrn_comparison_formal_v2/`：`02_volrn_solver/volrn_coefficients.npz`、`03_volrn_corrected_scenes/`、`04_volrn_weighted_mosaic/bagrn_volrn_iter200_weighted.tif`、`05_volrn_end_to_end/ours/mosaic.tif`、`05_volrn_end_to_end/volrn/mosaic.tif` 和 `05_radiometric_metrics/` 可作现有比较依据。`task16_status.json` 为 `COMPLETED_FINITE_NONCONVERGED`：结果有限且已输出，但 200 次迭代未达到严格收敛，不应标为已收敛 VOLRN 解。

这些产物对应 **B9** 的现有输入与冻结协议。把 B9 全局变换或辐射系数直接迁移至其他波段，需先验证跨波段共址和物理单位，不能仅凭文件头同网格就认定有效。

## 研究者需确认

- 师兄处理链输出的 B9 及其他波段究竟是 DN、辐亮度还是反射率；MTL 的辐亮度系数是否已应用、应用于哪个版本。
- 三个 `QUALITY_STATUS=RISK` 景的判定细节、是否应纳入所有 Benchmark 协议，以及有无独立云/质量掩膜。
- 独立检查点的坐标/误差、辐射参考观测和典型山区测试区位置；目前只有部分检查点计数标签，没有可重算的真值数据。
- 其他波段在相同 Affine 网格上的内容级配准精度，以及处理前后 B9 的具体工艺和重采样方法。
