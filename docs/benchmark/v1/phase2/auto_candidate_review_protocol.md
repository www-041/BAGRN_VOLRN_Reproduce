# 自动候选与独立人工真值边界

自动候选来自源 B9 的共享地理网格与双有效掩膜，按 8×8 空间网格做 Shi–Tomasi 角点抽取、局部梯度归一化互相关、双向一致性、唯一性、纹理/NoData 邻域过滤和像素距离去重。选择时优先覆盖不同网格，再考虑匹配分数；每对上限 40，不能为了满额降低阈值或输出低纹理区域的猜测。其位置是建议，不是人工真值，也不引用 Task15 EfficientLoFTR 点、RANSAC 内点或 Translation-L2 残差。每对原源 B9 SHA256 与 Task15 当时的记录相等才生成；候选文件存在时拒绝重新生成覆盖。

人工审核可 `APPROVED/REJECTED/DOUBTFUL/UNVERIFIED`。点击进入候选会在左右窗分别居中放大；人工修正保留原始坐标，修正后撤销原审批。APPROVED 需观察双窗、填写人工置信度和定位不确定度。没有冻结的空间划分时，split 保持 `UNASSIGNED`：人工可确认对应关系，但该点不进入正式评价真值。冻结划分后才能建议 DEVELOPMENT/TEST；TEST 需不同第二审核人复核。自动候选的 `matching_score`、`forward_backward_error` 绝不自动触发 APPROVED、HIGH 置信度或固定定位不确定度，且不得称作定位精度。候选 CSV 与正式人工检查点 CSV 独立；导出的自动来源真值只含符合 split 与审核规则的记录。

GUI 只预填可由现有记录确定的场景身份、采集时间及间隔、输入质量状态、原始像素和投影坐标、推荐来源/方法。`PAIR_GRID:<pair>:<grid_cell_id>` 是原候选推荐网格内的成对局部分区，不是已核准的稳定地物区、地形类别或 DEV/TEST 划分。当前无经核实的 DEM/土地覆盖图，因此地物建议为 `UNKNOWN`、地形建议为 `TERRAIN_UNKNOWN`；灰度和匹配分数不得推断类别。只有研究者提供经事先冻结的 GeoJSON（顶层 `metadata.status=FROZEN`、`version`、`crs=EPSG:32650`，每个 Polygon/MultiPolygon feature 有 `role=DEVELOPMENT|TEST` 与 `split_id`）并经本工具的基本几何校验后，工具才根据两侧有效坐标落在同一分区建议 split。该文件仍须由研究者独立确认空间隔离与版本，工具的自声明格式校验不等于科学审核；未提供文件时统一 `UNASSIGNED`。

批量编辑必须由研究者 Ctrl/Shift 明确选择目标候选、指定 Terrain/Landcover 值并二次确认；只允许修改仍为 `UNVERIFIED` 的候选。若选择集含已审核点，则整批拒绝，不修改任何点。Review、Confidence、Split、坐标和定位不确定度绝不批量填写，自动建议也不覆盖人工标签。首次保存候选修改时，原 CSV 快照保存在同目录的 `<pair>.baseline.csv`；每次保存的逐点修改前后记录保存在 `<pair>.audit.jsonl`，建议及来源快照保存在 `<pair>.context.json`。历史仅从此功能首次保存起可追溯，不能凭这些文件复原更早的修改；保存/重新打开应核对审核状态和修正坐标。

为避免单一角点策略的选择偏差，保留 GUI 自主点击。建议正式检查点中 20%–30% 来自独立自主选点，比例在开发期可调整、正式测试前冻结；若低纹理、阴影、复杂地形或不同地物没有可靠推荐，优先人工补充，不能从正式数据中忽略。结果应报告推荐来源与自主来源的分层误差、覆盖及失败数，不只报告混合总体。所有正式 TEST 点严禁进入被评估配准算法的求解、RANSAC 筛选或参数调优。当前四对候选均未获人工批准，不能进入正式 Benchmark。
