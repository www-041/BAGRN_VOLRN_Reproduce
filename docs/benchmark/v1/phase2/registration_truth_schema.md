# 独立配准检查点 schema 与评分边界

CSV 由 `scripts/annotate_benchmark_v1.py` 人工保存；目前**没有自动生成的 APPROVED 点**。主键 `point_id` 在 pair 内唯一；`pair_id` 如 `00_01`；`scene_i/j` 为完整产品 ID，`band=B9`。浮点 `row_i,col_i,row_j,col_j` 是原生 TIFF 的中心索引约定，`x_i,y_i,x_j,y_j` 是同一 EPSG:32650 投影坐标。无经纬度推断。`coordinate_precision_note` 应写人工定位不确定度，不能把小数位数当精度。

`landcover_type`、`terrain_class`、`confidence`、`notes` 用于审核和分层；`review_status` 仅允许人工置 `APPROVED/UNVERIFIED/REJECTED`。首次点击总是 `UNVERIFIED`；修改坐标后自动回退 `UNVERIFIED`。`reviewer` 和 `review_time_utc` 记录最后审核者和 UTC 时间。`split` 为 `DEVELOPMENT/TEST/UNASSIGNED`。正式误差只读取 `APPROVED` 且 `TEST` 的点；`DEVELOPMENT` 可以供拟合/调参，不能混入正式测试。审核轨迹目前是单行最新状态，并非不可篡改历史；正式冻结前需保留版本化原件和第二人审核记录。

同一地物、邻近裁剪、重叠地理窗口不得跨开发/测试。配准点空间块隔离需在 GIS 中结合位置审核；区域分割可用 `check_benchmark_phase2_leakage.py` 检查 DEVELOPMENT 与 TEST 多边形相交或距离 ≤140 m（默认 10 个 B9 像元），但这**不替代检查点级空间块核查**。

完整 78 对分为：48 对原生 B9 双有效掩膜重叠（具备方法尝试资格）；18 对 bbox 不交、12 对 bbox 交但有效掩膜不交（共 30 对 `NO_VALID_OVERLAP`，不得记 Matcher `FAIL`）。对具备重叠但方法未产出有效变换者记 `FAIL`；未尝试记 `NOT_ATTEMPTED`；无 APPROVED TEST 点记 `NO_INDEPENDENT_TRUTH`。报告候选总数、已尝试数、失败数、无真值数和成功率分母；成功对的 RMSE/P95/MAX/PCK 作 pair 级宏平均，并同时列出完整逐对分布与失败状态。不得把测试点用于拟合，也不得把无真值成功对伪造成 0 误差。

评分接口 `scripts/benchmark_phase2_core.py` 的 `evaluate_pair()` 只接受外部已冻结的**坐标映射函数**和 APPROVED TEST 点，不执行匹配/拟合。误差为右景预测像素中心坐标与人工右景像素坐标的欧氏距离；`RMSE=sqrt(mean(e²))`、`P95=np.percentile(e,95)`、`MAX=max(e)`、`PCK@t=mean(e≤t)`，单位均为原生 B9 像元。局部变形方法须另外声明映射模型，不能混入平移主榜。

自动推荐使用独立的 `auto_candidates/<pair>.csv`，所有生成行都是 `recommendation_status=SUGGESTED`、`human_review_status=UNVERIFIED`。无前缀 `row/col/x/y` 是**原始建议坐标**，人工调整只写 `adjusted_*` 并标 `human_adjusted=True`，不得擦除原建议。候选 TEST 第一次审核为 `PENDING_SECOND_REVIEW`，不同第二审核人明确确认后才 `APPROVED`。只有通过审核的点才导出到 `auto_reviewed_truth/<pair>.csv`，带 `point_source=AUTO_SUGGESTED` 与 `origin_candidate_id`；手工文件沿用原 schema，不强制迁移或重编号。来源映射同时记录在 `truth_provenance/<pair>.csv`，供正式报告按推荐/自主来源分层。候选点的 NCC 分数与双向误差只是自动筛选依据，不是独立精度真值。
