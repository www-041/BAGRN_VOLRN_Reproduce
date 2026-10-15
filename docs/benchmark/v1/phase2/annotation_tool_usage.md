# B9 独立同名点人工标注工具

## 客观信息预填、批量标签与审核留痕（当前版本）

**两个 Apply 按钮对应两类不同记录。**审核顶部 40 个自动候选点时，先在顶部候选列表单击一条，查看左右影像，再修改下方的 `Review` 等字段；点击顶部的 `Apply candidate review`，最后点击 `Save CSV`。下方的 `Apply manual-point review` 仅用于研究者通过 `Manual add` 自主新增、并在下方手工点列表选中的记录。若点错手工按钮而没有选中手工点，状态栏会明确指向顶部候选审核按钮。`Apply` 只更新内存，关闭窗口前必须再按 `Save CSV`；未提交或未保存的标签不会写入候选 CSV。

若编辑下方字段后，顶部列表的高亮消失，但顶部信息栏与左右图仍显示同一个候选 ID（如 `00_01_C0001`），现在可直接点 `Apply candidate review`，无需重新选点；程序以已查看的唯一候选 ID 为准。若顶部同时选中了多条，仍会拒绝批量审核。选 `APPROVED` 时必须填写人工 Confidence 和坐标定位不确定度，并确认确实检查过左右图；当前未冻结空间划分的点继续保留 `UNASSIGNED`，即使人工批准也不进入正式 TEST 评分。

打开候选点时，界面显示 `pair_id`、两景 ID、采集时间及相隔天数、输入质量状态、原始左右像素/投影坐标、推荐来源与方法。`PAIR_GRID:<pair>:<grid_cell_id>` 仅表示候选生成时的成对局部网格，不是地形、地物或冻结的空间分组。当前没有经核实的 DEM/土地覆盖分类图，建议值保持 `Terrain=TERRAIN_UNKNOWN`、`Landcover=UNKNOWN`；研究者可以保留 UNKNOWN 或逐点覆盖，程序不会根据灰度/NCC 猜测确定类别。`Confidence` 和人工定位不确定度也不会被自动填写。顶部进度显示已审核、待审核、待二审数量；`DOUBTFUL` 可用于无法确认的位置。

默认没有冻结的开发/测试空间划分，因此 Split 为 `UNASSIGNED`。人工可以将确实核对过的点标为 `APPROVED + UNASSIGNED`，但它**不是正式 TEST 真值**，不会被正式评分读取。待研究者独立冻结空间隔离文件后，可用 `--frozen-split <path.geojson>` 启用 DEVELOPMENT/TEST 建议。文件须为 GeoJSON FeatureCollection，顶层 `metadata` 含 `status: FROZEN`、非空 `version`、`crs: EPSG:32650`；各 Polygon/MultiPolygon feature 的 `properties` 含 `role: DEVELOPMENT` 或 `TEST` 及非空 `split_id`。只有左右两点都明确落在同一分区时才建议角色；仍须人工确认，TEST 点仍须不同审核人二审。格式校验不证明划分科学独立性。未冻结文件请勿自行给相邻点随机分组。

逐点审核：选候选→核对左右原生影像细节→必要时 `Adjust candidate` 分别修正左右位置→保留/修改 Landcover、Terrain，填写人工 Confidence 与定位不确定度→选择 `APPROVED`、`REJECTED` 或 `DOUBTFUL`→`Apply candidate review`→`Save CSV`。明确的人工确认是 APPROVED 唯一入口；算法分数不触发审批。人工自主选点仍用 `Manual add`。批量标签：在候选列表用 Ctrl/Shift 多选仍为 `UNVERIFIED` 的点→`Bulk labels...`→选择 Terrain 和/或 Landcover（`KEEP` 表示不改）→核对确认框→`Save CSV`。如果选中任何已审核点，整批拒绝；不能批量更改 Review、Confidence、Split、坐标或不确定度。候选仍须逐点复核，批量标签不等于批量批准。

候选 CSV 的原始坐标与 `adjusted_*` 人工修正坐标分别保留。首次使用本功能保存修改时，同目录创建 `<pair>.baseline.csv` 原 CSV 快照；后续每次保存的逐点前后差异与审核人/UTC 时间记入 `<pair>.audit.jsonl`，元数据与建议来源快照写入 `<pair>.context.json`。这些文件**只从首次使用本功能保存时开始留痕**，不能倒推过去未记录的操作。不会重生成候选坐标，也不会覆盖独立的手工标注 CSV。

在仓库根目录 PowerShell 中启动或只读查看计数：

```powershell
.\.venv-registration\Scripts\python.exe scripts/annotate_benchmark_v1.py --pair 00_01 --reviewer "你的姓名"
.\.venv-registration\Scripts\python.exe scripts/annotate_benchmark_v1.py --pair 00_01 --count-approved
```

`--count-approved` 分别给出 `auto_APPROVED_UNASSIGNED`（人工同意但未分组）、`auto_formal_APPROVED`（符合分组/审核规则）和 `auto_APPROVED_TEST_second_reviewed`。旧的 `auto_APPROVED` 字段保留兼容，含义与 `auto_formal_APPROVED` 相同。

本工具只读取 Task15 `scene_manifest.csv` 记录的两景 **源 B9 GeoTIFF**，不读取任何 Matcher/RANSAC 匹配点。拉伸发生在显示缓冲区，不写回原始像元。四对试验：`00_01`、`02_03`、`05_07`、`08_10`。不要把工具启动成功等同于检查点已审核。

## 自动候选推荐与人工复核（新增）

候选生成器只用源 B9、GeoTIFF Affine 与双有效掩膜；Shi–Tomasi 角点经局部梯度 NCC、双向检查及网格分散选点。先确认候选 CSV 尚不存在：生成器**拒绝覆盖**已有候选审核记录。默认每对最多 40 个，低纹理/阴影等情况允许不足。

```powershell
.\.venv-registration\Scripts\python.exe scripts/generate_benchmark_v1_candidates.py --pair 00_01 --max-candidates 40
.\.venv-registration\Scripts\python.exe scripts/annotate_benchmark_v1.py --pair 00_01 --reviewer "你的姓名"
```

其他试验对可将 `--pair` 改成 `02_03`、`05_07`、`08_10`。四个候选 CSV 位于 `auto_candidates/<pair>.csv`，旁边的 `.manifest.json` 记录输入 B9 SHA256 核验和实际候选数。脚本不会运行 EfficientLoFTR、LoFTR、LightGlue、Task15/16。候选分数只用于排序，**不代表匹配正确或定位精度**。

现有四份候选 CSV 已生成；日常审核只运行第二条 GUI 命令，**不要重新运行生成器覆盖审核记录**。候选文件按脚本所在仓库根目录定位，不依赖 PowerShell 当前目录。`00_01` 正常启动时终端会打印 `Loaded 40 auto candidates`，窗口顶部固定显示 `Candidates (40 loaded)`、筛选框、候选列表和审核按钮。两幅影像及手工标注控件位于下方可滚动区域；小屏幕可用右侧/底部滚动条查看，不会把候选控件挤出屏幕。若指定的候选文件不存在、为空或格式错误，程序会在终端打印路径及原因并弹出错误框，**不会进入纯手工模式**。可用 `--candidates 路径` 显式指定另一份候选 CSV，但必须与该影像对匹配。

打开 GUI 后，候选列表可按 `PENDING/REVIEWED/ALL` 筛选；点击候选或使用 `Previous/Next`，左右窗口会分别定位并放大到候选点，青色十字标记原始或人工调整位置。`Skip candidate` 将当前点记为 `SKIPPED` 并移到下一个，保存后可在待审核列表重新访问，但不产生审批。若坐标偏移，点击 `Adjust candidate`，在左右图分别点修正位置；原自动坐标保留在 CSV 的无前缀字段，修正坐标写入 `adjusted_*`，原有审批自动撤销。选候选时普通点击不会偷偷新增人工点；按 `Manual add` 后才可继续自主选点。

填写 `Landcover/Terrain/Confidence/Split/Coordinate precision/Notes` 后，将 `Review` 设为 `APPROVED`、`REJECTED`、`DOUBTFUL` 或 `UNVERIFIED`，按 `Apply candidate review`。APPROVED 必须明确确认已检查两窗细节并填写人工定位不确定度；TEST 点第一次同意会变为 `PENDING_SECOND_REVIEW`，须由**不同审核人**再次打开并确认后才正式 APPROVED。`Save CSV` 才保存候选审核状态。真正通过的候选另存于 `auto_reviewed_truth/<pair>.csv`，与手工 `annotations/<pair>.csv` 分离；`truth_provenance/<pair>.csv` 标明 `MANUAL` 或 `AUTO_SUGGESTED`。打开候选和只审核候选不会重写既有手工标注文件，也不会覆盖 `00_01_P0001` 一类已有手工 ID。

只读统计命令：

```powershell
.\.venv-registration\Scripts\python.exe scripts/annotate_benchmark_v1.py --pair 00_01 --count-approved
```

它分别报告手工 APPROVED、候选 APPROVED、已由第二人复核的候选 TEST 点及待二审数。正式配准评价只能使用 APPROVED TEST 点，且测试点不得参与任何被测算法的求解/RANSAC/参数选择。推荐器偏向高纹理，研究者应补充低纹理、山区、不同地物的自主选点；建议自主点约占正式检查点 20%–30%，正式评测前冻结比例，并按来源分层报告误差。此比例是待确认的设计目标，不应为了凑比例收录不可靠点。

GUI 小规模自检可用 `--pair 00_01 --smoke-candidates`：它在系统临时目录复制候选及现有手工 CSV，自动试验定位、双窗调整、REJECTED 审核、保存/重载，**不会修改真实候选或手工文件，也不会产生真实 APPROVED 点**。该自检不等于研究者的视觉核验。

在仓库根目录的 PowerShell 中运行：

```powershell
.\.venv-registration\Scripts\python.exe scripts/annotate_benchmark_v1.py --pair 00_01 --reviewer "你的姓名"
```

替换 `--pair` 可打开另外三对。`--check-env` 只验证两个源文件头和 Tk 窗口创建，不进入标注、不保存文件。默认输出为本目录 `annotations/<pair_id>.csv`；再次打开同一对会加载已有 CSV。可用 `--output` 指定其他**独立**标注 CSV；文件 schema 或影像对不一致时拒绝覆盖。建议每位标注者各保存一份，再由另一人复核合并，避免覆盖彼此记录。

两窗显示同一投影中心的原生 B9 窗口。鼠标滚轮/“Zoom ±”缩放，箭头按钮平移共同地理中心；窗口读取原生分辨率，放大只改变显示尺寸。左图点一个地物，再在右图点相同地物，才会新增 `UNVERIFIED` 点。红十字表示待完成点击；“Skip”丢弃待完成点。列表中选择点后，“Edit selected”允许重新点两边位置，并**自动撤销原审批**；“Delete selected”删除内存记录，“Doubtful”标记疑难。下方录入地物、地形、置信度、DEVELOPMENT/TEST、审核状态和备注；“Apply review”保存到内存，“Save CSV”才持久化。关闭窗口前若有未保存改动，请先保存。

`APPROVED` 必须由人明确选择、指定 split 和置信度，并再次确认不是直接采用 Matcher 点。建议独立第二审核人复核坐标及可辨识地物；工具本身不能证明独立性。未经审核点一律 `UNVERIFIED`，不得进入正式 RMSE/P95。测试点绝不可用于变换拟合、参数调整或筛点。

坐标约定：浮点 `row/col` 的整数 `0,0` 表示左上像元**中心**；投影坐标由 TIFF Affine 对 `(col+0.5,row+0.5)` 计算。裁剪原点、显示缩放、像素中心与投影坐标的往返误差有合成测试覆盖。鼠标浮点坐标表示记录精度，**不代表地物定位精度**；请在 `coordinate_precision_note`/备注中记录人工估计的不确定度。NoData 点击会被忽略，显示拉伸不能作为辐射判断依据。

若研究者要标注 15 对中其他影像对，应先扩展工具的 `TRIAL_PAIRS` 白名单并完成同样的显示/坐标验证；本阶段只开放四对试验。更多字段规则见 `registration_truth_schema.md`。
