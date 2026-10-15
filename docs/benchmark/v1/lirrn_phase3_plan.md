# Phase 3.2 实战计划：LIRRN (2024) 的 Python 忠实重实现与 DZ01V 两组真实 B9 裁剪实验

> **交给 Codex 直接执行；目标是实际 GeoTIFF + 真实统计，不是再写一轮可行性报告。**
>
> 计划版本：`LIRRN_PYTHON_PAIR_SMOKE_V1_20261014`  
> 固定评估协议：`DZ01V_UNIFIED_A_B_V1_20261013`  
> 实验性质：`PAIR_ONLY / DIAGNOSTIC_ONLY / NOT_FORMAL_13_SCENE_RANKING`  
> 冻结目录：`data/output/benchmark_v1_unified/freeze_20261013_v3/`

## 0. 执行决定、交付物和止损规则

本轮**必须以真实运行结果为第一优先级**：在同一套冻结 13 景 B9 未归一化对齐输入上，独立复现官方 LIRRN 双时相辐射归一化的执行逻辑，实际运行预先约定的 `00_01`、`00_10` 两组 `1024×1024` 裁剪，保存**每组至少一张真实校正后的 B9 GeoTIFF**、掩膜、前后对比图、指标 JSON/CSV 和执行日志。允许指标变差，不允许为了指标而改算法或筛掉难像元。

本轮不需要 MATLAB、`osgeo.gdal`、Spectralmatch 新环境或新 Benchmark 框架；优先使用已通过全仓测试的 `.venv-registration` Python。**不允许运行全景影像对、13 景、多波段或正式排名**。不修改 Ours、Task15/16、R1/R1.5、Phase1 冻结目录、锁、原始输入和既有输出。

**工作时限/止损：**先用不超过 15 分钟做源码与环境最小预检；随后直接写算法纯函数和必要测试，力争在本轮 Codex 会话内完成一对真实影像输出，再做第二对。不得自动发起“环境修复 Phase 3.3”；若无法运行，停止并给出阻塞函数、错误信息、已产物，而不是用新长篇设计文档替代输出。

**成功的最小定义**：`00_01` 与 `00_10` 均得到有效校正 GeoTIFF，两个输入共同支持域的校正输出覆盖完整（100%），定量指标确实由保存的输出像元重新计算，而不是示例值或合成值。第一次成功运行不要求 MAE 改善。单组成功也要保存并明确另一组失败，不能把整体状态写成成功。

## 1. 论文、官方源码与复现身份

- 论文：Moghimi et al., *LIRRN: Location-Independent Relative Radiometric Normalization of Bitemporal Remote-Sensing Images*, **Sensors (2024)**，DOI `10.3390/s24072272`。
- 作者公开代码：<https://github.com/ArminMoghimi/LIRRN>；本计划核查主分支提交 **`38b930000795c12a7373a738e12a3035c6e164e1`**（2024-04-03）。Codex应固定使用这一提交，并记录其实际文件 SHA256；若源代码无法联网获得但已有受信任的本地拷贝，必须核对版本/内容，不能假定与核查版本相同。
- 必看：`Function/LIRRN.m`、`Function/LinearReg.m`、`main.m`。源码根目录含 `LICENSE`，文本标示 **GNU LGPL 2.1**；公开再分发衍生代码前按许可证要求保留来源与声明。
- **命名**：`LIRRN_2024_PYTHON_SOURCE_FAITHFUL_PORT`（独立 Python 重实现）。目前没有 MATLAB 参考数值输出，因此标记 `matlab_numeric_equivalence_verified=false`；不要写“MATLAB 逐像元完全一致/严格数值复现”。

### 1.1 两个必须忠实遵守的非直觉源码事实

1. `Function/LIRRN.m` 内有一个**局部函数** `SampleSelection(P_N, A, B, ID)`，实际运行会解析到它；同仓库另一个 `Function/SampleSelection.m` 接受两参数并调用 `pdist2`，**不是该主算法调用的函数**。Python 不能照搬错误的外部同名函数。
2. `Function/LIRRN.m` 内的 `ComputeSample` 虽然注释说分别按最大、最小、均值采样，但代码**连续三次调用完全相同的 `ComputeDistances`**；三组都选择“接近最大值的前 P_N 个候选”，之后按相同随机 `ID` 抽取。V1 **保留这三次重复**，不得自行改为 max/min/mean 三策略，也不得把它优化成另一种PIF算法。可在审计报告中指出该行为与注释不一致。

`main.m` 中 `thresh = multithresh(SubImg(:,:,1),3)` 的变量没有用于 `LIRRN()`，Python V1不应额外加入这一预处理；展示RGB的代码也不是 B9 单波段算法的一部分。

## 2. 数据契约、分组和冻结边界

### 2.1 真实输入

- 仅从 `freeze_20261013_v3/aligned_inputs/scene_000_B9_input.tif`、`scene_001_B9_input.tif`、`scene_010_B9_input.tif` 及各自 `scene_XXX_input_mask.tif` 读取；所有输入**只读**。
- **不可**使用 `stages/06_bagrn` 已校正影像、历史Task15/16的算法输出、Spectralmatch的工作副本或任何生成过辐射校正的影像。
- 冻结规范网格：EPSG:32650，14 m；对齐影像是 `float32`，原始波段是 uint16 DN，辐射物理单位目前为 `UNKNOWN`。输出必须保留地理参考，不能称为地表反射率校正精度。
- 13景/48正重叠/30无有效重叠是全局协议，但本轮仅两对诊断裁剪。先核查 `freeze_lock.json` 和 `code_manifest.json` 的内容SHA；不以 Git HEAD 是否等于历史锁定 HEAD 代替内容锁检查。
- 最新冻结代码应在已恢复的 Git 分支（最终恢复提交 `7252df68c2ec9335d25c33930bce366daf031379`）上。先看 `git status`；不得强制 `checkout/reset/stash` 用户的未提交工作。若要新建工作分支，可用 `feat/lirrn-python-b9-pair-smoke`，只新增文件。

### 2.2 两个裁剪：严格复用此前Spectralmatch的空间选择规则

对于每对 `pair_id∈{00_01,00_10}`：

1. 从冻结 `fixed_windows.csv` 读取**该对第一条已冻结 B2 窗口**（文件顺序固定），获取 `row_off,col_off,height,width`。
2. 以该 B2 窗口的中心为中心，取 `1024×1024`，在规范总网格边界以内做确定性平移/裁剪（与 `spectralmatch_phase2_worker.py` 的 `_run` 选择规则一致）。
3. 对两景**使用同一个绝对规范网格窗口**，读两景数值及独立冻结有效掩膜；输入像元不得再重采样。
4. 保存 `crop_spec.json`：每景ID、冻结窗口ID、全局行列偏移、GeoTIFF 仿射变换、原图 SHA256/冻结锁SHA、有效像元数、共同重叠像元数。
5. `00_01`此前同规则诊断的交集约为 **360,981** 像元；`00_10`约 **461,041** 像元。这两个是**历史交叉验证参考值**，应根据当前冻结数据重新计算。如不一致，停止该对并输出差异来源；不可硬写结果或通过改窗口强行匹配。

**方向固定**：scene `i` 作为 `RefImg`（参考景，像元不变），scene `j` 作为 `SubImg`（待校正景）。`00_01`即 `Ref=000, Sub=001`；`00_10`即 `Ref=000, Sub=010`。这与Spectralmatch允许两景同时变换不同，应注明它是固定参考的双时相算法，**比较质量指标时不改变公共评分域**。

### 2.3 训练样本域 ≠ 公共评分域

- **LIRRN原生样本选择**：在当前固定裁剪内，分别对参考景和待校正景的**各自冻结有效像元**计算亮度分层、取样和回归，不强制空间对应；这才符合 location-independent 的源码设计。两景冻结有效掩膜只在该数据加载与 NoData 处理中起作用，不作为人工PIF标签。
- **Benchmark评分**：只使用冻结 `M_i ∧ M_j` 的相同地理位置重叠像元。校正输出不能改变这一评分域。`REF_i`仍采用冻结原值，`OUT_j`采用校正值；与校正前的 `REF_i,INPUT_j` 计算完全相同的指标。
- **不得**在训练/评分前人为挑选“稳定地物”区域，不做人为ROI标签，不把结果说成变化真值。由于算法可能使用裁剪内像元拟合，裁剪指标属于**样本内的兼容性诊断**，不是独立的泛化测试。

## 3. 官方 LIRRN V1 算法的数学过程及可复现实质

对单波段 B9，令参考景数据为 `R(p)`，待校正景数据为 `S(p)`，各自有效掩膜为 `M_R, M_S`。

### 3.1 每景独立三等级亮度分层

官方每波段调用 `multithresh(nonzeros(image),2)` 后 `imquantize(image, thresholds)`。对有效且非零值做 3 类 Multi-Otsu，选择两个阈值 `t1<t2`，常见目标为最大化类间方差：

\[
(t_1,t_2)=\arg\max_{t_1<t_2}\sum_{c=1}^{3}\omega_c(\mu_c-\mu)^2.
\]

分层定义（阈值边界规则须与选定实现一致）：

\[
L(x)=\begin{cases}1&x\le t_1\\2&t_1<x\le t_2\\3&x>t_2.\end{cases}
\]

**实现约束**：既有Python环境能导入`scikit-image`时可参考其 `threshold_multiotsu(classes=3,nbins=256)`，但不能默认它与 MATLAB 的 `multithresh` 在直方图和并列最优阈值处理上逐数相同。为避免环境依赖反复阻塞，V1允许使用**独立、确定性的 NumPy 256-bin Multi-Otsu**实现；在 `method_metadata.json` 固定记录 `otsu_backend=NUMPY_256BIN_PORT_V1`、边界、取值范围和并列极值处理规则。不得根据两对跑分结果切换阈值实现。若已有 `skimage`，可将其用于单元对照，不得按得分高低二选一。

输入数据不应该隐式改为8位或uint16，不允许对冻结有效像元的原始DN/浮点值作显示拉伸再参加拟合。`main.m`在读取时用了 `abs(double(...))`；对DZ01V首先检查有效值是否均非负，如发现负值，应记录并阻塞该对，而不是悄悄abs后声称使用冻结原值。

### 3.2 按每一类“靠近该类最大值”取样（忠实源码）

对类别 `c`：

- `A_c = nonzeros(R · 1[L_R=c])`；`B_c = nonzeros(S · 1[L_S=c])`；按照MATLAB列主序 `order='F'` 提取；原代码不使用像元空间一一对应。
- 分别求最大值 `a_max, b_max`，按 `|a-a_max|` 和 `|b-b_max|` **升序**排序，取前 `P_N` 个候选。对普通非负数据，相当于各自数值从大到小排序。
- 在所有类别及两景之间复用**同一个**随机索引向量：

\[
ID=(k_1,\ldots,k_m),\quad k_l\in\{1,\ldots,P_N\},\quad m=\operatorname{round}(0.1P_N).
\]

从候选中按 `ID` **有放回**抽取 `m` 个参考和待校正值，组成匹配样本对 `(b_{k_l},a_{k_l})`。
- 源码 `ComputeSample` 把上面同一 `ComputeDistances` **重复3次**并拼接；因此默认 `P_N=1000` 时每个类别得到 `3×100=300` 对样本，三个类别合计 **900对**（可重复）。不能替换为“最大/最小/均值”混合抽样。
- `ID` 在**一次算法调用内生成一次**，全类别复用；V1固定记录 `numpy_rng_seed=20261014`，且报告中注明NumPy RNG不会生成与MATLAB `randi`相同的随机序列，不能声称跨语言逐样本等价。

**不足样本的处理**：三类中任意一景可用样本数 `<P_N` 时，先写 `INSUFFICIENT_STRATUM_SAMPLES`，不得越界、重抽类别或悄悄忽略该类。预先允许**仅一次**显式降档 `P_N=200`，标记 `PN_200_PREDECLARED_FALLBACK` 并独立输出；不得在已经看过校正评分后改P_N。若`P_N=200`仍不满足则停止该对。正式报告清楚区分两对参数是否相同；参数不同不得伪称完全相同配置。

注意：外部 `Function/SampleSelection.m` 的两参数 `pdist2` 匹配**不是**这里的抽样流程。第一版不要将其混入。

### 3.3 单变量线性回归

将三个类别抽得的配对样本拼成向量 `x`（待校正）与 `y`（参考）；官方 `LinearReg.m` 调用MATLAB `fitlm` 拟合带截距的一元线性模型，未启用RobustOpts。Python采用 `float64` 普通最小二乘：

\[
(\hat a,\hat b)=\arg\min_{a,b}\sum_{k=1}^{n}(y_k-a x_k-b)^2,
\qquad S^{out}(p)=\hat a\,S(p)+\hat b.
\]

`scale=a, offset=b`；仅在冻结待校正景有效像元应用；无效像元仍为NoData，不因回归变成有效。保存 `rank/condition_number`，`n>=3`，设计矩阵rank必须为2；否则明确 `REGRESSION_DEGENERATE`，不能捏造系数。

可计算辅助拟合统计：

\[
SSE=\sum_k(y_k-\hat y_k)^2,\qquad
RMSE_{fit}=\sqrt{SSE/(n-2)},\qquad
R^2_{adj}=1-\frac{SSE/(n-2)}{\sum_k(y_k-\bar y)^2/(n-1)}.
\]

这些是**拟合诊断**，与Unified Benchmark中的跨景 `RMSE_pair` 不是同一个量，字段必须分别命名。`R^2_adj`若分母为0则为`null`并记录状态。

写输出时保持所有合法的负值或大值，不得任意裁剪为0~65535以获取好看结果；按指定 `float32` 导出并检查溢出/NaN/Inf，像元值超范围则报告数值问题。最终仅在可视化时拉伸。

## 4. 必须实现的最小函数接口（不要另建大型框架）

放在新目录：`src/benchmark_v1_lirrn/`，不改 `src/benchmark_v1_unified/` 已锁定代码。Python 3.11，使用NumPy、Rasterio、现有Unified Benchmark计算函数；SciPy已由Benchmark依赖提供。采用`dataclass`记录输入/输出。

```python
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from affine import Affine

@dataclass(frozen=True)
class CropSpec:
    pair_id: str
    anchor_window_id: str
    row_off: int
    col_off: int
    height: int  # 1024
    width: int   # 1024

@dataclass(frozen=True)
class RasterCrop:
    values: np.ndarray   # (H,W), float64 calculation copy
    valid: np.ndarray    # (H,W), bool, frozen source mask
    transform: Affine
    crs: str
    scene_index: int

@dataclass(frozen=True)
class RegressionResult:
    scale: float
    offset: float
    fit_rmse: float
    adjusted_r2: float | None
    rank: int
    condition_number: float
    sample_count: int

# 1. Check actual lock/content, no modification.
def verify_frozen_b9(freeze_root: Path) -> dict: ...

# 2. Reproduce first-B2-anchor crop selection; no method-output-based selection.
def select_pair_crop(freeze_root: Path, pair_id: str, side_px: int = 1024) -> CropSpec: ...

# 3. Read frozen canonical B9 crop and corresponding mask; no resampling.
def read_frozen_crop(freeze_root: Path, scene_index: int, spec: CropSpec) -> RasterCrop: ...

# 4. Multi-Otsu 3-class labels only on valid/nonzero samples.
def classify_three_levels(values: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, dict]: ...

# 5. One fixed shared ID vector, generated once per paired call.
def draw_shared_ids(p_n: int, seed: int) -> np.ndarray: ...

# 6. Faithfully implement local SampleSelection / three duplicate max calls.
def select_lirrn_samples(
    ref: RasterCrop, sub: RasterCrop,
    ref_labels: np.ndarray, sub_labels: np.ndarray,
    shared_ids: np.ndarray, p_n: int
) -> tuple[np.ndarray, np.ndarray, dict]: ...
# Output tuple x_sub, y_ref, audit; vectors length 9*round(.1*p_n).

# 7. Plain OLS: x_sub -> y_ref, intercept included.
def fit_lirrn_linear(x_sub: np.ndarray, y_ref: np.ndarray) -> RegressionResult: ...

# 8. Mask-safe float32 correction without clipping.
def apply_lirrn_linear(sub: RasterCrop, model: RegressionResult) -> np.ndarray: ...

# 9. Cropped GeoTIFF + explicit validity mask, same CRS/affine as input crop.
def write_corrected_geotiff(dst: Path, crop: RasterCrop, corrected: np.ndarray) -> dict: ...

# 10. Both input and corrected metrics on exact frozen pair intersection.
def score_pair_crop(ref: RasterCrop, sub: RasterCrop, corrected: np.ndarray) -> dict: ...

# 11. One pair end-to-end; returns output paths, metrics, status, stage times.
def run_pair(freeze_root: Path, pair_id: str, out_dir: Path,
             seed: int = 20261014, p_n: int = 1000) -> dict: ...
```

约束：`read_frozen_crop`读取网格、CRS、仿射变换并强制两景相等。对NoData仅以冻结显式mask为权威；不要因校正像元等于0或为负就自动改变valid。所有`output_valid`必须等于冻结待校正景 `sub.valid`。输出栅格nodata建议使用安全的`float32`有限哨兵并写GDAL mask；记录该哨兵与有效值不冲突。任何结果写出前先检测全部有限性。

### 4.1 函数之间的数据流

`verify_frozen_b9 → select_pair_crop → read_frozen_crop ×2 → classify_three_levels ×2 → draw_shared_ids → select_lirrn_samples → fit_lirrn_linear → apply_lirrn_linear → write_corrected_geotiff → score_pair_crop → report`。

失败要返回明确阶段与错误；不允许捕获任意异常后生成“成功”的空结果。

## 5. 辐射指标的计算：仅调用统一冻结函数

**不得重写自己的MAE/RDD以制造第二套标准。** 使用：

- `src.benchmark_v1_unified.radiometric.compute_pair_direct_metrics`；
- `src.benchmark_v1_unified.radiometric.compute_pair_distribution_metrics`；
- 结构诊断`compute_sobel_structure`、`compute_range_retention`；
- 若存在可复用的冻结输出支持审计函数，直接调用；不修改其源码。

固定公共支持：

\[
\Omega=M_R\land M_S,\qquad N=|\Omega|.
\]

输入与校正都在同一 `Ω`：`before=(R[Ω],S[Ω])`，`after=(R[Ω],S_out[Ω])`。

\[
MAE=\frac1N\sum_{p\in\Omega}|R_p-S_p|,\quad
RMSE=\sqrt{\frac1N\sum_{p\in\Omega}(R_p-S_p)^2},
\]

\[
MAMD=|\overline R-\overline S|,\qquad
MSDD=|\sigma_R-\sigma_S|\quad(\text{ddof}=0),
\]

\[
RDD=W_1(\{R_p\}_{p\in\Omega},\{S_p\}_{p\in\Omega}).
\]

保存`BIAS`、`P95AE`及局部结构/动态范围（固定第一B2窗口）；报告绝对数值与前后差、相对变化百分比（输入值为0时相对变化为null），均标为 `UNKNOWN_PIXEL_UNIT`。**不允许称为地表反射率RMSE**。

必须在**重新打开实际写出的 GeoTIFF**后，重复读取和评分，验证与写出前的计算结果在指定浮点容差内一致；只有保存前内存数组得分不算完成。

若输出在`Ω`上有任一无效像元，`FULL_SUPPORT=false`，相关主评分设置`null`且保留`PARTIAL_OUTPUT`诊断，不能悄悄缩小样本数。评分只依赖输入冻结掩膜，不能用算法输出反向筛选。

正式的B1需要完整影像对的冻结重叠区，**本轮只有crop B1-like诊断**；B2指标仅在裁剪内实际包含的冻结窗口统计，不填装成全1,533个窗口；B3本轮不运行。不得执行或伪造实验A最终Mosaic质量。

## 6. 目录、真实输出与可追溯性

只在新增位置工作：

```text
src/benchmark_v1_lirrn/
    __init__.py
    official_port.py        # 3-class Otsu, shared random draw, sampling, OLS
    frozen_crop.py         # immutable input, same crop as Spectralmatch
    pair_runner.py         # output, audit, metrics, visualization
scripts/benchmark/run_lirrn_phase3.py
tests/benchmark_v1_lirrn/
    test_official_port.py
    test_crop_io.py
    test_pair_scoring.py
    test_real_entry_contract.py

# 数据输出不提交到GitHub
 data/output/benchmark_v1_unified/lirrn_phase3_20261014/
    run_manifest.json
    00_01/
        crop_spec.json
        reference_input_B9.tif           # masked, unmodified reference crop
        subject_input_B9.tif             # masked, unmodified subject crop
        subject_lirrn_corrected_B9.tif  # real corrected raster
        subject_output_mask.tif
        before_after_fixed_stretch.png
        pair_metrics.json
        regression_audit.json
        execution.log
    00_10/
        ...
    summary.csv
    summary.json
    phase3_result.md
```

输出每景参考/待校正身份、原始冻结SHA、裁剪位置、源版本、算法参数、随机种子、Otsu阈值及各类候选像元数、回归样本数/coef/rank、NoData定义、文件SHA256、是否真实运行、输出有效支持率、指标及执行时长。

若为了避免重复存储大量GeoTIFF，只允许用路径引用冻结的裁剪输入（不必复制），但**至少必须实际保存 `subject_lirrn_corrected_B9.tif`**和掩膜、预览及结果JSON。

视觉展示：固定显示拉伸`P02/P98`只根据两景**未校正输入的有效像元**确定，然后全部`before/after`共用；在无效位置显示明确掩膜颜色，不得让哨兵值进入拉伸分位数。禁止对保存的数值GeoTIFF执行图像拉伸。

### 6.1 最小JSON结果契约

```json
{
  "protocol_id": "DZ01V_UNIFIED_A_B_V1_20261013",
  "method_id": "LIRRN_2024_PYTHON_SOURCE_FAITHFUL_PORT",
  "method_kind": "PAIR_ONLY",
  "benchmark_scope": "1024_CROP_DIAGNOSTIC_ONLY",
  "formal_ranking_eligible": false,
  "pair_id": "00_01",
  "status": "SUCCESS_DIAGNOSTIC",
  "reference_scene": "scene_000",
  "subject_scene": "scene_001",
  "frozen_input_sha256": "<read from frozen manifest>",
  "crop_spec": {"row_off": 0, "col_off": 0, "height": 1024, "width": 1024},
  "implementation": {
    "upstream_commit": "38b930000795c12a7373a738e12a3035c6e164e1",
    "matlab_numeric_equivalence_verified": false,
    "three_identical_max_sampling_calls": true,
    "otsu_backend": "NUMPY_256BIN_PORT_V1",
    "p_n": 1000,
    "rng_seed": 20261014
  },
  "input_overlap_pixels": null,
  "output_overlap_pixels": null,
  "full_support": null,
  "regression": {"scale": null, "offset": null, "fit_rmse": null, "adjusted_r2": null},
  "metrics_before": {"mae": null, "rmse": null, "mamd": null, "msdd": null, "rdd": null},
  "metrics_after": {"mae": null, "rmse": null, "mamd": null, "msdd": null, "rdd": null},
  "output_files": {"corrected_tif": null, "mask_tif": null},
  "timing": {"total_wall_s": null, "algorithm_wall_s": null, "metrics_wall_s": null, "peak_rss_bytes": null, "rss_status": "NOT_MEASURED"}
}
```

**以上JSON只是字段示例，不能把其中的null/坐标0作为实际运行结果！** 实际提交中必须由真实输出填值；失败时结果`status`不得是`SUCCESS_DIAGNOSTIC`。`eligible`如果冻结checker返回true也仅表示结构/哈希合法，不得变成正式排名资格。

## 7. 可运行环境：采用已有Python，不建立新环境

第一条Codex命令只需确认解释器与必需基础库：

```powershell
.\.venv-registration\Scripts\python.exe -c "import sys, numpy, rasterio, scipy; print(sys.version); print(numpy.__version__, rasterio.__version__, scipy.__version__)"
```

不导入`osgeo`、不安装MATLAB、不创建conda/micromamba环境；三等级 Otsu 的V1实现只依赖NumPy，因而无须强制安装`scikit-image`。统计指标的W1调用现有SciPy。

监控：裁剪固定`1024²`且只准逐对运行，不因未装某个可选监控包而再构造安装长流程。若已有`psutil`，记录进程树RSS；若没有，设置 `rss_status=NOT_MEASURED`，但**不得声称已强制2GiB限制**。本轮不得扩到全景或13景。若使用带2GiB轮询限制的运行器，必须具备相应监控能力、失败关闭并记录是否OS硬限制。

如果 `numpy/rasterio/scipy` 任一核心依赖缺失，记录错误并停止，不自动替换成WSL/Conda长环境安装任务。

## 8. 最小测试清单（测试不能替代真实影像）

1. **源代码解释**：单元测试断言每类3次是**相同的最大值距离排序**；禁止使用外部`Function/SampleSelection.m`的`pdist2`。
2. **3类阈值**：可构造的三峰正值数据能得到2个递增阈值、3类非空；全常数、有效像元过少、阈值不可分时明确失败。
3. **共享ID与有放回抽样**：同一seed结果相同、换seed有可能不同、类与两景共享同一ID、`P_N=1000`默认样本数900；不能用每类独立随机序列。
4. **MATLAB列优先与重复值**：有效像元按`order='F'`展平；数值相同的排序结果确定，记录稳定排序策略。
5. **分层样本不足**：1000不足时返回明确状态；唯一预先声明的200降档可运行，不能偷偷修改配置。
6. **已知仿射真值回归**：人工构造`Ref=1.25*Sub+20`的样本，`scale/offset`在合理容差内恢复；检查OLS秩退化时拒绝。
7. **NoData保护**：无效值不参与阈值/回归，校正后无效仍无效；合法的负输出不能被截断或当作NoData。
8. **GeoTIFF地理参考**：CRS、crop仿射、尺寸、dtype、mask完全保留；拒绝输入两景变换不相等。
9. **公共评分支持**：Input与After的交集像元完全一致；若故意删除一个输出像元，结果必须`PARTIAL_OUTPUT`，不得仅用剩余像元算成功主分数。
10. **回写一致性**：保存输出再打开评分与内存评分接近；产物SHA可核对。
11. **非正式排名防护**：`method_kind=PAIR_ONLY`、`formal_ranking_eligible=false`；无法使用裁剪结果充当13景A/B正式结果。
12. **再运行一致性**：相同seed与输入，校正影像有效像元、参数和核心指标数值可重现（文件元数据/压缩字节不一定逐字节相等，应单独核对像元与hash）。

如果源MATLAB环境未来可用，可额外加入golden numerical test；**本轮不能将没有MATLAB作为阻塞条件**。

## 9. Codex执行顺序与实际命令

### P0（先行15分钟，失败早停）

1. `git status --short`，确认未覆盖用户其他工作，确定从包含38/38锁文件的工作树开始。
2. 验证`freeze_20261013_v3`内容锁和两对冻结B2窗口可以读取。
3. 运行上一节Python核心依赖import。记录官方源码提交与算法说明，确认本地已有源码或依据受核查版本准备独立实现。
4. 创建新目录与输出目录，**不要创建新评测框架**。

### P1（算法纯函数，先小数组测试）

按§4接口实现 3-level Otsu、共享索引、3次重复最大值候选、OLS、mask-safe应用；只运行对应的5–8个核心单元测试。若算法实现一直阻塞，应优先定位**哪一个函数**，而不是写数十页阶段报告。

### P2（立即运行第一对真实影像）

建议CLI：

```powershell
.\.venv-registration\Scripts\python.exe scripts\benchmark\run_lirrn_phase3.py --freeze data\output\benchmark_v1_unified\freeze_20261013_v3 --pair 00_01 --crop-size 1024 --pn 1000 --seed 20261014 --output-root data\output\benchmark_v1_unified\lirrn_phase3_20261014
```

- 首先尝试官方 `P_N=1000`；若事先统计证明个别亮度类不足1000，允许自动启动**唯一预先声明**的`P_N=200`新run_id（在报告中明确不是相同参数），不能覆盖原失败结果。
- 检查**实际GeoTIFF已写出**，用重新打开的像元计算Input/After的MAE、RMSE、MAMD、MSDD、RDD。
- 即使指标严重恶化，也要保存并标记成功运行；算法质量结论另外表述。

### P3（运行第二对真实影像）

```powershell
.\.venv-registration\Scripts\python.exe scripts\benchmark\run_lirrn_phase3.py --freeze data\output\benchmark_v1_unified\freeze_20261013_v3 --pair 00_10 --crop-size 1024 --pn 1000 --seed 20261014 --output-root data\output\benchmark_v1_unified\lirrn_phase3_20261014
```

注意每对使用独立子目录；再次执行相同run_id不允许覆盖旧产物，需显式产生唯一新ID。对于`00_10`，长达61天的差异可能包含真实变化，不能根据MAE下降宣称真实辐射恢复正确。

### P4（简短收尾，不重复造轮子）

- 执行所有新增测试、冻结锁只读审计、`compileall`、新增文件`git diff --check`；锁定历史文件存在原始格式空格时保留字节，不为了消除告警更改。
- 生成两对并排对比表、真实预览和输出SHA；**明确什么指标改善、什么恶化**。
- 只提交**新增源码/测试/文档**；不得把真实GeoTIFF、冻结输入、缓存、日志、临时环境推到公开GitHub。是否commit/push由用户确认；不要自动push。

## 10. 最终验收表（Codex必须填写真实数值）

| Pair | 状态 | P_N | valid overlap | output support | Input MAE | LIRRN MAE | Input RMSE | LIRRN RMSE | MAMD前→后 | MSDD前→后 | RDD前→后 | wall s | corrected TIFF |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|---:|---|
| 00_01 | 待运行 | 1000 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 |
| 00_10 | 待运行 | 1000 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 |

**判定**：

- `SUCCESS_DIAGNOSTIC`：至少该对真实裁剪的校正GeoTIFF已实际生成并可读，格网一致、完整支持、回读后指标可复算、参数/seed/源版本清楚；即使指标变差仍可为成功运行。
- `FAILED_INPUT_OR_DEPENDENCY`：冻结路径/掩膜或必要核心库不可用，记录首次阻塞。
- `FAILED_INSUFFICIENT_SAMPLES`：1000与预声明200均不足样本。
- `FAILED_NUMERICAL`：多阈值不可分、OLS秩退化、非有限校正结果。
- `PARTIAL_OUTPUT`：应覆盖支持域存在失效像元，结果不能标成完成。
- `NO_REAL_RUN`：只有函数测试、报告或合成输出；**不视为本任务成功**。

本轮不产生正式13景比较结论；完成两对后暂停，向研究者交付真实产物路径与比较表，并询问是否继续扩展成完整影像对试验。

## 11. 给Codex的最终约束提醒

**“先有真实结果，再谈进一步优化。”** 不开发人工ROI、不继续安装MATLAB/Miniforge、不修改Frozen Benchmark、不修Spectralmatch、不重做Harmoni-Planet、不为了好分数任意改原LIRRN三次重复采样逻辑。若不能运行，必须证明具体失败条件，保留所有已生成的有效产物，停止，而不是再申请新的长工程阶段。
