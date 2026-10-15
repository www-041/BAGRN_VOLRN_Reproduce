# Task 10C Core Repair and Scientific Rerun Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不改变当前冻结的B9五景实验主问题、matcher/RANSAC/Global算法定义和Task11接缝线范围的前提下，一次性修复配准公平性、Global完整性、BAGRN/VOLRN科学语义、Task10评价与provenance问题，并在通过严格科学gate后自动完成正式B9辐射实验重跑。

**Architecture:** 保持现有“4 matcher → shared affine RANSAC → Global-ready → MST/Translation-L2 → canonical warp/mosaic → BAGRN → VOLRN → audit”的架构。先用TDD修核心库和runner，再对冻结artifact做replay；真实数据重跑采用两级gate：先只跑主几何组合的BAGRN+VOLRN科学有效性gate，gate通过后才自动跑最终2种冻结geometry × 3种radiometric方法。所有结果均可追溯到配置hash、输入hash、Global transform hash和代码commit。

**Tech Stack:** Python, NumPy, SciPy, rasterio, OpenCV, PyTorch/Kornia, EfficientLoFTR, LightGlue+DISK, pytest, Git.

**Spec:** 本计划即Task10C实施规范；执行开始后复制到 `docs/superpowers/plans/2026-09-26-task10c-core-repair-and-rerun.md` 并纳入版本控制。

## Global Constraints

- Repository: `www-041/BAGRN_VOLRN_Reproduce`.
- Existing branch: `exp/2026-09-24-b9-five-scene-validation`.
- Starting GitHub commit to verify: `8bbf830e69f2ede66398ad226f103a12672075a3` (short `8bbf830`).
- 使用现有repo/worktree；**不要创建新repo或新worktree**。
- **禁止push**。允许在当前分支频繁小commit；不要merge到`main`。
- 保留用户未跟踪文件和已有实验产物；不得用`git clean -fd`、不得覆盖非空结果目录。
- 本轮不改变：B9五景集合、band=B9、`match_max_side=1024`、shared affine RANSAC阈值、matcher接受规则、MST定义、Equal-L2 Translation Global目标、canonical grid、weighted-feather公式。
- 本轮不做：Task11 seamline创新、graph-cut、Poisson/multiband blending、新matcher、matcher特定阈值调参、BAGRN/VOLRN结果导向调参。
- 4个matcher仍为：`sift`, `loftr`, `efficient_loftr`, `lightglue_disk`。
- 最终radiometric geometry只保留两组：主实验 `efficient_loftr + translation_l2`；基线 `sift + mst`。
- Task10C严格VOLRN baseline固定：`lambda=0.5`, `block_size_pixels=400`, `rho=1.0`, `max_iter=200`上限；停止必须由明确的ADMM residual规则决定，不能仅靠迭代数。
- Radiometric control固定为实验scene index `0`，作为**预先声明的radiometric reference**，不得根据结果优劣自动选择；在metadata中明确它独立于geometry reference。
- 科学输出中的归一化中间结果与正式radiometric mosaic统一使用`float32`；若以后需要UInt16产品，必须走独立export步骤，本计划不做该产品化转换。
- 任何“PASS”必须有对应验证证据。`COMPLETED_NONCONVERGED`、`FAILED_SCIENCE_GATE`等不得伪装成PASS。
- 遇到普通代码错误、测试失败、环境可恢复问题时，Codex自行诊断、修复并继续，不询问用户。
- 只有以下情况允许硬停：①论文/项目源材料不足以唯一确定一个会改变科学定义的公式；②冻结数据/权重/依赖不可访问且无法从现有环境恢复；③继续执行会改变冻结协议；④输入或provenance损坏且无法可信恢复。
- 硬停时仍要提交已完成且独立正确的修复，写出`TASK10C_BLOCKED.md`，列出阻塞点、已完成项、未运行实验，**不得猜测或自动改科学定义**。
- 每个任务遵循TDD：先新增失败测试，确认FAIL，再最小实现，确认PASS，再commit。
- 所有真实数据实验必须顺序运行，避免8GB GPU并发OOM。

## Autonomous Execution Policy

执行本计划时不要逐步等待用户确认。按任务顺序自动完成，持续维护 `data/output/b9_five_scene_validation/task10c_execution_ledger.json`，每个task记录：开始/结束时间、Git HEAD、命令、exit code、产物路径、科学状态、异常与处理方式。

允许自动恢复：
- 非空partial output：重命名为 `_incomplete_<timestamp>` 后重跑；禁止静默覆盖。
- GPU OOM：先释放模型/缓存并重试同一冻结设置一次；不得自动降低分辨率、改precision、减match数量或换模型。
- 单个测试暴露真实bug：修bug、补测试、继续。
- 单个matcher真实不可用：记录依赖错误；若是LoFTR修复验证必需，则硬停真实实验阶段，但先完成不依赖该matcher的核心代码修复。

禁止自动恢复：
- 改`lambda/block_size/RANSAC/matcher confidence threshold`来“让结果更好”。
- 跳过未收敛状态并继续宣称VOLRN有效。
- 由旧summary反推或伪造inlier点。
- 用视觉效果替代科学gate。

## Review Focus

1. **VOLRN未收敛**：必须保留最后有限解供诊断，但正式science status不能PASS；不得再强制替换成identity后悄悄继续。
2. **无效像素/NoData**：归一化后的合法0值、负值、小数不能被旧DN nodata规则误删；统计mask必须来自finite/显式valid mask。
3. **LoFTR公平性**：四matcher进入shared RANSAC前必须执行同一valid-mask contract；修复后必须做旧/新replay并只重跑受影响下游。
4. **冻结协议与resume**：参数、source config、canonical grid、Global transforms或代码commit任一provenance变化时，旧结果不得被resume成PASS。
5. **论文指标语义**：ADM/ADSD/CD/GL/RDOA/Ave只有在与项目中的论文原文/既有可信实现核对后才能标记为paper metric；若Eq.(38)等定义不能从项目源材料确认，禁止猜测，真实科学实验阶段硬停并记录。

---

## File Structure / Ownership Map

优先修改现有文件，不做无关重构：

- `src/registration_benchmark/models.py`：统一matcher输出契约与shared valid-match过滤入口。
- `src/registration_benchmark/matchers/{sift,loftr,efficient_loftr,lightglue_disk}.py`：matcher adapter；只做统一contract、session/model复用和计时，不改算法阈值。
- `src/registration_benchmark/geometry.py`：shared affine RANSAC退化/失败structured status。
- `src/multiscene_sift/pairwise.py`：pair执行、per-pair GPU peak、session注入。
- `src/multiscene_sift/b9_runner.py`：显式scene count、geometry bundle provenance、matcher session生命周期。
- `src/multiscene_sift/geometry_artifacts.py`：NPZ/sidecar内容hash与验证。
- `src/multiscene_sift/global_registration.py`：MST/global transform hard-fail，不再identity fallback。
- `src/multiscene_sift/global_ready*.py`：canonical pair key、replay/provenance一致性。
- `src/multiscene_sift/global_geometric_adjustment.py`：metric frame命名、legacy B14参数化。
- `src/multiscene_sift/global_comparison.py`：world metric count/value对齐。
- `src/bagrn.py`：无有效overlap pair处理和明确有效像素权重语义。
- `src/volrn.py`：ADMM诊断、收敛状态、最后有限解、数值缩放语义、valid mask。
- `src/metrics.py`：paper metrics核对、GL边界mask、诊断命名。
- `src/mosaic.py`：radiometric scientific dtype policy；本轮不改`narrow_feather`算法。
- `src/multiscene_sift/radiometric_protocol.py`：Task10C冻结配置、hash/provenance、science status规则。
- `src/multiscene_sift/radiometric_runner.py`：paper metrics+diagnostics、histogram修复、float32输出、VOLRN状态传播。
- `src/multiscene_sift/radiometric_audit.py`：只基于有效PASS科学行生成结论；非收敛单独展示。
- `scripts/run_b9_registration.py`：保持CLI稳定，只接入session/contract后的runner。
- `scripts/run_b9_radiometric.py`, `scripts/run_b9_radiometric_batch.py`, `scripts/audit_b9_radiometric.py`：strict protocol、resume、gate、batch执行。
- `tests/`：为每个修复增加最小科学不变量测试。

不要把`src/mosaic.py`里现有`narrow_feather`升级为Task11正式算法；只允许为避免radiometric dtype损坏做局部修复。

---

### Task 0: Preflight, Baseline, and Execution Ledger

**Files:**
- Create: `docs/superpowers/plans/2026-09-26-task10c-core-repair-and-rerun.md`
- Create: `data/output/b9_five_scene_validation/task10c_execution_ledger.json`
- Create if needed: `docs/audits/2026-09-26-task10c-preflight.md`

**Interfaces:**
- Consumes: current Git checkout, existing `.venv-registration`, frozen B9 configs/artifacts.
- Produces: immutable baseline identity and autonomous execution ledger used by all later tasks.

- [ ] **Step 1: Verify repository identity and branch without modifying files**

Run:
```bash
git rev-parse --show-toplevel
git remote -v
git branch --show-current
git rev-parse HEAD
git status --short
```

Expected:
- branch is `exp/2026-09-24-b9-five-scene-validation`;
- remote points to `www-041/BAGRN_VOLRN_Reproduce`;
- HEAD descends from `8bbf830` (HEAD may be newer if local commits already exist);
- no destructive cleanup is required.

If HEAD is not descended from `8bbf830`, hard stop before edits.

- [ ] **Step 2: Copy this plan into repo and create the execution ledger**

Ledger minimum schema:
`schema_version`, `baseline_commit`, `current_branch`, `frozen_protocol`, `tasks`, `real_experiments`, `push_performed=false`.

- [ ] **Step 3: Record environment and dependency versions**

Run at least:
```bash
python --version
python -c "import numpy,scipy,rasterio,cv2; print(numpy.__version__, scipy.__version__, rasterio.__version__, cv2.__version__)"
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

Also record GPU name and VRAM if CUDA is available.

- [ ] **Step 4: Run baseline focused and full tests before edits**

Run:
```bash
python -m pytest -q
```

Expected baseline from user report: `767 passed, 4 skipped, 0 failed`.
If different, record exact delta. Do not assume user-reported status supersedes local evidence.

- [ ] **Step 5: Commit only plan/ledger/preflight files**

```bash
git add docs/superpowers/plans/2026-09-26-task10c-core-repair-and-rerun.md docs/audits/2026-09-26-task10c-preflight.md
git commit -m "docs: freeze Task10C repair protocol"
```

Do not add large generated experiment outputs unless repository convention already tracks the specific summary artifacts.

---

### Task 1: Shared Matcher Valid-Mask Contract and LoFTR Fairness Fix

**Files:**
- Modify: `src/registration_benchmark/models.py`
- Modify: `src/registration_benchmark/matchers/sift.py`
- Modify: `src/registration_benchmark/matchers/loftr.py`
- Modify: `src/registration_benchmark/matchers/efficient_loftr.py`
- Modify: `src/registration_benchmark/matchers/lightglue_disk.py`
- Test: add focused tests under `tests/registration_benchmark/` using existing naming conventions.

**Interfaces:**
- Consumes: match-view coordinates and `view.ref_valid/view.tgt_valid`.
- Produces: one shared helper, e.g. `filter_view_matches_by_valid_mask(...) -> (ref_xy, tgt_xy, confidence, keep_mask)`, used by all four adapters before `make_matchset_from_view()`.

- [ ] **Step 1: Write failing tests**

Tests must prove:
- a coordinate landing on invalid ref or tgt pixel is removed;
- out-of-bounds coordinates are removed;
- valid coordinates are unchanged;
- LoFTR now obeys exactly the same helper;
- SIFT/EfficientLoFTR/LightGlue output does not change on all-valid masks.

- [ ] **Step 2: Run new tests and confirm LoFTR-specific case fails on current code**

- [ ] **Step 3: Implement the shared helper and route all four adapters through it**

Do not change confidence thresholds, match caps, model weights, RANSAC threshold, or coordinate conversion.

- [ ] **Step 4: Run matcher contract + coordinate inversion tests**

- [ ] **Step 5: Commit**

```bash
git add src/registration_benchmark/models.py src/registration_benchmark/matchers tests/registration_benchmark
git commit -m "fix: unify matcher valid-mask filtering"
```

---

### Task 2: Matcher Session Reuse, Timing Semantics, and Per-Pair GPU Peak

**Files:**
- Modify: `src/registration_benchmark/matchers/loftr.py`
- Modify: `src/registration_benchmark/matchers/efficient_loftr.py`
- Modify: `src/registration_benchmark/matchers/lightglue_disk.py`
- Modify: `src/multiscene_sift/pairwise.py`
- Modify: `src/multiscene_sift/b9_runner.py`
- Test: matcher/session tests under `tests/registration_benchmark/` and `tests/multiscene_sift/`.

**Interfaces:**
- Consumes: one matcher name/device for an entire multi-pair run.
- Produces: reusable matcher session with one-time model initialization and per-pair inference, while keeping match coordinates/results algorithmically equivalent.

- [ ] **Step 1: Write failing session lifecycle tests**

Assert:
- model factory is called once for two or more pairs;
- per-pair call uses the same eval model;
- session close/release is safe;
- SIFT remains stateless;
- timing contains separate `model_init_runtime_sec` and `pair_inference_runtime_sec`.

- [ ] **Step 2: Add GPU peak test using a mocked CUDA interface**

Each pair must call `reset_peak_memory_stats()` before inference and read peak after that pair, so reported `peak_gpu_memory_mb` is pair-local rather than process-history peak.

- [ ] **Step 3: Implement minimal session abstraction**

Do not alter model type, precision, max keypoints, confidence threshold, resize policy, pretrained weights, or output coordinates.

- [ ] **Step 4: Run deterministic/replay unit tests**

On fixed synthetic/mock matcher output, cached and uncached paths must be identical.

- [ ] **Step 5: Commit**

```bash
git add src/registration_benchmark/matchers src/multiscene_sift/pairwise.py src/multiscene_sift/b9_runner.py tests
git commit -m "perf: reuse matcher models across pair runs"
```

---

### Task 3: Geometry and Global Hard-Fail Invariants

**Files:**
- Modify: `src/registration_benchmark/geometry.py`
- Modify: `src/multiscene_sift/global_registration.py`
- Modify: `src/multiscene_sift/b9_runner.py`
- Test: geometry/global focused tests.

**Interfaces:**
- Consumes: shared RANSAC outputs, accepted graph, MST edges.
- Produces: explicit failure statuses/errors instead of silent identity transforms.

- [ ] **Step 1: Write failing RANSAC-degeneracy tests**

Cover insufficient/collinear/invalid affine point sets. Expected result is a structured geometry failure (`INVALID_GEOMETRY` or existing equivalent), never an exception leak and never an identity model marked OK.

- [ ] **Step 2: Write failing Global integrity tests**

Assert:
- missing accepted pair for a required MST edge raises a hard error;
- scene without a composed Global transform raises a hard error;
- `build_accepted_graph` receives explicit `n_scenes` so a fully isolated scene cannot disappear from connectivity accounting.

- [ ] **Step 3: Implement minimal invariant changes**

Remove identity fallback branches used for missing MST pair/scene transform. Preserve legitimate identity only for the chosen reference scene.

- [ ] **Step 4: Run Global-focused regression suite**

- [ ] **Step 5: Commit**

```bash
git add src/registration_benchmark/geometry.py src/multiscene_sift/global_registration.py src/multiscene_sift/b9_runner.py tests
git commit -m "fix: hard fail invalid global geometry"
```

---

### Task 4: Geometry Artifact Integrity, Canonical Pair Keys, and Metric Provenance

**Files:**
- Modify: `src/multiscene_sift/geometry_artifacts.py`
- Modify: `src/multiscene_sift/global_ready.py`
- Modify: `src/multiscene_sift/global_ready_validation.py`
- Modify: `src/multiscene_sift/global_geometric_adjustment.py`
- Modify: `src/multiscene_sift/global_comparison.py`
- Test: related artifact/replay/metric tests.

**Interfaces:**
- Consumes: geometry NPZ+sidecars, replay summaries, Global residuals.
- Produces: content-hashed bundles, canonical `(min(i,j), max(i,j))` pair identity, correctly aligned world-metric aggregation and truthful metric-frame metadata.

- [ ] **Step 1: Write failing artifact hash tests**

Save a geometry bundle, mutate NPZ bytes/array, then require validation to fail on SHA256 mismatch.

- [ ] **Step 2: Write failing canonical-pair replay test**

Old/new pair direction reversal with identical geometry must map to the same canonical edge identity for comparison; coordinate orientation still remains explicit in the sidecar and must not be silently inverted.

- [ ] **Step 3: Write world-metric alignment test**

If one edge has missing world RMSE/P95, its `n_points` must be filtered together with its metric value; no positional slicing such as `counts[:len(rmse)]` is allowed.

- [ ] **Step 4: Write metric-frame/legacy band test**

`summary["metric_frame"]` must reflect the actual supplied global method rather than always say `mst_global_frame`. Any legacy loader that hard-codes `B14` must accept an explicit band or be clearly isolated/deprecated so B9 cannot silently inherit B14 metadata.

- [ ] **Step 5: Implement and run focused tests**

- [ ] **Step 6: Commit**

```bash
git add src/multiscene_sift/geometry_artifacts.py src/multiscene_sift/global_ready.py src/multiscene_sift/global_ready_validation.py src/multiscene_sift/global_geometric_adjustment.py src/multiscene_sift/global_comparison.py tests
git commit -m "fix: strengthen geometry artifact provenance"
```

---

### Task 5: BAGRN Valid-Overlap Semantics

**Files:**
- Modify: `src/bagrn.py`
- Test: `tests/test_bagrn.py` and/or `tests/test_cloud_aware_bagrn.py` following repository conventions.

**Interfaces:**
- Consumes: arrays, nodata/finite/cloud masks, geometric overlap records.
- Produces: BAGRN equations built only from statistically valid overlap observations.

- [ ] **Step 1: Write failing tests for empty-valid overlap**

A geometric overlap whose one side has zero valid pixels in a band must not be converted into `(mean=0,std=0)` and fed into WLS as if it were a real observation.

- [ ] **Step 2: Write disconnected-after-filtering test**

If removing invalid statistical overlaps disconnects the radiometric network for a band, BAGRN must return/raise an explicit scientific failure, not solve a misleading system.

- [ ] **Step 3: Freeze overlap-weight semantics**

For the strict paper baseline, keep the paper/project-defined geometric overlap-area weighting unless the bundled paper source explicitly states another rule. Record both geometric pixel count and clear-valid count for audit, but do not silently switch weighting based on outcome quality.

- [ ] **Step 4: Implement minimal changes and run BAGRN tests**

- [ ] **Step 5: Commit**

```bash
git add src/bagrn.py tests
git commit -m "fix: exclude invalid BAGRN overlap observations"
```

---

### Task 6: VOLRN Solver Diagnostics and Nonconvergence Semantics

**Files:**
- Modify: `src/volrn.py`
- Test: `tests/test_volrn.py`, `tests/test_cloud_aware_volrn.py`, and new focused solver tests.

**Interfaces:**
- Consumes: VOLRN sparse system and frozen solver parameters.
- Produces: final finite iterate plus explicit solver diagnostics per band, without replacing nonconverged solutions by identity.

- [ ] **Step 1: Write failing nonconvergence regression test**

Construct a deterministic system that reaches `max_iter` before convergence. Assert:
- returned coefficients are the last finite iterate, not forced `(a=1,b=0)`;
- `converged=False` is propagated;
- diagnostics contain `iterations`, `primal_residual`, `dual_residual`, `objective`, `x_relative_change`, and CG status/history summary.

- [ ] **Step 2: Write ADMM convergence-rule test**

Convergence requires explicit primal and dual residual thresholds (standard absolute+relative scaling is preferred) and finite solver state. `x_diff < tol` alone must not declare convergence.

Use the existing public `tol=1e-4` as the relative tolerance unless the repository already separates `abs_tol`/`rel_tol`; if splitting is necessary, freeze both in Task10C config and test them.

- [ ] **Step 3: Write CG failure propagation test**

Persistent nonzero CG status must appear in diagnostics and prevent `science_pass=True`; do not merely print a warning.

- [ ] **Step 4: Audit and fix internal scaling**

Locate any 1%-99% or other global/common rescaling applied before solving. The implementation must be mathematically equivalent to the documented objective:
`0.5 ||B x||_2^2 + lambda ||A x - b||_1`.

If scaling changes one objective term relative to the other, either:
- remove it for the strict baseline, or
- transform `B/A/b/lambda` analytically so the minimizer is invariant.

Add a synthetic test showing scaled and unscaled formulations return equivalent coefficients within tolerance.

- [ ] **Step 5: Run solver tests and commit**

```bash
git add src/volrn.py tests
git commit -m "fix: make VOLRN convergence scientifically explicit"
```

---

### Task 7: VOLRN Valid-Mask / NoData Semantics

**Files:**
- Modify: `src/volrn.py`
- Modify if needed: `src/multiscene_sift/radiometric_runner.py`
- Test: VOLRN/radiometric runner tests.

**Interfaces:**
- Consumes: registered float arrays and explicit valid/cloud masks.
- Produces: normalized float arrays where valid finite values remain valid regardless of whether their DN equals the source nodata sentinel.

- [ ] **Step 1: Write failing tests**

Cover:
- source nodata=0 but normalization legitimately maps a valid pixel to 0;
- valid normalized negative value;
- fractional float value;
- NaN/Inf remains invalid;
- cloud pixel excluded from estimation but transformed according to existing project semantics.

- [ ] **Step 2: Implement explicit valid-mask flow**

After geometric registration, use finite/registered valid masks as scientific validity authority. Do not reclassify normalized pixels by comparing transformed DN to the original source nodata value.

- [ ] **Step 3: Run cloud-aware and synthetic tests**

- [ ] **Step 4: Commit**

```bash
git add src/volrn.py src/multiscene_sift/radiometric_runner.py tests
git commit -m "fix: preserve normalized valid pixels"
```

---

### Task 8: Paper Metric Verification, GL Boundary Safety, and Histogram Bug Fix

**Files:**
- Modify: `src/metrics.py`
- Modify: `src/multiscene_sift/radiometric_runner.py`
- Test: metrics and radiometric runner tests.

**Interfaces:**
- Consumes: registered-before arrays, normalized-after arrays, overlaps, valid masks.
- Produces: six paper metrics plus clearly separated additional diagnostics.

- [ ] **Step 1: Verify formulas from project source material before changing definitions**

Search repository/project docs for the paper equations used by `src/metrics.py`, especially Eq.(38) CD and Eq.(39) GL. Record exact source path and equation interpretation in `docs/audits/2026-09-26-task10c-metric-formula-audit.md`.

If the source material cannot uniquely verify a formula that would require changing the current implementation, **do not guess**. Mark the paper metric as unverified and hard stop before Task 12 real science gate, while still completing all code changes that do not depend on the unknown formula.

- [ ] **Step 2: Write histogram regression test**

A known gain/offset applied only to `after` must change histogram diagnostics. This must fail on the current bug where histogram data are read from `before`.

- [ ] **Step 3: Fix histogram evaluator to use `after`**

Keep histogram TV/mean/std diagnostics separate from paper CD unless the formula audit proves they are identical.

- [ ] **Step 4: Write GL NoData-boundary test**

A sharp valid→NoData boundary must not create artificial GL on neighboring valid pixels. Erode/guard the gradient stencil around invalid pixels before aggregating orientation differences.

- [ ] **Step 5: Ensure Task10 can emit all six paper metrics**

For each method record:
`ADM, ADSD, CD, GL, RDOA, Ave` plus definitions/units/direction and the formula-audit source.

- [ ] **Step 6: Run metrics tests and commit**

```bash
git add src/metrics.py src/multiscene_sift/radiometric_runner.py docs/audits/2026-09-26-task10c-metric-formula-audit.md tests
git commit -m "fix: correct radiometric evaluation semantics"
```

---

### Task 9: Strict Task10C Protocol, Provenance, Resume, and Status Model

**Files:**
- Modify: `src/multiscene_sift/radiometric_protocol.py`
- Modify: `src/multiscene_sift/radiometric_runner.py`
- Modify: `scripts/run_b9_radiometric.py`
- Modify: `scripts/run_b9_radiometric_batch.py`
- Modify: `scripts/audit_b9_radiometric.py`
- Test: Task10 protocol/batch/resume tests.

**Interfaces:**
- Consumes: one frozen Task10C protocol and exact geometry artifacts.
- Produces: deterministic run identity and truthful status.

- [ ] **Step 1: Write failing frozen-parameter tests**

The strict protocol must reject runtime changes to:
- `lambda=0.5`
- `block_size_pixels=400`
- `rho=1.0`
- `max_iter=200`
- tolerance values
- `radiometric_control_idx=0`
- source config, canonical grid, geometry run identity.

CLI may expose `--protocol`, but strict-run scientific parameters must come from the protocol rather than free-form overrides.

- [ ] **Step 2: Write resume/provenance tests**

Resume must reject an old output if any of these differ:
- source config SHA256;
- canonical output grid SHA256;
- `global_transforms.json` SHA256;
- radiometric protocol SHA256;
- method;
- geometry run;
- code commit/protocol schema where repository conventions require it.

- [ ] **Step 3: Write retry-success error cleanup test**

When a previously failed row succeeds, stale `error` must be removed (`pop`) rather than coexisting with PASS.

- [ ] **Step 4: Introduce explicit science statuses**

Minimum statuses:
- `PASS_CONVERGED`
- `PASS_RAW`
- `PASS_BAGRN`
- `COMPLETED_NONCONVERGED`
- `FAILED_INPUT`
- `FAILED_PROVENANCE`
- `FAILED_SCIENCE_GATE`
- `FAILED_RUNTIME`

A generic `PASS` may be retained only as backward-compatible display alias, never as the sole scientific state.

- [ ] **Step 5: Freeze radiometric control metadata**

Record: `radiometric_control_idx=0`, scene ID, and rationale: “predeclared reference for reproducibility; not selected from outcome metrics; independent of geometry reference.”

- [ ] **Step 6: Commit**

```bash
git add src/multiscene_sift/radiometric_protocol.py src/multiscene_sift/radiometric_runner.py scripts/run_b9_radiometric.py scripts/run_b9_radiometric_batch.py scripts/audit_b9_radiometric.py tests
git commit -m "feat: freeze strict Task10C protocol"
```

---

### Task 10: Scientific Mosaic Dtype Policy

**Files:**
- Modify: `src/mosaic.py`
- Modify: `src/multiscene_sift/radiometric_runner.py`
- Test: `tests/test_mosaic.py` plus radiometric integration test.

**Interfaces:**
- Consumes: normalized float arrays.
- Produces: `float32` scientific mosaics without silent uint16 quantization/wrap/clipping.

- [ ] **Step 1: Write failing float preservation tests**

Input must include negative, fractional, and >65535 normalized values. Scientific output must remain finite `float32` and preserve values within resampling tolerance.

- [ ] **Step 2: Add explicit output dtype policy**

Do not let `create_mosaic()` infer scientific radiometric dtype solely from `arrays[0].dtype`/nodata. Preserve existing geometry-only weighted-mosaic contract for raw UInt16 runs unless caller requests scientific float output.

- [ ] **Step 3: Confirm no change to weighted feather formula**

Existing fixed weighted-feather tests must remain bitwise/numerically equivalent for their old inputs.

- [ ] **Step 4: Commit**

```bash
git add src/mosaic.py src/multiscene_sift/radiometric_runner.py tests/test_mosaic.py tests
git commit -m "fix: preserve float radiometric mosaics"
```

---

### Task 11: Task10C Scientific Invariant Test Suite

**Files:**
- Create/modify focused tests under `tests/`.
- Optional create: `tests/test_task10c_science_invariants.py` if repository conventions allow a consolidated integration file.

**Interfaces:**
- Consumes: repaired BAGRN/VOLRN/metrics/protocol code.
- Produces: synthetic proof that the repaired system cannot silently reproduce the old failure modes.

- [ ] **Step 1: Add synthetic local-radiometry case**

Create at least 2–3 overlapping scenes with a known spatially varying local mismatch that BAGRN alone cannot fully eliminate but VOLRN can model.

Required assertions:
- BAGRN coefficients are finite;
- VOLRN produces at least one non-identity local coefficient where mismatch exists;
- converged run reports `converged=True`;
- chosen local discrepancy metric improves from BAGRN to BAGRN+VOLRN;
- no claim relies on mosaic appearance.

- [ ] **Step 2: Add nonconverged science-status case**

Force too-small iteration budget and assert result is `COMPLETED_NONCONVERGED`, never scientific PASS.

- [ ] **Step 3: Add Task10 six-metric schema test**

RAW/BAGRN/BAGRN_VOLRN summaries all expose the six paper metric keys plus additional diagnostic namespace.

- [ ] **Step 4: Add strict protocol mutation tests**

Mutation of one parameter or hash must be rejected.

- [ ] **Step 5: Run all focused tests**

Suggested:
```bash
python -m pytest tests/test_bagrn.py tests/test_volrn.py tests/test_cloud_aware_bagrn.py tests/test_cloud_aware_volrn.py tests/test_cloud_aware_metrics.py tests/test_mosaic.py tests/multiscene_sift -q
```

Adjust paths only to existing repository layout; record exact command/result in ledger.

- [ ] **Step 6: Commit**

```bash
git add tests
git commit -m "test: lock Task10C scientific invariants"
```

---

### Task 12: LoFTR Replay Gate and Conditional Downstream Rerun

**Files / Outputs:**
- Use existing frozen config: `data/output/b9_five_scene_validation/04_frozen_five_scene_config_1024.json`.
- New output namespace: `data/output/b9_five_scene_validation/task10c_loftr_replay/`.
- Do not overwrite historical `matcher_runs_1024_globalready/loftr` or historical Global/mosaic outputs.

**Interfaces:**
- Consumes: repaired LoFTR valid-mask pipeline and frozen B9 config.
- Produces: old/new pair and graph comparison, with conditional downstream rerun only if geometry changed.

- [ ] **Step 1: Run only LoFTR five-scene matcher under the unchanged frozen protocol**

Use the same B9 scenes, `match_max_side=1024`, RANSAC threshold, seed, device policy and match parameters.

- [ ] **Step 2: Compare old vs repaired LoFTR**

For each of 10 edges record:
- raw matches;
- post-valid-mask matches;
- RANSAC inliers;
- RMSE/P95/coverage;
- accepted/rejected status;
- graph connectivity;
- geometry bundle hashes.

- [ ] **Step 3: Conditional action**

If accepted graph and inlier point bundles are byte/semantic equivalent after the intended valid-mask change, record `LOFTR_DOWNSTREAM_RERUN_NOT_REQUIRED`.

If any accepted edge, inlier set, affine transform or Global-ready geometry changes, rerun **only**:
- LoFTR MST;
- LoFTR Translation-L2;
- LoFTR weighted-feather mosaic audit required to keep downstream artifacts consistent.

Do not rerun SIFT/EfficientLoFTR/LightGlue merely because LoFTR changed.

- [ ] **Step 4: Verify no threshold was changed to restore replay**

- [ ] **Step 5: Commit only code/report summaries if repository convention tracks them**

Suggested commit:
```bash
git commit -m "audit: replay LoFTR after valid-mask fix"
```

---

### Task 13: Main-Geometry VOLRN Science Gate

**Outputs:**
- New root: `data/output/b9_five_scene_validation/task10c_radiometric_gate/`.
- Geometry: `efficient_loftr + translation_l2` only.
- Radiometric method: first BAGRN, then BAGRN+VOLRN under strict Task10C protocol.

**Interfaces:**
- Consumes: repaired code, strict protocol, existing/fresh valid EfficientLoFTR Translation-L2 transforms.
- Produces: go/no-go decision for the final 2×3 formal experiment.

- [ ] **Step 1: Pre-gate provenance validation**

Verify source config, five scene IDs, B9 band, canonical grid, Global transform hash, pixel size 14m, protocol hash, and code HEAD.

- [ ] **Step 2: Run BAGRN baseline once**

Record six paper metrics, diagnostics, theta parameters, runtime and output hashes.

- [ ] **Step 3: Run BAGRN+VOLRN once with strict baseline**

Frozen values:
```text
lambda = 0.5
block_size_pixels = 400
rho = 1.0
max_iter = 200
tol = frozen protocol value (default 1e-4 unless Task 6 introduced separate abs/rel tolerances)
radiometric_control_idx = 0
```

- [ ] **Step 4: Apply science gate automatically**

Gate passes only if all are true:
1. every processed band has finite solver state;
2. VOLRN reports convergence by primal+dual residual criteria before or at max_iter;
3. no persistent CG failure;
4. at least one valid block coefficient differs from identity by more than a numerical epsilon documented in the audit (epsilon is for detecting exact identity fallback, not an optimization target);
5. BAGRN+VOLRN output differs measurably from BAGRN on valid pixels;
6. at least one predeclared local-radiometry discrepancy metric improves, and no metric is fabricated from visual inspection;
7. paper metric and diagnostic schemas are complete and provenance-valid.

Do **not** require every metric to improve. Do **not** tune parameters if the gate fails.

- [ ] **Step 5: On gate failure**

Write `task10c_radiometric_gate/GATE_FAILED.md` with solver traces, coefficients, metrics and reason. Stop before final real experiment. Do not try alternate lambda/block sizes in this plan.

- [ ] **Step 6: On gate success**

Write `task10c_radiometric_gate/GATE_PASS.json` and continue automatically to Task 14.

---

### Task 14: Final Formal B9 2-Geometry × 3-Radiometric Rerun

**Outputs:**
- New root: `data/output/b9_five_scene_validation/task10c_radiometric_final/`.

**Interfaces:**
- Consumes: strict Task10C protocol and only the two frozen geometry solutions.
- Produces: six formal runs with uniform metrics, provenance and scientific statuses.

Run exactly these six rows, sequentially:

| Geometry | Radiometric |
|---|---|
| `efficient_loftr + translation_l2` | RAW |
| `efficient_loftr + translation_l2` | BAGRN |
| `efficient_loftr + translation_l2` | BAGRN_VOLRN |
| `sift + mst` | RAW |
| `sift + mst` | BAGRN |
| `sift + mst` | BAGRN_VOLRN |

- [ ] **Step 1: Create an immutable run manifest**

For every row persist:
- scene IDs/manifest indices;
- source config hash;
- canonical grid hash;
- geometry transform hash;
- protocol hash;
- Git commit;
- method;
- control scene;
- VOLRN params where applicable.

- [ ] **Step 2: Execute all six rows serially**

Each row gets separate logs and output directory. Non-empty output is renamed to `_incomplete_<timestamp>` before a retry.

- [ ] **Step 3: Validate each row immediately**

Check:
- expected CRS/transform/width/height;
- float32 policy for normalized scientific mosaics;
- finite valid pixels;
- no unexpected coverage holes;
- six paper metrics present;
- diagnostic metrics present;
- solver convergence for BAGRN_VOLRN;
- provenance hashes match.

If one BAGRN_VOLRN row is nonconverged, mark it non-PASS and continue only far enough to produce an honest comparison table; do not relabel it as PASS and do not retune.

- [ ] **Step 4: Build final comparison artifacts**

Produce at least:
- `01_run_inventory.csv/json`
- `02_paper_metrics.csv/json`
- `03_additional_diagnostics.csv/json`
- `04_solver_diagnostics.csv/json`
- `05_pairwise_radiometric_metrics.csv/json`
- `06_provenance_manifest.json`
- `07_task10c_summary.md`

Do not collapse metrics into a single “winner” score. Report RAW→BAGRN and BAGRN→BAGRN_VOLRN deltas separately for each geometry.

---

### Task 15: Final Audit, Regression, and Branch Verification

**Files:**
- Modify: `src/multiscene_sift/radiometric_audit.py` as needed for truthful final reporting.
- Create: `docs/audits/2026-09-26-task10c-final-audit.md`.

**Interfaces:**
- Consumes: repaired code + final experiment artifacts.
- Produces: review-ready branch with explicit evidence and no push.

- [ ] **Step 1: Ensure audit ignores non-PASS rows for universal improvement statements**

Statements such as “BAGRN+VOLRN is below BAGRN for every geometry” may only be computed over rows that passed the required science/convergence status, and the denominator must be shown.

- [ ] **Step 2: Run focused regression suite**

Include matcher contract, geometry/global, BAGRN/VOLRN, metrics, mosaic, protocol/resume/batch tests.

- [ ] **Step 3: Run full repository suite**

```bash
python -m pytest -q
```

Target: `0 failed`.

If failures occur:
- fix failures caused by Task10C;
- if a failure is environment-only or demonstrably pre-existing, reproduce evidence and classify it in the audit;
- do not claim the suite is green unless exit code is zero.

- [ ] **Step 4: Static checks**

Run at least:
```bash
git diff --check
python -m compileall -q src scripts
```

If the repo has formatter/linter config already, run it; do not introduce a new formatting regime merely for this task.

- [ ] **Step 5: Review generated scientific artifacts**

Programmatically verify row counts, statuses, hashes, no duplicated experiment IDs, no stale `error` on success, and no historical output overwritten.

- [ ] **Step 6: Final code review pass**

Use `superpowers:requesting-code-review` if available. Review specifically against the five `## Review Focus` items and every Global Constraint.

- [ ] **Step 7: Final commit**

```bash
git add src scripts tests docs/superpowers/plans docs/audits
git commit -m "fix: finalize Task10C scientific pipeline"
```

Add only small summary artifacts if repository policy already tracks them. Do not add large GeoTIFF/NPZ/log directories unless already required by repository conventions.

- [ ] **Step 8: Stop without push**

Final response/report must include:
- branch and final HEAD;
- commit list created by this plan;
- focused test result;
- full pytest result;
- LoFTR replay outcome and whether downstream rerun was needed;
- VOLRN gate PASS/FAIL;
- six final experiment statuses if gate passed;
- exact paths to final metrics/audit artifacts;
- explicit statement `push_performed=false`.

Do not merge, push, open PR, or start Task11 automatically.

---

## Expected Commit Sequence

Commit messages may vary slightly to fit repository style, but keep task boundaries approximately:

1. `docs: freeze Task10C repair protocol`
2. `fix: unify matcher valid-mask filtering`
3. `perf: reuse matcher models across pair runs`
4. `fix: hard fail invalid global geometry`
5. `fix: strengthen geometry artifact provenance`
6. `fix: exclude invalid BAGRN overlap observations`
7. `fix: make VOLRN convergence scientifically explicit`
8. `fix: preserve normalized valid pixels`
9. `fix: correct radiometric evaluation semantics`
10. `feat: freeze strict Task10C protocol`
11. `fix: preserve float radiometric mosaics`
12. `test: lock Task10C scientific invariants`
13. optional audit commit for LoFTR replay if tracked summaries change
14. `fix: finalize Task10C scientific pipeline`

Do not squash automatically; small commits make scientific regressions traceable.

## Formal Stop/Go Logic

```text
core TDD repairs
      |
      v
focused tests PASS
      |
      v
LoFTR replay
      |
      +-- unchanged --> keep old downstream LoFTR artifacts
      |
      +-- changed ----> rerun only LoFTR Global + weighted mosaic
      |
      v
metric-formula source audit valid?
      |
      +-- no --> write BLOCKED report; stop real science run
      |
      v
EfficientLoFTR + Translation-L2
BAGRN -> BAGRN+VOLRN strict gate
      |
      +-- nonconverged / identity fallback / invalid provenance --> GATE_FAILED; stop
      |
      v
final 2 geometry x 3 radiometric runs
      |
      v
focused tests + full pytest + audit
      |
      v
final commit, NO PUSH, STOP
```

## Self-Review

- **Spec coverage:** 本计划覆盖此前代码审计中的25类问题；Task11旧`narrow_feather`明确排除在本轮科学修复范围外。
- **Step scan:** 每个实现任务均包含失败测试→实现→验证→commit；真实实验使用两个明确gate，避免边修边跑和重复大实验。
- **Type consistency:** matcher输出统一进入`MatchSet`; geometry artifacts维持`pair_common_grid`/`target_to_reference`; radiometric status与solver diagnostics从VOLRN向runner/batch/audit单向传播。
- **Review Focus:** 五个高风险点均在Task 1、4、6–9、11–15中有明确测试或gate。
- **Proportion:** 计划只规定接口、科学常量、测试断言和执行顺序；不预写算法实现体。

## Completion Definition

Task10C只有在以下条件同时满足时才可称为“完成”：

1. 修复后的focused suite通过；
2. full pytest本地exit code为0，或最终报告明确列出并证明与本轮无关的环境/既有失败（不得写成全绿）；
3. LoFTR公平性修复完成并有replay证据；
4. VOLRN不存在“未收敛→identity→PASS”的路径；
5. histogram评价使用after；
6. Task10C参数/provenance/resume被冻结并可验证；
7. paper metrics的公式来源已核对；
8. EfficientLoFTR+Translation-L2的VOLRN gate通过后才运行正式六行实验；
9. 正式BAGRN_VOLRN行只有真实收敛才是科学PASS；
10. 没有push、merge、Task11扩展或结果导向调参。
