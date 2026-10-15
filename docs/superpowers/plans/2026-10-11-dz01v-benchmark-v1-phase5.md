# DZ01V Benchmark V1 Phase5 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver an independent, auditable single-pair six-method 2D-translation DEVELOPMENT runner plus a separate `GEOREF_ONLY` diagnostic baseline, verify it synthetically and with at most a bounded `00_01` smoke, and leave the four-pair experiment for the researcher to launch.

**Architecture:** A frozen-input controller creates one common B9 overlap descriptor without human checkpoints. Five correspondence adapters feed a shared deterministic translation estimator; AROSICS supplies its documented direct global shift. Results are persisted before a separate DEVELOPMENT-only evaluator loads 43 first-review points; TEST files never enter the matcher interface.

**Tech Stack:** Python, rasterio, NumPy, OpenCV, existing Kornia/PyTorch/LightGlue adapters, optional isolated KARIOS/AROSICS processes, PyYAML, psutil (already present in `.venv-registration` for parent-side RSS monitoring), pytest; PowerShell commands on Windows.

**Spec:** `docs/superpowers/specs/2026-10-11-dz01v-benchmark-v1-phase5-design.md` (approved 2026-10-11), with the researcher's three plan-stage additions below.

## Global Constraints

- Frozen pairs: `00_01`, `02_03`, `05_07`, `08_10`; DEVELOPMENT `43` (11/9/9/14), pending TEST `84`, UNASSIGNED `33`.
- Phase3 remains EPSG:32650, native 14 m, fixed origin `(0,0)` m, `ix=floor(x/10000)`, `iy=floor(y/10000)`, `DEVELOPMENT if (ix+2*iy)%3==0 else TEST`, 10 km squares and strict 500 m inset (`500<local_x,local_y<9500`). Never rewrite its point assignments, Phase1–4 data, candidate CSVs, reviews, original B9 or formal outputs.
- Only six Matcher IDs enter the main table: `SIFT`, `KARIOS`, `AROSICS_GLOBAL`, `LoFTR`, `EfficientLoFTR`, `LightGlue+DISK`. `GEOREF_ONLY` is a separate, non-ranked diagnostic; `AROSICS_LOCAL` is out of scope.
- All six use the same frozen B9 pair, true valid mask, predeclared overlap footprint, common view and independent DEVELOPMENT scorer. Method-specific required normalization is recorded; no outcome-driven common-footprint/metric change.
- Five correspondence methods use the *same* 2D translation estimator. `AROSICS_GLOBAL` retains its native `COREG` solver; do not claim the ranking isolates feature extraction from solver choice.
- The matcher interface receives only B9 pixels/masks and preprocessing metadata. It receives no DEVELOPMENT or TEST coordinates, labels, split assignments, second-review events, or Task15 transform. Reading overlap image pixels is allowed; independent checkpoint scoring is **not** a completely spatial-held-out generalization test.
- DEVELOPMENT truth is loaded only after transform persistence and may inform documented future development decisions, never the current fitted correspondences. TEST is never used for fitting, RANSAC, threshold selection, tuning, or Phase5 scoring.
- Separate `ENV_UNAVAILABLE`, `WEIGHTS_UNAVAILABLE`, `NO_VALID_OVERLAP`, `UNSUPPORTED_GRID`, `ALGORITHM_FAIL`, `SUCCESS`, and `NOT_ATTEMPTED`; absent independent truth is an *evaluation* state, not algorithm FAIL.
- New Phase5 code/docs/config/tests only. Do not modify Task15/Task16, old matcher modules, Phase3/Phase4, the parallel radiometric R1 worktree, or `.venv-registration` dependencies. Do not run all four pairs or any 13-scene network/Task15/Task16/radiometric experiment.
- Small reports go under `docs/benchmark/v1/phase5/`; run outputs and optional isolated environments go under Git-ignored `data/output/benchmark_v1_phase5/`. No overwrite of existing runs. Preserve all existing untracked files. In the current checkout `.git` is read-only; never force a commit, stage R1/Phase1–4 files, or call a task committed if Git refuses it.

## Frozen protocol values (before any real smoke)

These values are design choices, **not official method thresholds**. Put the same immutable values in `configs/benchmark_v1_phase5.yaml`, its SHA256 in every result, and a protocol-lock JSON emitted before real inference. Changing a value creates a new config/version; never silently revise it after seeing DEVELOPMENT scores.

| Item | Frozen V1 value |
|---|---|
| Direction/unit | `scene_i -> scene_j`, `(delta_row, delta_col)` in native B9 pixels, zero-based pixel centers |
| Common view | EPSG:32650, 14 m axis-aligned overlap grid; data bilinear, mask nearest; only true joint-valid pixels |
| Input stretch/size | 2nd–98th percentile per image on valid overlap, linear `[0,1]`; max side `1600` pixels; preserve aspect ratio and exact realized x/y resize factors |
| Matcher result cap | `5000` correspondences for methods supporting a cap; no confidence-based cross-method threshold |
| Shared fit | One-point displacement RANSAC, NumPy `default_rng(20261011)`, at most `1000` hypotheses, Euclidean inlier radius `2.0` native pixels |
| Acceptance | At least `12` valid matches; at least `8` inliers and `0.10` inlier fraction; inliers occupy at least `4` cells of a fixed `4x4` overlap grid, including at least `2` distinct grid rows and `2` columns |
| Tie/final fit | Maximum inlier count, then lower median inlier residual, then deterministic hypothesis index; final componentwise median displacement of chosen inliers; record final residual/inlier mask once |
| SIFT | Existing adapter `nfeatures=8000`, Lowe ratio `0.75` |
| LoFTR | Existing Kornia outdoor adapter, confidence floor `0.2`, max matches `5000` |
| EfficientLoFTR | Existing official outdoor full model, `fp32`, confidence floor `0.0`, explicit local repo/checkpoint paths and SHA256 |
| LightGlue+DISK | Existing official adapter, `2048` DISK keypoints, extractor resize `1024`, confidence floor `0.0` |
| KARIOS | Isolated official KLT; large-shift preprocessing **off**; reference=`scene_i`, monitored=`scene_j`; parse semicolon `x0,y0,dx,dy` output and preserve official version/default KLT settings in run manifest |
| AROSICS | Isolated `COREG` global, one fixed center window `256x256` on shared overlap, `max_iter=5`, `max_shift=5` view pixels; `COREG_LOCAL` prohibited; raw shift/sign verified on synthetic data before any real result |
| Smoke budget | Exactly one explicitly named method per invocation; one pair `00_01`; wall timeout `900 s`, host-process RSS ceiling `12 GiB`; GPU device/capability and peak allocated VRAM logged, OOM recorded. No claim of a hard cross-library VRAM cap. |

Any external version lacking the named KARIOS/AROSICS parameters or output contract is `ENV_UNAVAILABLE` with reason `UNSUPPORTED_ADAPTER_VERSION`, never silently substituted. If the fixed center AROSICS window lacks valid support, use `ALGORITHM_FAIL` with its reason, **not** a hand-picked alternate window. For all methods, invalid support is filtered before fit by the common rule. A global 2D native translation is permitted only when both native affine linear parts agree within `1e-9` map units per pixel; otherwise `UNSUPPORTED_GRID`.

## File map

| File | Responsibility |
|---|---|
| `configs/benchmark_v1_phase5.yaml` | Frozen four-pair identity, shared values above, per-method settings, smoke budget; no source-coordinate fields |
| `src/benchmark_v1_phase5/protocol.py` | Typed config, method/status enums, result validation and serialization |
| `src/benchmark_v1_phase5/inputs.py` | Read-only frozen-image identity, bounded common overlap view, masks, provenance; no truth loader |
| `src/benchmark_v1_phase5/coordinates.py` | View/common/native pixel-center round trips and `GEOREF_ONLY` affine-origin mapping |
| `src/benchmark_v1_phase5/translation.py` | Deterministic shared displacement RANSAC and coverage gate |
| `src/benchmark_v1_phase5/adapters.py` | Thin wrappers for four existing local matcher implementations and normalized correspondence output |
| `src/benchmark_v1_phase5/external.py` | Isolated KARIOS/AROSICS invocation and strict output/sign parsers; no installation side effect |
| `src/benchmark_v1_phase5/evaluation.py` | 43-point DEVELOPMENT-only score and status-preserving summaries; baseline uses same metric routine |
| `src/benchmark_v1_phase5/runner.py` | Explicit single-pair/method orchestration, protocol lock, atomic outputs and resource accounting |
| `scripts/run_benchmark_v1_phase5.py` | CLI for `--preflight`, `--synthetic`, `--smoke`, and researcher-only `--development` one pair/method per call |
| `docs/benchmark/v1/phase5/` | Small protocol, environment/weight/license ledger, adapter statuses, synthetic/smoke logs, manual commands and readiness |
| `data/output/benchmark_v1_phase5/` | Git-ignored, non-overwriting smoke/development run outputs and optional isolated env prefixes; never Task15/Task16/R1 output |
| `tests/test_benchmark_phase5_*.py` | Synthetic, no-model, identity/leakage, adapter and CLI regression tests |

Use Phase4 `scripts/benchmark_phase4_freeze.py::load_frozen_trial` and `verify_b9_identity` in the controller; use `scripts/benchmark_phase4_development.py::load_development_truth` only in the evaluation path. No Phase4 source modification. Official KARIOS output/requirements are documented at [KARIOS](https://github.com/telespazio-tim/karios); AROSICS global COREG API at [AROSICS docs](https://danschef.git-pages.gfz-potsdam.de/arosics/doc/usage/global_coreg.html). Verify exact installed revisions and licenses in the environment ledger rather than assuming a current website revision equals the installed one.

## Review Focus

1. Same-world-point native grids with offset origins: `GEOREF_ONLY` and fitted translation must include the nonzero native offset (Task 2 test).
2. All-valid bounding rectangles but disjoint NoData masks: return `NO_VALID_OVERLAP`, never launch a matcher (Task 1 test).
3. KARIOS `dx/dy` and AROSICS corrective shift have potentially opposite semantics: synthetic sign test must block an unverified direct-shift result (Task 5 test).
4. Candidate CSV/second-review path accidentally reaching the matcher process: reject unexpected descriptor keys and verify no truth import/open in adapter tests (Tasks 1, 6 and 7 tests).
5. Absent independent truth or an unavailable environment: keep transform or method status and null metrics without miscounting an algorithm FAIL (Task 6 test).

---

### Task 1: Frozen config and truth-free common input

**Files:** Create `configs/benchmark_v1_phase5.yaml`, `src/benchmark_v1_phase5/__init__.py`, `protocol.py`, `inputs.py`; test `tests/test_benchmark_phase5_inputs.py`.

**Interfaces:** `load_config(path: Path) -> Phase5Config`; `build_pair_manifest(snapshot: FrozenTrial, pair_id: str) -> PairManifest` (only frozen B9 headers/hash, never candidates); `prepare_pair(manifest: PairManifest, config: Phase5Config) -> PairInput`; `PairInput` contains common `MatchView`, per-scene native affines/masks, and reversible view provenance. Return `NO_VALID_OVERLAP` before method invocation if joint valid support is zero; do not emit giant union rasters.

- [ ] **RED:** Test config rejects a fifth pair, TEST/candidate/second-review keys, non-finite transforms and changed B9 hashes; test true-mask-disjoint synthetic GeoTIFFs return `NO_VALID_OVERLAP` despite intersecting bounds; test shared crop/validity and bounded reads. Run `python -m pytest tests/test_benchmark_phase5_inputs.py -q`; expect failure because Phase5 API is absent.
- [ ] **GREEN:** Implement the exact public interfaces above, an immutable config hash, Phase4 frozen-identity check, overlap-window reprojection (bilinear pixels/nearest valid masks), and one shared 1600-side view with recorded affine/resize/stretch. Re-run the focused test; expect all pass.
- [ ] **Verify:** Run `python -m pytest tests/test_benchmark_phase5_inputs.py tests/test_benchmark_phase4_freeze.py -q`; verify no original/checkpoint file changed. Stage/commit only new Phase5 paths if Git permissions and Phase4 dependency reproducibility permit; otherwise record why no commit occurred.

### Task 2: Pixel-center mapping and non-ranked GEOREF_ONLY

**Files:** Create `src/benchmark_v1_phase5/coordinates.py`, `tests/test_benchmark_phase5_coordinates.py`.

**Interfaces:** `view_xy_to_native_rc(xy: np.ndarray, pair: PairInput, side: Literal['i','j']) -> np.ndarray`; `georef_only_translation(i_affine: Affine, j_affine: Affine) -> tuple[float,float]`; `to_native_correspondences(matchset: MatchSet, pair: PairInput) -> NativeMatches`.

- [ ] **RED:** Synthetic 14 m GeoTIFFs with equal linear affine parts but nonzero origin offset must yield `GEOREF_ONLY` equal to `A_j^{-1}(t_i-t_j)` in `(row,col)`, not `(0,0)`; `view -> world -> native -> view` round trip within `1e-6` px across crop/resizes, positive/negative subpixel shifts, and different origins. Rotated/scaled relative grids return `UNSUPPORTED_GRID`; invalid-mask coordinates are rejected. Run `python -m pytest tests/test_benchmark_phase5_coordinates.py -q`; expect missing API failure.
- [ ] **GREEN:** Implement explicit zero-based pixel-center formulas (`Affine*(col+0.5,row+0.5)`), invert separately for scenes i/j, preserve complete native offset and direction. Treat existing `MatchSet.ref_xy` as scene i and `.tgt_xy` as scene j only after checking metadata frame. Re-run focused test; expect pass.
- [ ] **Verify:** Run coordinate and Task 1 tests together. The baseline is tagged `GEOREF_ONLY`, `diagnostic_only=true`, with no fake match/inlier count and is absent from six-method method registry.

### Task 3: Frozen common robust 2D translation fit

**Files:** Create `src/benchmark_v1_phase5/translation.py`, `tests/test_benchmark_phase5_translation.py`; reference the already-frozen config values from Task 1.

**Interfaces:** `estimate_translation(matches: NativeMatches, config: Phase5Config) -> TranslationFit` with native i→j `(delta_row,delta_col)`, inlier mask, match/inlier/coverage counts, seed and failure code.

- [ ] **RED:** Test known `(+2.25,-3.5)` and reverse shifts with outliers recover within `0.25` px; same input/seed returns bit-identical fit; 11 matches, 7 inliers, `<0.10` inlier ratio and single-cell clusters fail for their named gates. Test confidence magnitude changes do not alter shared RANSAC rule. Run `python -m pytest tests/test_benchmark_phase5_translation.py -q`; expect missing API failure.
- [ ] **GREEN:** Implement one-point displacement hypotheses, 1000 seeded trials, 2 px Euclidean inlier radius, fixed tie-break and componentwise-median final fit; use the 4×4 overlap coverage gate and values in the protocol table. Re-run focused test; expect pass.
- [ ] **Verify:** Run Tasks 1–3 tests together and inspect serialized protocol SHA. No affine/RANSAC matrix fitting from `src/registration_benchmark/geometry.py` may enter this result.

### Task 4: Four in-repository Matcher wrappers

**Files:** Create `src/benchmark_v1_phase5/adapters.py`, `tests/test_benchmark_phase5_adapters.py`; no edits in `src/registration_benchmark/`.

**Interfaces:** `run_local_matcher(method: str, pair: PairInput, config: Phase5Config) -> AdapterMatches`; outputs existing `MatchSet` converted by Task 2, plus model-init/feature/match timings and provenance. Dispatch only `SIFT`, `LoFTR`, `EfficientLoFTR`, `LightGlue+DISK` here.

- [ ] **RED:** Synthetic small-image/controlled adapter-output tests verify each wrapper's i/j direction, common-mask filtering, existing model resize inverse, missing weights→`WEIGHTS_UNAVAILABLE`, missing package→`ENV_UNAVAILABLE`, and exceptions→`ALGORITHM_FAIL`. Check wrappers never open a candidate CSV or truth file. Run `python -m pytest tests/test_benchmark_phase5_adapters.py -q`; expect missing API failure.
- [ ] **GREEN:** Call existing `match_sift`, `match_loftr`, `match_efficient_loftr`, `match_lightglue_disk` with frozen explicit arguments; no hidden Task15 transform and no alternative fallback. Validate their `pair_common_grid` metadata and invert the Phase5 view/native chain exactly once. Re-run focused test; expect pass.
- [ ] **Verify:** Run `python -m pytest tests/test_benchmark_phase5_adapters.py tests/registration_benchmark/test_sift_matcher.py tests/registration_benchmark/test_loftr_adapter.py tests/registration_benchmark/test_efficient_loftr_adapter.py tests/registration_benchmark/test_lightglue_disk_adapter.py -q`; document which real weights and environments are only importable, not end-to-end verified.

### Task 5: Isolated KARIOS correspondence and AROSICS direct-shift adapters

**Files:** Create `src/benchmark_v1_phase5/external.py`, `tests/test_benchmark_phase5_external.py`, `docs/benchmark/v1/phase5/environment_and_versions.md`.

**Interfaces:** `parse_karios_csv(path: Path, pair: PairInput) -> NativeMatches`; `parse_arosics_result(path: Path, pair: PairInput) -> DirectTranslation`; `run_external(method: str, pair: PairInput, config: Phase5Config, python_exe: Path) -> AdapterMatches | DirectTranslation`. External tool invocation is a subprocess in an isolated Python environment named in config; never `pip install` inside `.venv-registration`.

- [ ] **RED:** KARIOS semicolon `x0,y0,dx,dy` fixture must restore scene-i/scene-j points, including negative dx/dy and crop origin. Missing/duplicate/non-finite CSV columns must fail. AROSICS fixture must preserve raw X/Y shift, declared target→reference correction convention and map to full native i→j, with a deliberate wrong-sign fixture rejected. An unsupported external revision or no executable yields environment status, not algorithm FAIL. Run `python -m pytest tests/test_benchmark_phase5_external.py -q`; expect missing API failure.
- [ ] **GREEN:** Implement strict parsers and isolated subprocess request/response, record command/version/config/stdout/stderr and explicit view GeoTIFF/mask paths. KARIOS uses common view as reference=i, monitored=j; its official `x0,y0,dx,dy` convention is verified synthetically. AROSICS uses the fixed `COREG` center window and retains native solver provenance. Re-run focused tests; expect pass.
- [ ] **Environment check:** Inspect official license/version/requirements and installed interpreters without modifying `.venv-registration`; if feasible, create isolated KARIOS/AROSICS env prefixes only under `data/output/benchmark_v1_phase5/envs/` and run synthetic sign tests. If network/ABI/license/weights block installation, document exact blocker and keep real adapter `ENV_UNAVAILABLE`; do not report fixture-only tests as library end-to-end PASS.
- [ ] **Verify:** Run Task 5 tests and a no-write `--preflight` check; KARIOS/AROSICS statuses must reflect actual executable/library availability. Record the observed installed revision, not a presumed upstream latest.

### Task 6: Independent DEVELOPMENT scoring, baseline and status accounting

**Files:** Create `src/benchmark_v1_phase5/evaluation.py`, `tests/test_benchmark_phase5_evaluation.py`.

**Interfaces:** `score_development(result: Phase5Result, dev_points: Sequence[dict]) -> EvaluationResult`; `summarize_development(results: Sequence[EvaluationResult], expected_pairs: Sequence[str]) -> dict`; `score_georef_only(pair: PairInput, dev_points: Sequence[dict]) -> EvaluationResult`. Reuse the repository's `pair_metrics()` definition without changing it.

- [ ] **RED:** A fixed transform on known points yields RMSE/P95/max/PCK@1/2/3 identical to `pair_metrics`; positive/negative direction failures are detected. TEST/UNASSIGNED records are rejected; absent truth leaves execution `SUCCESS` and evaluation `NO_INDEPENDENT_TRUTH`; unavailable/no-overlap/algorithm-failed rows keep distinct denominators and null metrics. `GEOREF_ONLY` uses the same 43-point scorer but is not in six-method macro ranking. Run `python -m pytest tests/test_benchmark_phase5_evaluation.py -q`; expect missing API failure.
- [ ] **GREEN:** Implement pure scoring with no matcher import or fit call; enforce frozen DEVELOPMENT counts from Phase4 when real snapshot is used; keep per-pair and success-only macro metrics plus all attempted/unavailable/failure/no-overlap counts. A real `00_01` smoke may score only its 11 DEV points; scoring all 43 requires the later researcher-initiated four-pair run. Re-run focused test; expect pass.
- [ ] **Verify:** Run evaluation, Phase4 development and truth-isolation tests together. Report clearly: DEV checkpoint precision is human-assessed, algorithms may read overlap pixels, and this is not a fully spatial-held-out generalization claim.

### Task 7: Single-pair CLI, provenance lock and synthetic end-to-end tests

**Files:** Create `src/benchmark_v1_phase5/runner.py`, `scripts/run_benchmark_v1_phase5.py`, `tests/test_benchmark_phase5_runner.py`, `docs/benchmark/v1/phase5/geometry_output_protocol.md`.

**Interfaces:** `run_one(pair_id: str, method: str, config_path: Path, output_root: Path, mode: Literal['synthetic','smoke','development']) -> Phase5Result`; CLI has explicit `--pair`, `--method`, `--config`, `--output-root`, one of `--preflight|--synthetic|--smoke|--development`; no all-pairs/all-methods default.

- [ ] **RED:** CLI rejects an implicit/default run, a real `--smoke` pair other than `00_01`, unknown method, attempt to overwrite an output, unfrozen config/hash, and any matcher descriptor carrying checkpoint coordinates. A synthetic single-pair run must persist translation before importing/reading DEV truth; outputs contain no TEST data and include method/pair/status/config hash. Test fake subprocess timeout/RSS budget classification and atomic write cleanup. Run `python -m pytest tests/test_benchmark_phase5_runner.py -q`; expect missing CLI/API failure.
- [ ] **GREEN:** Implement an isolated one-pair worker, output `protocol_lock.json`, `input_provenance.json`, `method_result.json`, separate `georef_only_metrics.json` and optional `development_metrics.json`, warning/resource log, and status manifest. Lock config values/hash before inference, then persist fit/direct result before separate scorer call. Enforce 900 s timeout and 12 GiB RSS limit from the parent with psutil; report GPU memory as observed or UNKNOWN with reason. Re-run focused tests; expect pass.
- [ ] **Verify:** Run `python scripts/run_benchmark_v1_phase5.py --config configs/benchmark_v1_phase5.yaml --preflight`; test that it performs no model inference. Run all Phase5 synthetic tests and inspect the output schema, coordinate sign and `GEOREF_ONLY` separation.

### Task 8: Bounded `00_01` smoke, manual four-pair instructions and readiness

**Files:** Create `docs/benchmark/v1/phase5/adapter_status.md`, `synthetic_and_smoke_log.md`, `four_pair_development_commands.md`, `benchmark_phase5_readiness.md`; update only Phase5 environment ledger and protocol/config if **no** real smoke has yet occurred (a change after smoke requires a new config/version and recorded rationale). This task is a verification/report handoff, not a new product-code change.

**Interfaces:** Manual single-pair command is:

```powershell
python -u scripts/run_benchmark_v1_phase5.py --config configs/benchmark_v1_phase5.yaml --pair 00_01 --method SIFT --smoke --output-root data/output/benchmark_v1_phase5/runs/00_01_sift_smoke
```

Researcher-only four-pair DEVELOPMENT invocation (each iteration is still one explicit pair/method process; run **only after** reviewing Phase5 readiness):

```powershell
$pairs = @('00_01','02_03','05_07','08_10')
$methods = @('SIFT','KARIOS','AROSICS_GLOBAL','LoFTR','EfficientLoFTR','LightGlue+DISK')
foreach ($pair in $pairs) { foreach ($method in $methods) { python -u scripts/run_benchmark_v1_phase5.py --config configs/benchmark_v1_phase5.yaml --pair $pair --method $method --development --output-root ("data/output/benchmark_v1_phase5/runs/dev_${pair}_${method}") } }
```

- [ ] **Pre-smoke gate:** Re-run Task 7's CLI scope tests (`--smoke --pair 02_03` must exit before opening B9), all synthetic positive/negative/native-origin/no-mask-overlap tests, input-hash checks and read-only environment/weight/license preflight. Freeze and hash the exact config in the log **before** real inference; a failed gate forbids the smoke.
- [ ] **Documentation:** Write the two explicit PowerShell commands above, the six-method availability matrix, the non-ranked `GEOREF_ONLY` procedure, and a warning that the loop is a researcher-only next action. Record all method-specific preprocessing and actual version evidence.
- [ ] **Smoke:** Run **only** the SIFT `00_01` command above if frozen B9 identity/valid overlap and resource preflight pass. Do not silently change input size, RANSAC values, or footprint to make it pass. If even this bounded attempt is unsafe/unavailable, record the exact reason and do not substitute another pair/method. No four-pair/15-pair/Task15/Task16 run.
- [ ] **Final verification:** Run `python -m pytest tests -k benchmark_phase5 -q`, full `python -m pytest -q`, `python -m compileall -q src/benchmark_v1_phase5 scripts/run_benchmark_v1_phase5.py`, and `git diff --check`; inspect new untracked files separately because `git diff --check` does not cover them. Read exit codes, failure/skip counts and status logs; verify Phase3/Phase4 and original B9 hashes unchanged. Write adapter status per method (`END_TO_END_VERIFIED`, `CONTRACT_ONLY`, `ENV_UNAVAILABLE`, etc.), any `00_01` `GEOREF_ONLY` DEV diagnostic, actual `00_01` smoke outcome, and truthful readiness. Stop and hand the researcher the manual command; do not launch it.

## Stage acceptance and handoff

Tasks 1–3 pass only with true-overlap, round-trip and known-translation synthetic proof plus unchanged frozen data. Tasks 4–5 pass only with honest per-method environment/weight/version evidence and no algorithm substitution. Task 6 passes only with zero TEST leakage, separate GEOREF_ONLY results and preserved status denominators. Task 7 passes only when every CLI call requires an explicit method/pair and a persisted config lock before inference. Task 8 passes only after focused/full regression results are reported without hiding failures, and the `00_01` smoke is either executed within budget or honestly marked blocked.

Do not describe an import, stub parser, or synthetic fixture as a real KARIOS/AROSICS end-to-end pass. Do not describe DEVELOPMENT checkpoint scoring as an independent spatial holdout: the methods see the same overlap image pixels, while TEST point labels/coordinates remain isolated. The researcher decides whether to start the four-pair DEVELOPMENT loop after reviewing the final report.
