# Phase 2.3 Freeze and Environment Recovery Implementation Plan

> **For agentic workers:** execute inline task-by-task; preserve the unrelated dirty worktree and do not push.

**Goal:** Persist the exact six frozen file bytes in a dedicated branch, attest clean-worktree reproducibility, and determine whether an isolated Spectralmatch environment can be established without touching protected research artifacts.

**Architecture:** Commit only the six hash-locked restorations in a new branch, then verify every entry in the freeze code manifest from a clean linked worktree. Probe the approved Windows Miniforge/conda-forge and WSL2 routes once; if both are blocked, record `ENVIRONMENT_BLOCKED` and stop environment troubleshooting.

**Tech Stack:** Git worktrees, PowerShell, Python 3.11, pytest, SHA256, Miniforge/conda-forge, WSL2.

**Spec:** User task, “Phase 2.3：冻结版本持久化与Spectralmatch环境恢复” (2026-10-14).

## Global Constraints

- Preserve `freeze_20261013_v3`, its lock files, protocol formulas, source imagery, and historical results unchanged.
- The six source files must match their recorded SHA256 bytes exactly; do not format or alter line endings.
- Do not modify the existing `.venv-registration`, prior temporary Spectralmatch environments, Task15/16, Ours, R1/R1.5, or formal outputs.
- Do not run real crops, PIF comparisons, full image pairs, or 13-scene experiments.
- Do not push the new branch.
- If both Windows package acquisition and WSL2 are inaccessible, record `ENVIRONMENT_BLOCKED` and stop environment work.

## Review Focus

- Dirty user changes in the current worktree must not enter the recovery commit; stage only the six exact-hash files.
- Git clean/smudge filters must not alter the staged locked bytes; verify staged SHA256 before commit.
- The freeze manifest contains absolute original-worktree paths; clean-worktree verification must resolve each manifest entry relative to that clean worktree, not trust the original path.
- A clean file-hash PASS must not be misrepresented as proof that historical commit `3c9f4e3` contained those files.
- WSL `E_ACCESSDENIED` must not be reported as a usable or installed Ubuntu route.

---

### Task 1: Persist exact frozen bytes in an isolated branch

**Files:** six lock-listed files only: the two Markdown files, three Python modules, and `tests/benchmark_v1_unified/test_core.py`.

- [ ] Back up the six current files and record their hashes in a new temporary backup directory.
- [ ] Create branch `recovery/spectralmatch-phase2-3-20261014` without discarding existing unrelated worktree changes.
- [ ] Stage only the six lock-listed files; verify staged SHA256 values against `freeze_20261013_v3/code_manifest.json`.
- [ ] Commit as `recovery: persist exact Unified Benchmark V1 frozen bytes`.

### Task 2: Verify clean-worktree reproducibility and attest

**Files:** add `docs/benchmark/v1/unified_v1/recovery_attestation.json` and `phase2_3_environment_attempt.json` on the recovery branch.

- [ ] Add a clean worktree at the recovery commit and verify every code-manifest entry (38/38 expected) against relative paths inside that clean worktree.
- [ ] Run frozen-core tests in the clean worktree; do not treat absolute-path root checks as clean-worktree evidence.
- [ ] Record original HEAD `3c9f4e3` missing the six paths, recovery branch/commit, SHA results, lock immutability, and clean-worktree test results.
- [ ] Record the official environment constraints, one-shot Windows/WSL route results, exact blockers, and `ENVIRONMENT_BLOCKED` if both routes fail.
- [ ] Commit only the two Phase 2.3 attestation files in a follow-up commit; verify the recovery bytes remain identical.

### Task 3: Final verification and stop

- [ ] Run the frozen audit and focused Phase 2.1 tests where the accepted environment permits.
- [ ] Confirm no real-data experiments started and protected data/results were not modified.
- [ ] Report the recovery commit, attestation commit, clean-worktree results, environment status, and GO/NO-GO for 00_01/00_10.
