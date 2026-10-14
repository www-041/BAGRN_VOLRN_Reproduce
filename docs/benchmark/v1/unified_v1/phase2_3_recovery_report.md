# Phase 2.3 recovery and environment report

Date: 2026-10-14
Branch: `recovery/spectralmatch-phase2-3-20261014`
Recovery commit: `a6352037a43125fa64a83a32053463db1e4ff1ae`

## Freeze persistence

The six named files were backed up before staging. The backup, staged index, recovery commit, and original worktree each match their frozen SHA256 values byte-for-byte. The recovery commit contains exactly those six paths. The historical freeze head `3c9f4e30888445ce783f56f4f26cb064e1add887` does not contain the six paths; the bytes were recovered from Git objects. The commit persists the recovered bytes without changing the old lock or manifest.

A detached clean worktree was checked out at the recovery commit. The machine-wide `core.autocrlf=true` initially caused raw SHA mismatches: after LF normalization, all 29 present files matched their lock values. A second clean worktree was then created with `core.autocrlf=false`; it is clean and its frozen-core tests pass (`27 passed`). In that byte-preserving checkout, **29/38 files match exactly, 0 mismatch, and 9 are missing**. The six restored files all match exactly. By comparison, the original working tree currently matches all 38 entries. Thus the six-file recovery is reproducible, but the complete 38-file locked source set is not reproducible from Git alone. No additional locked files were overwritten or committed in this phase.

The 9 absent paths are R1/R1.5 scripts present in the original working tree but not committed in the recovery branch:

- `scripts/benchmark/annotate_radiometric_rois.py`
- `scripts/benchmark/build_radiometric_r1_5_report.py`
- `scripts/benchmark/build_radiometric_r1_band_readiness.py`
- `scripts/benchmark/build_radiometric_r1_inventory.py`
- `scripts/benchmark/generate_radiometric_r1_5_auto_rois.py`
- `scripts/benchmark/generate_radiometric_r1_synthetic.py`
- `scripts/benchmark/prepare_radiometric_r1_rois.py`
- `scripts/benchmark/suggest_radiometric_r1_5_labels.py`
- `scripts/benchmark/summarize_radiometric_r1_5_machine_agreement.py`

The 29 mismatches are the remaining locked protocol, audit/evaluation, and `src/benchmark_v1_unified` implementation/test files. Exact per-file hashes remain in the immutable freeze manifest; this phase did not alter it.

## Environment recovery

No new acceptable Spectralmatch environment was built. The rejected pre-existing temporary environment has Python 3.11.17, `osgeo.gdal` 3.11.5, Rasterio 1.4.4 linked to GDAL 3.10.3, PROJ 9.5.1, Spectralmatch 1.6.0, and lacks `psutil` and `scikit-image`. The GDAL/Rasterio mismatch disqualifies it; its successful repeated imports and small raster round-trip do not establish compatibility.

The Miniforge/conda-forge attempt could not proceed because available micromamba 2.9.0 crashes even for `--version` (`0xC0000409`); GitHub release downloads were unreachable, and the configured PyPI index has no compatible GDAL 3.11.5 Windows binary wheel. WSL status and distro enumeration returned `E_ACCESSDENIED`. Per the stop condition, environment troubleshooting stopped here. The official Spectralmatch example was not run because prerequisites were not met.

The official inspected Spectralmatch source is version 1.6.0, commit `2c35fef0d8c324cd2c06fbc5b1a78e36e2853ff3`; its published requirements are Python 3.10–3.12, GDAL >=3.11, and PROJ >=9.3. See the [official repository](https://github.com/spectralmatch/spectralmatch).

## Resource monitor and tests

The Phase 2.1 resource-monitor suite passed (6 tests). Synthetic subprocess measurements from the existing registration environment: normal exit peak RSS 27,475,968 B (7 samples); 96 MiB polling threshold peak RSS 182,657,024 B (3 samples, process tree terminated); 0.7 s timeout peak RSS 40,275,968 B (15 samples, process tree terminated). These are sampled polling safeguards, not operating-system hard memory limits.

Regression evidence from Phase 2.2: Phase 2.1 `42 passed`; frozen core `27 passed`; combined `69 passed`; full repository `1,325 passed, 4 skipped`; compileall passed. In this clean recovery worktree, frozen-core tests passed again (`27 passed`). Exact frozen whitespace was retained, so `git diff --check` reports whitespace encoded in those locked bytes; normalizing it would invalidate SHA256.

## Decision

**ENVIRONMENT_BLOCKED / NO-GO.** Do not restart the `00_01` or `00_10` real-crop experiments. Do not run the 13-scene experiment. Two blockers remain: nine locked files are not in Git, so the full 38-file freeze is not reproducible from a clean checkout; and no compatible isolated Spectralmatch environment is available. No real crop, PIF comparison, Task15/16, or 13-scene experiment was run in Phase 2.3.

Machine-readable evidence: [`phase2_3_recovery_attestation.json`](phase2_3_recovery_attestation.json).
