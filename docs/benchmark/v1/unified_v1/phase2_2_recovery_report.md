# Phase 2.2 — Freeze recovery and Spectralmatch environment audit

Date: 2026-10-14  
Repository HEAD: `85429ba5e221b5ed3901569a86d5658de12603ad`  
Outcome: freeze file hashes restored in the current worktree; a clean, coherent Spectralmatch environment could not be built. **Real-crop restart: NO-GO.**

## A. Freeze-file audit and recovery

The read-only audit found all six exact locked file contents as Git blobs. The frozen manifest names `3c9f4e30888445ce783f56f4f26cb064e1add887` as its Git head, but none of the six paths exists in that commit. The exact blobs are not present in a branch/tag-reachable history according to `git log --all --find-object`; therefore the freeze-time Git commit does not provide their source, even though the lock SHA values were recoverable from the object database.

After completing that audit, the six worktree files were restored byte-for-byte to the SHA256 values already present in the existing freeze manifest. No lock, manifest, evaluation rule, or raster was regenerated or weakened. The restoration is currently uncommitted. It makes the current worktree pass strict file-hash verification, but does not repair the historical provenance gap: a checkout of the manifest's named commit alone cannot reproduce these six files.

| File | Freeze SHA256 | Recovered blob | Current committed `85429ba` SHA256 | Diff and impact |
|---|---|---|---|---|
| `docs/benchmark/v1/unified_v1/benchmark_protocol_draft.md` | `36fb02eee8a35d4dd1f3605aa07b5702fe10b811a2de777ab4f9281957edc836` | `a6bca5aa2677df94d1b632d3517d4d2715e71652` | `e88ff951aaed097d23cad20611c795a591ca2cde0971f35b9eb5327a889d100e` | One Markdown hard-line-break/blank-line formatting change; protocol text and metric rules unchanged. |
| `docs/benchmark/v1/unified_v1/plan.md` | `54215b6a5f118e03ff702f5ef27843500fbfdc962b775ff560efc7b216e9dc25` | `c4e464812c22fc910bd470403a797f3e051343c3` | `ba89f53a30ece7b0da2cdde43a85a2e69971c933b602240b4d634584294b378c` | Only Markdown hard-line-break/blank-line formatting in the header metadata; no algorithm or metric definition change. |
| `src/benchmark_v1_unified/__init__.py` | `e10aced210e4132341a1781d627eac04a425167f4c6e39fb53c203124b58bf54` | `a73a354970ac9981818a26db67482dbd8adc9e9d` | `89dfcef37a06a6682af382c5917f2670f9c60e573d2043225fc640c738477a9a` | One final blank line only; no runtime change. |
| `src/benchmark_v1_unified/identity.py` | `5dcdc517dc3286c0d16acd9600c6d32dd2020d5b4670d25dda75c454803970ba` | `3f1127cb6d1e5e404a345a34492a8bfac460eed8` | `507c608e56894472a052f799e596062879ab175679748d82c8e48f61c6b85616` | One final blank line only; streaming SHA256 logic unchanged. |
| `src/benchmark_v1_unified/structure.py` | `8f39a40e0bd1bdf1e1ee7da9350d14938350295f79333b66baa9b0c048da67b0` | `9cc2acce92343b8f9e864209ca819e73ad75db73` | `c575ffbf9f2cd6a409bdcf550ea1f3b279eca7bf0d25c641c97f7320e4ed6706` | One final blank line only; no structural metric formula or result change. |
| `tests/benchmark_v1_unified/test_core.py` | `c5252aa9e2418df560d125582119fb279b982f721c8647cea63a84425bd17bea` | `ec72b3ca2671bce7fe04231818824ecbdcf5ec22` | `33f17160f47f593e1e39707068051653e6f3d58bef55f58035b4467a604006ab` | One final blank line only; assertions unchanged. |

All six recovered and current files are UTF-8 without BOM and LF-only. The three Python modules and test differ from the locked bytes only by a final blank line; the two Markdown differences are formatting whitespace. Current worktree SHA256s now equal all six locked SHA256s.

**Freeze audit after recovery:** `PASS`; `code_hash_mismatch_count=0`; all 13 frozen raster/mask pairs readable; inventory remains 78 total pairs, 48 positive valid-overlap pairs, 30 `NO_VALID_OVERLAP`. `00_01` and `00_10` remain in the fixed inventory with 1,295,654 and 1,374,076 valid overlap pixels respectively. Audit reads only `freeze_20261013_v3/aligned_inputs`, not Stage06 outputs.

Important scope of this PASS: it verifies the existing lock hashes, manifests, pair inventory and raster grids in this worktree. It does **not** prove the named Git head contains the locked files, and the restoration has not been committed. The safe migration option, if a clean commit provenance is required, is a separately reviewed new freeze version with an explicit parent/version link; retain `freeze_20261013_v3` unchanged as historical evidence. No new freeze was created here.

## B. Spectralmatch environment

Official installation requirements are Python 3.10–3.12, GDAL >=3.11 and PROJ >=9.3. The inspected upstream checkout is Spectralmatch 1.6.0, commit `2c35fef0d8c324cd2c06fbc5b1a78e36e2853ff3`. The official README installation example uses conda-forge for GDAL/PROJ before installing the Python package. The local upstream checkout already had a pre-existing uncommitted change in `spectralmatch/mask/mask.py`; it was not modified.

No valid new environment was produced. The existing temporary environment was not modified and is rejected for use:

| Component in existing temporary environment | Observed | Required/status |
|---|---:|---|
| Python | 3.11.17 | Meets official range |
| `osgeo.gdal` | 3.11.5 | Meets official minimum |
| Rasterio | 1.4.4, linked GDAL 3.10.3 | **Mismatch; reject** |
| PROJ via `pyproj` | 9.5.1 | Meets stated minimum, but not proven consistent with both GDAL stacks |
| NumPy | 2.4.6 | Present |
| Spectralmatch | 1.6.0 | Imports |
| psutil | Missing | Required by fail-closed runner |
| scikit-image | Missing | Required by Phase 2.1 audit set |

Five independent fresh Python processes imported `osgeo.gdal`, Rasterio and Spectralmatch successfully (5/5), but each reported GDAL 3.11.5 vs Rasterio-linked GDAL 3.10.3. A 4x3 in-memory Rasterio GeoTIFF round-trip and GDAL MEM-driver check passed in this environment; this does not cure the version mismatch. The official end-to-end sample was not run because the environment failed the prerequisite consistency check.

A clean environment creation was attempted in a new temporary prefix. The available micromamba 2.9.0 executable failed even on `--version`, returning Windows status `0xC0000409` (`-1073740791`); no new prefix was created. A direct GitHub release download was unreachable. PyPI access works, but `pip download --only-binary=:all: GDAL==3.11.5` confirms no compatible Windows wheel is available from that index. No existing `.venv-registration`, old temporary Spectralmatch environment, or Phase 2.1 source checkout was modified to work around this.

## C. Resource-monitor validation

Unit suite: `tests/benchmark_v1_spectralmatch_phase2/test_resource_monitor.py` — **6 passed**. This includes fail-closed behavior when psutil is absent, zero RSS samples, process-handle initialization failure, RSS threshold termination, timeout termination and normal completion.

Additional short synthetic subprocess runs used the existing `.venv-registration` psutil 7.2.2 monitor. Values below are observed process-tree RSS, not estimates:

| Case | Status | Peak RSS | Samples | Process tree | Polling guard | OS hard limit |
|---|---|---:|---:|---|---|---|
| Normal exit | COMPLETED | 27,475,968 B | 7 | Natural exit; not forcibly terminated | Active | No |
| RSS threshold (96 MiB test threshold) | RESOURCE_LIMIT | 182,657,024 B | 3 | Terminated and verified exited | Triggered | No |
| Timeout (0.7 s) | TIMEOUT | 40,275,968 B | 15 | Parent and child terminated and verified exited | Active | No |

The actual 2 GiB threshold was not reached by synthetic runs; the runner’s 2 GiB guard is sampled polling, not an operating-system hard memory limit. The code does not claim otherwise. In the rejected temporary environment, psutil is absent, so the runner precondition must block before worker launch.

## D. Regression and readiness

Executed with `.venv-registration` Python 3.11.6 (read-only use):

- Phase 2.1 suite: **42 passed**, 3 warnings.
- Unified Benchmark freeze/core tests: **27 passed**, 1 pytest-cache permission warning.
- After exact-byte restoration, combined Phase 2.1 + frozen-core suite: **69 passed**, 2 warnings.
- After exact-byte restoration, full repository suite: **1,325 passed, 4 skipped**, 13,400 warnings, 207.10 s. Skips were the two disconnected synthetic SIFT cases and two unconfigured optional matcher integrations. No test failure or Windows native exception occurred in this run.
- Freeze audit after exact-byte restoration: **PASS** as scoped above.
- `python -m compileall -q src scripts tests`: **PASS**.
- `git diff --check`: **reports the exact frozen Markdown hard-break spaces and required final blank lines as whitespace warnings** in the six restored files. They are retained because removing them would break the locked SHA256s. Git also warns that LF may be converted to CRLF on a future checkout; current restored files are LF-only and hash-exact.

No crop smoke, PIF comparison, real Spectralmatch algorithm run, Task15/16 run, or 13-scene experiment was started.

### Decision

**NO-GO for restarting `00_01`/`00_10` real crop tests.** The frozen worktree gate now passes, but the independent Spectralmatch environment is still not reproducible or GDAL-consistent, and the freeze-time Git provenance gap remains uncommitted. Resume only after a clean isolated Python/GDAL/PROJ/Rasterio environment is built and passes the official minimal example and repeated-process checks. Do not change or regenerate the existing freeze to get past this block.
