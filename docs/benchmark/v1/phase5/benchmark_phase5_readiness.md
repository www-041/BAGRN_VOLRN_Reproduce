# DZ01V Benchmark V1 Phase5 readiness

Snapshot: 2026-10-11. Phase5 implements a separate six-method adapter and single-pair development runner; it does not modify Task15/Task16.

## Decision

**READY FOR RESEARCHER-CONTROLLED DEVELOPMENT RUNS WITH LIMITATIONS; real `00_01` smoke remains blocked by current resource preflight.** This readiness means the protocol/CLI/tests are prepared, not that six methods have completed real-data runs. The researcher should review method availability and available RAM before starting even the optional smoke.

## Frozen scope and isolation

- Four pairs only: `00_01`, `02_03`, `05_07`, `08_10`; 43 DEVELOPMENT points (11/9/9/14), 84 TEST candidates awaiting independent second review, 33 UNASSIGNED.
- CRS EPSG:32650, native 14 m imagery; Phase3 fixed 10 km grid and 500 m buffer are unchanged.
- Algorithms see the common declared overlap imagery and masks. Matcher workers receive no checkpoint coordinates. DEVELOPMENT truth is loaded only after the method result is persisted. TEST labels/coordinates/second-review records are not used for fitting, RANSAC, selection or tuning.
- Independent points evaluate the transform but do not constitute a fully spatial-held-out generalization test because methods see the same overlap pixels.
- `GEOREF_ONLY` is a separate affine diagnostic. AROSICS global COREG uses its native solver, so solver differences remain a comparison limitation. Local COREG is out of the six-method main table.

## Method/environment status

SIFT, LoFTR, EfficientLoFTR and LightGlue+DISK modules were discoverable in `.venv-registration`; no real B9 inference or weight identity validation was done in this task. KARIOS and AROSICS plus their GDAL/GeoArray dependencies are absent, and their external interpreter paths are null in the frozen config. These must remain explicit unavailable attempts until isolated environments and provenance are reviewed.

## Verification evidence

- Phase5-focused selection (`pytest tests -k benchmark_phase5`): **100 passed, 1062 deselected**, 1208 warnings.
- Full suite (`pytest -q`): **1158 passed, 4 skipped, 0 failed**, 13238 warnings in 166.92 s. Skip reasons reported by pytest: two synthetic full-radiometric-runner cases have a disconnected SIFT graph; the EfficientLoFTR official-import compatibility test requires unset `EFFICIENT_LOFTR_REPO`; the LightGlue negative-path unit test skips its “package unavailable should raise” assertion when LightGlue is installed (the test's installed-package branch delegates verification to separate integration tests).
- Phase5 `compileall`: passed for package, CLI and both workers.
- `git diff --check`: exit 0. Separate trailing-whitespace scan of Phase5 source, scripts, tests and reports: clean.
- CLI read-only preflight: passed; verified 8 B9 identities and frozen assignment/config hashes; inference false.
- Phase3 assignments, Phase3 leakage audit and all four Phase2 candidate CSV SHA256 values match the previously recorded frozen hashes. No frozen file was written.
- Optional real SIFT `00_01` smoke: not run (`BLOCKED_RESOURCE_PREFLIGHT`), about 2.28 GiB available against 12 GiB frozen RSS ceiling.
- Original B9 SHA identities were revalidated by the preflight. No image data or formal experiment output was changed.

## Still required before formal claims

1. Repeat resource preflight and run only the single `00_01` SIFT smoke if safe; do not change frozen parameters to fit available resources.
2. Establish isolated KARIOS/AROSICS environments, pin versions/licenses and validate the library-level synthetic sign/API gates. Validate actual model weights for deep methods.
3. Researcher may then explicitly start the four-pair DEVELOPMENT commands in [the run guide](four_pair_development_commands.md), inspecting every status. The commands have not been run here.
4. Complete second-person TEST review and later freeze/verify all relevant input identities before any formal test benchmark. Never use TEST points for method development.

No formal six-algorithm comparison, 15-pair benchmark, Task15/Task16, or radiometric experiment ran. No claim of accuracy/performance is made.
