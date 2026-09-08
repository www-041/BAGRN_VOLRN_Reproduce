# Final Review Fix Report

Date: 2026-09-09

Branch: `fix/dz01v-mosaic-series-readiness-20260907`

## Findings closed

1. Final validation now treats every required edge as a hard gate. A required
   edge with a non-`None` `failure_reason`, fewer than `final_min_blocks`
   accepted blocks, missing validation, or unusable statistics is recorded in
   `required_edge_failures` and forces aggregate quality to `fail`. Existing
   `unavailable_edges` and top-level failure compatibility are preserved.
2. Initial block matching now reads `global_block_size`, `max_global_shift`,
   and `global_confidence_threshold` from `registration_params`. The phase
   fallback uses the same configured confidence and shift limits.
3. Smoke-mode validation receives a copied set of pair controls translated from
   the full-frame pixel origin into the recropped frame. Full-frame registration
   measurements remain unchanged for diagnostics and downstream consumers.
4. Original-based global residual rematching catches exceptions per edge,
   records the warning and `keep_current_global_shift` fallback, and continues
   processing other edges.
5. Registration results now expose canonical `status`/`failure` fields for
   normal, single-scene, blocked, and smoke paths. The diagnostic CLI gates
   raster artifact generation on connectivity and configured quality, writes a
   failure JSON record only when that gate fails, and returns exit code 1.
   Fallback arrays are never presented as registered artifacts.
6. The active robust pair estimator now repeats the existing joint Euclidean
   MAD rejection up to five rounds until the retained set stabilizes, matching
   the plan without changing its thresholds or minimum-count failure semantics.

The ledger checklist contradiction was corrected: Tasks 1 and 2 are marked
complete, matching their detailed completion history. The quality-gate guide
was also synchronized with the per-edge and diagnostic failure behavior.

## Tests and exact outputs

TDD red run before production changes:

```text
FFFFFFE
6 failed, 1 error
```

The six failures were the new aggregator, global-rematch, initial-control,
phase-limit, and smoke-coordinate regressions; the diagnostic test reached
pytest teardown but was blocked by the host temporary-directory ACL.

The additional iterative-MAD regression was then run against the pre-fix
estimator and produced `1 failed`; it passed after the bounded estimator fix.

Final registration-focused verification:

```text
.\.venv\Scripts\python.exe -m pytest tests/test_registration_quality.py tests/test_coregistration_regressions.py tests/test_registration_refinement.py tests/test_multiband_pipeline_regressions.py tests/test_script_contracts.py -k "not analyze_displacement_spikes_handles_distance_field" -q -p no:cacheprovider
...................................................
52 passed, 1 deselected in 2.97s
```

Diagnostic regression verification:

```text
.\.venv\Scripts\python.exe -m pytest tests/test_diagnostics_regressions.py -k "not registration_diagnostic_distinguishes_raw_and_robust_pair_matches and not no_overlap_writes_structured_failure_payload and not failed_overlapping_registration_writes_failure_only_and_returns_nonzero and not registration_diagnostic_uses_actual_registration_schema" -q -p no:cacheprovider
....
4 passed, 4 deselected in 2.59s
```

The new failed-overlap diagnostic test itself passed during the selected
regression run (`......`), but pytest exited during session teardown while
scanning the protected `--basetemp` directory. The teardown error was:

```text
PermissionError: [WinError 5] ... pytest-final-review-diag
```

Full suite:

```text
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
246 passed, 23 failed, 13 errors in 11.49s
```

All 23 failures and 13 errors were blocked writes, reads, or cleanup under
the host's protected temporary directory (including pytest `tmp_path`,
`tempfile.TemporaryDirectory`, and raster output). No assertion failure was
reported outside those environment-limited cases.

Additional checks:

```text
.\.venv\Scripts\python.exe -c "import ast, pathlib; paths=['src/coregistration.py','src/multiband_pipeline.py','scripts/diagnose_registration_pair.py','tests/test_registration_quality.py','tests/test_registration_refinement.py','tests/test_diagnostics_regressions.py']; [ast.parse(pathlib.Path(path).read_text(encoding='utf-8'), filename=path) for path in paths]; print('syntax-ok')"
syntax-ok
git diff --check
```

The diff check passed; Git only reported the repository's existing LF/CRLF
conversion warnings.

```text
.\.venv\Scripts\python.exe scripts/diagnose_registration_pair.py --help
exit 0

.\.venv\Scripts\python.exe -m src.multiband_pipeline configs/dz01_mosaic_series_b14.yaml --dry-run
exit 0
```

The preflight command inspected all six scenes and reported consistent
DZ01V sensors, CRS, resolution, and connected N=2/N=4/N=6 overlap graphs, but
the process returned 1 while rendering its final check-mark summary because
the Windows `gbk` console codec could not encode `\u2713`.

## Concerns

- The host temporary-directory and console-encoding restrictions remain
  external environment issues; rerun the temp-writing tests in a normal
  writable/UTF-8 environment before release.
- The remaining review notes about broader mocked orchestration coverage and
  local smoothing-candidate selection were not changed in this bounded wave;
  changing those would broaden behavior beyond the requested Critical and
  Important fixes. Its existing thresholds, output schema, and normal
  registration behavior were otherwise preserved.
- Pre-existing untracked review artifacts and `data/report_figures/` were not
  staged or modified.
