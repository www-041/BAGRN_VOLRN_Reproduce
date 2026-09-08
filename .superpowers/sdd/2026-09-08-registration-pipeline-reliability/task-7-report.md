# Task 7 report — registration diagnostic outputs

## Summary

Implemented Task 7 with the exact requested commit message:

`fix: repair registration diagnostic outputs`

The registration-only diagnostic now consumes the actual `register_scenes()` result schema, writes a JSON diagnostic payload, sets the selected first scene as the control scene before pipeline construction, and writes registered-image, overlay, and diagnostic-mosaic artifacts from `registered_arrays`. The quality-gate documentation now identifies independent final validation as the authoritative quality evidence.

## Files changed

- `scripts/diagnose_registration_pair.py`
  - Added `build_diagnostic_payload(registration, scene_ids, output_dir)`.
  - Preserves supplied `scene_ids`, `quality`, and `final_validation` in the returned payload.
  - Serializes raw block matches, robust pair measurements, initial/final shifts, refinement history, local CV/field diagnostics, final validation, and uppercase PASS/WARN/FAIL classification.
  - Writes `registration_diagnostics.json`.
  - Sets `config.control_scene` before constructing `MultibandPipeline`.
  - Writes registered reference/target GeoTIFFs, a red-green overlay, and a weighted diagnostic mosaic by default.
  - Adds explicit `--mosaic-mode source_selection` support without using that mode implicitly.
- `tests/test_diagnostics_regressions.py`
  - Added the plan’s actual-registration-schema regression using `build_diagnostic_payload()`.
- `docs/REGISTRATION_QUALITY_GATE.md`
  - Documents final independent validation as final quality evidence.
  - Removes the implication that pre-warp pair shifts are final quality evidence.
  - Documents diagnostic artifacts and explicit mosaic-mode selection.
- `.superpowers/sdd/2026-09-08-registration-pipeline-reliability/task-7-report.md`
  - This report.

## Tests and verification

- `pytest tests/test_diagnostics_regressions.py::test_registration_diagnostic_uses_actual_registration_schema -v`
  - Blocked during pytest fixture setup by the known Windows `tmp_path` permission error (`WinError 5` while scanning `C:\Users\wang\AppData\Local\Temp\pytest-of-wang`).
- `pytest tests/test_diagnostics_regressions.py -v`
  - 3 existing diagnostics tests passed; the new test hit the same environment-level `tmp_path` error.
- `pytest tests/test_diagnostics_regressions.py -k "identity_gain or dynamic_range" -v`
  - 3 passed.
- `pytest tests/test_script_contracts.py -v`
  - 1 passed.
- `E:\python3\python3.exe -m py_compile scripts/diagnose_registration_pair.py`
  - Passed.
- `E:\python3\python3.exe scripts/diagnose_registration_pair.py --help`
  - Passed and shows the new `--mosaic-mode` option.
- Direct schema check with the plan’s registration fields
  - Passed; confirmed returned `quality`/`final_validation` preservation and creation of `registration_diagnostics.json`.
- Synthetic artifact check with two registered arrays
  - Passed; confirmed registered reference, registered target, red-green overlay, and diagnostic mosaic files were written.
- Synthetic CLI/control-scene check
  - Passed; confirmed the selected first scene ID is assigned to `config.control_scene` before pipeline construction.
- `git diff --check`
  - Passed; only normal line-ending conversion warnings were reported.

## Self-review

- No registration or radiometric-normalization formulas were changed.
- Diagnostic mosaicking receives the registered arrays returned by registration, never normalized arrays or intermediate warp inputs.
- `source_selection` is only passed when selected through the explicit CLI option; default behavior remains weighted.
- Non-finite diagnostics are converted to JSON `null` only for on-disk serialization so the returned quality and validation dictionaries remain unchanged.
- Prior registration work and unrelated untracked artifacts were preserved. No subagents were spawned, and nothing was pushed or merged.

## Concerns

The focused pytest regression cannot complete in this managed environment because pytest’s temporary-directory fixture is denied access before the test body runs. The equivalent schema assertion and JSON output were executed directly with the project’s rasterio-enabled Python runtime, and the adjacent diagnostics tests passed.
