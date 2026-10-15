# Phase5 adapter status

Snapshot: 2026-10-11. Status is based on source inspection, dependency preflight and synthetic contract tests. Module detection is not end-to-end verification.

| Method | Adapter state | Current evidence and limitation |
|---|---|---|
| SIFT | `CONTRACT_TESTED; REAL_SMOKE_BLOCKED` | In-repository matcher wrapper and synthetic/common-coordinate tests. Real `00_01` smoke was not launched because available memory was below the frozen resource budget. |
| KARIOS (KLT) | `ENV_UNAVAILABLE` | Isolated worker/parser contract tests only. KARIOS, GDAL bindings and Conda are absent; no compatible isolated Python 3.12+ environment is configured. |
| AROSICS global `COREG` | `ENV_UNAVAILABLE` | Isolated worker/parser and synthetic shift-sign contract tests only. AROSICS, GeoArray and GDAL bindings are absent; external interpreter is unconfigured. Uses native COREG solver, not shared feature RANSAC. |
| LoFTR | `CONTRACT_TESTED; NOT_END_TO_END_VERIFIED` | Adapter/dependency discovery and synthetic interface tests; no B9 inference or weight validation run. |
| EfficientLoFTR | `CONTRACT_TESTED; WEIGHTS_UNAVAILABLE_OR_UNVERIFIED` | Interface tests only. Required repository/weight identity was not verified as available for this Phase5 protocol. |
| LightGlue + DISK | `CONTRACT_TESTED; NOT_END_TO_END_VERIFIED` | Adapter/dependency discovery and synthetic interface tests; no B9 inference or weight validation run. |
| `GEOREF_ONLY` | `DIAGNOSTIC_BASELINE` | Affine-based native-grid mapping and scoring interface; not a Matcher, no fabricated matches, excluded from the six-method ranking. |

The read-only Phase5 preflight found the SIFT, LoFTR, EfficientLoFTR and LightGlue+DISK Python modules in `.venv-registration`; KARIOS and AROSICS were unavailable. Module importability alone does not establish correct model weights, successful inference or scientific performance. External methods require separately configured environments and reviewed provenance. See [environment and versions](environment_and_versions.md) and [geometry output protocol](geometry_output_protocol.md).

No algorithm was substituted or silently omitted from a run. Future unavailable/failed attempts must be retained with their classified status and null metrics.
