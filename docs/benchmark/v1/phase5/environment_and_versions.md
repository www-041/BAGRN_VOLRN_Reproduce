# Phase5 external method environment and adapter record

Snapshot date: 2026-10-11. This is an environment/preflight record, not an algorithm run.

## Observed local environment

The interpreter used for this check is `E:\python3.11\python.exe` (Python 3.11.6). `karios`, `arosics`, `geoarray`, and `osgeo` are not importable and no matching distributions are installed. `conda` is not on `PATH`. No packages were installed and `.venv-registration` was not modified.

| Method | Interface implemented | Local environment | Verification status |
|---|---|---|---|
| KARIOS (KLT) | Isolated worker, shared-view GeoTIFFs and 0/1 joint-valid mask, strict semicolon CSV parser, native-coordinate restoration | `ENV_UNAVAILABLE`; KARIOS, GDAL bindings and Conda absent; upstream currently documents Python 3.12+ | Synthetic parser/subprocess contract only; not library end-to-end verified |
| AROSICS `COREG` global | Isolated worker, fixed 256x256 center window, max_iter=5, max_shift=5, direct shift parser gated by a synthetic native sign check | `ENV_UNAVAILABLE`; AROSICS, GeoArray and GDAL bindings absent | Synthetic conversion/parser only; library sign check and real data call not run |
| AROSICS `COREG_LOCAL` | Not part of this adapter or main table | Not checked/installed | Out of scope; distinct local model |

The frozen config currently records `external_python: {KARIOS: null, AROSICS_GLOBAL: null}` because neither external environment exists. `run_external` refuses an unconfigured interpreter, requires any configured executable to be an absolute path exactly matching the requested executable, and rejects the active interpreter and `.venv-registration`. Setting a path requires a reviewed new config/version and a new config SHA256 before use. The runner writes staged inputs and logs into a new method-specific output directory; it does not create or mutate an environment. A future isolated environment should live under the Phase5 output tree, outside `.venv-registration`, with Python/package versions, source revision, dependencies, and license metadata recorded before use. The resolved external interpreter path is retained in attempted-run provenance, including failures.

## KARIOS contract

The official KARIOS README documents a dedicated Conda environment and Python 3.12+, same footprint/geotransform/projection/resolution for monitored, reference, and mask inputs, and integer-valued image inputs. The adapter therefore stages both images on the same Phase5 overlap grid, scales the already-frozen display view to uint8 [0,255], and writes a joint validity mask with 0=excluded and 1=included. Invocation uses scene_j as monitored and scene_i as reference, omits `--enable-large-shift-detection`, and preserves the package's default processing JSON and its SHA256 in worker provenance.

The documented CSV is semicolon-delimited: `x0,y0` are zero-based reference-image column/row, and `dx,dy` are monitored-minus-reference displacement, with positive dx east/column and positive dy south/row. The parser reconstructs the scene-j point as `(x0+dx, y0+dy)`, then converts each side independently to its native pixel-center coordinates and checks each scene's own validity mask. Score is preserved only as match metadata; it does not weight the common translation fit.

Source: [KARIOS official repository and README](https://github.com/telespazio-tim/karios) (Apache-2.0, as stated by the repository). Upstream main is not treated as an installed/pinned version; the local package version is `NOT_INSTALLED`.

## AROSICS contract

The official AROSICS 1.13.2 documentation describes `COREG` as a global X/Y shift estimated from one matching window. This adapter uses only `COREG`, with one fixed shared-overlap center window, and never calls `COREG_LOCAL`. AROSICS performs its native global solver; its result is not produced by the shared feature-correspondence RANSAC, so the six-way table must not be interpreted as isolating feature extraction alone.

`x_shift_px`/`y_shift_px` are treated as the target-geocoding correction, not observed scene-i-to-scene-j feature motion. The isolated worker first runs a deterministic synthetic +2 row/+3 column target-content shift and requires the returned correction to be approximately (-3,-2) pixels; a failed sign check blocks processing. The controller then negates that correction, maps through the common-view affine, and combines it with the GEOREF_ONLY native-origin offset. This rule remains subject to first verification in the pinned AROSICS environment before any real smoke.

Source: [AROSICS official global COREG documentation](https://danschef.git-pages.gfz-potsdam.de/arosics/doc/usage/global_coreg.html). The documentation currently identifies version 1.13.2; local AROSICS version is `NOT_INSTALLED`.

## Limits and next compatibility gate

- No KARIOS or AROSICS package was imported in this process; pure parser tests and a mocked controller subprocess do not establish third-party compatibility.
- Before either method is used on B9, install into a separate, explicitly named environment outside `.venv-registration`; pin and record the exact source/package revision and license, run `--help`/API smoke checks, and verify the staged mask/georeferencing and output schema on synthetic data.
- If an installed version's arguments/output or shift convention differs from the adapter contract, classify it `ENV_UNAVAILABLE` / `UNSUPPORTED_ADAPTER_VERSION`; do not silently change protocol values or substitute another algorithm.
- No model, KARIOS, AROSICS, Task15, Task16, or radiometric experiment was run for this record.
