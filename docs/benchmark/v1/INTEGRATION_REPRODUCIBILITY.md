# Integration branch package and reproducibility boundary

This branch consolidates the existing BAGRN/VOLRN project with the DZ01V Benchmark V1 additions, LIRRN Python port, LSSTAA adapted Python implementation, Spectralmatch Phase 2 adapter/diagnostics, Phase 5 registration adapters, R1/R1.5 tools, tests, configs, and their compact protocols/reports. The historical freeze recovery commits are retained in branch history.

## Data boundary

The original 13-scene GeoTIFFs, aligned raster inputs/masks, algorithm outputs, temporary files, virtual environments, and run logs are not part of this Git branch. They remain external inputs and must be transferred to the cloud host separately with SHA256 verification. No `data/output` experiment raster was staged.

The small files in `unified_v1/freeze_20261013_v3/` are byte-for-byte copies of the freeze audit manifests and fixed inventories, not a replacement freeze and not a regenerated protocol. The original frozen lock keeps its original absolute file paths; the local 54-file lock check therefore requires the original data workspace, while the 38-entry code manifest is checked against this clean checkout. Do not rewrite those paths or regenerate the freeze to make a cloud check pass.

The phase-2/phase-3 checkpoint CSVs and review logs are included as small benchmark records. They preserve the recorded first-review state; TEST candidates that still require an independent second reviewer are not thereby promoted to final test truth.

## Reproducibility status

- Core BAGRN/VOLRN implementations and historical Task15/Task16 entry points are retained from the repository history.
- Unified V1 protocol, score paths, freeze manifests, and test suites are included.
- LIRRN is the Python port and retains its explicit official-equivalence status; do not label it official-equivalent unless its comparison gate passes.
- LSSTAA is an adapted Python implementation marked `LSSTAA_PY_ADAPTED_DN_PROXY_UNVERIFIED`; its three-scene exploratory output is diagnostic only. The external official source is identified by repository URL/commit in the LSSTAA reports and is not vendored here.
- Spectralmatch Phase 2 remains environment-blocked/no-go per its report. Its diagnostics and adapter code do not imply a successful full-data run.
- No claim of Linux compatibility or cloud resource readiness is made by this integration commit. Run the documented test suite and environment checks on the target Linux image before experiments.

## Suggested clean-checkout checks

Use the provided `requirements.txt` and, only when registration adapters are needed, `requirements-registration.txt`. Install into a disposable environment. Run `pytest -q`, `python -m compileall -q src scripts tests`, and `git diff --check`; run the Unified V1 freeze/code-manifest audit against the mounted, read-only original freeze separately. Do not include external raster data in Git.
